# -*- coding: utf-8 -*-
"""
VO 前端 + 地图点维护(高效版, 为 BA 提供多帧重复观测)
关键优化: 地图点数量上限 + 描述子用 numpy 数组 + 只在需要时匹配
用法: python vo_map.py <TUM序列目录> [最大帧数=200] [地图点上限=6000]
"""
import sys, os
import numpy as np
import cv2

TUM_FR1_K = np.array([[517.3, 0, 318.6],
                      [0, 516.5, 255.3],
                      [0, 0, 1]], np.float64)


def read_tum_list(path):
    items = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split()
        if len(p) >= 2:
            items.append((float(p[0]), p[1]))
    return items


def read_groundtruth(path):
    gt = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = line.split()
        if len(p) >= 8:
            gt.append((float(p[0]),
                       np.array([float(p[1]), float(p[2]), float(p[3])]),
                       np.array([float(p[4]), float(p[5]), float(p[6]), float(p[7])])))
    return gt


def match_gt(ts, gt):
    arr = np.array([g[0] for g in gt])
    return gt[int(np.argmin(np.abs(arr - ts)))]


def umeyama_align(src, dst):
    n = src.shape[0]
    mu_s = src.mean(0); mu_d = dst.mean(0)
    Sc = src - mu_s; Dc = dst - mu_d
    C = Dc.T @ Sc / n
    U, S, Vt = np.linalg.svd(C)
    d = np.ones(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        d[2] = -1
    R = U @ np.diag(d) @ Vt
    s = (S * d).sum() / ((Sc ** 2).sum() / n)
    return R, mu_d - s * R @ mu_s, s


def main(seq_dir, max_frames=200, max_map=6000):
    rgb = read_tum_list(os.path.join(seq_dir, "rgb.txt"))
    gt = read_groundtruth(os.path.join(seq_dir, "groundtruth.txt"))
    rgb = rgb[:max_frames]
    img_paths = [os.path.join(seq_dir, r[1]) for r in rgb]
    img_paths = [p for p in img_paths if os.path.exists(p)]
    print("frames to process:", len(img_paths), flush=True)

    orb = cv2.ORB_create(nfeatures=1500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    K = TUM_FR1_K

    pose = np.eye(4)
    poses = [pose.copy()]

    mp_xyz = []                 # list of (3,)
    mp_des = []                 # list of uint8(32,)
    mp_lastobs = []             # 上次被观测的帧号
    obs = []

    prev = cv2.imread(img_paths[0], cv2.IMREAD_GRAYSCALE)
    kp1, des1 = orb.detectAndCompute(prev, None)

    for i in range(1, len(img_paths)):
        cur = cv2.imread(img_paths[i], cv2.IMREAD_GRAYSCALE)
        kp2, des2 = orb.detectAndCompute(cur, None)
        if des1 is None or des2 is None:
            poses.append(pose.copy()); prev, kp1, des1 = cur, kp2, des2; continue

        # --- 位姿: 邻帧匹配 ---
        ms = bf.knnMatch(des1, des2, k=2)
        good = [m for m, n in ms if m.distance < 0.75 * n.distance]
        if len(good) < 8:
            poses.append(pose.copy()); prev, kp1, des1 = cur, kp2, des2; continue
        p1 = np.float32([kp1[m.queryIdx].pt for m in good])
        p2 = np.float32([kp2[m.trainIdx].pt for m in good])
        E, mask = cv2.findEssentialMat(p1, p2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None:
            poses.append(pose.copy()); prev, kp1, des1 = cur, kp2, des2; continue
        _, R, t, _ = cv2.recoverPose(E, p1, p2, K)
        T_rel = np.eye(4); T_rel[:3, :3] = R; T_rel[:3, 3] = t.ravel()
        pose_prev = pose.copy()
        pose = pose @ T_rel
        poses.append(pose.copy())

        # --- 已有地图点重投影匹配(只取最近被观测的 max_map 个点) ---
        if mp_xyz:
            recent = np.argsort(np.array(mp_lastobs))[-max_map:]
            W = np.array([mp_xyz[j] for j in recent])          # (M,3)
            Rcw = pose[:3, :3]; tcw = pose[:3, 3]
            C = (Rcw.T @ (W - tcw).T).T
            ok = C[:, 2] > 0.3
            if ok.sum() > 5:
                cp = C[ok]
                uv = np.stack([K[0,0]*cp[:,0]/cp[:,2] + K[0,2],
                               K[1,1]*cp[:,1]/cp[:,2] + K[1,2]], axis=1)
                H, Wd = cur.shape
                iv = (uv[:,0]>=0)&(uv[:,0]<Wd)&(uv[:,1]>=0)&(uv[:,1]<H)
                sel = np.where(ok)[0][iv]
                if len(sel) > 5:
                    sub = recent[sel]
                    desM = np.array([mp_des[j] for j in sub], np.uint8)
                    try:
                        mm = bf.match(desM, des2)
                    except cv2.error:
                        mm = []
                    for m in mm:
                        if m.distance > 40:
                            continue
                        gi = int(sub[m.queryIdx])
                        u, v = kp2[m.trainIdx].pt
                        obs.append([i, gi, float(u), float(v)])
                        mp_lastobs[gi] = i

        # --- 三角化新点 ---
        inl = mask.ravel() == 1
        ms_inl = [good[j] for j in np.where(inl)[0]]
        P1 = K @ np.hstack([np.eye(3), np.zeros((3,1))])
        P2 = K @ np.hstack([R, t])
        pts4d = cv2.triangulatePoints(P1, P2, p1[inl].T.astype(np.float64), p2[inl].T.astype(np.float64))
        pts3d = (pts4d[:3] / pts4d[3]).T
        Rwp = pose_prev[:3,:3]; twp = pose_prev[:3,3]
        pts_world = (Rwp @ pts3d.T).T + twp
        for k in range(len(pts_world)):
            if not np.all(np.isfinite(pts_world[k])):
                continue
            pid = len(mp_xyz)
            mp_xyz.append(pts_world[k])
            mp_des.append(des2[ms_inl[k].trainIdx])
            mp_lastobs.append(i)
            obs.append([i-1, pid, float(p1[inl][k][0]), float(p1[inl][k][1])])
            obs.append([i,   pid, float(p2[inl][k][0]), float(p2[inl][k][1])])

        if i % 20 == 0:
            print("  frame %d/%d landmarks=%d obs=%d" % (i, len(img_paths)-1, len(mp_xyz), len(obs)), flush=True)
        prev, kp1, des1 = cur, kp2, des2

    poses_arr = np.array(poses)
    points_arr = np.array(mp_xyz, np.float64) if mp_xyz else np.zeros((0,3))
    obs_arr = np.array(obs, np.float64) if obs else np.zeros((0,4))
    # 统计多点观测
    from collections import Counter
    cc = Counter(int(o[1]) for o in obs)
    multi = sum(1 for v in cc.values() if v >= 3)
    print("landmarks=%d obs=%d  landmarks_with_>=3_obs=%d" % (len(points_arr), len(obs_arr), multi), flush=True)

    np.savez(os.path.join(seq_dir, "ba_data.npz"), poses=poses_arr, points=points_arr, obs=obs_arr, K=K)

    est = poses_arr[:, :3, 3]
    gt_xyz = np.array([match_gt(ts, gt)[1] for ts, _ in rgb[:len(est)]])
    np.savetxt(os.path.join(seq_dir, "vo_traj_gt.txt"), gt_xyz)
    R_, t_, s_ = umeyama_align(est, gt_xyz)
    aligned = (s_ * (R_ @ est.T).T) + t_
    err = np.linalg.norm(aligned - gt_xyz, axis=1)
    rmse = float(np.sqrt((err**2).mean()))
    print("\n===== VO 前端 =====")
    print("scale=%.5f ATE_RMSE=%.4f m ATE_mean=%.4f m" % (s_, rmse, err.mean()))
    print("traj_len(gt)=%.3f m" % np.linalg.norm(np.diff(gt_xyz, axis=0), axis=1).sum())
    np.savetxt(os.path.join(seq_dir, "vo_traj_est_before.txt"), aligned)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7,6))
    plt.plot(gt_xyz[:,0], gt_xyz[:,2], 'g-', lw=2, label="ground truth")
    plt.plot(aligned[:,0], aligned[:,2], 'b--', lw=2, label="VO front-end")
    plt.scatter(gt_xyz[0,0], gt_xyz[0,2], c='red', s=70, zorder=5)
    plt.xlabel("X (m)"); plt.ylabel("Z (m)")
    plt.title("VO vs GT (ATE RMSE=%.3f m)" % rmse)
    plt.legend(); plt.grid(True); plt.axis("equal")
    plt.savefig(os.path.join(seq_dir, "vo_vs_gt.png"), dpi=150)
    print("saved:", os.path.join(seq_dir, "vo_vs_gt.png"))


if __name__ == "__main__":
    seq = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\r26304\Documents\codex-shop\slam_vo\data\rgbd_dataset_freiburg1_xyz"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    mm = int(sys.argv[3]) if len(sys.argv) > 3 else 6000
    main(seq, n, mm)
