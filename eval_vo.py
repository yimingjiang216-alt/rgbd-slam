# -*- coding: utf-8 -*-
"""
在 TUM RGB-D 数据集上评测视觉里程计(VO)前端 + 保存 BA 所需数据。
- 跑单目 VO 得到估计轨迹
- 与 groundtruth 做 Umeyama 对齐(解决单目尺度不确定性 + 坐标系差异)
- 计算 ATE(绝对轨迹误差) 并画轨迹对比图
- 额外保存 poses / points / observations 到 npz，供 ba.py 做光束法平差

用法: python eval_vo.py <TUM序列目录> [最大帧数]
"""
import sys, os
import numpy as np
import cv2

TUM_FR1_K = np.array([[517.3, 0, 318.6],
                      [0, 516.5, 255.3],
                      [0, 0, 1]], dtype=np.float64)


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
            ts = float(p[0])
            t = np.array([float(p[1]), float(p[2]), float(p[3])])
            q = np.array([float(p[4]), float(p[5]), float(p[6]), float(p[7])])
            gt.append((ts, t, q))
    return gt


def quat_to_R(q):
    x, y, z, w = q
    n = np.sqrt(x*x + y*y + z*z + w*w)
    x, y, z, w = x/n, y/n, z/n, w/n
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)],
    ])


def match_gt(ts, gt):
    arr = np.array([g[0] for g in gt])
    i = int(np.argmin(np.abs(arr - ts)))
    return gt[i]


def run_vo(img_paths, K, orb=None):
    """跑单目 VO，返回估计轨迹(N,3) 以及供 BA 使用的观测数据"""
    if orb is None:
        orb = cv2.ORB_create(nfeatures=2000)
    pose = np.eye(4)
    traj = [pose[:3, 3].copy()]

    # --- BA 数据容器 ---
    poses = [pose.copy()]          # 每帧 4x4 位姿(世界<-相机)
    points = []                    # 三维点
    observations = []              # [frame_idx, point_idx, u, v]

    prev = cv2.imread(img_paths[0], cv2.IMREAD_GRAYSCALE)
    kp1, des1 = orb.detectAndCompute(prev, None)

    for i in range(1, len(img_paths)):
        cur = cv2.imread(img_paths[i], cv2.IMREAD_GRAYSCALE)
        kp2, des2 = orb.detectAndCompute(cur, None)
        if des1 is None or des2 is None:
            traj.append(pose[:3, 3].copy()); poses.append(pose.copy())
            prev, kp1, des1 = cur, kp2, des2; continue

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        ms = sorted(bf.match(des1, des2), key=lambda m: m.distance)
        if len(ms) < 8:
            traj.append(pose[:3, 3].copy()); poses.append(pose.copy())
            prev, kp1, des1 = cur, kp2, des2; continue

        p1 = np.float32([kp1[m.queryIdx].pt for m in ms])
        p2 = np.float32([kp2[m.trainIdx].pt for m in ms])
        E, mask = cv2.findEssentialMat(p1, p2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None:
            traj.append(pose[:3, 3].copy()); poses.append(pose.copy())
            prev, kp1, des1 = cur, kp2, des2; continue

        _, R, t, _ = cv2.recoverPose(E, p1, p2, K)
        T_rel = np.eye(4); T_rel[:3, :3] = R; T_rel[:3, 3] = t.ravel()
        pose_prev = pose.copy()
        pose = pose @ T_rel
        traj.append(pose[:3, 3].copy())
        poses.append(pose.copy())

        # ---- 三角化本帧观测(用于 BA) ----
        inl = mask.ravel() == 1
        P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = K @ np.hstack([R, t])
        pts4d = cv2.triangulatePoints(P1, P2, p1[inl].T.astype(np.float64), p2[inl].T.astype(np.float64))
        pts3d = (pts4d[:3] / pts4d[3]).T
        # 把「上一帧相机坐标」下的点转到世界坐标
        Rwp = pose_prev[:3, :3]; twp = pose_prev[:3, 3]
        pts_world = (Rwp @ pts3d.T).T + twp
        for k in range(len(pts_world)):
            if not np.all(np.isfinite(pts_world[k])):
                continue
            pid = len(points)
            points.append(pts_world[k])
            observations.append([i-1, pid, float(p1[inl][k][0]), float(p1[inl][k][1])])
            observations.append([i,   pid, float(p2[inl][k][0]), float(p2[inl][k][1])])

        if i % 20 == 0:
            print(f"  frame {i}/{len(img_paths)-1}  matched={len(ms)} inliers={int(mask.sum())}")
        prev, kp1, des1 = cur, kp2, des2

    return np.array(traj), np.array(poses), np.array(points, dtype=np.float64), np.array(observations, dtype=np.float64)


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
    var_s = (Sc ** 2).sum() / n
    s = (S * d).sum() / var_s
    t = mu_d - s * R @ mu_s
    return R, t, s


def ate_of(est, gt_xyz, tag=""):
    R, t, s = umeyama_align(est, gt_xyz)
    aligned = (s * (R @ est.T).T) + t
    err = np.linalg.norm(aligned - gt_xyz, axis=1)
    rmse = float(np.sqrt((err ** 2).mean()))
    print(f"[{tag}] scale={s:.4f}  ATE_RMSE={rmse:.4f} m  ATE_mean={err.mean():.4f} m")
    return aligned, rmse


def main(seq_dir, max_frames=200):
    rgb = read_tum_list(os.path.join(seq_dir, "rgb.txt"))
    gt  = read_groundtruth(os.path.join(seq_dir, "groundtruth.txt"))
    print("rgb frames:", len(rgb), "| gt samples:", len(gt))

    rgb = rgb[:max_frames]
    img_paths = [os.path.join(seq_dir, r[1]) for r in rgb]
    img_paths = [p for p in img_paths if os.path.exists(p)]
    print("将处理帧数:", len(img_paths))

    print("running VO ...")
    est, poses, points, obs = run_vo(img_paths, TUM_FR1_K)
    print("VO done. poses=%d points=%d obs=%d" % (len(poses), len(points), len(obs)))

    # 真值对齐
    gt_xyz = []
    for ts, _ in [(r[0], r[1]) for r in rgb[:len(est)]]:
        g = match_gt(ts, gt)
        gt_xyz.append(g[1])
    gt_xyz = np.array(gt_xyz)

    print("\n===== 优化前 (VO 前端) =====")
    aligned_before, rmse_before = ate_of(est, gt_xyz, "before")

    # 保存 BA 数据
    np.savez(os.path.join(seq_dir, "ba_data.npz"),
             poses=poses, points=points, obs=obs, K=TUM_FR1_K)
    np.savetxt(os.path.join(seq_dir, "vo_traj_gt.txt"), gt_xyz)
    np.savetxt(os.path.join(seq_dir, "vo_traj_est_before.txt"), aligned_before)
    print("BA 数据已存: ba_data.npz")

    # 画对比图
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 6))
    plt.plot(gt_xyz[:, 0], gt_xyz[:, 2], 'g-', lw=2, label="ground truth")
    plt.plot(aligned_before[:, 0], aligned_before[:, 2], 'b--', lw=2, label="VO front-end")
    plt.scatter(gt_xyz[0, 0], gt_xyz[0, 2], c='red', s=70, zorder=5)
    plt.xlabel("X (m)"); plt.ylabel("Z (m)")
    plt.title("VO vs Ground Truth (ATE RMSE=%.3f m)" % rmse_before)
    plt.legend(); plt.grid(True); plt.axis("equal")
    out = os.path.join(seq_dir, "vo_vs_gt.png")
    plt.savefig(out, dpi=150)
    print("对比图:", out)


if __name__ == "__main__":
    seq = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\r26304\Documents\codex-shop\slam_vo\data\rgbd_dataset_freiburg1_xyz"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    main(seq, n)
