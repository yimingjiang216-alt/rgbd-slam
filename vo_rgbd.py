# -*- coding: utf-8 -*-
"""
RGB-D 视觉里程计 (修真正版)
关键修正: 上一版 vo_map.py 用单目 recoverPose, 返回的 t 是单位向量(只有方向没有尺度),
          导致每帧位移恒为 1.0, 位姿是随机游走, 根本不是真轨迹。
本版: 用深度图把像素反投影成真实三维点, 用 3D-2D PnP 求位姿, 尺度由深度给出, 无尺度歧义。
流程: ORB 特征 -> 与上一帧匹配 -> 用上一帧深度反投影得 3D 点 -> solvePnPRansac 求当前帧位姿
      -> 三角化/深度新点入库 -> 多帧观测记录(供 BA 使用)
用法: python vo_rgbd.py <序列目录> [最大帧数] [地图点上限]
"""
import sys, os
import numpy as np
import cv2
from collections import defaultdict

TUM_FR1_K = np.array([[517.3, 0, 318.6],
                      [0, 516.5, 255.3],
                      [0, 0, 1]], np.float64)
DEPTH_SCALE = 5000.0   # TUM 深度 PNG: 深度值/5000 = 米


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


def main(seq_dir, max_frames=798, max_map=40000, verbose=True):
    rgb = read_tum_list(os.path.join(seq_dir, "rgb.txt"))
    depth = read_tum_list(os.path.join(seq_dir, "depth.txt"))
    gt = read_groundtruth(os.path.join(seq_dir, "groundtruth.txt"))
    rgb = rgb[:max_frames]

    # 深度时间戳索引, 每帧配最近深度图
    dts = np.array([d[0] for d in depth])
    K = TUM_FR1_K
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    orb = cv2.ORB_create(nfeatures=2000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def load_depth(ts):
        i = int(np.argmin(np.abs(dts - ts)))
        p = os.path.join(seq_dir, depth[i][1])
        d = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if d is None:
            return None
        return d.astype(np.float32) / DEPTH_SCALE

    # 位姿 = 相机在世界系下的 T_wc (4x4)。世界系 = 第一帧相机系
    Twc = np.eye(4)
    poses = [Twc.copy()]

    map_xyz = []      # 世界系三维点
    map_des = []      # 描述子
    obs = []          # [frame, point_id, u, v]
    mp_last = []

    img0_path = os.path.join(seq_dir, rgb[0][1])
    prev = cv2.imread(img0_path, cv2.IMREAD_GRAYSCALE)
    kp1, des1 = orb.detectAndCompute(prev, None)
    depth1 = load_depth(rgb[0][0])

    n_pnp_fail = 0
    for i in range(1, len(rgb)):
        cur = cv2.imread(os.path.join(seq_dir, rgb[i][1]), cv2.IMREAD_GRAYSCALE)
        if cur is None:
            poses.append(Twc.copy()); continue
        kp2, des2 = orb.detectAndCompute(cur, None)
        depth2 = load_depth(rgb[i][0])

        if des1 is None or des2 is None or depth1 is None or des2 is None or len(kp1) < 10:
            poses.append(Twc.copy())
            prev, kp1, des1, depth1 = cur, kp2, des2, depth2
            continue

        # --- 邻帧匹配 ---
        ms = bf.knnMatch(des1, des2, k=2)
        good = [m for m, nb in ms if m.distance < 0.75 * nb.distance]

        # --- 用"上一帧深度"把匹配点反投影为 3D (在上一帧相机系), 转到世界系 ---
        obj = []; img = []; dsc = []
        for m in good:
            u1, v1 = kp1[m.queryIdx].pt
            u2, v2 = kp2[m.trainIdx].pt
            dval = depth1[int(round(v1)), int(round(u1))]
            if not np.isfinite(dval) or dval <= 0.1 or dval > 12.0:
                continue
            Xc = np.array([(u1 - cx) * dval / fx,
                           (v1 - cy) * dval / fy,
                           dval])
            Xw = Twc[:3, :3] @ Xc + Twc[:3, 3]      # 世界系
            obj.append(Xw); img.append([u2, v2]); dsc.append(des2[m.trainIdx])

        # 记录内点对应的 keypoint 下标, 供取描述子用
        kp2_idx = []
        ok_frame = False
        if len(obj) >= 12:
            obj = np.array(obj, np.float64); img = np.array(img, np.float64); dsc = np.array(dsc, np.uint8)
            ok, rvec, tvec, inl = cv2.solvePnPRansac(
                obj, img, K, None, flags=cv2.SOLVEPNP_ITERATIVE,
                reprojectionError=2.0, iterationsCount=200, confidence=0.999)
            if ok and inl is not None and len(inl) >= 10:
                R, _ = cv2.Rodrigues(rvec)
                T = np.eye(4); T[:3, :3] = R; T[:3, 3] = tvec.ravel()
                Twc_prev = Twc
                Twc = T                      # T 就是当前帧相机->世界 = T_wc
                poses.append(Twc.copy())
                ok_frame = True
                # 用内点建立新地图点(反投影到世界系, 用自己的深度更准)
                inl = inl.ravel()
                if depth2 is not None:
                    for j in inl:
                        u2, v2 = img[j]
                        ui, vi = int(round(u2)), int(round(v2))
                        if not (0 <= vi < depth2.shape[0] and 0 <= ui < depth2.shape[1]):
                            continue
                        dval = depth2[vi, ui]
                        if not np.isfinite(dval) or dval <= 0.1 or dval > 12.0:
                            continue
                        Xc = np.array([(u2 - cx) * dval / fx,
                                       (v2 - cy) * dval / fy,
                                       dval])
                        Xw = Twc[:3, :3] @ Xc + Twc[:3, 3]
                        pid = len(map_xyz)
                        map_xyz.append(Xw)
                        map_des.append(dsc[j])
                        mp_last.append(i)
                        obs.append([i, pid, float(u2), float(v2)])
        if not ok_frame:
            n_pnp_fail += 1
            poses.append(Twc.copy())

        # --- 已有地图点重投影匹配, 追加多帧观测 ---
        if len(map_xyz) > 10 and depth2 is not None:
            recent = np.argsort(np.array(mp_last))[-max_map:]
            W = np.array([map_xyz[j] for j in recent])
            Rcw = Twc[:3, :3].T; tcw = -Rcw @ Twc[:3, 3]
            C = (Rcw @ W.T).T + tcw
            ok2 = C[:, 2] > 0.3
            if ok2.sum() > 5:
                cp = C[ok2]
                uv = np.stack([fx * cp[:, 0] / cp[:, 2] + cx,
                               fy * cp[:, 1] / cp[:, 2] + cy], 1)
                H, Wd = cur.shape
                iv = (uv[:, 0] >= 0) & (uv[:, 0] < Wd) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
                sel = np.where(ok2)[0][iv]
                if len(sel) > 5:
                    sub = recent[sel]
                    desM = np.array([map_des[j] for j in sub], np.uint8)
                    try:
                        mm = bf.match(desM, des2)
                    except cv2.error:
                        mm = []
                    for m in mm:
                        if m.distance > 40:
                            continue
                        gi = int(sub[m.queryIdx])
                        u, v = kp2[m.trainIdx].pt
                        dval = depth2[int(round(v)), int(round(u))]
                        # 深度一致性检查: 观测深度应与三角化深度接近
                        Cc = Rcw @ map_xyz[gi] + tcw
                        if Cc[2] > 0 and abs(dval - Cc[2]) / max(Cc[2], 1e-3) < 0.15:
                            obs.append([i, gi, float(u), float(v)])
                            mp_last[gi] = i

        if verbose and i % 50 == 0:
            print("  frame %4d/%4d landmarks=%d obs=%d pnp_fail=%d" %
                  (i, len(rgb) - 1, len(map_xyz), len(obs), n_pnp_fail), flush=True)
        prev, kp1, des1, depth1 = cur, kp2, des2, depth2

    poses_arr = np.array(poses)
    points_arr = np.array(map_xyz, np.float64) if map_xyz else np.zeros((0, 3))
    obs_arr = np.array(obs, np.float64) if obs else np.zeros((0, 4))
    print("landmarks=%d obs=%d pnp_fail=%d" % (len(points_arr), len(obs_arr), n_pnp_fail), flush=True)

    np.savez(os.path.join(seq_dir, "rgbd_data.npz"),
             poses=poses_arr, points=points_arr, obs=obs_arr, K=K)

    est = poses_arr[:, :3, 3]
    gt_xyz = np.array([match_gt(ts, gt)[1] for ts, _ in rgb[:len(est)]])
    np.savetxt(os.path.join(seq_dir, "rgbd_traj_gt.txt"), gt_xyz)
    R_, t_, s_ = umeyama_align(est, gt_xyz)
    aligned = (s_ * (R_ @ est.T).T) + t_
    err = np.linalg.norm(aligned - gt_xyz, axis=1)
    rmse = float(np.sqrt((err ** 2).mean()))
    print("\n===== RGB-D VO =====")
    print("scale=%.5f ATE_RMSE=%.4f m ATE_mean=%.4f m" % (s_, rmse, err.mean()))
    print("traj_len(est)=%.3f m  traj_len(gt)=%.3f m" %
          (np.linalg.norm(np.diff(aligned, axis=0), axis=1).sum(),
           np.linalg.norm(np.diff(gt_xyz, axis=0), axis=1).sum()))
    np.savetxt(os.path.join(seq_dir, "rgbd_traj_est.txt"), aligned)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 6))
    plt.plot(gt_xyz[:, 0], gt_xyz[:, 2], 'g-', lw=2, label="ground truth")
    plt.plot(aligned[:, 0], aligned[:, 2], 'b--', lw=2, label="RGB-D VO (ATE=%.3f m)" % rmse)
    plt.xlabel("X (m)"); plt.ylabel("Z (m)")
    plt.title("RGB-D VO vs Ground Truth (TUM fr1/xyz)")
    plt.legend(); plt.grid(True); plt.axis("equal")
    plt.savefig(os.path.join(seq_dir, "rgbd_vo_vs_gt.png"), dpi=150)
    print("图:", os.path.join(seq_dir, "rgbd_vo_vs_gt.png"))



if __name__ == "__main__":
    seq = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\r26304\Documents\codex-shop\slam_vo\data\rgbd_dataset_freiburg1_xyz"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 798
    mm = int(sys.argv[3]) if len(sys.argv) > 3 else 40000
    main(seq, n, mm)
