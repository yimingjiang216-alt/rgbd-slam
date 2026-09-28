# -*- coding: utf-8 -*-
"""
RGB-D 视觉里程计 v2 —— 帧到关键帧(而非帧到帧链式)
上一版失败的根因:
  1) 用"上一帧"做参考做 PnP, 且把上一帧自己已经漂掉的位姿当真值 ->
     误差正反馈, 平移量级被压成 GT 的 1/12, 且逐帧正负抖动(step sum 100m, net 0.068m)
  2) solvePnPRansac 返回的是 T_cw(世界->相机), 代码当成 T_wc 用了, 方向反了
本版修正:
  - 维护关键帧: 关键帧的位姿 = 关键帧建立时的固定位姿, 不随帧漂移
  - 对普通帧, 用"关键帧的世界点 + 当前帧2D观测"做 PnP, 解得 T_cw 后取逆得 T_wc
  - 关键帧切换: 当内点率下降或位移超过阈值时, 用当前帧建立新关键帧
  - 深度图给出真实尺度
用法: python vo_rgbd2.py <序列目录> [最大帧数]
"""
import sys, os
import numpy as np
import cv2

TUM_FR1_K = np.array([[517.3, 0, 318.6],
                      [0, 516.5, 255.3],
                      [0, 0, 1]], np.float64)
DEPTH_SCALE = 5000.0
MIN_DEPTH, MAX_DEPTH = 0.15, 8.0


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
            gt.append((float(p[0]), np.array([float(p[1]), float(p[2]), float(p[3])])))
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


def main(seq_dir, max_frames=798, verbose=True):
    rgb = read_tum_list(os.path.join(seq_dir, "rgb.txt"))[:max_frames]
    depth = read_tum_list(os.path.join(seq_dir, "depth.txt"))
    gt = read_groundtruth(os.path.join(seq_dir, "groundtruth.txt"))
    dts = np.array([x[0] for x in depth])
    K = TUM_FR1_K
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    def load_gray(ts):
        return cv2.imread(os.path.join(seq_dir, rgb_by_ts[ts]), cv2.IMREAD_GRAYSCALE)

    def load_depth(ts):
        i = int(np.argmin(np.abs(dts - ts)))
        d = cv2.imread(os.path.join(seq_dir, depth[i][1]), cv2.IMREAD_UNCHANGED)
        if d is None:
            return None
        return d.astype(np.float32) / DEPTH_SCALE

    rgb_by_ts = {t: p for t, p in rgb}
    orb = cv2.ORB_create(nfeatures=2500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def backproject(u, v, dep):
        """像素+深度 -> 相机系三维点"""
        z = dep[int(round(v)), int(round(u))]
        if not np.isfinite(z) or z < MIN_DEPTH or z > MAX_DEPTH:
            return None
        return np.array([(u - cx) * z / fx, (v - cy) * z / fy, z])

    # ---- 初始化: 第0帧为第一个关键帧, 世界系 = 第0帧相机系 ----
    kf = 0
    Twc_kf = np.eye(4)                 # 关键帧位姿 T_wc
    g0 = load_gray(rgb[0][0]); d0 = load_depth(rgb[0][0])
    kp_kf, des_kf = orb.detectAndCompute(g0, None)
    # 关键帧地图点(世界系)
    kf_pts = []; kf_des = []; kf_uv = []
    for j, k in enumerate(kp_kf):
        u, v = k.pt
        Xc = backproject(u, v, d0)
        if Xc is None:
            continue
        Xw = Twc_kf[:3, :3] @ Xc + Twc_kf[:3, 3]
        kf_pts.append(Xw); kf_des.append(des_kf[j]); kf_uv.append([u, v])
    kf_pts = np.array(kf_pts, np.float64)
    kf_des = np.array(kf_des, np.uint8)
    print("keyframe 0: %d map points" % len(kf_pts), flush=True)

    poses = [Twc_kf.copy()]
    obs_all = []       # 用于 BA: [frame, kf_point_id, u, v]
    for j in range(len(kf_pts)):
        obs_all.append([0, j, kf_uv[j][0], kf_uv[j][1]])
    global_pts = [kf_pts]
    kf_base = 0      # 当前关键帧点在 global_pts 里的起始下标

    n_reloc = 0
    for i in range(1, len(rgb)):
        g = load_gray(rgb[i][0]); dep = load_depth(rgb[i][0])
        kp, des = orb.detectAndCompute(g, None)
        if des is None or len(kp) < 10:
            poses.append(poses[-1].copy()); continue

        # ---- 与关键帧匹配 ----
        ms = bf.knnMatch(kf_des, des, k=2)
        good = [m for m, nb in ms if m.distance < 0.75 * nb.distance]
        obj = np.array([kf_pts[m.queryIdx] for m in good], np.float64)
        img = np.array([kp[m.trainIdx].pt for m in good], np.float64)

        Twc = poses[-1].copy()
        n_in = 0
        if len(obj) >= 12:
            ok, rvec, tvec, inl = cv2.solvePnPRansac(
                obj, img, K, None, flags=cv2.SOLVEPNP_ITERATIVE,
                reprojectionError=3.0, iterationsCount=300, confidence=0.999)
            if ok and inl is not None and len(inl) >= 12:
                R, _ = cv2.Rodrigues(rvec)
                # solvePnPRansac 解出的是 X_c = R*X_w + t (即 T_cw)
                T_cw = np.eye(4); T_cw[:3, :3] = R; T_cw[:3, 3] = tvec.ravel()
                Twc = np.linalg.inv(T_cw)       # 取逆 -> T_wc
                n_in = len(inl)
        poses.append(Twc.copy())

        # ---- 决定是否新建关键帧 ----
        moved = np.linalg.norm(Twc[:3, 3] - Twc_kf[:3, 3])
        inlier_ratio = n_in / max(len(obj), 1)
        if n_in < 60 or inlier_ratio < 0.5 or moved > 0.25:
            # 用当前帧建立新关键帧(位姿用当前估计)
            new_pts = []; new_des = []; new_uv = []
            for j, k in enumerate(kp):
                u, v = k.pt
                Xc = backproject(u, v, dep)
                if Xc is None:
                    continue
                Xw = Twc[:3, :3] @ Xc + Twc[:3, 3]
                new_pts.append(Xw); new_des.append(des[j]); new_uv.append([u, v])
            if len(new_pts) >= 100:
                kf_pts = np.array(new_pts, np.float64)
                kf_des = np.array(new_des, np.uint8)
                Twc_kf = Twc.copy()
                kf_base = sum(len(p) for p in global_pts)
                global_pts.append(kf_pts)
                for j in range(len(kf_pts)):
                    obs_all.append([i, kf_base + j, new_uv[j][0], new_uv[j][1]])
                kf = i
                n_reloc += 1

        # ---- 用当前位姿把关键帧点投到当前帧, 严格验证对应关系后追加观测 ----
        # 关键: 只记录"几何上确实对应上"的观测, 否则 BA 会被错误对应带偏
        if dep is not None and len(kf_pts) > 0 and n_in > 0 and kf_base >= 0:
            Rcw = Twc[:3, :3].T; tcw = -Rcw @ Twc[:3, 3]
            Xc = (Rcw @ kf_pts.T).T + tcw
            okz = (Xc[:, 2] > MIN_DEPTH) & (Xc[:, 2] < MAX_DEPTH)
            if okz.sum() > 10:
                cp = Xc[okz]
                uv = np.stack([fx * cp[:, 0] / cp[:, 2] + cx,
                               fy * cp[:, 1] / cp[:, 2] + cy], 1)
                H, Wd = g.shape
                iv = (uv[:, 0] >= 0) & (uv[:, 0] < Wd) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
                sel = np.where(okz)[0][iv]
                if len(sel) > 10:
                    proj = uv[iv]
                    # 用预测位置直接找最近特征点, 并做距离 + 描述子 + 深度三重校验
                    kp_pts = np.array([k.pt for k in kp], np.float64)
                    d2 = ((proj[:, None, :] - kp_pts[None, :, :]) ** 2).sum(-1) if len(kp) else None
                    for si in range(len(sel)):
                        gi = int(sel[si])
                        if len(kp) == 0:
                            break
                        j = int(np.argmin(d2[si]))
                        if d2[si, j] > 9.0:            # 预测位置 3px 内必须有特征点
                            continue
                        # 描述子校验
                        if int(np.unpackbits(np.bitwise_xor(kf_des[gi], des[j])).sum()) > 40:
                            continue
                        u, v = kp[j].pt
                        z_meas = float(dep[int(round(v)), int(round(u))])
                        z_pred = float(Xc[gi, 2])
                        if z_meas > MIN_DEPTH and z_meas < MAX_DEPTH and \
                           abs(z_meas - z_pred) / max(z_pred, 1e-3) < 0.10:
                            obs_all.append([i, kf_base + gi, float(u), float(v)])

        if verbose and i % 50 == 0:
            print("  frame %4d  kf=%d/reloc=%d  kfpts=%d obs=%d inl=%d" %
                  (i, kf, n_reloc, len(kf_pts), len(obs_all), n_in), flush=True)

    poses_arr = np.array(poses)
    points_arr = np.vstack(global_pts) if global_pts else np.zeros((0, 3))
    obs_arr = np.array(obs_all, np.float64) if obs_all else np.zeros((0, 4))
    print("total map points=%d obs=%d relocalizations=%d" % (len(points_arr), len(obs_arr), n_reloc), flush=True)
    np.savez(os.path.join(seq_dir, "rgbd_data.npz"),
             poses=poses_arr, points=points_arr, obs=obs_arr, K=K)

    est = poses_arr[:, :3, 3]
    gt_xyz = np.array([match_gt(ts, gt)[1] for ts, _ in rgb[:len(est)]])
    np.savetxt(os.path.join(seq_dir, "rgbd_traj_gt.txt"), gt_xyz)
    R_, t_, s_ = umeyama_align(est, gt_xyz)
    aligned = (s_ * (R_ @ est.T).T) + t_
    err = np.linalg.norm(aligned - gt_xyz, axis=1)
    rmse = float(np.sqrt((err ** 2).mean()))
    print("\n===== RGB-D VO v2 =====")
    print("scale=%.5f  ATE_RMSE=%.4f m  ATE_mean=%.4f m" % (s_, rmse, err.mean()))
    print("traj_len(est)=%.3f m   traj_len(gt)=%.3f m" %
          (np.linalg.norm(np.diff(aligned, axis=0), axis=1).sum(),
           np.linalg.norm(np.diff(gt_xyz, axis=0), axis=1).sum()))
    np.savetxt(os.path.join(seq_dir, "rgbd_traj_est.txt"), aligned)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    ax[0].plot(gt_xyz[:, 0], gt_xyz[:, 2], 'g-', lw=2, label="ground truth")
    ax[0].plot(aligned[:, 0], aligned[:, 2], 'b--', lw=2, label="RGB-D VO")
    ax[0].set_xlabel("X (m)"); ax[0].set_ylabel("Z (m)")
    ax[0].set_title("Trajectory (ATE RMSE=%.3f m)" % rmse)
    ax[0].legend(); ax[0].grid(True); ax[0].axis("equal")
    ax[1].plot(err, 'r-'); ax[1].set_xlabel("frame"); ax[1].set_ylabel("position error (m)")
    ax[1].set_title("Error over time"); ax[1].grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(seq_dir, "rgbd_vo_vs_gt.png"), dpi=150)
    print("图:", os.path.join(seq_dir, "rgbd_vo_vs_gt.png"))


if __name__ == "__main__":
    seq = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\r26304\Documents\codex-shop\slam_vo\data\rgbd_dataset_freiburg1_xyz"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 798
    main(seq, n)
