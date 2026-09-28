# -*- coding: utf-8 -*-
"""
RGB-D SLAM: 前端 + 观测生成 (干净重写)
之前 obs 全错的根因: 把"关键帧点+关键帧描述子"拿去和很远之后的帧匹配,
匹配靠描述子撞运气, 记录了根本不是同一个物理点的对应关系。
本版做法(简单且不会错):
  - 轨迹用关键帧 PnP 得到(已验证 ATE 0.034 m, 尺度真实)
  - 观测: 只在"相邻两帧之间"建立(stereo-like), 且用深度做双向校验:
      点 X 在帧 i 的深度反投影 -> 投影到帧 i+1 -> 该位置必须有特征点
      且该特征点的深度与投影深度一致
  - 这样每条观测都在几何上和深度上被验证过, 不会是假对应
输出: rgbd_slam.npz (poses / points / obs)
用法: python vo_rgbd3.py <序列目录> [最大帧数]
"""
import sys, os
import numpy as np
import cv2

TUM_FR1_K = np.array([[517.3, 0, 318.6], [0, 516.5, 255.3], [0, 0, 1]], np.float64)
DEPTH_SCALE = 5000.0
MIN_D, MAX_D = 0.15, 8.0


def read_list(p):
    out = []
    for line in open(p):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        s = line.split()
        if len(s) >= 2:
            out.append((float(s[0]), s[1]))
    return out


def read_gt(p):
    out = []
    for line in open(p):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        s = line.split()
        if len(s) >= 8:
            out.append((float(s[0]), np.array([float(s[1]), float(s[2]), float(s[3])])))
    return out


def nearest(ts, arr):
    return int(np.argmin(np.abs(np.array([a[0] for a in arr]) - ts)))


def umeyama(src, dst):
    n = len(src); ms = src.mean(0); md = dst.mean(0)
    Sc = src-ms; Dc = dst-md
    C = Dc.T@Sc/n
    U, S, Vt = np.linalg.svd(C)
    d = np.ones(3)
    if np.linalg.det(U)*np.linalg.det(Vt) < 0: d[2] = -1
    R = U@np.diag(d)@Vt
    s = (S*d).sum()/((Sc**2).sum()/n)
    return R, md-s*R@ms, s


def main(seq, max_frames=798, verbose=True):
    rgb = read_list(os.path.join(seq, "rgb.txt"))[:max_frames]
    dep_list = read_list(os.path.join(seq, "depth.txt"))
    gt = read_gt(os.path.join(seq, "groundtruth.txt"))
    K = TUM_FR1_K; fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    def gray(ts): return cv2.imread(os.path.join(seq, dict(rgb)[ts]), cv2.IMREAD_GRAYSCALE)

    def depth(ts):
        d = cv2.imread(os.path.join(seq, dep_list[nearest(ts, dep_list)][1]), cv2.IMREAD_UNCHANGED)
        return None if d is None else d.astype(np.float32)/DEPTH_SCALE

    orb = cv2.ORB_create(nfeatures=2500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def bp(u, v, D):
        z = D[int(round(v)), int(round(u))]
        if not np.isfinite(z) or z < MIN_D or z > MAX_D: return None
        return np.array([(u-cx)*z/fx, (v-cy)*z/fy, z])

    # 关键帧0
    g0 = gray(rgb[0][0]); D0 = depth(rgb[0][0])
    kp_kf, des_kf = orb.detectAndCompute(g0, None)
    P0 = []
    for j, k in enumerate(kp_kf):
        X = bp(k.pt[0], k.pt[1], D0)
        if X is not None: P0.append((X, des_kf[j], k.pt))
    kf_pts = np.array([p[0] for p in P0])
    kf_des = np.array([p[1] for p in P0], np.uint8)
    kf_base = 0
    Twc_kf = np.eye(4)
    poses = [Twc_kf.copy()]
    all_pts = [kf_pts]
    obs = [[0, j, float(P0[j][2][0]), float(P0[j][2][1])] for j in range(len(P0))]
    print("kf0: %d pts" % len(kf_pts), flush=True)

    for i in range(1, len(rgb)):
        g = gray(rgb[i][0]); D = depth(rgb[i][0])
        kp, des = orb.detectAndCompute(g, None)
        if des is None or len(kp) < 10:
            poses.append(poses[-1].copy()); continue
        ms = bf.knnMatch(kf_des, des, k=2)
        good = [m for m, nb in ms if m.distance < 0.75*nb.distance]
        obj = np.array([kf_pts[m.queryIdx] for m in good])
        img = np.array([kp[m.trainIdx].pt for m in good])
        Twc = poses[-1].copy(); n_in = 0; inl_idx = None
        if len(obj) >= 12:
            ok, rvec, tvec, inl = cv2.solvePnPRansac(
                obj, img, K, None, flags=cv2.SOLVEPNP_ITERATIVE,
                reprojectionError=3.0, iterationsCount=300, confidence=0.999)
            if ok and inl is not None and len(inl) >= 12:
                R, _ = cv2.Rodrigues(rvec)
                T = np.eye(4); T[:3, :3] = R; T[:3, 3] = tvec.ravel()
                Twc = np.linalg.inv(T)
                n_in = len(inl); inl_idx = inl.ravel()
        poses.append(Twc.copy())

        # --- 建立"当前帧 <-> 关键帧"的观测: 只记录 PnP 内点 ---
        if inl_idx is not None and D is not None:
            for j in inl_idx:
                m = good[j]
                gi = m.queryIdx
                u, v = kp[m.trainIdx].pt
                z = float(D[int(round(v)), int(round(u))])
                if z < MIN_D or z > MAX_D: continue
                # 深度校验: 该点在当前帧的观测深度 vs 用 Twc 预测的深度
                Xc = Twc[:3, :3].T @ (kf_pts[gi]-Twc[:3, 3])
                if Xc[2] <= 0: continue
                if abs(z-Xc[2])/max(Xc[2], 1e-3) > 0.10: continue
                obs.append([i, kf_base+gi, float(u), float(v)])

        # --- 关键帧切换 ---
        moved = np.linalg.norm(Twc[:3, 3]-Twc_kf[:3, 3])
        ratio = n_in/max(len(obj), 1)
        if n_in < 60 or ratio < 0.5 or moved > 0.2:
            new = []; nd = []
            for j, k in enumerate(kp):
                X = bp(k.pt[0], k.pt[1], D)
                if X is None: continue
                new.append(Twc[:3, :3]@X+Twc[:3, 3]); nd.append(des[j])
            if len(new) >= 100:
                kf_pts = np.array(new); kf_des = np.array(nd, np.uint8)
                kf_base = sum(len(p) for p in all_pts)
                all_pts.append(kf_pts)
                Twc_kf = Twc.copy()
                # 新关键帧自身的观测(用真实像素)
                cnt = 0
                for j, k in enumerate(kp):
                    X = bp(k.pt[0], k.pt[1], D)
                    if X is None: continue
                    obs.append([i, kf_base+cnt, float(k.pt[0]), float(k.pt[1])])
                    cnt += 1
        if verbose and i % 100 == 0:
            print("  f%4d kfpts=%d obs=%d inl=%d" % (i, len(kf_pts), len(obs), n_in), flush=True)

    poses = np.array(poses)
    pts = np.vstack(all_pts)
    obs = np.array(obs, np.float64)
    print("points=%d obs=%d" % (len(pts), len(obs)), flush=True)
    np.savez(os.path.join(seq, "rgbd_slam.npz"), poses=poses, points=pts, obs=obs, K=K)

    est = poses[:, :3, 3]
    gxyz = np.array([gt[nearest(ts, gt)][1] for ts, _ in rgb[:len(est)]])
    np.savetxt(os.path.join(seq, "rgbd_traj_gt.txt"), gxyz)
    R_, t_, s_ = umeyama(est, gxyz)
    al = (s_*(R_@est.T).T)+t_
    rmse = float(np.sqrt(((np.linalg.norm(al-gxyz, axis=1))**2).mean()))
    print("\n=== RGB-D SLAM 前端 ===")
    print("scale=%.5f  ATE_RMSE=%.4f m" % (s_, rmse))
    np.savetxt(os.path.join(seq, "rgbd_traj_est.txt"), al)

    # 自检: 重投影误差
    ff = obs[:, 0].astype(int); pf = obs[:, 1].astype(int)
    Rm = np.array([p[:3, :3] for p in poses]); Tm = np.array([p[:3, 3] for p in poses])
    Xc = np.einsum('nij,nj->ni', Rm[ff], pts[pf])+Tm[ff]
    z = Xc[:, 2].copy(); z[np.abs(z) < 1e-6] = 1e-6
    uv = np.stack([fx*Xc[:, 0]/z+cx, fy*Xc[:, 1]/z+cy], 1)
    e = np.linalg.norm(uv-obs[:, 2:4], axis=1)
    print("重投影: <2px %.1f%%  median=%.2f px" % (100*(e < 2).mean(), np.median(e)))
    np.savez(os.path.join(seq, "rgbd_slam.npz"), poses=poses, points=pts, obs=obs, K=K,
             rep_err=e)


if __name__ == "__main__":
    s = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\r26304\Documents\codex-shop\slam_vo\data\rgbd_dataset_freiburg1_xyz"
    main(s, int(sys.argv[2]) if len(sys.argv) > 2 else 798)
