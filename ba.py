# -*- coding: utf-8 -*-
"""
光束法平差(Bundle Adjustment) 后端 —— 小规模局部窗口优化。
读取 eval_vo.py 产生的 ba_data.npz，优化相机位姿 + 三维点，最小化重投影误差。

用法: python ba.py <TUM序列目录> [窗口帧数=8] [最小观测次数=3]
"""
import sys, os
import numpy as np
from scipy.optimize import least_squares
import cv2
from collections import defaultdict


def rodrigues(r):
    R, _ = cv2.Rodrigues(np.asarray(r, dtype=np.float64).reshape(3, 1))
    return R


def rot_to_rvec(R):
    r, _ = cv2.Rodrigues(R)
    return r.ravel()


def project(K, rvec, tvec, X):
    R = rodrigues(rvec)
    Xc = R @ X + tvec
    z = Xc[2]
    if z <= 1e-6:
        z = 1e-6
    return np.array([K[0, 0] * Xc[0] / z + K[0, 2],
                     K[1, 1] * Xc[1] / z + K[1, 2]])


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
    return R, mu_d - s * R @ mu_s, s


def main(seq_dir, win=8, min_obs=3, max_iter=25):
    d = np.load(os.path.join(seq_dir, "ba_data.npz"))
    poses = d["poses"]; points = d["points"]; obs = d["obs"]; K = d["K"]
    F, P, O = len(poses), len(points), len(obs)
    print("frames=%d points=%d obs=%d" % (F, P, O))

    sel = obs[obs[:, 0] < win]
    pcount = defaultdict(int)
    for row in sel:
        pcount[int(row[1])] += 1
    keep_pts = sorted([p for p, c in pcount.items() if c >= min_obs])
    p_index = {p: i for i, p in enumerate(keep_pts)}
    keep_set = set(keep_pts)
    sel = np.array([row for row in sel if int(row[1]) in keep_set])
    npose, npt = win, len(keep_pts)
    print("window: frames=%d points=%d obs=%d" % (npose, npt, len(sel)))
    print("参数个数: %d (pose %d + points %d)" % (npose*6 + npt*3, npose*6, npt*3))

    x0 = np.zeros(npose * 6 + npt * 3)
    for i in range(npose):
        x0[i*6:i*6+3] = rot_to_rvec(poses[i][:3, :3])
        x0[i*6+3:i*6+6] = poses[i][:3, 3]
    for j, p in enumerate(keep_pts):
        x0[npose*6 + j*3: npose*6 + j*3 + 3] = points[p]

    obs_f = sel[:, 0].astype(int)
    obs_p = np.array([p_index[int(p)] for p in sel[:, 1]])
    obs_uv = sel[:, 2:4]
    nres = len(obs_f) * 2

    def residuals(x):
        rvecs = x[:npose*6].reshape(npose, 6)
        pts = x[npose*6:].reshape(npt, 3)
        res = np.zeros(nres)
        for k in range(len(obs_f)):
            uv = project(K, rvecs[obs_f[k], :3], rvecs[obs_f[k], 3:6], pts[obs_p[k]])
            res[2*k]   = uv[0] - obs_uv[k, 0]
            res[2*k+1] = uv[1] - obs_uv[k, 1]
        return res

    print("running BA (小规模稠密)...")
    r = least_squares(residuals, x0, method="trf", loss="huber", f_scale=2.0,
                      max_nfev=max_iter * 40, verbose=2)
    print("BA done. cost=%.4f nfev=%d" % (r.cost, r.nfev))

    rvecs = r.x[:npose*6].reshape(npose, 6)
    opt_pos = np.array([-rodrigues(rvecs[i, :3]).T @ rvecs[i, 3:6] for i in range(npose)])
    vo_pos = np.array([poses[i][:3, 3] for i in range(npose)])

    gt = np.loadtxt(os.path.join(seq_dir, "vo_traj_gt.txt"))[:npose]

    def ate(est):
        R, t, s = umeyama_align(est, gt)
        al = (s * (R @ est.T).T) + t
        e = np.linalg.norm(al - gt, axis=1)
        return float(np.sqrt((e**2).mean())), al

    rb, _ = ate(vo_pos)
    ra, aligned_after = ate(opt_pos)
    print("\n===== BA 结果 (窗口前 %d 帧) =====" % npose)
    print("优化前 ATE RMSE = %.4f m" % rb)
    print("优化后 ATE RMSE = %.4f m" % ra)
    if rb > 0:
        print("提升: %.1f%%" % ((rb - ra) / rb * 100))

    np.savetxt(os.path.join(seq_dir, "vo_traj_est_after.txt"), aligned_after)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 6))
    plt.plot(gt[:, 0], gt[:, 2], 'g-', lw=2, label="ground truth")
    plt.plot(aligned_after[:, 0], aligned_after[:, 2], 'r--', lw=2, label="after BA")
    plt.scatter(gt[0, 0], gt[0, 2], c='red', s=70, zorder=5)
    plt.xlabel("X (m)"); plt.ylabel("Z (m)")
    plt.title("After BA (ATE RMSE=%.3f m)" % ra)
    plt.legend(); plt.grid(True); plt.axis("equal")
    plt.savefig(os.path.join(seq_dir, "ba_vs_gt.png"), dpi=150)
    print("对比图:", os.path.join(seq_dir, "ba_vs_gt.png"))


if __name__ == "__main__":
    seq = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\r26304\Documents\codex-shop\slam_vo\data\rgbd_dataset_freiburg1_xyz"
    w = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    mn = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    main(seq, w, mn)
