# -*- coding: utf-8 -*-
"""
光束法平差(BA) 后端 —— 向量化加速版。
用 numpy 批量计算所有观测的投影残差，替代逐点 Python 循环。
读取 eval_vo.py 产生的 ba_data.npz。

用法: python ba_fast.py <TUM序列目录> [窗口帧数=60] [最小观测次数=2] [最大点数=4000]
"""
import sys, os
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
import cv2
from collections import defaultdict


def rodrigues_batch(rvecs):
    """批量旋转向量 -> 旋转矩阵. rvecs:(N,3) -> R:(N,3,3)"""
    N = len(rvecs)
    R = np.zeros((N, 3, 3))
    for i in range(N):
        R[i], _ = cv2.Rodrigues(rvecs[i].reshape(3, 1))
    return R


def rot_to_rvec(R):
    r, _ = cv2.Rodrigues(R)
    return r.ravel()


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


def main(seq_dir, win=60, min_obs=2, max_pts=4000, use_sparse=True, max_iter=30):
    d = np.load(os.path.join(seq_dir, "ba_data.npz"))
    poses = d["poses"]; points = d["points"]; obs = d["obs"]; K = d["K"]
    F, P, O = len(poses), len(points), len(obs)
    print("frames=%d points=%d obs=%d" % (F, P, O))

    sel = obs[obs[:, 0] < win]
    pcount = defaultdict(int)
    for row in sel:
        pcount[int(row[1])] += 1

    # 优先保留观测次数多的点，并限制总点数
    cand = sorted([p for p, c in pcount.items() if c >= min_obs], key=lambda p: -pcount[p])
    keep_pts = cand[:max_pts]
    p_index = {p: i for i, p in enumerate(keep_pts)}
    keep_set = set(keep_pts)
    sel = np.array([row for row in sel if int(row[1]) in keep_set])

    npose, npt = win, len(keep_pts)
    nobs = len(sel)
    print("window: frames=%d points=%d obs=%d" % (npose, npt, nobs))
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
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    def residuals(x):
        # 正确切分: 每帧 6 个数 = rvec(3)+tvec(3)
        params = x[:npose*6].reshape(npose, 6)
        rv = params[:, :3]
        tv = params[:, 3:6]
        pts = x[npose*6:].reshape(npt, 3)

        R = rodrigues_batch(rv)                     # (npose,3,3)
        # 批量投影: Xc = R[f] @ X[p] + t[f]
        Xp = pts[obs_p]                             # (nobs,3)
        Rf = R[obs_f]                               # (nobs,3,3)
        tf = tv[obs_f]                              # (nobs,3)
        Xc = np.einsum('nij,nj->ni', Rf, Xp) + tf   # (nobs,3)
        z = Xc[:, 2].copy()
        z[np.abs(z) < 1e-6] = 1e-6
        u = fx * Xc[:, 0] / z + cx
        v = fy * Xc[:, 1] / z + cy
        res = np.empty(nobs * 2)
        du = u - obs_uv[:, 0]
        dv = v - obs_uv[:, 1]
        CLIP = 20.0
        res[0::2] = np.clip(du, -CLIP, CLIP)
        res[1::2] = np.clip(dv, -CLIP, CLIP)
        return res

    # 稀疏结构
    jac_sp = None
    if use_sparse:
        npar = npose*6 + npt*3
        jac_sp = lil_matrix((nobs*2, npar), dtype=int)
        for k in range(nobs):
            f = obs_f[k]; pi = obs_p[k]
            pose_cols = list(range(f*6, f*6+6))
            pt_cols = list(range(npose*6 + pi*3, npose*6 + pi*3 + 3))
            jac_sp[2*k, pose_cols] = 1
            jac_sp[2*k, pt_cols] = 1
            jac_sp[2*k+1, pose_cols] = 1
            jac_sp[2*k+1, pt_cols] = 1
        jac_sp = jac_sp.tocsr()
        print("sparse jacobian: shape=%s nnz=%d" % (jac_sp.shape, jac_sp.nnz))

    print("running BA (vectorized)...")
    r = least_squares(residuals, x0, method="trf", loss="linear",
                      jac_sparsity=jac_sp, max_nfev=200,
                      xtol=1e-12, ftol=1e-12, gtol=1e-12, verbose=1)
    print("BA done. cost=%.4f nfev=%d" % (r.cost, r.nfev))

    params = r.x[:npose*6].reshape(npose, 6)
    rv = params[:, :3]; tv = params[:, 3:6]
    R = rodrigues_batch(rv)
    opt_pos = np.array([-R[i].T @ tv[i] for i in range(npose)])
    vo_pos = np.array([poses[i][:3, 3] for i in range(npose)])

    gt = np.loadtxt(os.path.join(seq_dir, "vo_traj_gt.txt"))[:npose]

    def ate(est):
        R_, t_, s_ = umeyama_align(est, gt)
        al = (s_ * (R_ @ est.T).T) + t_
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
    w = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    mn = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    mp = int(sys.argv[4]) if len(sys.argv) > 4 else 4000
    main(seq, w, mn, mp)
