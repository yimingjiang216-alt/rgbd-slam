# -*- coding: utf-8 -*-
"""
BA 后端 —— 解析雅可比版 (标准做法, 快速)
- 参数: 每帧位姿(rvec 3 + tvec 3) + 每个三维点(3)
- 残差: 重投影误差 (u,v)
- 雅可比: 用解析式(投影函数偏导 + 旋转扰动), 不再用数值差分
用法: python ba_analytic.py <TUM序列目录> [窗口帧数=200] [最小观测=3] [最大点=8000]
"""
import sys, os
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix, csr_matrix
import cv2
from collections import defaultdict


def rodrigues_batch(rvecs):
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
    s = (S * d).sum() / ((Sc ** 2).sum() / n)
    return R, mu_d - s * R @ mu_s, s


def main(seq_dir, win=200, min_obs=3, max_pts=8000, max_nfev=60):
    d = np.load(os.path.join(seq_dir, "ba_data.npz"))
    poses = d["poses"]; points = d["points"]; obs = d["obs"]; K = d["K"]
    F, P, O = len(poses), len(points), len(obs)
    print("frames=%d points=%d obs=%d" % (F, P, O), flush=True)

    sel = obs[obs[:, 0] < win]
    pcount = defaultdict(int)
    for row in sel:
        pcount[int(row[1])] += 1
    cand = sorted([p for p, c in pcount.items() if c >= min_obs], key=lambda p: -pcount[p])
    keep_pts = cand[:max_pts]
    p_index = {p: i for i, p in enumerate(keep_pts)}
    keep_set = set(keep_pts)
    sel = np.array([row for row in sel if int(row[1]) in keep_set])

    npose, npt, nobs = win, len(keep_pts), len(sel)
    npar = npose*6 + npt*3
    print("window: frames=%d points=%d obs=%d  params=%d" % (npose, npt, nobs, npar), flush=True)

    x0 = np.zeros(npar)
    for i in range(npose):
        x0[i*6:i*6+3] = rot_to_rvec(poses[i][:3, :3])
        x0[i*6+3:i*6+6] = poses[i][:3, 3]
    for j, p in enumerate(keep_pts):
        x0[npose*6 + j*3: npose*6 + j*3 + 3] = points[p]

    obs_f = sel[:, 0].astype(int)
    obs_p = np.array([p_index[int(p)] for p in sel[:, 1]])
    obs_uv = sel[:, 2:4]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    def unpack(x):
        params = x[:npose*6].reshape(npose, 6)
        pts = x[npose*6:].reshape(npt, 3)
        return params[:, :3], params[:, 3:6], pts

    def residuals(x):
        rv, tv, pts = unpack(x)
        R = rodrigues_batch(rv)
        Xp = pts[obs_p]
        Rf = R[obs_f]; tf = tv[obs_f]
        Xc = np.einsum('nij,nj->ni', Rf, Xp) + tf
        z = Xc[:, 2].copy(); z[np.abs(z) < 1e-6] = 1e-6
        u = fx * Xc[:, 0] / z + cx
        v = fy * Xc[:, 1] / z + cy
        du = np.clip(u - obs_uv[:, 0], -50, 50)
        dv = np.clip(v - obs_uv[:, 1], -50, 50)
        res = np.empty(nobs*2)
        res[0::2] = du; res[1::2] = dv
        return res

    def jacobian(x):
        rv, tv, pts = unpack(x)
        R = rodrigues_batch(rv)
        Xp = pts[obs_p]
        Rf = R[obs_f]; tf = tv[obs_f]
        Xc = np.einsum('nij,nj->ni', Rf, Xp) + tf      # (nobs,3)
        X, Y, Z = Xc[:,0], Xc[:,1], Xc[:,2]
        Zs = Z.copy(); Zs[np.abs(Zs) < 1e-6] = 1e-6
        # 投影对相机坐标的雅可比 (2x3)
        invZ = 1.0/Zs; invZ2 = invZ*invZ
        # d(u,v)/d(Xc,Yc,Zc)
        du_dXc = np.stack([fx*invZ, np.zeros_like(Z), -fx*X*invZ2], axis=1)  # (nobs,3)
        dv_dXc = np.stack([np.zeros_like(Z), fy*invZ, -fy*Y*invZ2], axis=1)  # (nobs,3)

        rows = np.repeat(np.arange(nobs), 2)
        rows[0::2] = np.arange(nobs); rows[1::2] = np.arange(nobs)
        row_u = np.arange(0, 2*nobs, 2)
        row_v = np.arange(1, 2*nobs, 2)

        data = []; ri = []; ci = []
        # --- 对三维点的偏导: d(r)/dXp = Jproj @ Rf ---
        JR = np.einsum('nij,njk->nik', np.stack([du_dXc, dv_dXc], axis=1).reshape(nobs,2,3), Rf)  # (nobs,2,3)
        for k in range(nobs):
            pcol = npose*6 + obs_p[k]*3
            for a in range(3):
                data.append(JR[k,0,a]); ri.append(row_u[k]); ci.append(pcol+a)
                data.append(JR[k,1,a]); ri.append(row_v[k]); ci.append(pcol+a)
        # --- 对位姿平移的偏导 ---
        for k in range(nobs):
            f = obs_f[k]; pcol = f*6+3
            for a in range(3):
                data.append(du_dXc[k,a]); ri.append(row_u[k]); ci.append(pcol+a)
                data.append(dv_dXc[k,a]); ri.append(row_v[k]); ci.append(pcol+a)
        # --- 对位姿旋转的偏导(用扰动近似: dXc/dw = -[Xc]_x) ---
        for k in range(nobs):
            f = obs_f[k]; pcol = f*6
            Xc_k = Xc[k]
            # dXc/dw = -skew(Xc)
            dXc_dw = np.array([[0, Xc_k[2], -Xc_k[1]],
                               [-Xc_k[2], 0, Xc_k[0]],
                               [Xc_k[1], -Xc_k[0], 0]])   # -skew
            # 链式: d(r)/dw = Jproj @ dXc_dw
            Ju = du_dXc[k] @ dXc_dw
            Jv = dv_dXc[k] @ dXc_dw
            for a in range(3):
                data.append(Ju[a]); ri.append(row_u[k]); ci.append(pcol+a)
                data.append(Jv[a]); ri.append(row_v[k]); ci.append(pcol+a)

        J = csr_matrix((np.array(data), (np.array(ri), np.array(ci))), shape=(2*nobs, npar))
        return J

    print("running BA (analytic jacobian)...", flush=True)
    r = least_squares(residuals, x0, jac=jacobian, method="trf",
                      max_nfev=max_nfev, xtol=1e-10, ftol=1e-10, gtol=1e-10, verbose=0)
    print("BA done. cost=%.4e nfev=%d" % (r.cost, r.nfev), flush=True)

    rv, tv, pts = unpack(r.x)
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
    print("\n===== BA 结果 (窗口 %d 帧, %d 点) =====" % (npose, npt))
    print("优化前 ATE RMSE = %.4f m" % rb)
    print("优化后 ATE RMSE = %.4f m" % ra)
    if rb > 0: print("提升: %.1f%%" % ((rb - ra)/rb*100))

    np.savetxt(os.path.join(seq_dir, "vo_traj_est_after.txt"), aligned_after)

    # 三线对比图
    before = np.loadtxt(os.path.join(seq_dir, "vo_traj_est_before.txt"))[:npose]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 7))
    plt.plot(gt[:,0], gt[:,2], 'g-', lw=2.5, label="ground truth")
    plt.plot(before[:,0], before[:,2], 'b--', lw=1.5, label="VO front-end (ATE=%.3f)" % rb)
    plt.plot(aligned_after[:,0], aligned_after[:,2], 'r-', lw=1.5, label="after BA (ATE=%.3f)" % ra)
    plt.scatter(gt[0,0], gt[0,2], c='k', s=80, marker='o', zorder=5, label="start")
    plt.xlabel("X (m)"); plt.ylabel("Z (m)")
    plt.title("VO vs BA vs Ground Truth")
    plt.legend(); plt.grid(True); plt.axis("equal")
    plt.savefig(os.path.join(seq_dir, "vo_ba_gt.png"), dpi=150)
    print("图:", os.path.join(seq_dir, "vo_ba_gt.png"))


if __name__ == "__main__":
    seq = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\r26304\Documents\codex-shop\slam_vo\data\rgbd_dataset_freiburg1_xyz"
    w = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    mn = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    mp = int(sys.argv[4]) if len(sys.argv) > 4 else 8000
    main(seq, w, mn, mp)
