# -*- coding: utf-8 -*-
"""
RGB-D 数据的滑窗 BA —— 修好前端之后重做后端
数据: rgbd_data.npz (位姿尺度真实, 地图点由深度反投影得到, 不再是退化的三角化点)
检查: 优化前后重投影误差 + 轨迹 ATE
用法: python ba_rgbd.py <序列目录> [窗口=60] [步长=30] [每帧点配额=300] [最小观测=3]
"""
import sys, os
import numpy as np
import cv2
from collections import defaultdict
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix


def rodrigues_batch(rvecs):
    R = np.zeros((len(rvecs), 3, 3))
    for i in range(len(rvecs)):
        R[i], _ = cv2.Rodrigues(rvecs[i].reshape(3, 1))
    return R


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


def build_system(obs_sel, fidx, pidx, poses, points, N, npt, K, npar):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    obs_f = np.array([fidx[int(o[0])] for o in obs_sel], int)
    obs_p = np.array([pidx[int(o[1])] for o in obs_sel], int)
    obs_uv = obs_sel[:, 2:4]
    nobs = len(obs_sel)

    x0 = np.zeros(npar)
    for f, i in fidx.items():
        P = poses[f]
        r, _ = cv2.Rodrigues(P[:3, :3])
        x0[i*6:i*6+3] = r.ravel(); x0[i*6+3:i*6+6] = P[:3, 3]
    for p, j in pidx.items():
        x0[N*6 + j*3: N*6 + j*3 + 3] = points[p]

    def residuals(x):
        pr = x[:N*6].reshape(N, 6); pts = x[N*6:].reshape(npt, 3)
        R = rodrigues_batch(pr[:, :3])
        Xc = np.einsum('nij,nj->ni', R[obs_f], pts[obs_p]) + pr[:, 3:6][obs_f]
        z = Xc[:, 2].copy(); z[np.abs(z) < 1e-6] = 1e-6
        du = np.clip(fx*Xc[:, 0]/z + cx - obs_uv[:, 0], -100, 100)
        dv = np.clip(fy*Xc[:, 1]/z + cy - obs_uv[:, 1], -100, 100)
        res = np.empty(2*nobs); res[0::2] = du; res[1::2] = dv
        return res

    def jacobian(x):
        pr = x[:N*6].reshape(N, 6); pts = x[N*6:].reshape(npt, 3)
        R = rodrigues_batch(pr[:, :3])
        Xc = np.einsum('nij,nj->ni', R[obs_f], pts[obs_p]) + pr[:, 3:6][obs_f]
        X, Y, Z = Xc[:, 0], Xc[:, 1], Xc[:, 2]
        Zs = Z.copy(); Zs[np.abs(Zs) < 1e-6] = 1e-6
        invZ = 1.0/Zs; invZ2 = invZ*invZ
        du = np.stack([fx*invZ, np.zeros(nobs), -fx*X*invZ2], 1)
        dv = np.stack([np.zeros(nobs), fy*invZ, -fy*Y*invZ2], 1)
        ru = np.arange(0, 2*nobs, 2); rv_ = np.arange(1, 2*nobs, 2)
        J2 = np.stack([du, dv], 1).reshape(nobs, 2, 3)

        JR = np.einsum('nij,njk->nik', J2, R[obs_f])
        pcol = N*6 + obs_p*3
        r_pt = np.empty(nobs*6, int); c_pt = np.empty(nobs*6, int); d_pt = np.empty(nobs*6)
        idx = np.arange(0, nobs*6, 6)
        for a in range(3):
            r_pt[idx+2*a] = ru;   r_pt[idx+2*a+1] = rv_
            c_pt[idx+2*a] = pcol+a; c_pt[idx+2*a+1] = pcol+a
            d_pt[idx+2*a] = JR[:, 0, a]; d_pt[idx+2*a+1] = JR[:, 1, a]

        fcol = obs_f*6
        r_t = np.empty(nobs*6, int); c_t = np.empty(nobs*6, int); d_t = np.empty(nobs*6)
        for a in range(3):
            r_t[idx+2*a] = ru;   r_t[idx+2*a+1] = rv_
            c_t[idx+2*a] = fcol+3+a; c_t[idx+2*a+1] = fcol+3+a
            d_t[idx+2*a] = du[:, a]; d_t[idx+2*a+1] = dv[:, a]

        S = np.zeros((nobs, 3, 3))
        S[:, 0, 1] = Xc[:, 2]; S[:, 0, 2] = -Xc[:, 1]
        S[:, 1, 0] = -Xc[:, 2]; S[:, 1, 2] = Xc[:, 0]
        S[:, 2, 0] = Xc[:, 1]; S[:, 2, 1] = -Xc[:, 0]
        Jw = np.einsum('nij,njk->nik', J2, S)
        r_w = np.empty(nobs*6, int); c_w = np.empty(nobs*6, int); d_w = np.empty(nobs*6)
        for a in range(3):
            r_w[idx+2*a] = ru;   r_w[idx+2*a+1] = rv_
            c_w[idx+2*a] = fcol+a; c_w[idx+2*a+1] = fcol+a
            d_w[idx+2*a] = Jw[:, 0, a]; d_w[idx+2*a+1] = Jw[:, 1, a]

        ri = np.concatenate([r_pt, r_t, r_w]); ci = np.concatenate([c_pt, c_t, c_w])
        dd = np.concatenate([d_pt, d_t, d_w])
        return csr_matrix((dd, (ri, ci)), shape=(2*nobs, npar))

    return residuals, jacobian, x0, obs_f, obs_p, obs_uv


def main(seq_dir, W=60, step=30, quota=300, min_obs=3, max_nfev=30):
    d = np.load(os.path.join(seq_dir, 'rgbd_data.npz'))
    G_poses, G_points, G_obs, K = d['poses'], d['points'], d['obs'], d['K']
    F = len(G_poses)
    gt_all = np.loadtxt(os.path.join(seq_dir, 'rgbd_traj_gt.txt'))
    print('frames=%d points=%d obs=%d' % (F, len(G_points), len(G_obs)), flush=True)

    by_frame = defaultdict(list)
    for o in G_obs:
        by_frame[int(o[0])].append(o)

    opt = {i: G_poses[i].copy() for i in range(F)}
    wins = 0
    for start in range(0, F-5, step):
        end = min(start+W, F)
        frames = list(range(start, end))
        obs_w = np.array([o for f in frames for o in by_frame[f]], float)
        if len(obs_w) < 100:
            continue
        pc = defaultdict(int); first = {}
        for o in obs_w:
            p = int(o[1]); pc[p] += 1
            if p not in first or int(o[0]) < first[p]:
                first[p] = int(o[0])
        cand = sorted([p for p, c in pc.items() if c >= min_obs], key=lambda p: -pc[p])
        q = defaultdict(int); keep = []
        for p in cand:
            f0 = first[p]
            if q[f0] < quota:
                keep.append(p); q[f0] += 1
        if len(keep) < 20:
            continue
        ks = set(keep)
        obs_sel = np.array([o for o in obs_w if int(o[1]) in ks], float)
        fidx = {f: i for i, f in enumerate(frames)}
        pidx = {p: i for i, p in enumerate(keep)}
        N = len(frames); npt = len(keep); npar = N*6 + npt*3
        res, jac, x0, obs_f, obs_p, obs_uv = build_system(
            obs_sel, fidx, pidx, {f: opt[f] for f in frames},
            {p: G_points[p] for p in keep}, N, npt, K, npar)
        r0 = res(x0)
        rms0 = float(np.sqrt((r0**2).mean()))
        out = least_squares(res, x0, jac=jac, method='trf', max_nfev=max_nfev,
                            xtol=1e-8, ftol=1e-8, gtol=1e-8, verbose=0)
        rms1 = float(np.sqrt((out.fun**2).mean()))
        for i, f in enumerate(frames):
            R, _ = cv2.Rodrigues(out.x[i*6:i*6+3].reshape(3, 1))
            P = np.eye(4); P[:3, :3] = R; P[:3, 3] = out.x[i*6+3:i*6+6]
            opt[f] = P
        wins += 1
        seg_gt = gt_all[frames]
        def seg_ate(pl):
            e = np.array([pl[f][:3, 3] for f in frames])
            R_, t_, s_ = umeyama_align(e, seg_gt)
            al = (s_*(R_@e.T).T)+t_
            return float(np.sqrt(((np.linalg.norm(al-seg_gt, axis=1))**2).mean()))
        print('  win%3d f%4d-%4d pts=%5d obs=%6d nfev=%2d  repl %.1f->%.1f px  segATE %.4f->%.4f' %
              (wins, frames[0], frames[-1], npt, len(obs_sel), out.nfev, rms0, rms1,
               seg_ate({i: G_poses[i] for i in frames}), seg_ate(opt)), flush=True)
        if end >= F:
            break

    def global_ate(pl):
        e = np.array([pl[i][:3, 3] for i in range(F)])
        R_, t_, s_ = umeyama_align(e, gt_all)
        al = (s_*(R_@e.T).T)+t_
        err = np.linalg.norm(al-gt_all, axis=1)
        return float(np.sqrt((err**2).mean())), al, err, s_

    rb, _, errb, _ = global_ate({i: G_poses[i] for i in range(F)})
    ra, aligned, erra, s_ = global_ate(opt)
    print('\n===== 滑窗 BA (RGB-D) =====')
    print('窗口=%d 步长=%d 窗口数=%d 每帧点配额=%d' % (W, step, wins, quota))
    print('优化前 ATE RMSE = %.4f m  (max %.4f)' % (rb, errb.max()))
    print('优化后 ATE RMSE = %.4f m  (max %.4f)' % (ra, erra.max()))
    print('提升 = %.1f%%' % ((rb-ra)/rb*100))
    np.savetxt(os.path.join(seq_dir, 'rgbd_traj_ba.txt'), aligned)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    before = np.loadtxt(os.path.join(seq_dir, 'rgbd_traj_est.txt'))
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    ax[0].plot(gt_all[:, 0], gt_all[:, 2], 'g-', lw=2.5, label='ground truth')
    ax[0].plot(before[:, 0], before[:, 2], 'b--', lw=1.6, label='VO (ATE=%.3f m)' % rb)
    ax[0].plot(aligned[:, 0], aligned[:, 2], 'r-', lw=1.6, label='VO+BA (ATE=%.3f m)' % ra)
    ax[0].set_xlabel('X (m)'); ax[0].set_ylabel('Z (m)')
    ax[0].set_title('RGB-D SLAM: VO vs VO+BA vs Ground Truth')
    ax[0].legend(); ax[0].grid(True); ax[0].axis('equal')
    ax[1].plot(errb, 'b--', label='VO (mean %.3f)' % errb.mean())
    ax[1].plot(erra, 'r-', label='VO+BA (mean %.3f)' % erra.mean())
    ax[1].set_xlabel('frame'); ax[1].set_ylabel('position error (m)')
    ax[1].set_title('Error over time'); ax[1].legend(); ax[1].grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(seq_dir, 'rgbd_vo_ba_vs_gt.png'), dpi=150)
    print('图:', os.path.join(seq_dir, 'rgbd_vo_ba_vs_gt.png'))


if __name__ == '__main__':
    seq = sys.argv[1] if len(sys.argv) > 1 else r'C:\Users\r26304\Documents\codex-shop\slam_vo\data\rgbd_dataset_freiburg1_xyz'
    W = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    st = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    q = int(sys.argv[4]) if len(sys.argv) > 4 else 300
    mo = int(sys.argv[5]) if len(sys.argv) > 5 else 3
    main(seq, W, st, q, mo)
