# -*- coding: utf-8 -*-
"""
滑窗 BA (sliding-window bundle adjustment)
思路: 不再对全部 798 帧做一次性全局优化(点数一多就退化成病态问题),
      而是沿时间轴滚动一个 W 帧的窗口, 每段独立做 BA, 用上一窗口末帧作为参考固定,
      只把窗口内点集与该窗口强相关的观测纳入优化。
每个窗口内部:
  - 参数 = W 帧位姿(6) + 窗口内三维点(3)
  - 残差 = 重投影误差
  - 解析雅可比
用法: python ba_window.py <序列目录> [窗口=50] [步长=25] [每帧最大点=400] [最小观测=3]
"""
import sys, os
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix
import cv2
from collections import defaultdict


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


def solve_window(obs_w, poses_w, points_w, K, min_obs=3, max_per_frame=400, max_nfev=40):
    """obs_w: (n,4) 全局帧号/点号/u/v ; poses_w: dict 帧号->4x4 ; points_w: dict 点号->xyz"""
    frames = sorted(poses_w.keys())
    fidx = {f: i for i, f in enumerate(frames)}
    N = len(frames)

    # 每个点在窗口内的观测次数
    pc = defaultdict(int); fc = defaultdict(int)
    for o in obs_w:
        pc[int(o[1])] += 1
        fc[int(o[0])] += 1
    keep = [p for p, c in pc.items() if c >= min_obs]
    if not keep:
        return None
    # 按每帧配额裁剪, 保证帧间均衡
    quota = defaultdict(int)
    keep2 = []
    keep_sorted = sorted(keep, key=lambda p: -pc[p])
    # 每个点的首帧用于配额
    first = {}
    for o in obs_w:
        p = int(o[1])
        if p not in first or int(o[0]) < first[p]:
            first[p] = int(o[0])
    for p in keep_sorted:
        f0 = first[p]
        if quota[f0] < max_per_frame:
            keep2.append(p); quota[f0] += 1
    keep = keep2
    pidx = {p: i for i, p in enumerate(keep)}
    keepset = set(keep)

    rows = [o for o in obs_w if int(o[1]) in keepset]
    if len(rows) < 20:
        return None
    sel = np.array(rows, float)
    npt = len(keep); nobs = len(sel)
    npar = N * 6 + npt * 3

    x0 = np.zeros(npar)
    for f in frames:
        i = fidx[f]
        P = poses_w[f]
        r, _ = cv2.Rodrigues(P[:3, :3])
        x0[i*6:i*6+3] = r.ravel()
        x0[i*6+3:i*6+6] = P[:3, 3]
    for j, p in enumerate(keep):
        x0[N*6 + j*3: N*6 + j*3 + 3] = points_w[p]

    obs_f = np.array([fidx[int(o[0])] for o in sel], int)
    obs_p = np.array([pidx[int(o[1])] for o in sel], int)
    obs_uv = sel[:, 2:4]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    def unpack(x):
        pr = x[:N*6].reshape(N, 6)
        pts = x[N*6:].reshape(npt, 3)
        return pr[:, :3], pr[:, 3:6], pts

    def residuals(x):
        rv, tv, pts = unpack(x)
        R = rodrigues_batch(rv)
        Xp = pts[obs_p]
        Xc = np.einsum('nij,nj->ni', R[obs_f], Xp) + tv[obs_f]
        z = Xc[:, 2].copy(); z[np.abs(z) < 1e-6] = 1e-6
        du = np.clip(fx * Xc[:, 0] / z + cx - obs_uv[:, 0], -50, 50)
        dv = np.clip(fy * Xc[:, 1] / z + cy - obs_uv[:, 1], -50, 50)
        res = np.empty(nobs * 2)
        res[0::2] = du; res[1::2] = dv
        return res

    def jacobian(x):
        rv, tv, pts = unpack(x)
        R = rodrigues_batch(rv)
        Xc = np.einsum('nij,nj->ni', R[obs_f], pts[obs_p]) + tv[obs_f]
        X, Y, Z = Xc[:, 0], Xc[:, 1], Xc[:, 2]
        Zs = Z.copy(); Zs[np.abs(Zs) < 1e-6] = 1e-6
        invZ = 1.0 / Zs; invZ2 = invZ * invZ
        du = np.stack([fx*invZ, np.zeros(nobs), -fx*X*invZ2], 1)
        dv = np.stack([np.zeros(nobs), fy*invZ, -fy*Y*invZ2], 1)
        ru = np.arange(0, 2*nobs, 2); rv_ = np.arange(1, 2*nobs, 2)

        # 解析 + 向量化组装 COO
        JR = np.einsum('nij,njk->nik', np.stack([du, dv], 1).reshape(nobs, 2, 3), R[obs_f])  # (nobs,2,3)
        ptcol = N*6 + obs_p*3
        r_pt = np.empty(nobs * 6, int); c_pt = np.empty(nobs * 6, int); d_pt = np.empty(nobs * 6)
        r_pt[0::6] = ru; r_pt[1::6] = rv_; r_pt[2::6] = ru; r_pt[3::6] = rv_; r_pt[4::6] = ru; r_pt[5::6] = rv_
        c_pt[0::6] = ptcol; c_pt[1::6] = ptcol; c_pt[2::6] = ptcol+1; c_pt[3::6] = ptcol+1
        c_pt[4::6] = ptcol+2; c_pt[5::6] = ptcol+2
        d_pt[0::6] = JR[:,0,0]; d_pt[1::6] = JR[:,1,0]
        d_pt[2::6] = JR[:,0,1]; d_pt[3::6] = JR[:,1,1]
        d_pt[4::6] = JR[:,0,2]; d_pt[5::6] = JR[:,1,2]

        fcol = obs_f * 6
        r_t = np.empty(nobs * 6, int); c_t = np.empty(nobs * 6, int); d_t = np.empty(nobs * 6)
        r_t[0::6] = ru; r_t[1::6] = rv_; r_t[2::6] = ru; r_t[3::6] = rv_; r_t[4::6] = ru; r_t[5::6] = rv_
        c_t[0::6] = fcol + 3; c_t[1::6] = fcol + 3; c_t[2::6] = fcol + 4; c_t[3::6] = fcol + 4
        c_t[4::6] = fcol + 5; c_t[5::6] = fcol + 5
        d_t[0::6] = du[:, 0]; d_t[1::6] = dv[:, 0]
        d_t[2::6] = du[:, 1]; d_t[3::6] = dv[:, 1]
        d_t[4::6] = du[:, 2]; d_t[5::6] = dv[:, 2]

        # 旋转: dXc/dw = -skew(Xc)
        S = np.zeros((nobs, 3, 3))
        S[:, 0, 1] = Xc[:, 2]; S[:, 0, 2] = -Xc[:, 1]
        S[:, 1, 0] = -Xc[:, 2]; S[:, 1, 2] = Xc[:, 0]
        S[:, 2, 0] = Xc[:, 1]; S[:, 2, 1] = -Xc[:, 0]
        Jw = np.einsum('nij,njk->nik', np.stack([du, dv], 1).reshape(nobs, 2, 3), S)  # (nobs,2,3)
        r_w = np.empty(nobs * 6, int); c_w = np.empty(nobs * 6, int); d_w = np.empty(nobs * 6)
        r_w[0::6] = ru; r_w[1::6] = rv_; r_w[2::6] = ru; r_w[3::6] = rv_; r_w[4::6] = ru; r_w[5::6] = rv_
        c_w[0::6] = fcol; c_w[1::6] = fcol; c_w[2::6] = fcol+1; c_w[3::6] = fcol+1
        c_w[4::6] = fcol+2; c_w[5::6] = fcol+2
        d_w[0::6] = Jw[:,0,0]; d_w[1::6] = Jw[:,1,0]
        d_w[2::6] = Jw[:,0,1]; d_w[3::6] = Jw[:,1,1]
        d_w[4::6] = Jw[:,0,2]; d_w[5::6] = Jw[:,1,2]

        ri = np.concatenate([r_pt, r_t, r_w])
        ci = np.concatenate([c_pt, c_t, c_w])
        dd = np.concatenate([d_pt, d_t, d_w])
        return csr_matrix((dd, (ri, ci)), shape=(2*nobs, npar))

    r = least_squares(residuals, x0, jac=jacobian, method='trf', max_nfev=max_nfev,
                      xtol=1e-8, ftol=1e-8, gtol=1e-8, verbose=0)
    return r, frames, keep, obs_f, obs_p


def main(seq_dir, W=50, step=25, max_per_frame=400, min_obs=3):
    d = np.load(os.path.join(seq_dir, 'ba_data.npz'))
    G_poses, G_points, G_obs, K = d['poses'], d['points'], d['obs'], d['K']
    F = len(G_poses)
    print('total frames=%d points=%d obs=%d' % (F, len(G_points), len(G_obs)), flush=True)

    by_frame = defaultdict(list)
    for o in G_obs:
        by_frame[int(o[0])].append(o)

    opt_poses = {i: G_poses[i].copy() for i in range(F)}
    seg_before, seg_after = [], []
    nwin = 0
    for start in range(0, F - 5, step):
        end = min(start + W, F)
        frames = list(range(start, end))
        obs_w = [o for f in frames for o in by_frame[f]]
        if len(obs_w) < 100:
            continue
        pw = {i: opt_poses[i] for i in frames}
        out = solve_window(np.array(obs_w, float), pw, {int(p): G_points[int(p)] for p in set(int(o[1]) for o in obs_w)},
                           K, min_obs=min_obs, max_per_frame=max_per_frame)
        if out is None:
            continue
        r, fr, keep, obs_f, obs_p = out
        N = len(fr); npt = len(keep)
        for i, f in enumerate(fr):
            rv = r.x[i*6:i*6+3]; tv = r.x[i*6+3:i*6+6]
            R, _ = cv2.Rodrigues(rv.reshape(3,1))
            P = np.eye(4); P[:3,:3] = R; P[:3,3] = tv
            opt_poses[f] = P
        nwin += 1
        # 分段 ATE
        gt_all = np.loadtxt(os.path.join(seq_dir, 'vo_traj_gt.txt'))
        gt_seg = gt_all[fr]
        def ate_of(plist):
            est = np.array([plist[f][:3,3] for f in fr])
            R_, t_, s_ = umeyama_align(est, gt_seg)
            al = (s_*(R_@est.T).T)+t_
            e = np.linalg.norm(al-gt_seg, axis=1)
            return float(np.sqrt((e**2).mean()))
        seg_before.append(ate_of({i: G_poses[i] for i in fr}))
        seg_after.append(ate_of({i: opt_poses[i] for i in fr}))
        print('  win %3d frames %4d-%4d pts=%5d obs=%6d nfev=%d  ATE %.4f -> %.4f' %
              (nwin, fr[0], fr[-1], npt, len(obs_f), r.nfev, seg_before[-1], seg_after[-1]), flush=True)
        if end >= F:
            break

    # 全局 ATE
    gt = np.loadtxt(os.path.join(seq_dir, 'vo_traj_gt.txt'))[:F]
    def global_ate(plist):
        est = np.array([plist[i][:3,3] for i in range(F)])
        R_, t_, s_ = umeyama_align(est, gt)
        al = (s_*(R_@est.T).T)+t_
        e = np.linalg.norm(al-gt, axis=1)
        return float(np.sqrt((e**2).mean())), al, s_
    rb, _, _ = global_ate({i: G_poses[i] for i in range(F)})
    ra, aligned, s_ = global_ate(opt_poses)
    print('\n===== 滑窗 BA 结果 =====')
    print('窗口=%d 步长=%d 窗口数=%d' % (W, step, nwin))
    print('优化前(全序列) ATE RMSE = %.4f m' % rb)
    print('优化后(全序列) ATE RMSE = %.4f m' % ra)
    if rb > 0:
        print('提升: %.1f%%' % ((rb-ra)/rb*100))
    print('尺度 s = %.5f' % s_)

    np.savetxt(os.path.join(seq_dir, 'vo_traj_est_wba.txt'), aligned)
    before = np.loadtxt(os.path.join(seq_dir, 'vo_traj_est_before.txt'))[:F]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,7))
    plt.plot(gt[:,0], gt[:,2], 'g-', lw=2.5, label='ground truth')
    plt.plot(before[:,0], before[:,2], 'b--', lw=1.5, label='VO front-end (ATE=%.3f m)' % rb)
    plt.plot(aligned[:,0], aligned[:,2], 'r-', lw=1.5, label='after sliding-window BA (ATE=%.3f m)' % ra)
    plt.scatter(gt[0,0], gt[0,2], c='k', s=80, marker='o', zorder=5, label='start')
    plt.xlabel('X (m)'); plt.ylabel('Z (m)')
    plt.title('VO vs Sliding-Window BA vs Ground Truth (TUM fr1/xyz)')
    plt.legend(); plt.grid(True); plt.axis('equal')
    plt.savefig(os.path.join(seq_dir, 'vo_ba_gt_window.png'), dpi=150)
    print('图:', os.path.join(seq_dir, 'vo_ba_gt_window.png'))


if __name__ == '__main__':
    seq = sys.argv[1] if len(sys.argv) > 1 else r'C:\Users\r26304\Documents\codex-shop\slam_vo\data\rgbd_dataset_freiburg1_xyz'
    W = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    st = int(sys.argv[3]) if len(sys.argv) > 3 else 25
    mpf = int(sys.argv[4]) if len(sys.argv) > 4 else 400
    mo = int(sys.argv[5]) if len(sys.argv) > 5 else 3
    main(seq, W, st, mpf, mo)
