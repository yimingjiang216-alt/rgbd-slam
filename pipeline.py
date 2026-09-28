# -*- coding: utf-8 -*-
"""
RGB-D SLAM 完整流水线 (重写)
============================================================
设计原则(针对之前失败的原因):
  1. 尺度: 用深度图直接给, 不用单目 recoverPose (它无尺度, t 恒为1)
  2. 位姿: 每帧都对"关键帧点云"做 PnP, 关键帧位姿固定不漂 -> 不累积正反馈
  3. 观测: 每条观测都过三重校验(重投影距离 + 描述子 + 深度一致性),
          且只记录当帧 PnP 的内点 -> 不产生假对应
  4. 坐标系: 内部统一用第一帧相机系; 评估时用 SE(3) 对齐(而非含尺度的 Umeyama)
            -> 旋转误差不会被平移掩盖
流程:
  stage 1 前端  -> 位姿 + 地图点 + 观测
  stage 2 BA    -> 滑窗局部优化
  stage 3 评估  -> ATE(SE3对齐) + 旋转误差 + 重投影误差
用法:
  python pipeline.py front  <序列目录>
  python pipeline.py ba     <序列目录>
  python pipeline.py eval   <序列目录>
  python pipeline.py all    <序列目录>
"""
import sys, os, json
import numpy as np
import cv2

SEQ = r"C:\Users\r26304\Documents\codex-shop\slam_vo\data\rgbd_dataset_freiburg1_xyz"
K = np.array([[517.3, 0, 318.6], [0, 516.5, 255.3], [0, 0, 1]], np.float64)
DEPTH_SCALE = 5000.0
MIN_D, MAX_D = 0.2, 6.0
FX, FY, CX, CY = K[0, 0], K[1, 1], K[0, 2], K[1, 2]


# ---------------- 工具 ----------------
def read_list(path):
    out = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        s = line.split()
        if len(s) >= 2:
            out.append((float(s[0]), s[1]))
    return out


def read_gt(path):
    out = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        s = line.split()
        if len(s) >= 8:
            out.append((float(s[0]), np.array([float(x) for x in s[1:8]])))
    return out


def q2R(q):
    x, y, z, w = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
                     [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]])


def q2T(q):
    T = np.eye(4); T[:3, :3] = q2R(q[3:7]); T[:3, 3] = q[0:3]
    return T


def nearest(ts, arr):
    return int(np.argmin(np.abs(np.array([a[0] for a in arr]) - ts)))


def se3_align(src, dst, with_scale=False):
    """src->dst 对齐。with_scale=False 时是纯 SE(3)(保长度), True 时才估尺度。"""
    n = len(src); ms = src.mean(0); md = dst.mean(0)
    Sc = src - ms; Dc = dst - md
    C = Dc.T @ Sc / n
    U, S, Vt = np.linalg.svd(C)
    d = np.ones(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        d[2] = -1
    R = U @ np.diag(d) @ Vt
    if with_scale:
        s = (S * d).sum() / max(((Sc**2).sum()/n), 1e-12)
    else:
        s = 1.0
    return R, md - s * R @ ms, s


def rot_angle(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R)-1)/2, -1, 1))))


# ---------------- stage 1: 前端 ----------------
def front(seq=SEQ, max_frames=798, verbose=True):
    print("[stage 1] RGB-D 前端", flush=True)
    rgb = read_list(os.path.join(seq, "rgb.txt"))[:max_frames]
    dl = read_list(os.path.join(seq, "depth.txt"))
    gt = read_gt(os.path.join(seq, "groundtruth.txt"))
    rgb_ts = [r[0] for r in rgb]
    d_ts = np.array([d[0] for d in dl])

    def load_rgb(ts):
        return cv2.imread(os.path.join(seq, dict(rgb)[ts]), cv2.IMREAD_GRAYSCALE)

    def load_D(ts):
        img = cv2.imread(os.path.join(seq, dl[int(np.argmin(np.abs(d_ts-ts)))][1]),
                         cv2.IMREAD_UNCHANGED)
        return None if img is None else img.astype(np.float32)/DEPTH_SCALE

    orb = cv2.ORB_create(nfeatures=3000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def backproj(u, v, D):
        ui, vi = int(round(u)), int(round(v))
        if not (0 <= vi < D.shape[0] and 0 <= ui < D.shape[1]):
            return None
        z = float(D[vi, ui])
        if not np.isfinite(z) or z < MIN_D or z > MAX_D:
            return None
        return np.array([(u-CX)*z/FX, (v-CY)*z/FY, z]), z

    # 第 0 帧做参考 (世界系 = 第0帧相机系)
    Twc = np.eye(4)
    poses = [Twc.copy()]
    g = load_rgb(rgb_ts[0]); D = load_D(rgb_ts[0])
    kp, des = orb.detectAndCompute(g, None)
    kf_pts, kf_des = [], []
    base_obs = []
    for j, k in enumerate(kp):
        r = backproj(k.pt[0], k.pt[1], D)
        if r is None:
            continue
        kf_pts.append(r[0]); kf_des.append(des[j]); base_obs.append(k.pt)
    kf_pts = np.array(kf_pts, np.float64)
    kf_des = np.array(kf_des, np.uint8)
    Twc_kf = Twc.copy()
    kf_base = 0
    all_pts = [kf_pts]
    obs = [[0, j, float(base_obs[j][0]), float(base_obs[j][1])] for j in range(len(kf_pts))]
    n_kf = 1
    print("  kf#1 frame 0: %d pts" % len(kf_pts), flush=True)

    stat = []
    for i in range(1, len(rgb)):
        g = load_rgb(rgb_ts[i]); D = load_D(rgb_ts[i])
        kp, des = orb.detectAndCompute(g, None)
        if des is None or len(kp) < 10:
            poses.append(poses[-1].copy()); stat.append((i, 0, 0)); continue

        # --- 对当前关键帧点云做 PnP ---
        ms = bf.knnMatch(kf_des, des, k=2)
        good = [m for m, nb in ms if m.distance < 0.8*nb.distance]
        Twc_new = poses[-1].copy()
        n_in = 0
        inl = None
        if len(good) >= 12:
            obj = np.array([kf_pts[m.queryIdx] for m in good], np.float64)
            img = np.array([kp[m.trainIdx].pt for m in good], np.float64)
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                obj, img, K, None, flags=cv2.SOLVEPNP_ITERATIVE,
                reprojectionError=2.0, iterationsCount=500, confidence=0.9995)
            if ok and inliers is not None and len(inliers) >= 12:
                R, _ = cv2.Rodrigues(rvec)
                T_cw = np.eye(4); T_cw[:3, :3] = R; T_cw[:3, 3] = tvec.ravel()
                Twc_new = np.linalg.inv(T_cw)
                n_in = int(len(inliers)); inl = inliers.ravel()
        poses.append(Twc_new.copy())

        # --- 记录观测: 只记录 PnP 内点, 且深度一致 ---
        if inl is not None and D is not None:
            Rcw = Twc_new[:3, :3].T; tcw = -Rcw @ Twc_new[:3, 3]
            for j in inl:
                gi = good[j].queryIdx
                u, v = kp[good[j].trainIdx].pt
                z_meas = D[int(round(v)), int(round(u))] if (0 <= int(round(v)) < D.shape[0]
                                                            and 0 <= int(round(u)) < D.shape[1]) else 0
                if not (MIN_D < z_meas < MAX_D):
                    continue
                z_pred = (Rcw @ kf_pts[gi] + tcw)[2]
                if z_pred <= 0 or abs(z_meas-z_pred)/max(z_pred, 1e-3) > 0.08:
                    continue
                obs.append([i, kf_base+gi, float(u), float(v)])

        # --- 关键帧切换: 视野重叠不足或走太远 ---
        moved = float(np.linalg.norm(Twc_new[:3, 3]-Twc_kf[:3, 3]))
        ratio = n_in/max(len(good), 1)
        if (n_in < 120 or ratio < 0.55 or moved > 0.15) and D is not None:
            new_pts, new_des, new_uv = [], [], []
            for j, k in enumerate(kp):
                r = backproj(k.pt[0], k.pt[1], D)
                if r is None:
                    continue
                new_pts.append(Twc_new[:3, :3] @ r[0] + Twc_new[:3, 3])
                new_des.append(des[j]); new_uv.append(k.pt)
            if len(new_pts) >= 150:
                kf_pts = np.array(new_pts, np.float64)
                kf_des = np.array(new_des, np.uint8)
                kf_base = sum(len(p) for p in all_pts)
                all_pts.append(kf_pts)
                Twc_kf = Twc_new.copy()
                n_kf += 1
                for j in range(len(new_pts)):
                    obs.append([i, kf_base+j, float(new_uv[j][0]), float(new_uv[j][1])])
        if verbose and i % 100 == 0:
            print("  f%4d kf=%d kfpts=%d inl=%d obs=%d" % (i, n_kf, len(kf_pts), n_in, len(obs)), flush=True)
        stat.append((i, n_in, len(kf_pts)))

    poses = np.array(poses)
    pts = np.vstack(all_pts)
    obs = np.array(obs, np.float64)
    np.savez(os.path.join(seq, "slam_front.npz"), poses=poses, points=pts, obs=obs, K=K)

    gt_xyz = np.array([gt[nearest(t, gt)][1][0:3] for t in rgb_ts[:len(poses)]])
    np.savetxt(os.path.join(seq, "gt_xyz.txt"), gt_xyz)
    np.savetxt(os.path.join(seq, "gt_quat.txt"), np.array([gt[nearest(t, gt)][1] for t in rgb_ts[:len(poses)]]))
    print("  points=%d obs=%d keyframes=%d" % (len(pts), len(obs), n_kf), flush=True)
    return poses, pts, obs, gt_xyz


# ---------------- stage 2: 滑窗 BA ----------------
def rodrigues_many(rv):
    R = np.zeros((len(rv), 3, 3))
    for i in range(len(rv)):
        R[i], _ = cv2.Rodrigues(rv[i].reshape(3, 1))
    return R


def ba(seq=SEQ, W=40, step=20, quota=250, min_obs=3, max_nfev=50):
    print("[stage 2] 滑窗 BA", flush=True)
    from scipy.optimize import least_squares
    from scipy.sparse import csr_matrix
    from collections import defaultdict

    d = np.load(os.path.join(seq, "slam_front.npz"))
    G_poses, G_pts, G_obs, Kk = d["poses"], d["points"], d["obs"], d["K"]
    F = len(G_poses)
    gt = np.loadtxt(os.path.join(seq, "gt_xyz.txt"))
    byf = defaultdict(list)
    for o in G_obs:
        byf[int(o[0])].append(o)

    opt = {i: G_poses[i].copy() for i in range(F)}
    print("  frames=%d pts=%d obs=%d" % (F, len(G_pts), len(G_obs)), flush=True)

    nwin = 0
    for start in range(0, F-3, step):
        end = min(start+W, F)
        frames = list(range(start, end))
        if len(frames) < 5:
            break
        obs_w = np.array([o for f in frames for o in byf[f]], np.float64)
        if len(obs_w) < 200:
            continue
        # 重叠帧(除第一轮外)保持上一轮全局解不变, 只优化新帧:
        # 这样相邻窗口在重叠区不会给出互相矛盾的结果, 消除拼接跳变
        fixed = set()
        if nwin > 0 and start > 0:
            for f in frames:
                if f < start + step:
                    fixed.add(f)
        # 每帧配额选点, 保证帧间均衡
        # 统计每个点在"全序列"里的观测次数, 与在"窗口内"的次数
        cnt = defaultdict(int); first = {}
        for o in obs_w:
            p = int(o[1]); cnt[p] += 1
            f0 = int(o[0])
            if p not in first or f0 < first[p]:
                first[p] = f0
        global_cnt = defaultdict(int)
        for o in G_obs:
            global_cnt[int(o[1])] += 1
        # 只优化"观测全部落在窗口内"的点:
        # 否则点会被窗口外(位姿被固定)的观测拽走, 反而把轨迹带偏
        cand = sorted([p for p, c in cnt.items()
                       if c >= min_obs and cnt[p] == global_cnt[p]],
                      key=lambda p: -cnt[p])
        q = defaultdict(int); keep = []
        for p in cand:
            f0 = first[p]
            if q[f0] < quota:
                keep.append(p); q[f0] += 1
        if len(keep) < 30:
            continue
        ks = set(keep)
        # 关键: 只保留"两端都在窗口内"的观测.
        # obs_w 已经只含窗口内帧的观测, 所以直接用 obs_w 过滤即可(不要再去查全局 obs).
        sel = np.array([o for o in obs_w if int(o[1]) in ks], np.float64)
        fidx = {f: i for i, f in enumerate(frames)}
        pidx = {p: i for i, p in enumerate(keep)}
        N, npt = len(frames), len(keep)
        npar = N*6 + npt*3
        of = np.array([fidx[int(o[0])] for o in sel], int)
        op = np.array([pidx[int(o[1])] for o in sel], int)
        ouv = sel[:, 2:4]
        nobs = len(sel)

        x0 = np.zeros(npar)
        for f, i in fidx.items():
            P = opt[f]
            r, _ = cv2.Rodrigues(P[:3, :3])
            x0[i*6:i*6+3] = r.ravel(); x0[i*6+3:i*6+6] = P[:3, 3]
        # 固定帧(重叠区)的在参数向量里的下标, 优化时用大幅阻尼锁住
        fix_idx = []
        for f in fixed:
            if f in fidx:
                i = fidx[f]
                fix_idx.extend(range(i*6, i*6+6))
        fix_idx = np.array(fix_idx, int)
        for p, j in pidx.items():
            x0[N*6+j*3:N*6+j*3+3] = G_pts[p]

        def residual(x):
            pr = x[:N*6].reshape(N, 6); pts = x[N*6:].reshape(npt, 3)
            R = rodrigues_many(pr[:, :3])
            # 位姿存的是 T_wc = [R|t], 观测模型必须是 X_cam = R^T (X_w - t)
            Xc = np.einsum("nij,nj->ni", R[of].transpose(0, 2, 1), pts[op] - pr[:, 3:6][of])
            z = Xc[:, 2].copy(); z[z < 1e-6] = 1e-6
            du = FX*Xc[:, 0]/z + CX - ouv[:, 0]
            dv = FY*Xc[:, 1]/z + CY - ouv[:, 1]
            r = np.empty(2*nobs); r[0::2] = du; r[1::2] = dv
            return r

        def jacobian(x):
            pr = x[:N*6].reshape(N, 6); pts = x[N*6:].reshape(npt, 3)
            R = rodrigues_many(pr[:, :3])
            Xc = np.einsum("nij,nj->ni", R[of].transpose(0, 2, 1), pts[op] - pr[:, 3:6][of])
            X, Y, Z = Xc[:, 0], Xc[:, 1], Xc[:, 2]
            zz = np.where(np.abs(Z) < 1e-6, 1e-6, Z)
            iz = 1.0/zz; iz2 = iz*iz
            du = np.stack([FX*iz, np.zeros(nobs), -FX*X*iz2], 1)
            dv = np.stack([np.zeros(nobs), FY*iz, -FY*Y*iz2], 1)
            J2 = np.stack([du, dv], 1).reshape(nobs, 2, 3)
            ru = np.arange(0, 2*nobs, 2); rv = np.arange(1, 2*nobs, 2)
            idx = np.arange(0, nobs*6, 6)

            JR = np.einsum("nij,njk->nik", J2, R[of].transpose(0, 2, 1))
            pc = N*6 + op*3
            a = np.empty(nobs*6, int); b = np.empty(nobs*6, int); c = np.empty(nobs*6)
            for t in range(3):
                a[idx+2*t] = ru; a[idx+2*t+1] = rv
                b[idx+2*t] = pc+t; b[idx+2*t+1] = pc+t
                c[idx+2*t] = JR[:, 0, t]; c[idx+2*t+1] = JR[:, 1, t]
            r_pt, c_pt, d_pt = a.copy(), b.copy(), c.copy()

            # dXc/dt = -R^T, 故 d(Xc)/dt 的链式: J2 @ (-R^T)
            JT = np.einsum("nij,njk->nik", J2, -R[of].transpose(0, 2, 1))
            fc = of*6
            for t in range(3):
                a[idx+2*t] = ru; a[idx+2*t+1] = rv
                b[idx+2*t] = fc+3+t; b[idx+2*t+1] = fc+3+t
                c[idx+2*t] = JT[:, 0, t]; c[idx+2*t+1] = JT[:, 1, t]
            r_t, c_t, d_t = a.copy(), b.copy(), c.copy()

            # Y = Xw - t, dXc/dw = R^T [Y]_x
            Yk = pts[op] - pr[:, 3:6][of]
            S = np.zeros((nobs, 3, 3))
            S[:, 0, 1] = -Yk[:, 2]; S[:, 0, 2] = Yk[:, 1]
            S[:, 1, 0] = Yk[:, 2]; S[:, 1, 1] = 0; S[:, 1, 2] = -Yk[:, 0]
            S[:, 2, 0] = -Yk[:, 1]; S[:, 2, 1] = Yk[:, 0]
            SR = np.einsum("nij,njk->nik", S, R[of].transpose(0, 2, 1))
            Jw = np.einsum("nij,njk->nik", J2, SR)
            for t in range(3):
                a[idx+2*t] = ru; a[idx+2*t+1] = rv
                b[idx+2*t] = fc+t; b[idx+2*t+1] = fc+t
                c[idx+2*t] = Jw[:, 0, t]; c[idx+2*t+1] = Jw[:, 1, t]
            r_w, c_w, d_w = a.copy(), b.copy(), c.copy()

            return csr_matrix((np.concatenate([d_pt, d_t, d_w]),
                               (np.concatenate([r_pt, r_t, r_w]),
                                np.concatenate([c_pt, c_t, c_w]))), shape=(2*nobs, npar))

        r0 = residual(x0)
        rms0 = float(np.sqrt((r0**2).mean()))

        # --- 先验阻尼: 把窗口内位姿轻微拉向"上一轮全局解", 避免重叠区两个窗口
        #     给出互相矛盾的位姿. 帧在重叠区被优化过多次时, 先验保证结果平滑.
        PRIOR_W = float(os.environ.get("SLAM_PRIOR_W", "3.0"))
        prior = x0.copy()
        if len(fix_idx) > 0:
            # 只对"重叠区帧"加强先验(权重 1e4 相当于固定), 其余帧不加
            M = len(fix_idx)
            def residual_aug(x):
                return np.concatenate([residual(x), PRIOR_W*(x[fix_idx]-prior[fix_idx])])
            def jacobian_aug(x):
                from scipy.sparse import csr_matrix as _csr, vstack as spvstack
                Jb = jacobian(x)
                Jp = _csr((np.full(M, PRIOR_W),
                           (np.arange(M), fix_idx)), shape=(M, npar))
                return spvstack([Jb, Jp], format="csr")
            out = least_squares(residual_aug, x0, jac=jacobian_aug, method="trf",
                                max_nfev=max_nfev, xtol=1e-10, ftol=1e-10, gtol=1e-10)
            rms1 = float(np.sqrt((out.fun[:2*nobs]**2).mean()))
        else:
            out = least_squares(residual, x0, jac=jacobian, method="trf",
                                max_nfev=max_nfev, xtol=1e-10, ftol=1e-10, gtol=1e-10)
            rms1 = float(np.sqrt((out.fun**2).mean()))
        for i, f in enumerate(frames):
            R, _ = cv2.Rodrigues(out.x[i*6:i*6+3].reshape(3, 1))
            P = np.eye(4); P[:3, :3] = R; P[:3, 3] = out.x[i*6+3:i*6+6]
            opt[f] = P
        nwin += 1
        gseg = gt[frames]
        def sate(pl):
            e = np.array([pl[f][:3, 3] for f in frames])
            R_, t_, _ = se3_align(e, gseg)
            al = (R_ @ e.T).T + t_
            return float(np.sqrt(((np.linalg.norm(al-gseg, axis=1))**2).mean()))
        print("  win%3d f%3d-%3d pts=%5d obs=%6d nfev=%2d  reproj %6.2f->%6.2f px  segATE %.4f->%.4f"
              % (nwin, frames[0], frames[-1], npt, nobs, out.nfev, rms0, rms1,
                 sate({i: G_poses[i] for i in frames}), sate(opt)), flush=True)
        if end >= F:
            break

    Pa = np.array([opt[i] for i in range(F)])
    np.savez(os.path.join(seq, "slam_ba.npz"), poses=Pa, points=G_pts, obs=G_obs, K=Kk)
    print("  窗口数=%d" % nwin, flush=True)
    return Pa


# ---------------- stage 3: 评估 ----------------
def evaluate(seq=SEQ):
    print("[stage 3] 评估 (SE3 对齐, 不含尺度)", flush=True)
    gt = np.loadtxt(os.path.join(seq, "gt_xyz.txt"))
    res = {}
    for tag, f in [("VO", "slam_front.npz"), ("VO+BA", "slam_ba.npz")]:
        if not os.path.exists(os.path.join(seq, f)):
            continue
        d = np.load(os.path.join(seq, f))
        P = d["poses"]; pts = d["points"]; obs = d["obs"]
        est = P[:, :3, 3]
        R_, t_, s_ = se3_align(est, gt, with_scale=False)
        al = (R_ @ est.T).T + t_
        e = np.linalg.norm(al-gt, axis=1)
        ate = float(np.sqrt((e**2).mean()))
        # 重投影必须在 SLAM 自己的坐标系里算(位姿与地图点自然一致).
        # 注意: 一旦引入尺度 s(se3_align with_scale) 观测模型就不再等价,
        # 所以重投影只用原始的量, 不做任何对齐.
        ff = obs[:, 0].astype(int); pf = obs[:, 1].astype(int)
        Rm = np.array([p[:3, :3] for p in P]); Tm = np.array([p[:3, 3] for p in P])
        # X_cam = R_wc^T (X_w - t_wc)
        Xc = np.einsum("nij,nj->ni", Rm[ff].transpose(0, 2, 1), pts[pf] - Tm[ff])
        z = Xc[:, 2].copy(); z[np.abs(z) < 1e-6] = 1e-6
        uv = np.stack([FX*Xc[:, 0]/z+CX, FY*Xc[:, 1]/z+CY], 1)
        re = np.linalg.norm(uv-obs[:, 2:4], axis=1)
        # 旋转误差: 与真值在同一参考系下比 (Tg[0]^-1 Tg[f])
        Tq = np.loadtxt(os.path.join(seq, "gt_quat.txt"))
        Rg = np.array([q2R(Tq[k][3:7]) for k in range(len(P))])
        Rg0 = Rg[0]
        rot_err = []
        for k in range(0, len(P), 10):
            rel = Rg0.T @ Rg[k]                 # 真值相对第一帧
            rot_err.append(rot_angle(P[k][:3, :3].T @ rel))
        rot_err = np.array(rot_err)
        res[tag] = dict(ate=ate, ate_mean=float(e.mean()), ate_max=float(e.max()),
                        reproj_med=float(np.median(re)),
                        reproj_in2=float(100*(re < 2).mean()),
                        rot_med=float(np.median(rot_err)), rot_max=float(rot_err.max()),
                        n=len(P))
    for tag, v in res.items():
        print("  %-6s ATE=%.4f m (mean %.4f, max %.4f)  重投影中位数=%.2f px  <2px=%.1f%%  旋转误差中位数=%.2f deg"
              % (tag, v["ate"], v["ate_mean"], v["ate_max"], v["reproj_med"], v["reproj_in2"], v["rot_med"]), flush=True)
    if "VO" in res and "VO+BA" in res:
        a, b = res["VO"]["ate"], res["VO+BA"]["ate"]
        print("  ATE 变化: %.4f -> %.4f  (%.1f%%)" % (a, b, (a-b)/a*100), flush=True)
    with open(os.path.join(seq, "eval_result.json"), "w") as fp:
        json.dump(res, fp, indent=2, ensure_ascii=False)
    return res


def make_plot(seq=SEQ):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    gt = np.loadtxt(os.path.join(seq, "gt_xyz.txt"))
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    ax[0].plot(gt[:, 0], gt[:, 2], "g-", lw=2.5, label="ground truth")
    for tag, f, st in [("VO", "slam_front.npz", "b--"), ("VO+BA", "slam_ba.npz", "r-")]:
        p = os.path.join(seq, f)
        if not os.path.exists(p):
            continue
        P = np.load(p)["poses"]; est = P[:, :3, 3]
        R_, t_, _ = se3_align(est, gt)
        al = (R_ @ est.T).T + t_
        e = np.linalg.norm(al-gt, axis=1)
        ate = np.sqrt((e**2).mean())
        ax[0].plot(al[:, 0], al[:, 2], st, lw=1.8, label="%s (ATE=%.3f m)" % (tag, ate))
        ax[1].plot(e, st, label="%s (mean %.3f)" % (tag, e.mean()))
    ax[0].set_xlabel("X (m)"); ax[0].set_ylabel("Z (m)")
    ax[0].set_title("RGB-D SLAM on TUM fr1/xyz"); ax[0].legend(); ax[0].grid(True); ax[0].axis("equal")
    ax[1].set_xlabel("frame"); ax[1].set_ylabel("position error (m)")
    ax[1].set_title("Error over time"); ax[1].legend(); ax[1].grid(True)
    plt.tight_layout()
    out = os.path.join(seq, "slam_result.png")
    plt.savefig(out, dpi=150)
    print("  图:", out, flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    s = sys.argv[2] if len(sys.argv) > 2 else SEQ
    if cmd in ("front", "all"):
        front(s)
    if cmd in ("ba", "all"):
        ba(s)
    if cmd in ("eval", "all"):
        evaluate(s)
        make_plot(s)
