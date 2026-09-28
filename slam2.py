# -*- coding: utf-8 -*-
"""
RGB-D SLAM v2 —— 持久地图 + 局部BA (修掉"点一次固化"的根因)
============================================================
v1 失败根因:
  地图点在关键帧建立那一刻用当时位姿反投影 -> 位姿误差被固化进点坐标
  -> BA 把错误几何拟合得更紧 -> 重投影变好但轨迹变差
v2 设计:
  1. 持久地图: 点一旦建立就留在 map 里, 每个点记录它的所有观测(帧, u, v)
  2. 共视关系: 每帧记录它观测到的点集合; 用共视点数决定"局部窗口"包含哪些帧
  3. 局部BA: 每次关键帧切换后, 取 [当前帧 + 共视最强的若干关键帧] 组成局部窗口,
     窗口内 位姿与点 联合优化 -> 点会被其他帧的观测"拉正", 不再固化
  4. 滑动优化: 关键帧带固定窗口一起优化, 保证连续性
流程:
  stage front : 跟踪 + 建图 (输出持久地图)
  stage lba   : 局部BA (滚动优化, 每步只优化局部窗口)
  stage eval  : 评估
用法: python slam2.py <front|lba|eval|all> [序列目录]
"""
import sys, os, json
import numpy as np
import cv2

# Default sequence.  Override on the command line:
#   python slam2.py all <sequence-dir>
SEQ = (r"C:\Users\r26304\Documents\codex-shop\slam_vo\data\rgbd_dataset_freiburg1_xyz")
SEQ_DESK = (r"C:\Users\r26304\Documents\codex-shop\slam_vo\data\rgbd_dataset_freiburg1_desk")
K = np.array([[517.3, 0, 318.6], [0, 516.5, 255.3], [0, 0, 1]], np.float64)
DEPTH_SCALE = 5000.0
MIN_D, MAX_D = 0.2, 6.0
FX, FY, CX, CY = K[0, 0], K[1, 1], K[0, 2], K[1, 2]


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


def se3_align(src, dst):
    n = len(src); ms = src.mean(0); md = dst.mean(0)
    Sc = src - ms; Dc = dst - md
    C = Dc.T @ Sc / n
    U, S, Vt = np.linalg.svd(C)
    d = np.ones(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        d[2] = -1
    R = U @ np.diag(d) @ Vt
    return R, md - R @ ms, 1.0


def rot_angle(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R)-1)/2, -1, 1))))


def _online_lba(kf, kf_poses, map_xyz, map_obs, map_frames, poses,
                kf_window=8, min_obs=2, max_pts=900, prior_w=1e3):
    """在线局部BA: 取与当前关键帧共视最强的若干关键帧, 联合优化位姿与点.
    返回当前关键帧优化后的位姿(供跟踪继续使用)."""
    from collections import defaultdict
    kfs = sorted(kf_poses.keys())
    cur_pts = set()
    for p, fs in enumerate(map_frames):
        if kf in fs:
            cur_pts.add(p)
    covis = []
    for other in kfs:
        if other == kf:
            continue
        o_pts = set()
        for p, fs in enumerate(map_frames):
            if other in fs:
                o_pts.add(p)
        inter = len(cur_pts & o_pts)
        if inter > 0:
            covis.append((inter, other))
    covis.sort(reverse=True)
    win = sorted(set([kf] + [f for _, f in covis[:kf_window]]))
    wf = set(win)
    cnt = defaultdict(int)
    for p in cur_pts:
        c = sum(1 for (f, u, v) in map_obs[p] if f in wf)
        if c >= min_obs:
            cnt[p] = c
    for f in win:
        for p, fs in enumerate(map_frames):
            if f in fs and p not in cnt:
                c = sum(1 for (ff, u, v) in map_obs[p] if ff in wf)
                if c >= min_obs:
                    cnt[p] = c
    keep = sorted(cnt, key=lambda p: -cnt[p])[:max_pts]
    if len(keep) < 30:
        return None
    sel = np.array([(p, f, u, v) for p in keep for (f, u, v) in map_obs[p]
                    if f in wf], np.float64)
    if len(sel) < 100:
        return None
    pose0 = {f: kf_poses[f] for f in win}
    xyz0 = {p: map_xyz[p] for p in keep}
    prior_idx = np.array(range(0, 6), int)   # 固定窗口里最早那一帧, 保证基准
    npose, nxyz, r0, r1, nfev = solve_window(
        sel, win, keep, pose0, xyz0, max_nfev=40,
        prior_idx=prior_idx, prior_w=prior_w)
    for f, T in npose.items():
        kf_poses[f] = T
        poses[f] = T
    for p, X in nxyz.items():
        map_xyz[p] = X
    return kf_poses.get(kf)


# ============ stage 1: 前端 + 持久地图 ============
def front(seq=SEQ, max_frames=None, verbose=True, online_ba=True):
    print("[1] 前端 + 建图", flush=True)
    rgb_all = read_list(os.path.join(seq, "rgb.txt"))
    rgb = rgb_all if max_frames is None else rgb_all[:max_frames]
    dl = read_list(os.path.join(seq, "depth.txt"))
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
        return np.array([(u-CX)*z/FX, (v-CY)*z/FY, z])

    # 持久地图
    map_xyz = []        # list of np.array(3)  世界系
    map_des = []        # 描述子
    map_obs = []        # list of list[(frame_idx, u, v)]
    map_frames = []     # list of set(frame_idx)  用于共视
    # 关键帧
    kf_poses = {}       # frame_idx -> Twc
    kf_kp = {}          # frame_idx -> (kp, des, point_ids:list)
    Twc = np.eye(4)
    poses = {}
    poses[0] = Twc.copy()
    kf_poses[0] = Twc.copy()

    g = load_rgb(rgb[0][0]); D = load_D(rgb[0][0])
    kp, des = orb.detectAndCompute(g, None)
    ids0 = []
    for j, k in enumerate(kp):
        Xc = backproj(k.pt[0], k.pt[1], D)
        if Xc is None:
            continue
        pid = len(map_xyz)
        map_xyz.append(Twc[:3, :3] @ Xc + Twc[:3, 3])
        map_des.append(des[j])
        map_obs.append([(0, k.pt[0], k.pt[1])])
        map_frames.append({0})
        ids0.append(pid)
    kf_kp[0] = (kp, des, ids0)
    print("  kf0: %d points" % len(ids0), flush=True)

    last_kf = 0
    for i in range(1, len(rgb)):
        g = load_rgb(rgb[i][0]); D = load_D(rgb[i][0])
        kp, des = orb.detectAndCompute(g, None)
        if des is None or len(kp) < 10:
            poses[i] = poses[i-1].copy(); continue

        # --- 用最近关键帧的点做 PnP ---
        last_kp, last_des, last_ids = kf_kp[last_kf]
        # 只取有有效深度、且被观测>=2次的点(更稳)
        cand = [p for p in last_ids if len(map_obs[p]) >= 2]
        if len(cand) < 20:
            cand = last_ids
        cidx = {p: t for t, p in enumerate(cand)}
        sub_des = np.array([map_des[p] for p in cand], np.uint8)
        ms = bf.knnMatch(sub_des, des, k=2)
        good = [m for m, nb in ms if m.distance < 0.8*nb.distance]

        Twc_new = poses[i-1].copy()
        n_in = 0; inl = None
        if len(good) >= 12:
            obj = np.array([map_xyz[cand[m.queryIdx]] for m in good], np.float64)
            img = np.array([kp[m.trainIdx].pt for m in good], np.float64)
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                obj, img, K, None, flags=cv2.SOLVEPNP_ITERATIVE,
                reprojectionError=2.0, iterationsCount=500, confidence=0.9995)
            if ok and inliers is not None and len(inliers) >= 12:
                R, _ = cv2.Rodrigues(rvec)
                T_cw = np.eye(4); T_cw[:3, :3] = R; T_cw[:3, 3] = tvec.ravel()
                Twc_new = np.linalg.inv(T_cw)
                n_in = int(len(inliers)); inl = inliers.ravel()
        poses[i] = Twc_new.copy()

        # --- 把 PnP 内点加入地图观测(持久, 后续BA会用) ---
        added = 0
        if inl is not None and D is not None:
            Rcw = Twc_new[:3, :3].T; tcw = -Rcw @ Twc_new[:3, 3]
            for j in inl:
                gi = cand[good[j].queryIdx]
                u, v = kp[good[j].trainIdx].pt
                ui, vi = int(round(u)), int(round(v))
                if not (0 <= vi < D.shape[0] and 0 <= ui < D.shape[1]):
                    continue
                z = float(D[vi, ui])
                if not (MIN_D < z < MAX_D):
                    continue
                zp = (Rcw @ map_xyz[gi] + tcw)[2]
                if zp <= 0 or abs(z-zp)/max(zp, 1e-3) > 0.08:
                    continue
                # 避免同帧重复观测
                if map_obs[gi] and map_obs[gi][-1][0] == i:
                    continue
                map_obs[gi].append((i, float(u), float(v)))
                map_frames[gi].add(i)
                added += 1

        # --- 是否新建关键帧 ---
        moved = float(np.linalg.norm(Twc_new[:3, 3] - kf_poses[last_kf][:3, 3]))
        ratio = n_in/max(len(good), 1)
        if (n_in < 120 or ratio < 0.55 or moved > 0.2) and D is not None:
            new_ids = []
            for j, k in enumerate(kp):
                Xc = backproj(k.pt[0], k.pt[1], D)
                if Xc is None:
                    continue
                pid = len(map_xyz)
                map_xyz.append(Twc_new[:3, :3] @ Xc + Twc_new[:3, 3])
                map_des.append(des[j])
                map_obs.append([(i, k.pt[0], k.pt[1])])
                map_frames.append({i})
                new_ids.append(pid)
            if len(new_ids) >= 150:
                kf_poses[i] = Twc_new.copy()
                kf_kp[i] = (kp, des, new_ids)
                last_kf = i
                # ===== 在线局部BA: 关键帧建立后立刻优化局部窗口, 并写回跟踪状态 =====
                if len(kf_poses) >= 3 and online_ba:
                    try:
                        Twc_new = _online_lba(
                            i, kf_poses, map_xyz, map_obs, map_frames, poses)
                    except Exception as _e:
                        pass
        if verbose and i % 100 == 0:
            print("  f%4d kf=%d pts=%d obs=%d inl=%d"
                  % (i, len(kf_poses), len(map_xyz), added, n_in), flush=True)

    # 保存
    P = np.array([poses[i] for i in range(len(rgb))])
    np.savez(os.path.join(seq, "map_v2.npz"),
             poses=P, xyz=np.array(map_xyz),
             des=np.array(map_des, np.uint8),
             obs=np.array([(p, f, u, v) for p in range(len(map_xyz))
                           for (f, u, v) in map_obs[p]], np.float64),
             kf=np.array(sorted(kf_poses.keys())), K=K)
    gt = read_gt(os.path.join(seq, "groundtruth.txt"))
    gt_xyz = np.array([gt[nearest(t, gt)][1][0:3] for t, _ in rgb])
    np.savetxt(os.path.join(seq, "gt_xyz.txt"), gt_xyz)
    np.savetxt(os.path.join(seq, "gt_quat.txt"),
               np.array([gt[nearest(t, gt)][1] for t, _ in rgb]))
    print("  关键帧=%d 点=%d 观测=%d"
          % (len(kf_poses), len(map_xyz), sum(len(o) for o in map_obs)), flush=True)
    return P


# ============ stage 2: 局部 BA ============
def rod(rv):
    R = np.zeros((len(rv), 3, 3))
    for i in range(len(rv)):
        R[i], _ = cv2.Rodrigues(rv[i].reshape(3, 1))
    return R


def solve_window(sel, frames, keep, pose0, xyz0, max_nfev=60, prior_idx=None, prior_w=1e4):
    """局部BA: 优化 frames 的位姿 + keep 的点. sel = (pid, fidx, u, v)"""
    from scipy.optimize import least_squares
    from scipy.sparse import csr_matrix, vstack as spvstack
    N, npt = len(frames), len(keep)
    npar = N*6 + npt*3
    fidx = {f: i for i, f in enumerate(frames)}
    pidx = {p: i for i, p in enumerate(keep)}
    pf = np.array([fidx[int(o[1])] for o in sel], int)
    pp = np.array([pidx[int(o[0])] for o in sel], int)
    uv = sel[:, 2:4]
    nobs = len(sel)

    x0 = np.zeros(npar)
    for f, i in fidx.items():
        r, _ = cv2.Rodrigues(pose0[f][:3, :3])
        x0[i*6:i*6+3] = r.ravel(); x0[i*6+3:i*6+6] = pose0[f][:3, 3]
    for p, j in pidx.items():
        x0[N*6+j*3:N*6+j*3+3] = xyz0[p]

    def residual(x):
        pr = x[:N*6].reshape(N, 6); p3 = x[N*6:].reshape(npt, 3)
        R = rod(pr[:, :3])
        Xc = np.einsum("nij,nj->ni", R[pf].transpose(0, 2, 1), p3[pp] - pr[:, 3:6][pf])
        z = Xc[:, 2].copy(); z[np.abs(z) < 1e-6] = 1e-6
        du = FX*Xc[:, 0]/z + CX - uv[:, 0]
        dv = FY*Xc[:, 1]/z + CY - uv[:, 1]
        r = np.empty(2*nobs); r[0::2] = du; r[1::2] = dv
        return r

    def jacobian(x):
        pr = x[:N*6].reshape(N, 6); p3 = x[N*6:].reshape(npt, 3)
        R = rod(pr[:, :3])
        Xc = np.einsum("nij,nj->ni", R[pf].transpose(0, 2, 1), p3[pp] - pr[:, 3:6][pf])
        X, Y, Z = Xc[:, 0], Xc[:, 1], Xc[:, 2]
        zz = np.where(np.abs(Z) < 1e-6, 1e-6, Z)
        iz = 1/zz; iz2 = iz*iz
        du = np.stack([FX*iz, np.zeros(nobs), -FX*X*iz2], 1)
        dv = np.stack([np.zeros(nobs), FY*iz, -FY*Y*iz2], 1)
        J2 = np.stack([du, dv], 1).reshape(nobs, 2, 3)
        ru = np.arange(0, 2*nobs, 2); rv = np.arange(1, 2*nobs, 2)
        idx = np.arange(0, nobs*6, 6)
        RT = R[pf].transpose(0, 2, 1)
        JR = np.einsum("nij,njk->nik", J2, RT)
        pc = N*6 + pp*3
        a = np.empty(nobs*6, int); b = np.empty(nobs*6, int); c = np.empty(nobs*6)
        for t in range(3):
            a[idx+2*t] = ru; a[idx+2*t+1] = rv
            b[idx+2*t] = pc+t; b[idx+2*t+1] = pc+t
            c[idx+2*t] = JR[:, 0, t]; c[idx+2*t+1] = JR[:, 1, t]
        r_pt, c_pt, d_pt = a.copy(), b.copy(), c.copy()
        JT = np.einsum("nij,njk->nik", J2, -RT)
        fc = pf*6
        for t in range(3):
            a[idx+2*t] = ru; a[idx+2*t+1] = rv
            b[idx+2*t] = fc+3+t; b[idx+2*t+1] = fc+3+t
            c[idx+2*t] = JT[:, 0, t]; c[idx+2*t+1] = JT[:, 1, t]
        r_t, c_t, d_t = a.copy(), b.copy(), c.copy()
        Yk = p3[pp] - pr[:, 3:6][pf]
        S = np.zeros((nobs, 3, 3))
        S[:, 0, 1] = -Yk[:, 2]; S[:, 0, 2] = Yk[:, 1]
        S[:, 1, 0] = Yk[:, 2]; S[:, 1, 2] = -Yk[:, 0]
        S[:, 2, 0] = -Yk[:, 1]; S[:, 2, 1] = Yk[:, 0]
        Jw = np.einsum("nij,njk->nik", J2, np.einsum("nij,njk->nik", S, RT))
        for t in range(3):
            a[idx+2*t] = ru; a[idx+2*t+1] = rv
            b[idx+2*t] = fc+t; b[idx+2*t+1] = fc+t
            c[idx+2*t] = Jw[:, 0, t]; c[idx+2*t+1] = Jw[:, 1, t]
        r_w, c_w, d_w = a.copy(), b.copy(), c.copy()
        J = csr_matrix((np.concatenate([d_pt, d_t, d_w]),
                        (np.concatenate([r_pt, r_t, r_w]),
                         np.concatenate([c_pt, c_t, c_w]))), shape=(2*nobs, npar))
        if prior_idx is not None and len(prior_idx) > 0:
            M = len(prior_idx)
            Jp = csr_matrix((np.full(M, prior_w),
                             (np.arange(M), prior_idx)), shape=(M, npar))
            J = spvstack([J, Jp], format="csr")
        return J

    def residual_aug(x):
        r = residual(x)
        if prior_idx is not None and len(prior_idx) > 0:
            return np.concatenate([r, prior_w*(x[prior_idx]-x0[prior_idx])])
        return r

    out = least_squares(residual_aug, x0, jac=jacobian, method="trf",
                        max_nfev=max_nfev, xtol=1e-10, ftol=1e-10, gtol=1e-10)
    rms0 = float(np.sqrt((residual(x0)**2).mean()))
    rms1 = float(np.sqrt((residual(out.x)**2).mean()))
    new_pose = {}
    for f, i in fidx.items():
        R, _ = cv2.Rodrigues(out.x[i*6:i*6+3].reshape(3, 1))
        P = np.eye(4); P[:3, :3] = R; P[:3, 3] = out.x[i*6+3:i*6+6]
        new_pose[f] = P
    new_xyz = {p: out.x[N*6+j*3:N*6+j*3+3].copy() for p, j in pidx.items()}
    return new_pose, new_xyz, rms0, rms1, out.nfev


def lba(seq=SEQ, kf_window=10, min_obs=2, max_pts=1200, max_nfev=60, verbose=True):
    print("[2] 局部 BA (滚动, 点参与优化)", flush=True)
    d = np.load(os.path.join(seq, "map_v2.npz"))
    P = d["poses"]; xyz = d["xyz"]; obs = d["obs"]; kfs = d["kf"].tolist()
    F = len(P)
    from collections import defaultdict
    bypoint = defaultdict(list)
    for o in obs:
        bypoint[int(o[0])].append((int(o[1]), o[2], o[3]))
    frame_pts = defaultdict(set)
    for o in obs:
        frame_pts[int(o[1])].add(int(o[0]))
    gtp = {i: P[i].copy() for i in range(F)}
    gxy = {i: xyz[i].copy() for i in range(len(xyz))}
    print("  帧=%d 点=%d 观测=%d 关键帧=%d" % (F, len(xyz), len(obs), len(kfs)), flush=True)

    gt = np.loadtxt(os.path.join(seq, "gt_xyz.txt"))

    def cur_ate(idx=None):
        ii = idx if idx is not None else list(range(F))
        e = np.array([gtp[i][:3, 3] for i in ii])
        g = gt[ii]
        R_, t_, _ = se3_align(e, g)
        al = (R_ @ e.T).T + t_
        return float(np.sqrt(((np.linalg.norm(al-g, axis=1))**2).mean()))

    nrun = 0
    for ki, kf in enumerate(kfs):
        if ki == 0:
            continue
        # --- 共视选窗口帧: 与当前关键帧共视点最多的若干关键帧 ---
        cur_pts = frame_pts.get(kf, set())
        covis = []
        for other in kfs[:ki]:
            o_pts = frame_pts.get(other, set())
            if not o_pts:
                continue
            covis.append((len(cur_pts & o_pts), other))
        covis.sort(reverse=True)
        win_frames = [kf] + [f for _, f in covis[:kf_window]]
        win_frames = sorted(set(win_frames))
        wf = set(win_frames)
        # --- 选点: 被窗口内>=2帧观测到的点 ---
        cnt = defaultdict(int)
        for p in cur_pts:
            c = sum(1 for (f, u, v) in bypoint[p] if f in wf)
            if c >= min_obs:
                cnt[p] = c
        # 再补充窗口内其它帧的共视点, 保证约束充足
        for f in win_frames:
            for p in frame_pts.get(f, ()):               
                if p in cnt:
                    continue
                c = sum(1 for (ff, u, v) in bypoint[p] if ff in wf)
                if c >= min_obs:
                    cnt[p] = c
        keep = sorted(cnt, key=lambda p: -cnt[p])[:max_pts]
        if len(keep) < 30:
            continue
        ks = set(keep)
        sel = np.array([(p, f, u, v) for p in keep for (f, u, v) in bypoint[p]
                        if f in win_frames], np.float64)
        if len(sel) < 100:
            continue
        # 固定: 窗口内但非当前关键帧的帧? 不固定, 但为保证连续性,
        # 对"两端"(最老的3帧)加先验
        prior_idx = []
        for f in win_frames[:3]:
            i = win_frames.index(f)
            prior_idx.extend(range(i*6, i*6+6))
        prior_idx = np.array(prior_idx, int) if prior_idx else None
        try:
            npose, nxyz, r0, r1, nfev = solve_window(
                sel, win_frames, keep, gtp, gxy, max_nfev=max_nfev,
                prior_idx=prior_idx, prior_w=1e3)
        except Exception as e:
            print("    kf%4d 跳过 (%s)" % (kf, type(e).__name__), flush=True)
            continue
        for f, T in npose.items():
            gtp[f] = T
        for p, X in nxyz.items():
            gxy[p] = X
        nrun += 1
        if verbose and nrun % 5 == 0:
            print("  lba%3d kf%4d frames=%d pts=%d obs=%d nfev=%d reproj %.2f->%.2f px  ATE=%.4f"
                  % (nrun, kf, len(win_frames), len(keep), len(sel), nfev, r0, r1, cur_ate()), flush=True)

    Pa = np.array([gtp[i] for i in range(F)])
    np.savez(os.path.join(seq, "map_v2_ba.npz"), poses=Pa,
             xyz=np.array([gxy[i] for i in range(len(xyz))]), kf=d["kf"], K=d["K"])
    print("  优化次数=%d  最终ATE=%.4f m" % (nrun, cur_ate()), flush=True)
    return Pa


# ============ stage 3: 评估 ============
def evaluate(seq=SEQ):
    print("[3] 评估", flush=True)
    gt = np.loadtxt(os.path.join(seq, "gt_xyz.txt"))
    Tq = np.loadtxt(os.path.join(seq, "gt_quat.txt"))
    Rg = np.array([q2R(Tq[k][3:7]) for k in range(len(Tq))])
    out = {}
    for tag, f in [("VO", "map_v2.npz"), ("VO+localBA", "map_v2_ba.npz")]:
        p = os.path.join(seq, f)
        if not os.path.exists(p):
            continue
        P = np.load(p)["poses"]
        est = P[:, :3, 3]
        R_, t_, _ = se3_align(est, gt)
        al = (R_ @ est.T).T + t_
        e = np.linalg.norm(al-gt, axis=1)
        ate = float(np.sqrt((e**2).mean()))
        rot = []
        for k in range(0, len(P), 10):
            rot.append(rot_angle(P[k][:3, :3].T @ (Rg[0].T @ Rg[k])))
        rot = np.array(rot)
        acc = np.linalg.norm(np.diff(al, 2, axis=0), axis=1)
        out[tag] = dict(ate=ate, ate_mean=float(e.mean()), ate_max=float(e.max()),
                        rot_med=float(np.median(rot)), jitter=float(acc.mean()))
    for tag, v in out.items():
        print("  %-11s ATE=%.4f m (mean %.4f max %.4f)  旋转中位数=%.2f deg  抖动=%.5f"
              % (tag, v["ate"], v["ate_mean"], v["ate_max"], v["rot_med"], v["jitter"]), flush=True)
    if "VO" in out and "VO+localBA" in out:
        a, b = out["VO"]["ate"], out["VO+localBA"]["ate"]
        print("  ATE: %.4f -> %.4f  (%.1f%%)" % (a, b, (a-b)/a*100), flush=True)
    json.dump(out, open(os.path.join(seq, "eval_v2.json"), "w"), indent=2, ensure_ascii=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    ax[0].plot(gt[:, 0], gt[:, 2], "g-", lw=2.5, label="ground truth")
    for tag, f, st in [("VO", "map_v2.npz", "b--"), ("VO+localBA", "map_v2_ba.npz", "r-")]:
        p = os.path.join(seq, f)
        if not os.path.exists(p):
            continue
        P = np.load(p)["poses"]; est = P[:, :3, 3]
        R_, t_, _ = se3_align(est, gt)
        al = (R_ @ est.T).T + t_
        e = np.linalg.norm(al-gt, axis=1)
        ax[0].plot(al[:, 0], al[:, 2], st, lw=1.8, label="%s (ATE=%.3f m)" % (tag, np.sqrt((e**2).mean())))
        ax[1].plot(e, st, label="%s (mean %.3f)" % (tag, e.mean()))
    ax[0].set_xlabel("X (m)"); ax[0].set_ylabel("Z (m)")
    ax[0].set_title("RGB-D SLAM (TUM fr1/xyz)"); ax[0].legend(); ax[0].grid(True); ax[0].axis("equal")
    ax[1].set_xlabel("frame"); ax[1].set_ylabel("position error (m)")
    ax[1].set_title("Error over time"); ax[1].legend(); ax[1].grid(True)
    plt.tight_layout()
    o = os.path.join(seq, "slam_v2.png")
    plt.savefig(o, dpi=150)
    print("  图:", o, flush=True)
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    s = sys.argv[2] if len(sys.argv) > 2 else SEQ
    if cmd in ("front", "all"):
        front(s)
    if cmd in ("lba", "all"):
        lba(s)
    if cmd in ("eval", "all"):
        evaluate(s)
