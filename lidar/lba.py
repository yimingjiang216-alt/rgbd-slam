"""
lba.py — 激光几何 BA（point-to-plane bundle adjustment）

与视觉 BA 的区别：
  视觉 BA 残差： r = π(R·X + t) - uv           （重投影误差，2 维）
  激光 BA 残差： r = n · (R·p_kf - q_map)      （点到面距离，1 维）

优化变量：
  - 窗口内关键帧位姿 T_wc (SE(3))
  - 地图点位置 q_map (3 维)

使用 scipy.optimize.least_squares 做稀疏最小二乘。

数学约定
--------
世界坐标系下的地图点 X。
位姿 T_wc = [R | t] 把相机/雷达坐标系的点变到世界：X_w = R·x_c + t。
反投影：x_c = R^T (X_w - t)

残差定义（源点固定，只优化位姿和地图点）：
  对每个观测 (kf i, 地图点 j)：
    x_c = R_i^T (X_j - t_i)          # 地图点变到第 i 帧坐标系
    这是"第 i 帧应该看到的地图点位置"
  但激光不投影，改为：用第 i 帧的原始点云点 p（在该帧坐标系），
    它的世界坐标 = R_i p + t_i
    地图点 X_j 带法向 n_j
    残差 = n_j · (R_i p + t_i - X_j)
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


# ─────────────────────────────────────────────────────────────
# 李群工具
# ─────────────────────────────────────────────────────────────
def skew(w):
    return np.array([[0, -w[2], w[1]],
                     [w[2], 0, -w[0]],
                     [-w[1], w[0], 0]], dtype=np.float64)


def so3_exp(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = skew(k)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def so3_log(R):
    c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    th = np.arccos(c)
    if th < 1e-12:
        return np.zeros(3)
    if abs(th - np.pi) < 1e-6:
        A = (R + np.eye(3)) / 2.0
        k = np.sqrt(np.maximum(np.diag(A), 0.0))
        n = np.linalg.norm(k)
        return th * (k / n if n > 1e-12 else np.array([1.0, 0, 0]))
    v = np.array([R[2, 1] - R[1, 2],
                  R[0, 2] - R[2, 0],
                  R[1, 0] - R[0, 1]]) / (2.0 * np.sin(th))
    return th * v


def se3_exp(xi):
    """xi = [rho(3), phi(3)] -> 4x4，左乘更新约定 T <- exp(xi) T"""
    rho, phi = xi[:3], xi[3:]
    R = so3_exp(phi)
    th = np.linalg.norm(phi)
    if th < 1e-12:
        V = np.eye(3) + 0.5 * skew(phi)
    else:
        K = skew(phi)
        V = (np.eye(3) + (1 - np.cos(th)) / th ** 2 * K
             + (th - np.sin(th)) / th ** 3 * (K @ K))
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = V @ rho
    return T


# ─────────────────────────────────────────────────────────────
# 核心：几何 BA
# ─────────────────────────────────────────────────────────────
def _pack(poses, points, kf_ids, pt_ids):
    """把状态压成向量：每帧 6 维 [t(3), phi(3)]，每点 3 维 xyz

    ★ 旋转参数化：R = so3_exp(phi)（直接重建，等价于右乘扰动语义）
    雅可比必须与此一致，见 geometry_ba 里的说明。
    """
    x = []
    for k in kf_ids:
        T = poses[k]
        x.extend(np.concatenate([T[:3, 3], so3_log(T[:3, :3])]))
    for p in pt_ids:
        x.extend(points[p])
    return np.array(x, dtype=np.float64)


def _unpack(x, n_kf, kf_ids, pt_ids):
    poses = {}
    for i, k in enumerate(kf_ids):
        b = 6 * i
        T = np.eye(4)
        T[:3, 3] = x[b:b + 3]
        T[:3, :3] = so3_exp(x[b + 3:b + 6])
        poses[k] = T
    points = {}
    off = 6 * n_kf
    for i, p in enumerate(pt_ids):
        b = off + 3 * i
        points[p] = x[b:b + 3].copy()
    return poses, points


def geometry_ba(obs, poses, points, kf_ids=None, fix_first=True,
                max_nfev=40, min_obs=3, verbose=False):
    """几何 BA：优化位姿和地图点

    参数
    ----
    obs : list of (kf_id, point_id, p_local(3,), normal_world(3,))
          p_local 是在 kf 坐标系下的原始观测点
          normal_world 是地图点在世界系下的法向
    poses : dict kf_id -> 4x4 (camera-to-world)
    points : dict point_id -> 3-array (世界系)
    kf_ids : 参与优化的帧（None = 全部）
    fix_first : 固定第一帧（消除规范自由度）
    """
    kf_ids = sorted(kf_ids or poses.keys())
    kf_set = set(kf_ids)
    fix_set = set([kf_ids[0]]) if fix_first else set()

    # 只保留窗口内的观测
    obs = [o for o in obs if o[0] in kf_set]
    if not obs:
        return poses, points, None

    # 统计每个点的观测数，过滤太少的
    from collections import defaultdict
    pc = defaultdict(int)
    for o in obs:
        pc[o[1]] += 1
    valid_pts = {p for p, c in pc.items() if c >= min_obs}
    obs = [o for o in obs if o[1] in valid_pts]
    if not obs:
        return poses, points, None

    pt_ids = sorted(valid_pts)
    point_ids_set = set(pt_ids)

    # 构建索引
    kf_index = {k: i for i, k in enumerate(kf_ids)}
    pt_index = {p: i for i, p in enumerate(pt_ids)}

    # 观测按帧分组（加速残差计算）
    by_kf = defaultdict(list)
    for (k, p, pl, nw) in obs:
        by_kf[k].append((pt_index[p], pl, nw))

    # 固定帧：不加入优化变量（从状态向量里剔除），而不是用 bounds 卡住
    free_kf = [k for k in kf_ids if k not in fix_set]
    all_kf = kf_ids

    n_pt = len(pt_ids)
    n_res = len(obs)

    # 打包：只优化 free_kf 的位姿 + 全部点
    x_parts = []
    for k in free_kf:
        T = poses[k]
        x_parts.append(np.concatenate([T[:3, 3], so3_log(T[:3, :3])]))
    for p in pt_ids:
        x_parts.append(points[p])
    x0 = np.concatenate(x_parts).astype(np.float64)

    n_free = len(free_kf)

    def unpack(x):
        P = dict(poses)          # 固定帧保持原值
        for i, k in enumerate(free_kf):
            b = 6 * i
            T = np.eye(4)
            T[:3, 3] = x[b:b + 3]
            T[:3, :3] = so3_exp(x[b + 3:b + 6])
            P[k] = T
        X = {}
        off = 6 * n_free
        for i, p in enumerate(pt_ids):
            b = off + 3 * i
            X[p] = x[b:b + 3].copy()
        return P, X

    def residual(x):
        P, X = unpack(x)
        r = np.empty(n_res)
        row = 0
        for k in all_kf:
            T = P[k]
            R, t = T[:3, :3], T[:3, 3]
            for (pi, pl, nw) in by_kf[k]:
                pw = R @ pl + t
                r[row] = float(nw @ (pw - X[pt_ids[pi]]))
                row += 1
        return r

    res = least_squares(
        residual, x0,
        method="trf", loss="huber", f_scale=0.1,
        x_scale="jac", max_nfev=max_nfev, verbose=0)
    # 注：不传 jac，让 scipy 用数值差分。
    # 原因：参数化 R = so3_exp(phi) 下解析雅可比易错
    # （三种候选公式都与数值差分不符，见 resolve_jac.py / resolve_jac2.py），
    # 数值差分虽然慢，但正确性有保证。

    P_opt, X_opt = unpack(res.x)

    info = {
        "cost0": float(np.sum(residual(x0) ** 2)),
        "cost": float(np.sum(res.fun ** 2)),
        "nfev": int(res.nfev),
        "nframes": len(kf_ids),
        "n_free_frames": n_free,
        "npoints": n_pt,
        "nres": n_res,
        "status": int(res.status),
    }
    return P_opt, X_opt, info
