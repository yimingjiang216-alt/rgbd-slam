"""
lba2.py — 激光 BA（位姿主变量，地图点带先验）

问题：完全自由的点面 BA 里，地图点会把残差吸收掉，位姿几乎不动
      （实测位姿只移动 0.097cm，代价降 1e6 倍）。

解法：给地图点加「先验约束」——地图点不能随便动，只能小范围调整。
      用残差项：r_map = w_map * (X_j - X_j_prior)
      这样地图点有点「黏性」，位姿必须动起来才能降残差。

这也符合物理：地图点是多帧融合的结果，不该被单帧观测完全左右。
"""
import numpy as np
from scipy.optimize import least_squares

from lba import so3_exp, so3_log


def geometry_ba_prior(obs, poses, points, point_prior, kf_ids=None,
                      fix_first=True, max_nfev=40, min_obs=1,
                      w_map=1.0):
    """带地图点先验的几何 BA

    残差：
      r_obs = n_j^T (R_k pl + t_k - X_j)        （点面）
      r_prior = w_map * (X_j - X_j_prior)       （地图点黏性，3 维）

    w_map 越大，地图点越不动，位姿承担更多修正。
    """
    from collections import defaultdict

    kf_ids = sorted(kf_ids or poses.keys())
    kf_set = set(kf_ids)
    fix_set = set([kf_ids[0]]) if fix_first else set()

    obs = [o for o in obs if o[0] in kf_set]
    if not obs:
        return poses, points, None

    pc = defaultdict(int)
    for o in obs:
        pc[o[1]] += 1
    valid = {p for p, c in pc.items() if c >= min_obs}
    obs = [o for o in obs if o[1] in valid]
    if not obs:
        return poses, points, None

    pt_ids = sorted(valid)
    by_kf = defaultdict(list)
    pt_index = {p: i for i, p in enumerate(pt_ids)}
    for (k, p, pl, nw) in obs:
        by_kf[k].append((pt_index[p], pl, nw))

    free_kf = [k for k in kf_ids if k not in fix_set]
    n_pt = len(pt_ids)
    n_obs_res = len(obs)
    n_prior_res = 3 * n_pt
    n_res = n_obs_res + n_prior_res

    x_parts = []
    for k in free_kf:
        T = poses[k]
        x_parts.append(np.concatenate([T[:3, 3], so3_log(T[:3, :3])]))
    for p in pt_ids:
        x_parts.append(points[p])
    x0 = np.concatenate(x_parts).astype(np.float64)

    n_free = len(free_kf)
    priors = np.array([point_prior[p] for p in pt_ids])

    def unpack(x):
        P = dict(poses)
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
            X[p] = x[b:b + 3]
        return P, X

    def residual(x):
        P, X = unpack(x)
        r = np.empty(n_res)
        row = 0
        for k in kf_ids:
            T = P[k]
            R, t = T[:3, :3], T[:3, 3]
            for (pi, pl, nw) in by_kf[k]:
                pw = R @ pl + t
                r[row] = float(nw @ (pw - X[pt_ids[pi]]))
                row += 1
        # 地图点先验
        off = 6 * n_free
        for i in range(n_pt):
            b = off + 3 * i
            r[row:row + 3] = w_map * (x[b:b + 3] - priors[i])
            row += 3
        return r

    res = least_squares(residual, x0, method="trf", loss="linear",
                        x_scale="jac", max_nfev=max_nfev, verbose=0)

    P_opt, X_opt = unpack(res.x)
    info = {
        "cost0": float(np.sum(residual(x0) ** 2)),
        "cost": float(np.sum(res.fun ** 2)),
        "nfev": int(res.nfev),
        "nframes": len(kf_ids),
        "n_free_frames": n_free,
        "npoints": n_pt,
        "nres": n_res,
    }
    return P_opt, X_opt, info
