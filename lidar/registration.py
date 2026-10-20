"""
registration.py — 点云配准算法（自己实现）

包含：
  - point-to-point ICP
  - point-to-plane ICP
  - NDT（正态分布变换）
  - SVD 求解刚体变换

全部基于 numpy/scipy，不使用 open3d/PCL。

数学约定
--------
位姿 T = [R | t]，把 source 点变换到 target 坐标系：
    p_target = R @ p_source + t
"""
import numpy as np
from scipy.spatial import cKDTree


# ─────────────────────────────────────────────────────────────
# 基础：SVD 求最优刚体变换
# ─────────────────────────────────────────────────────────────
def best_fit_transform(A: np.ndarray, B: np.ndarray):
    """求 R, t 使 ||R A + t - B|| 最小（Kabsch / Umeyama 无尺度版）

    A, B: (N, 3)
    返回 R (3,3), t (3,)
    """
    centroid_A = A.mean(axis=0)
    centroid_B = B.mean(axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # H = A^T B，对 H 做 SVD
    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # 处理反射（det = -1 的情况）
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = centroid_B - R @ centroid_A
    return R, t


def transform_points(pts: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (R @ pts[:, :3].T).T + t


def pose_to_matrix(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def matrix_to_pose(T: np.ndarray):
    return T[:3, :3].copy(), T[:3, 3].copy()


# ─────────────────────────────────────────────────────────────
# 常量速度模型（给 ICP 一个初值）
# ─────────────────────────────────────────────────────────────
def identity_transform():
    return np.eye(3), np.zeros(3)


# ─────────────────────────────────────────────────────────────
# Point-to-Point ICP
# ─────────────────────────────────────────────────────────────
def icp_point_to_point(source: np.ndarray, target: np.ndarray,
                       target_tree: cKDTree = None,
                       R0: np.ndarray = None, t0: np.ndarray = None,
                       max_iter: int = 50, tol: float = 1e-4,
                       max_correspondence_dist: float = 1.0,
                       verbose: bool = False):
    """经典 ICP

    source: (N,3) 待配准点云
    target: (M,3) 参考点云
    返回 R, t, info(dict: iterations, final_rmse, n_pairs)
    """
    if R0 is None:
        R0, t0 = identity_transform()
    R, t = R0.copy(), t0.copy()

    src = source[:, :3].astype(np.float64)
    tgt = target[:, :3].astype(np.float64)
    if target_tree is None:
        target_tree = cKDTree(tgt)

    prev_rmse = np.inf
    n_pairs = 0
    for it in range(max_iter):
        # 1. 当前变换下的 source
        moved = transform_points(src, R, t)

        # 2. 找最近邻
        dists, idx = target_tree.query(moved, k=1, workers=-1)

        # 3. 剔除过远对应（outlier）
        mask = dists < max_correspondence_dist
        n_pairs = int(mask.sum())
        if n_pairs < 10:
            if verbose:
                print(f"  iter {it}: 有效对应太少 ({n_pairs})，停止")
            break

        src_sel = src[mask]
        tgt_sel = tgt[idx[mask]]

        # 4. 求增量变换（作用在原始 source 上）
        R_delta, t_delta = best_fit_transform(
            transform_points(src_sel, R, t), tgt_sel)

        # 5. 累乘
        R = R_delta @ R
        t = R_delta @ t + t_delta

        rmse = float(np.sqrt((dists[mask] ** 2).mean()))
        if verbose:
            print(f"  iter {it}: pairs={n_pairs} rmse={rmse:.5f}")
        if abs(prev_rmse - rmse) < tol:
            break
        prev_rmse = rmse

    info = {"iterations": it + 1, "final_rmse": float(rmse), "n_pairs": n_pairs}
    return R, t, info


# ─────────────────────────────────────────────────────────────
# Point-to-Plane ICP
# ─────────────────────────────────────────────────────────────
def icp_point_to_plane(source: np.ndarray, target: np.ndarray,
                       target_normals: np.ndarray,
                       target_tree: cKDTree = None,
                       R0: np.ndarray = None, t0: np.ndarray = None,
                       max_iter: int = 50, tol: float = 1e-4,
                       max_correspondence_dist: float = 1.0,
                       verbose: bool = False):
    """point-to-plane ICP

    最小化 sum( (n_i · (R p_i + t - q_i))^2 )
    对 (R,t) 线性化（小角度近似），解 6x6 线性系统 A x = b，
    x = [rx, ry, rz, tx, ty, tz]
    """
    if R0 is None:
        R0, t0 = identity_transform()
    R, t = R0.copy(), t0.copy()

    src = source[:, :3].astype(np.float64)
    tgt = target[:, :3].astype(np.float64)
    nrm = target_normals[:, :3].astype(np.float64)
    if target_tree is None:
        target_tree = cKDTree(tgt)

    prev_rmse = np.inf
    n_pairs = 0
    it = 0
    for it in range(max_iter):
        moved = transform_points(src, R, t)
        dists, idx = target_tree.query(moved, k=1, workers=-1)
        mask = dists < max_correspondence_dist
        n_pairs = int(mask.sum())
        if n_pairs < 10:
            break

        p = moved[mask]                 # 当前 source 位置
        q = tgt[idx[mask]]              # 对应 target 点
        n = nrm[idx[mask]]              # 对应法向

        # 残差标量
        r = np.einsum("ij,ij->i", n, p - q)

        # 构造 A: 每个对应一行 [ (p × n)^T , n^T ]
        cross = np.cross(p, n)          # (K,3)
        A = np.hstack([cross, n])       # (K,6)
        b = -r                          # (K,)

        # 解 x（带阻尼，避免病态）
        AtA = A.T @ A
        AtA += 1e-6 * np.eye(6)
        x = np.linalg.solve(AtA, A.T @ b)

        # 增量变换
        dR = _exp_so3(x[:3])
        dt = x[3:]

        R = dR @ R
        t = dR @ t + dt

        rmse = float(np.sqrt((dists[mask] ** 2).mean()))
        if verbose:
            print(f"  iter {it}: pairs={n_pairs} rmse={rmse:.5f} |x|={np.linalg.norm(x):.6f}")
        if abs(prev_rmse - rmse) < tol:
            break
        prev_rmse = rmse

    info = {"iterations": it + 1, "final_rmse": float(rmse), "n_pairs": n_pairs}
    return R, t, info


def _exp_so3(w: np.ndarray) -> np.ndarray:
    """so(3) -> SO(3)，罗德里格斯公式"""
    theta = np.linalg.norm(w)
    if theta < 1e-12:
        return np.eye(3)
    k = w / theta
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


# ─────────────────────────────────────────────────────────────
# NDT（正态分布变换）
# ─────────────────────────────────────────────────────────────
class NDT:
    """2D/3D NDT 配准

    思路：把 target 按体素划分，每格算高斯（均值+协方差）；
    优化时让 source 点落在目标格子的高斯分布里的概率最大。
    """

    def __init__(self, voxel_size: float = 1.0, max_iter: int = 30,
                 tol: float = 1e-4):
        self.voxel_size = voxel_size
        self.max_iter = max_iter
        self.tol = tol
        self.cells = {}          # 体素 key -> (mean, inv_cov, det_cov)

    def _key(self, p):
        return tuple(np.floor(p / self.voxel_size).astype(np.int64))

    def fit(self, target: np.ndarray):
        xyz = target[:, :3]
        buckets = {}
        for p in xyz:
            k = self._key(p)
            buckets.setdefault(k, []).append(p)

        self.cells = {}
        for k, pts in buckets.items():
            pts = np.array(pts)
            if len(pts) < 5:
                continue
            mean = pts.mean(axis=0)
            cov = np.cov(pts.T)
            if cov.ndim == 0 or np.linalg.det(cov) < 1e-9:
                continue
            cov += 1e-3 * np.eye(3)      # 正则化
            inv = np.linalg.inv(cov)
            det = np.linalg.det(cov)
            self.cells[k] = (mean, inv, det)
        return self

    def _score(self, p):
        """返回点 p 在 NDT 网格里的得分（对数似然），没有对应格返回 -inf"""
        k = self._key(p)
        if k not in self.cells:
            return -np.inf, None, None
        mean, inv, det = self.cells[k]
        d = p - mean
        return -0.5 * (d @ inv @ d) - 0.5 * np.log(det), mean, inv

    def align(self, source: np.ndarray, R0=None, t0=None):
        """用数值梯度做优化（简单但可跑）

        对 6 维参数 (rx,ry,rz,tx,ty,tz) 用有限差分求梯度，做梯度上升。
        """
        if R0 is None:
            R0, t0 = identity_transform()
        x = np.zeros(6)
        x[3:] = t0
        R, t = R0.copy(), t0.copy()
        # 记录当前 6 维表示的旋转
        r_vec = _log_so3(R)

        prev = -np.inf
        it = 0
        for it in range(self.max_iter):
            cur = x.copy()
            cur[:3] = r_vec
            score = self._total_score(source, cur)
            if score == -np.inf:
                break
            if abs(score - prev) < self.tol:
                break
            prev = score

            # 有限差分梯度
            g = np.zeros(6)
            eps = 1e-4
            for i in range(6):
                xp = x.copy()
                xp[:3] = r_vec
                xp[i] += eps
                s_p = self._total_score(source, xp)
                if s_p == -np.inf:
                    continue
                g[i] = (s_p - score) / eps

            # 梯度上升（步长随迭代衰减）
            lr = 0.05 / (1 + it * 0.2)
            r_vec = r_vec + lr * g[:3]
            x[3:] = x[3:] + lr * g[3:]

        R = _exp_so3(r_vec)
        t = x[3:]
        # 最终 rmse（用最近邻近似）
        info = {"iterations": it + 1, "final_rmse": float(self._nn_rmse(source, R, t))}
        return R, t, info

    def _total_score(self, source, x):
        R = _exp_so3(x[:3])
        t = x[3:]
        moved = transform_points(source, R, t)
        total = 0.0
        n_hit = 0
        for p in moved:
            s, _, _ = self._score(p)
            if s == -np.inf:
                continue
            total += s
            n_hit += 1
        if n_hit < len(source) * 0.3:     # 命中率太低认为失败
            return -np.inf
        return total / len(source)

    def _nn_rmse(self, source, R, t):
        # NDT 没有 target 点云数组，用格子均值近似
        means = np.array([v[0] for v in self.cells.values()]) if self.cells else np.zeros((1, 3))
        tree = cKDTree(means)
        moved = transform_points(source, R, t)
        d, _ = tree.query(moved, k=1, workers=-1)
        dd = d[np.isfinite(d)]
        return float(np.sqrt((dd ** 2).mean())) if len(dd) else np.inf


def _log_so3(R: np.ndarray) -> np.ndarray:
    """SO(3) -> so(3)"""
    c = (np.trace(R) - 1) / 2
    c = np.clip(c, -1.0, 1.0)
    theta = np.arccos(c)
    if theta < 1e-12:
        return np.zeros(3)
    if abs(theta - np.pi) < 1e-6:
        # 特殊情况，取对角
        A = (R + np.eye(3)) / 2
        k = np.sqrt(np.maximum(np.diag(A), 0))
        return theta * k / (np.linalg.norm(k) + 1e-12)
    w = np.array([R[2, 1] - R[1, 2],
                  R[0, 2] - R[2, 0],
                  R[1, 0] - R[0, 1]]) / (2 * np.sin(theta))
    return theta * w
