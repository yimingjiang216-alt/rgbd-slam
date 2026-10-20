"""
pointcloud.py — 点云基础操作（自己实现，不依赖 open3d/PCL）

包含：
  - 点云读取（KITTI .bin）
  - 体素下采样
  - KD-Tree 最近邻（基于 scipy，保证正确性）
  - 法向量估计（可选，用于 point-to-plane ICP）
"""
import os

import numpy as np
from scipy.spatial import cKDTree


# ─────────────────────────────────────────────────────────────
# 读取
# ─────────────────────────────────────────────────────────────
def load_kitti_bin(path: str) -> np.ndarray:
    """读 KITTI Velodyne .bin -> (N, 4) float32  [x, y, z, intensity]"""
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 4 != 0:
        raise ValueError(f"文件大小不是 4 的倍数: {path}")
    return raw.reshape(-1, 4)


def filter_range(pts: np.ndarray, max_range: float = 50.0) -> np.ndarray:
    """剔除过远的点（KITTI 有 80 米外的稀疏点，对配准是噪声）"""
    xyz = pts[:, :3]
    d = np.linalg.norm(xyz, axis=1)
    return pts[d < max_range]


def filter_ground(pts: np.ndarray, z_min: float = -1.5, z_max: float = 0.5) -> np.ndarray:
    """粗略去掉地面点（KITTI 车身坐标系 z 向上，地面大约在 z=-1.7）"""
    z = pts[:, 2]
    keep = (z < z_min) | (z > z_max)
    return pts[keep]


# ─────────────────────────────────────────────────────────────
# 下采样
# ─────────────────────────────────────────────────────────────
def voxel_downsample(pts: np.ndarray, voxel_size: float = 0.5,
                     keep_intensity: bool = False) -> np.ndarray:
    """体素下采样：每个体素保留点坐标均值（可选保留强度均值）

    实现方式：把坐标除以体素边长取整得到体素索引，
    再用 np.unique 分组（这是最朴素但可控的做法）。
    """
    if len(pts) == 0:
        return pts
    xyz = pts[:, :3]
    keys = np.floor(xyz / voxel_size).astype(np.int64)

    # 把 3 维索引压成 1 维，便于 unique
    # 加偏移避免负数
    keys = keys - keys.min(axis=0)
    dims = keys.max(axis=0) + 1
    flat = keys[:, 0] * dims[1] * dims[2] + keys[:, 1] * dims[2] + keys[:, 2]

    uniq, inv, counts = np.unique(flat, return_inverse=True, return_counts=True)

    # 每组的坐标和
    out = np.zeros((len(uniq), 4), dtype=np.float32)
    for i in range(3):
        out[:, i] = np.bincount(inv, weights=xyz[:, i], minlength=len(uniq)) / counts
    if keep_intensity and pts.shape[1] > 3:
        out[:, 3] = np.bincount(inv, weights=pts[:, 3], minlength=len(uniq)) / counts
    return out


# ─────────────────────────────────────────────────────────────
# 最近邻
# ─────────────────────────────────────────────────────────────
def build_kdtree(pts: np.ndarray) -> cKDTree:
    return cKDTree(pts[:, :3] if pts.ndim == 2 and pts.shape[1] >= 3 else pts)


def nearest_neighbors(tree: cKDTree, query: np.ndarray, k: int = 1):
    """返回 (distances, indices)"""
    q = query[:, :3] if query.ndim == 2 and query.shape[1] >= 3 else query
    d, i = tree.query(q, k=k, workers=-1)
    return d, i


# ─────────────────────────────────────────────────────────────
# 法向量（用于 point-to-plane ICP）
# ─────────────────────────────────────────────────────────────
def estimate_normals(pts: np.ndarray, tree: cKDTree = None, k: int = 20) -> np.ndarray:
    """用 k 邻域 PCA 估法向量。

    对每个点取最近的 k 个邻居，算协方差矩阵，最小特征值对应的特征向量即法向。
    """
    xyz = pts[:, :3] if pts.shape[1] >= 3 else pts
    if tree is None:
        tree = cKDTree(xyz)
    k = min(k, len(xyz))
    _, idx = tree.query(xyz, k=k, workers=-1)

    # idx: (N, k)
    neigh = xyz[idx]                      # (N, k, 3)
    centered = neigh - neigh.mean(axis=1, keepdims=True)

    # 批量算协方差 (N, 3, 3)
    cov = np.einsum("nki,nkj->nij", centered, centered) / k

    # 特征分解
    w, v = np.linalg.eigh(cov)            # w 升序, v 列是特征向量
    normals = v[:, :, 0]                  # 最小特征值对应列

    # 方向统一（朝上，便于可视化；对配准影响由 point-to-plane 的绝对值处理）
    flip = normals[:, 2] < 0
    normals[flip] *= -1
    return normals.astype(np.float32)


def write_ply(path: str, pts: np.ndarray, rgb=None) -> None:
    """写最简单的 ascii ply，便于用 MeshLab 打开检查"""
    n = len(pts)
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for p in pts[:, :3]:
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    from paths import KITTI as base
    p = load_kitti_bin(os.path.join(base, "velodyne_points", "data", "0000000000.bin"))
    print(f"原始: {p.shape}")

    p2 = filter_range(p, 50.0)
    print(f"限距 50m: {p2.shape}")

    p3 = filter_ground(p2)
    print(f"去地面: {p3.shape}")

    p4 = voxel_downsample(p3, 0.5)
    print(f"体素 0.5m: {p4.shape}")

    t = build_kdtree(p4)
    print(f"KDTree 建立完成, n={t.n}")

    normals = estimate_normals(p4)
    print(f"法向量: {normals.shape}, 前 3 个: {np.round(normals[:3], 3).tolist()}")
    print(f"法向量单位长度检查: {np.linalg.norm(normals, axis=1).mean():.4f}")
