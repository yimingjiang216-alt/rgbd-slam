"""depth_to_points.py — 深度图转点云 + TUM 数据加载

TUM 深度图是 16 位 PNG，单位 mm，因子 5000。
相机内参默认用 fr1 的标定值。
"""
import os

import numpy as np

from paths import imread_any


# TUM fr1 内参（官方标定）
FR1_FX = 517.3
FR1_FY = 516.5
FR1_CX = 318.6
FR1_CY = 255.3

DEPTH_FACTOR = 5000.0     # TUM depth: pixel / 5000 = meters


def read_tum_list(path):
    """读 rgb.txt / depth.txt，返回 [(timestamp, relpath), ...]"""
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                items.append((float(parts[0]), parts[1]))
    return items


def read_tum_groundtruth(path):
    """读 groundtruth.txt -> dict {timestamp: (tx,ty,tz,qx,qy,qz,qw)}"""
    gt = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            v = [float(x) for x in line.split()]
            gt[v[0]] = np.array(v[1:8])
    return gt


def quat_to_matrix(q):
    """四元数 (qx,qy,qz,qw) -> 3x3 旋转矩阵"""
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
    ])


def pose_matrix(pose7):
    """(tx,ty,tz,qx,qy,qz,qw) -> 4x4"""
    T = np.eye(4)
    T[:3, :3] = quat_to_matrix(pose7[3:7])
    T[:3, 3] = pose7[:3]
    return T


def depth_to_pointcloud(depth_img, fx=FR1_FX, fy=FR1_FY,
                        cx=FR1_CX, cy=FR1_CY,
                        depth_factor=DEPTH_FACTOR,
                        min_depth=0.3, max_depth=5.0,
                        step=2):
    """深度图 -> 点云 (N,3)，相机坐标系（x右 y下 z前）

    step: 像素采样步长（step=2 表示隔一个像素取一个，减少点数）
    """
    h, w = depth_img.shape
    # 构造像素网格
    us = np.arange(0, w, step)
    vs = np.arange(0, h, step)
    uu, vv = np.meshgrid(us, vs)          # (H', W')

    d = depth_img[::step, ::step].astype(np.float64) / depth_factor

    # 有效深度掩码
    valid = (d > min_depth) & (d < max_depth)

    z = d[valid]
    x = (uu[valid] - cx) * z / fx
    y = (vv[valid] - cy) * z / fy

    return np.stack([x, y, z], axis=1)


def load_depth(path):
    """读 TUM 16 位深度 PNG（imread_any 兼容 Windows 中文路径）"""
    import cv2
    img = imread_any(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        img = img[:, :, 0]
    return img


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from paths import TUM_DESK as D

    from paths import TUM_DESK as D, imread_any

    depths = read_tum_list(os.path.join(D, "depth.txt"))
    gts = read_tum_groundtruth(os.path.join(D, "groundtruth.txt"))

    print(f"深度帧: {len(depths)}, 真值帧: {len(gts)}")

    t0, p0 = depths[0]
    img = load_depth(os.path.join(D, p0))
    print(f"\n深度图: {img.shape} dtype={img.dtype} "
          f"范围 [{img.min()}, {img.max()}]")

    pts = depth_to_pointcloud(img)
    print(f"点云: {pts.shape}")
    print(f"  x: [{pts[:,0].min():.2f}, {pts[:,0].max():.2f}]")
    print(f"  y: [{pts[:,1].min():.2f}, {pts[:,1].max():.2f}]")
    print(f"  z: [{pts[:,2].min():.2f}, {pts[:,2].max():.2f}]")

    # 对齐真值
    print(f"\n时间戳 {t0:.3f} 的真值:")
    nearest = min(gts.keys(), key=lambda k: abs(k - t0))
    print(f"  最近真值 {nearest:.3f} (差 {abs(nearest-t0)*1000:.1f}ms)")
    print(f"  位姿: {np.round(gts[nearest], 3)}")
