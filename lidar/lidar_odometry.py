"""
lidar_odometry.py — LiDAR 里程计

流程：
  1. 逐帧读点云，预处理（限距/去地面/体素下采样）
  2. 相邻帧 ICP 配准，得到相对位姿
  3. 累乘得到全局轨迹
  4. 与 KITTI 真值（oxts -> ENU）对比，算 ATE

输出与 rgbd-slam 兼容的 TUM 轨迹格式（timestamp x y z qx qy qz qw）
"""
import glob
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

from pointcloud import (load_kitti_bin, filter_range, filter_ground,
                        voxel_downsample, build_kdtree, estimate_normals)
from paths import imread_any, KITTI as KITTI_DIR  # noqa: E402
from registration import (icp_point_to_point, icp_point_to_plane,
                          transform_points, pose_to_matrix, _exp_so3)


# ─────────────────────────────────────────────────────────────
# KITTI 真值
# ─────────────────────────────────────────────────────────────
def read_oxts_pose(path: str):
    """读 KITTI oxts，返回 (R, t) 在局部 ENU 坐标系（以第一帧为原点）

    oxts 字段: [0]lat [1]lon [2]alt [3]roll [4]pitch [5]yaw
    """
    with open(path) as f:
        v = [float(x) for x in f.read().split()]
    lat, lon, alt = v[0], v[1], v[2]
    roll, pitch, yaw = v[3], v[4], v[5]
    return lat, lon, alt, roll, pitch, yaw


def mercator_projection(lat, lon, scale=1.0):
    """KITTI devkit 官方的墨卡托投影（把经纬度转成米）"""
    er = 6378137.0
    mx = scale * lon * np.pi * er / 180.0
    my = scale * er * np.log(np.tan((90.0 + lat) * np.pi / 360.0))
    return mx, my


def rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def roty(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def load_ground_truth(oxts_dir: str):
    """返回 list of 4x4 变换（相对第一帧）"""
    files = sorted(glob.glob(os.path.join(oxts_dir, "*.txt")))
    poses = []
    origin = None
    for fp in files:
        lat, lon, alt, roll, pitch, yaw = read_oxts_pose(fp)
        mx, my = mercator_projection(lat, lon)
        # KITTI 官方：R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
        R = rotz(yaw) @ roty(pitch) @ rotx(roll)
        t = np.array([mx, my, alt])

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t

        if origin is None:
            origin = np.linalg.inv(T)
        poses.append(origin @ T)
    return poses


# ─────────────────────────────────────────────────────────────
# 预处理
# ─────────────────────────────────────────────────────────────
def preprocess(path, max_range=50.0, voxel=0.5, remove_ground=True):
    p = load_kitti_bin(path)
    p = filter_range(p, max_range)
    if remove_ground:
        p = filter_ground(p)
    p = voxel_downsample(p, voxel)
    return p


# ─────────────────────────────────────────────────────────────
# 里程计主循环
# ─────────────────────────────────────────────────────────────
def run_odometry(velo_dir, n_frames=None, method="p2pl",
                 max_range=50.0, voxel=0.5, max_corr=2.0,
                 use_constant_velocity=True, verbose=True):
    """跑 LiDAR 里程计

    method: 'p2p' point-to-point | 'p2pl' point-to-plane
    返回 list of 4x4 变换（相对第一帧），以及每帧耗时
    """
    files = sorted(glob.glob(os.path.join(velo_dir, "*.bin")))
    if n_frames:
        files = files[:n_frames]
    if not files:
        raise FileNotFoundError(f"没有点云文件: {velo_dir}")

    import time
    poses = []
    timings = []
    T_world_prev = np.eye(4)    # 上一帧的全局位姿（相对第 0 帧）
    vel = np.eye(4)             # 常量速度模型：上一帧->当前帧的相对位姿

    cloud_prev = None
    tree_prev = None
    nrm_prev = None

    for i, fp in enumerate(files):
        t0 = time.time()
        cloud = preprocess(fp, max_range, voxel)

        if i == 0:
            poses.append(np.eye(4))
            cloud_prev = cloud
            tree_prev = build_kdtree(cloud[:, :3])
            if method == "p2pl":
                nrm_prev = estimate_normals(cloud)
            timings.append(time.time() - t0)
            if verbose:
                print(f"  frame {i:3d}: 初始化, {len(cloud)} 点")
            continue

        # 预测初值：ICP 要求的是「当前帧在上一帧坐标系下的位姿」，
        # 即 T_prev_from_cur，而 vel 存的是 T_cur_from_prev，所以要取逆。
        if use_constant_velocity:
            T_pred = np.linalg.inv(vel)
        else:
            T_pred = np.eye(4)
        R0 = T_pred[:3, :3]
        t0v = T_pred[:3, 3]

        # 配准：当前帧(source) -> 上一帧(target)
        # 输出 R_rel,t_rel 满足：p_prev = R_rel @ p_cur + t_rel
        # 即「当前帧在上一帧坐标系下的位姿」
        if method == "p2pl":
            R_rel, t_rel, info = icp_point_to_plane(
                cloud[:, :3], cloud_prev[:, :3], nrm_prev,
                target_tree=tree_prev, R0=R0, t0=t0v,
                max_iter=40, max_correspondence_dist=max_corr)
        else:
            R_rel, t_rel, info = icp_point_to_point(
                cloud[:, :3], cloud_prev[:, :3],
                target_tree=tree_prev, R0=R0, t0=t0v,
                max_iter=40, max_correspondence_dist=max_corr)

        # ★ 方向修正：ICP 给的是 p_prev = T @ p_cur，
        #   但我们要的是「从上一帧到当前帧的运动」T_cur_from_prev = T^{-1}
        T_prev_from_cur = pose_to_matrix(R_rel, t_rel)
        T_cur_from_prev = np.linalg.inv(T_prev_from_cur)

        # 全局位姿 = 上一帧全局位姿 @ (上一帧->当前帧)
        T_world_cur = T_world_prev @ T_cur_from_prev
        poses.append(T_world_cur)

        vel = T_cur_from_prev

        dt = time.time() - t0
        timings.append(dt)
        if verbose and (i % 10 == 0 or i == len(files) - 1):
            tr = T_world_cur[:3, 3]
            print(f"  frame {i:3d}: 全局t=({tr[0]:7.2f},{tr[1]:7.2f},{tr[2]:6.2f}) "
                  f"{len(cloud):5d}点 rmse={info['final_rmse']:.3f} {dt*1000:.0f}ms")

        # 保存为下一帧的 reference
        cloud_prev = cloud
        tree_prev = build_kdtree(cloud[:, :3])
        if method == "p2pl":
            nrm_prev = estimate_normals(cloud)
        T_world_prev = T_world_cur

    return poses, timings


# ─────────────────────────────────────────────────────────────
# 评估
# ─────────────────────────────────────────────────────────────
def ate_rmse(est_poses, gt_poses):
    """ATE RMSE（只做刚体对齐，不做相似变换，与 rgbd-slam 同口径）

    1. 取每帧平移
    2. 去重心
    3. 用 SVD 求最优 R（只旋转）
    4. 算 RMSE
    """
    n = min(len(est_poses), len(gt_poses))
    est = np.array([p[:3, 3] for p in est_poses[:n]])
    gt = np.array([p[:3, 3] for p in gt_poses[:n]])

    ec = est - est.mean(axis=0)
    gc = gt - gt.mean(axis=0)

    H = ec.T @ gc
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    aligned = (R @ ec.T).T + gt.mean(axis=0)
    err = np.linalg.norm(aligned - gt, axis=1)
    return float(np.sqrt((err ** 2).mean())), err


def rotation_error(est_poses, gt_poses):
    """相对旋转误差的中位数（度）"""
    n = min(len(est_poses), len(gt_poses))
    errs = []
    for i in range(1, n):
        Re = est_poses[i][:3, :3]
        Rg = gt_poses[i][:3, :3]
        dR = Re @ Rg.T
        c = (np.trace(dR) - 1) / 2
        c = np.clip(c, -1, 1)
        errs.append(np.degrees(np.arccos(c)))
    return float(np.median(errs)), errs


def save_tum(poses, timestamps, path):
    """保存为 TUM 格式：timestamp tx ty tz qx qy qz qw"""
    from scipy.spatial.transform import Rotation
    with open(path, "w") as f:
        for i, T in enumerate(poses):
            t = T[:3, 3]
            q = Rotation.from_matrix(T[:3, :3]).as_quat()  # x,y,z,w
            f.write(f"{timestamps[i]:.6f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} "
                    f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")
