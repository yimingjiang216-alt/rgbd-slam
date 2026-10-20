"""tum_lidar_odometry.py — TUM 深度点云 LiDAR 里程计

与 KITTI 版的区别：
  - 数据源：深度图 -> 点云（不用 LiDAR 硬件）
  - 帧间位移小（1.6cm），零初值 ICP 可用
  - 输出与 rgbd-slam 兼容的 TUM 轨迹

这验证了「传感器前端可替换」的设计：后端（位姿累乘、评估）完全复用。
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

from tum_data import (read_tum_list, read_tum_groundtruth, load_depth,
                      depth_to_pointcloud, pose_matrix)
from pointcloud import voxel_downsample, build_kdtree
from registration import icp_point_to_point, icp_point_to_plane, pose_to_matrix
from pointcloud import estimate_normals

from paths import TUM_DESK as D_TUM, RESULTS as _RES, imread_any


def load_cloud(i, depths, voxel=0.02, max_depth=4.0):
    t, p = depths[i]
    img = load_depth(os.path.join(D_TUM, p))
    pts = depth_to_pointcloud(img, max_depth=max_depth)
    return voxel_downsample(pts, voxel), t


def run_odometry(method="p2pl", voxel=0.02, max_corr=0.3,
                 start=0, end=None, verbose=True):
    """跑 TUM 点云里程计"""
    depths = read_tum_list(os.path.join(D_TUM, "depth.txt"))
    depths = [(t, p) for t, p in depths if os.path.exists(os.path.join(D_TUM, p))]
    if end is None:
        end = len(depths)

    gt_map = read_tum_groundtruth(os.path.join(D_TUM, "groundtruth.txt"))

    import time
    poses = []
    timestamps = []
    timings = []
    T_world = np.eye(4)
    prev_cloud = None
    prev_tree = None
    prev_nrm = None

    for idx in range(start, end):
        t0 = time.time()
        cloud, ts = load_cloud(idx, depths, voxel)
        timestamps.append(ts)

        if prev_cloud is None:
            poses.append(np.eye(4))
            prev_cloud = cloud
            prev_tree = build_kdtree(cloud)
            if method == "p2pl":
                prev_nrm = estimate_normals(cloud)
            timings.append(time.time() - t0)
            if verbose:
                print(f"  [{idx}] 初始化 {len(cloud)} 点")
            continue

        # 零初值（TUM 位移小，够用）
        if method == "p2pl":
            R, t, info = icp_point_to_plane(
                cloud[:, :3], prev_cloud[:, :3], prev_nrm,
                target_tree=prev_tree,
                R0=np.eye(3), t0=np.zeros(3),
                max_iter=60, max_correspondence_dist=max_corr)
        else:
            R, t, info = icp_point_to_point(
                cloud[:, :3], prev_cloud[:, :3], target_tree=prev_tree,
                R0=np.eye(3), t0=np.zeros(3),
                max_iter=60, max_correspondence_dist=max_corr)

        # ICP 求 p_prev = R @ p_cur + t
        # 实测发现：这个变换的平移分量与「上一帧->当前帧」的真值位移同向，
        # 所以直接用它作为相对位姿，不要再取逆（取逆会翻转方向）。
        T_cur_from_prev = pose_to_matrix(R, t)

        T_world = T_world @ T_cur_from_prev
        poses.append(T_world)

        dt = time.time() - t0
        timings.append(dt)

        if verbose and (idx % 50 == 0 or idx == end - 1):
            tr = T_world[:3, 3]
            print(f"  [{idx}] t=({tr[0]:6.2f},{tr[1]:6.2f},{tr[2]:6.2f}) "
                  f"{len(cloud):6d}点 rmse={info['final_rmse']*100:.2f}cm "
                  f"{dt*1000:.0f}ms")

        prev_cloud = cloud
        prev_tree = build_kdtree(cloud)
        if method == "p2pl":
            prev_nrm = estimate_normals(cloud)

    return poses, timestamps, timings, gt_map


def ate_rmse(est_poses, gt_poses):
    """ATE RMSE，只做刚体对齐（与 rgbd-slam 同口径）"""
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


if __name__ == "__main__":
    method = sys.argv[1] if len(sys.argv) > 1 else "p2pl"

    print("=" * 70)
    print(f"TUM fr1/desk LiDAR 里程计（深度点云 + {method}）")
    print("=" * 70)

    n_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    poses, timestamps, timings, gt_map = run_odometry(
        method=method, end=n_frames, verbose=True)

    # 对齐真值
    est = []
    gt = []
    for p, ts in zip(poses, timestamps):
        k = min(gt_map.keys(), key=lambda x: abs(x - ts))
        if abs(k - ts) < 0.1:
            est.append(p)
            gt.append(pose_matrix(gt_map[k]))

    print()
    print(f"有效帧: {len(est)}/{len(poses)}")

    if len(est) > 10:
        rmse, errs = ate_rmse(est, gt)
        end_err = np.linalg.norm(est[-1][:3, 3] - gt[-1][:3, 3])
        path_len = np.sum(np.linalg.norm(
            np.diff(np.array([p[:3, 3] for p in gt]), axis=0), axis=1))

        print(f"\n结果:")
        print(f"  ATE RMSE:     {rmse*100:.2f} cm")
        print(f"  终点误差:     {end_err*100:.2f} cm")
        print(f"  轨迹长度:     {path_len:.2f} m")
        print(f"  平均耗时:     {np.mean(timings)*1000:.0f} ms/帧")
        print(f"  处理帧数:     {len(poses)}")

        # 保存
        os.makedirs(_RES, exist_ok=True)
        np.save(f"results/tum_lidar_{method}_poses.npy", np.array(est))
        np.save(f"results/tum_lidar_{method}_gt.npy", np.array(gt))
        with open(f"results/tum_lidar_{method}_summary.txt", "w") as f:
            f.write(f"method={method}\n")
            f.write(f"frames={len(poses)}\n")
            f.write(f"ATE_RMSE_cm={rmse*100:.2f}\n")
            f.write(f"end_error_cm={end_err*100:.2f}\n")
            f.write(f"path_len_m={path_len:.2f}\n")
        print(f"\n已保存至 results/")
