"""compare_frontends.py — 视觉前端 vs 激光前端对比

在同一段 TUM fr1/desk 数据上对比两种前端（无回环、无 BA 的纯里程计）。
这是「传感器前端可替换」设计的验证。
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

import cv2

from paths import TUM_DESK as D, RESULTS as _RES, imread_any
FX, FY, CX, CY = 517.3, 516.5, 318.6, 255.3
DEPTH_FACTOR = 5000.0


def ate_rmse(est_poses, gt_poses):
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
    return float(np.sqrt((err ** 2).mean()))


def load_data():
    depth_items = read_tum_list(os.path.join(D, "depth.txt"))
    depth_items = [(t, p) for t, p in depth_items
                   if os.path.exists(os.path.join(D, p))]
    gt_map = read_tum_groundtruth(os.path.join(D, "groundtruth.txt"))

    # 要对齐 RGB 和 depth（视觉前端需要图像）
    rgb_items = read_tum_list(os.path.join(D, "rgb.txt"))
    rgb_items = [(t, p) for t, p in rgb_items
                 if os.path.exists(os.path.join(D, p))]

    # 用 RGB 时间戳为主，找最近 depth
    pairs = []
    for rt, rp in rgb_items:
        best = min(depth_items, key=lambda x: abs(x[0] - rt))
        if abs(best[0] - rt) < 0.03:
            pairs.append((rt, rp, best[0], best[1]))

    return pairs, gt_map


def nearest_gt(gt_map, t):
    k = min(gt_map.keys(), key=lambda x: abs(x - t))
    return k, gt_map[k]


# ═══════════════════════════════════════════════════════════
def run_visual_odometry(pairs, gt_map, n_frames, verbose=False):
    """极简视觉里程计：ORB + 特征匹配 + PnP

    这是简化版（无 BA、无回环），用于和激光前端对比。
    """
    orb = cv2.ORB_create(nfeatures=1500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], dtype=np.float64)

    poses = []
    gt_list = []
    prev = None
    T_world = np.eye(4)

    for i in range(min(n_frames, len(pairs))):
        rt, rp, dt, dp = pairs[i]
        img = cv2.imread(os.path.join(D, rp), cv2.IMREAD_GRAYSCALE)
        k, gp = nearest_gt(gt_map, rt)
        gt_list.append(pose_matrix(gp))

        if img is None:
            poses.append(T_world.copy())
            continue

        kp, des = orb.detectAndCompute(img, None)

        if prev is None or des is None or prev[1] is None:
            poses.append(np.eye(4))
            prev = (kp, des)
            continue

        # 匹配
        matches = bf.match(des, prev[1])
        matches = sorted(matches, key=lambda m: m.distance)[:200]

        if len(matches) < 12:
            poses.append(T_world.copy())
            prev = (kp, des)
            if verbose:
                print(f"  [{i}] 匹配太少 ({len(matches)})")
            continue

        # 深度
        dep = cv2.imread(os.path.join(D, dp), cv2.IMREAD_UNCHANGED)
        if dep.ndim == 3:
            dep = dep[:, :, 0]
        dep = dep.astype(np.float64) / DEPTH_FACTOR

        # 当前帧的 3D 点
        pts3d = []
        pts2d = []
        for m in matches:
            u, v = kp[m.queryIdx].pt
            ui, vi = int(round(u)), int(round(v))
            if 0 <= vi < dep.shape[0] and 0 <= ui < dep.shape[1]:
                z = dep[vi, ui]
                if 0.3 < z < 4.0:
                    x = (u - CX) * z / FX
                    y = (v - CY) * z / FY
                    pts3d.append([x, y, z])
                    pts2d.append(prev[0][m.trainIdx].pt)

        if len(pts3d) < 8:
            poses.append(T_world.copy())
            prev = (kp, des)
            continue

        pts3d = np.array(pts3d, dtype=np.float64)
        pts2d = np.array(pts2d, dtype=np.float64)

        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts3d, pts2d, K, None, iterationsCount=100,
            reprojectionError=3.0, flags=cv2.SOLVEPNP_EPNP)

        if ok and inliers is not None and len(inliers) >= 8:
            R_rel, _ = cv2.Rodrigues(rvec)
            t_rel = tvec.ravel()
            # PnP: 把当前帧 3D 点变到上一帧坐标系，所以是 T_prev_from_cur
            T_prev_from_cur = np.eye(4)
            T_prev_from_cur[:3, :3] = R_rel
            T_prev_from_cur[:3, 3] = t_rel
            # 真值检验过：T_cur_from_prev = inv(T_prev_from_cur)
            T_cur_from_prev = np.linalg.inv(T_prev_from_cur)
            T_world = T_world @ T_cur_from_prev

        poses.append(T_world.copy())
        prev = (kp, des)

    return poses, gt_list


# ═══════════════════════════════════════════════════════════
def run_lidar_odometry(pairs, gt_map, n_frames, voxel=0.02, verbose=False):
    """激光前端：深度点云 ICP"""
    poses = []
    gt_list = []
    T_world = np.eye(4)
    prev_cloud = None
    prev_tree = None

    for i in range(min(n_frames, len(pairs))):
        rt, rp, dt, dp = pairs[i]
        dep = cv2.imread(os.path.join(D, dp), cv2.IMREAD_UNCHANGED)
        if dep.ndim == 3:
            dep = dep[:, :, 0]
        pts = depth_to_pointcloud(dep, max_depth=4.0)
        cloud = voxel_downsample(pts, voxel)

        k, gp = nearest_gt(gt_map, rt)
        gt_list.append(pose_matrix(gp))

        if prev_cloud is None:
            poses.append(np.eye(4))
            prev_cloud = cloud
            prev_tree = build_kdtree(cloud)
            continue

        R, t, info = icp_point_to_point(
            cloud[:, :3], prev_cloud[:, :3], target_tree=prev_tree,
            R0=np.eye(3), t0=np.zeros(3),
            max_iter=60, max_correspondence_dist=0.3)

        # 经严格验证：ICP 平移即 T_cur_from_prev
        T_cur_from_prev = pose_to_matrix(R, t)
        T_world = T_world @ T_cur_from_prev
        poses.append(T_world.copy())

        prev_cloud = cloud
        prev_tree = build_kdtree(cloud)

    return poses, gt_list


# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 150

    print("=" * 70)
    print(f"视觉前端 vs 激光前端（TUM fr1/desk，前 {n} 帧，均无 BA/回环）")
    print("=" * 70)

    pairs, gt_map = load_data()
    print(f"可用帧对: {len(pairs)}")

    print("\n[1/2] 视觉里程计（ORB + PnP）...")
    vp, vg = run_visual_odometry(pairs, gt_map, n)
    v_ate = ate_rmse(vp, vg)
    v_path = np.sum(np.linalg.norm(
        np.diff(np.array([p[:3, 3] for p in vg]), axis=0), axis=1))
    print(f"  ATE: {v_ate*100:.2f} cm, 轨迹 {v_path:.2f} m")

    print("\n[2/2] 激光里程计（深度点云 ICP）...")
    lp, lg = run_lidar_odometry(pairs, gt_map, n)
    l_ate = ate_rmse(lp, lg)
    l_path = np.sum(np.linalg.norm(
        np.diff(np.array([p[:3, 3] for p in lg]), axis=0), axis=1))
    print(f"  ATE: {l_ate*100:.2f} cm, 轨迹 {l_path:.2f} m")

    print()
    print("=" * 70)
    print("结果汇总")
    print("=" * 70)
    print(f"{'前端':<20}{'ATE (cm)':>12}{'轨迹长度 (m)':>15}")
    print("-" * 70)
    print(f"{'视觉 (ORB+PnP)':<20}{v_ate*100:>12.2f}{v_path:>15.2f}")
    print(f"{'激光 (深度点云 ICP)':<20}{l_ate*100:>12.2f}{l_path:>15.2f}")

    os.makedirs(_RES, exist_ok=True)
    with open(os.path.join(_RES, "frontend_comparison.txt"), "w") as f:
        f.write(f"TUM fr1/desk, {n} frames, no BA/loop closure\n")
        f.write(f"visual_ATE_cm={v_ate*100:.2f}, visual_path_m={v_path:.2f}\n")
        f.write(f"lidar_ATE_cm={l_ate*100:.2f}, lidar_path_m={l_path:.2f}\n")
    print("\n已保存 results/frontend_comparison.txt")
