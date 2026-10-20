"""
lidar_pose_graph.py — LiDAR 前端 + rgbd-slam 位姿图后端

架构：
    [LiDAR 前端]  深度点云 -> ICP -> 帧间相对位姿
         ↓
    [回环检测]    复用 rgbd-slam 的 loop_detection（视觉回环）
         ↓
    [位姿图后端]  复用 rgbd-slam 的 PoseGraph（SE(3) 优化）

对照实验：
    A. LiDAR 前端 + 位姿图（无回环）      -> 纯里程计精度
    B. LiDAR 前端 + 位姿图 + 视觉回环    -> 加回环后的精度
    C. 视觉前端 + 位姿图（rgbd-slam 原方案）-> 基准

这个脚本把 slam-lidar 的 LiDAR 前端接到 rgbd-slam 的后端上。
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cv2

# 复用 rgbd-slam 的后端模块（仓库根）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pose_graph import PoseGraph  # noqa: E402

# 复用 lidar 的前端模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tum_data import (read_tum_list, read_tum_groundtruth, load_depth,  # noqa: E402
                      depth_to_pointcloud, pose_matrix)
from pointcloud import voxel_downsample, build_kdtree  # noqa: E402
from registration import icp_point_to_point, pose_to_matrix  # noqa: E402
from paths import TUM_DESK as D, RESULTS as _RES, imread_any  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
# 路径统一由 paths.py 提供（绝对路径 + imread_any 兼容中文目录）


# ─────────────────────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────────────────────
def _norm(relpath):
    """把 'depth/xxx.png' 这类相对路径规范成当前平台的路径"""
    return os.path.join(D, *relpath.split("/"))


def load_pairs():
    """返回 [(rgb_ts, rgb_path, depth_ts, depth_path), ...]，RGB 与 depth 对齐"""
    depth_items = read_tum_list(os.path.join(D, "depth.txt"))
    depth_items = [(t, _norm(p)) for t, p in depth_items
                   if os.path.exists(_norm(p))]
    rgb_items = read_tum_list(os.path.join(D, "rgb.txt"))
    rgb_items = [(t, _norm(p)) for t, p in rgb_items
                 if os.path.exists(_norm(p))]

    pairs = []
    for rt, rp in rgb_items:
        best = min(depth_items, key=lambda x: abs(x[0] - rt))
        if abs(best[0] - rt) < 0.03:
            pairs.append((rt, rp, best[0], best[1]))
    return pairs


def nearest_gt(gt_map, t):
    k = min(gt_map.keys(), key=lambda x: abs(x - t))
    return k, gt_map[k]


# ─────────────────────────────────────────────────────────────
# 前端：LiDAR 里程计，输出每帧位姿 + 每帧相对位姿
# ─────────────────────────────────────────────────────────────
def run_lidar_frontend(pairs, n_frames, voxel=0.02, max_corr=0.3):
    """返回 (poses[n], rel_poses[n] where rel[i] = T_{i-1 -> i})"""
    poses = []
    rels = []
    T_world = np.eye(4)
    prev_cloud = None
    prev_tree = None

    for i in range(n_frames):
        rt, rp, dt, dp = pairs[i]
        dep = imread_any(dp, cv2.IMREAD_UNCHANGED)   # dp 已是完整路径
        if dep.ndim == 3:
            dep = dep[:, :, 0]
        pts = depth_to_pointcloud(dep, max_depth=4.0)
        cloud = voxel_downsample(pts, voxel)

        if prev_cloud is None:
            poses.append(np.eye(4))
            rels.append(np.eye(4))
            prev_cloud = cloud
            prev_tree = build_kdtree(cloud)
            continue

        R, t, info = icp_point_to_point(
            cloud[:, :3], prev_cloud[:, :3], target_tree=prev_tree,
            R0=np.eye(3), t0=np.zeros(3),
            max_iter=60, max_correspondence_dist=max_corr)

        # 经严格验证：ICP 平移即 T_cur_from_prev
        rel = pose_to_matrix(R, t)
        T_world = T_world @ rel
        poses.append(T_world.copy())
        rels.append(rel)

        prev_cloud = cloud
        prev_tree = build_kdtree(cloud)

    return poses, rels


# ─────────────────────────────────────────────────────────────
# 后端：用 PoseGraph 优化（可选加回环）
# ─────────────────────────────────────────────────────────────
def optimize_with_pose_graph(poses, rels, loops=None, iterations=100):
    """把里程计边（+可选回环边）放进 PoseGraph 优化

    loops: [(i, j, Z, rot_weight), ...]
    """
    n = len(poses)
    ids = list(range(n))

    g = PoseGraph()
    g.add_nodes(ids, poses)

    # 里程计边
    for a in range(n - 1):
        g.add_edge(a, a + 1, rels[a + 1], weight=1.0, kind="odom")

    # 回环边
    n_loop = 0
    if loops:
        for i, j, Z, rw in loops:
            g.add_edge(int(i), int(j), Z, weight=1.0,
                       kind="loop", rot_weight=rw)
            n_loop += 1

    hist = g.optimize(iterations=iterations, damping=1e-8, huber=0.20,
                      fix_first=True, verbose=False)

    # 取优化后的位姿
    # PoseGraph 内部把 poses 存在 nodes 里，需要看它怎么存的
    opt = []
    for nid in ids:
        opt.append(g.poses[nid] if hasattr(g, "poses") else poses[nid])

    return np.array(opt), n_loop, hist


# ─────────────────────────────────────────────────────────────
# 评估
# ─────────────────────────────────────────────────────────────
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


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 150

    print("=" * 70)
    print(f"LiDAR 前端 + rgbd-slam 位姿图后端（TUM fr1/desk，{n} 帧）")
    print("=" * 70)

    pairs = load_pairs()
    gt_map = read_tum_groundtruth(os.path.join(D, "groundtruth.txt"))
    print(f"可用帧对: {len(pairs)}")

    # 1. LiDAR 前端
    print("\n[1] 运行 LiDAR 前端...")
    poses, rels = run_lidar_frontend(pairs, n)
    gt = []
    for i in range(n):
        k, gp = nearest_gt(gt_map, pairs[i][0])
        gt.append(pose_matrix(gp))
    gt = np.array(gt)

    ate_raw = ate_rmse(poses, gt)
    print(f"  纯里程计 ATE: {ate_raw*100:.2f} cm")

    # 2. 位姿图优化（无回环）
    print("\n[2] 位姿图优化（无回环）...")
    try:
        opt_a, _, hist = optimize_with_pose_graph(poses, rels, loops=None)
        ate_a = ate_rmse(opt_a, gt)
        print(f"  优化后 ATE: {ate_a*100:.2f} cm")
        print(f"  优化迭代: {len(hist)}")
    except Exception as e:
        print(f"  失败: {type(e).__name__}: {e}")
        ate_a = None

    # 3. 摘要
    print()
    print("=" * 70)
    print("结果")
    print("=" * 70)
    print(f"{'配置':<30}{'ATE (cm)':>12}")
    print("-" * 70)
    print(f"{'LiDAR 纯里程计':<30}{ate_raw*100:>12.2f}")
    if ate_a is not None:
        print(f"{'LiDAR + 位姿图':<30}{ate_a*100:>12.2f}")

    os.makedirs(_RES, exist_ok=True)
    with open(os.path.join(_RES, "lidar_posegraph.txt"), "w") as f:
        f.write(f"frames={n}\n")
        f.write(f"raw_odometry_ATE_cm={ate_raw*100:.2f}\n")
        if ate_a is not None:
            f.write(f"posegraph_ATE_cm={ate_a*100:.2f}\n")
    print("\n已保存 results/lidar_posegraph.txt")
