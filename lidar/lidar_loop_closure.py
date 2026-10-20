"""
lidar_loop_closure.py — LiDAR 回环：几何回环 + ICP 验证

与 rgbd-slam 的视觉回环不同，LiDAR 回环用：
  1. 候选：位姿距离 < 阈值（不需要外观检索）
  2. 验证：当前帧点云 vs 候选帧点云做 ICP，看是否能配准上
  3. 约束：ICP 给出的相对位姿，加入位姿图

这避开了视觉回环「视点变化导致描述子失配」的问题（那是 rgbd-slam 里 desk 序列的失败原因）。
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))   # 仓库根
from pose_graph import PoseGraph  # noqa: E402

sys.path.insert(0, _HERE)
from tum_data import (read_tum_list, read_tum_groundtruth,  # noqa: E402
                      depth_to_pointcloud, pose_matrix)
from pointcloud import voxel_downsample, build_kdtree  # noqa: E402
from registration import icp_point_to_point, pose_to_matrix  # noqa: E402

from paths import TUM_DESK as D, RESULTS as _RES, imread_any


def read_list(path):
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            if len(p) >= 2:
                items.append((float(p[0]), p[1]))
    return items


def to_path(rel):
    return os.path.join(D, *rel.split("/"))


def load_pairs():
    depths = read_list(os.path.join(D, "depth.txt"))
    depths = [(t, to_path(p)) for t, p in depths if os.path.exists(to_path(p))]
    rgbs = read_list(os.path.join(D, "rgb.txt"))
    rgbs = [(t, to_path(p)) for t, p in rgbs if os.path.exists(to_path(p))]

    pairs = []
    for rt, rp in rgbs:
        best = min(depths, key=lambda x: abs(x[0] - rt))
        if abs(best[0] - rt) < 0.03:
            pairs.append((rt, rp, best[0], best[1]))
    return pairs


def load_cloud(depth_path, voxel=0.02, max_depth=4.0):
    dep = imread_any(depth_path, cv2.IMREAD_UNCHANGED)
    if dep is None:
        return None
    if dep.ndim == 3:
        dep = dep[:, :, 0]
    pts = depth_to_pointcloud(dep, max_depth=max_depth)
    return voxel_downsample(pts, voxel)


def nearest_gt(gt_map, t):
    k = min(gt_map.keys(), key=lambda x: abs(x - t))
    return k, gt_map[k]


def ate_rmse(est, gt):
    n = min(len(est), len(gt))
    e = np.array([p[:3, 3] for p in est[:n]])
    g = np.array([p[:3, 3] for p in gt[:n]])
    ec = e - e.mean(axis=0)
    gc = g - g.mean(axis=0)
    H = ec.T @ gc
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    aligned = (R @ ec.T).T + g.mean(axis=0)
    err = np.linalg.norm(aligned - g, axis=1)
    return float(np.sqrt((err ** 2).mean()))


def find_loop_candidates(poses, min_gap=90, max_dist=0.30,
                         min_path_dist=1.5):
    """位姿距离找候选对，但排除「假回环」

    关键：两个条件同时满足才算候选
      1. 位姿距离小（看起来回到了原处）
      2. 沿轨迹走过的路程足够长（排除相邻帧）

    只检查 1 会找到大量假回环：里程计低估运动时，
    走了很远但估计位置没怎么变，会误判成闭合。
    """
    from scipy.spatial import cKDTree

    n = len(poses)
    P = np.array([p[:3, 3] for p in poses])
    tree = cKDTree(P)

    # 累计路程（用于判断"真的走了多远"）
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])

    cands = []
    for i in range(n):
        idx = tree.query_ball_point(P[i], r=max_dist)
        for j in idx:
            if j < i + min_gap:
                continue
            d = float(np.linalg.norm(P[i] - P[j]))
            # ★ 关键过滤：沿轨迹的路程要够长，否则是相邻帧误判
            path_d = abs(cum[j] - cum[i])
            if path_d < min_path_dist:
                continue
            cands.append((i, j, d, path_d))

    # 按位姿距离排序，优先最接近的
    cands = sorted(set(cands), key=lambda x: x[2])
    return cands


def verify_loop_with_icp(cloud_i, cloud_j, max_corr=0.3, min_inlier_ratio=0.35):
    """用 ICP 验证回环：把 cloud_i 配准到 cloud_j

    返回 (ok, Z, rmse, inlier_ratio)
    Z 是 T_cur_from_prev 形式的相对位姿（i 在 j 坐标系）
    """
    tree = build_kdtree(cloud_j)
    R, t, info = icp_point_to_point(
        cloud_i[:, :3], cloud_j[:, :3], target_tree=tree,
        R0=np.eye(3), t0=np.zeros(3),
        max_iter=80, max_correspondence_dist=max_corr)

    # 内点比例：配准后距离 < max_corr 的点占比
    from registration import transform_points
    moved = transform_points(cloud_i[:, :3], R, t)
    d, _ = tree.query(moved, k=1, workers=-1)
    ratio = float((d < max_corr).mean())
    rmse = info["final_rmse"]

    ok = (ratio >= min_inlier_ratio) and (rmse < max_corr)
    Z = pose_to_matrix(R, t)
    return ok, Z, rmse, ratio


def main(n_frames=613, min_gap=90, max_dist=0.30, verbose=True):
    pairs = load_pairs()
    gt_map = read_tum_groundtruth(os.path.join(D, "groundtruth.txt"))

    n = min(n_frames, len(pairs))
    print("=" * 72)
    print(f"LiDAR 回环 + 位姿图（TUM fr1/desk，{n} 帧）")
    print("=" * 72)

    # 1. 前端跑一遍，得到里程计位姿 + 点云
    clouds = []
    poses = []
    rels = []
    T_world = np.eye(4)
    prev_cloud = None
    prev_tree = None

    for i in range(n):
        rt, rp, dt, dp = pairs[i]
        cl = load_cloud(dp)
        if cl is None or len(cl) < 100:
            clouds.append(None)
            poses.append(T_world.copy())
            rels.append(np.eye(4))
            continue
        clouds.append(cl)

        if prev_cloud is None:
            poses.append(np.eye(4))
            rels.append(np.eye(4))
            prev_cloud, prev_tree = cl, build_kdtree(cl)
            continue

        R, t, info = icp_point_to_point(
            cl[:, :3], prev_cloud[:, :3], target_tree=prev_tree,
            R0=np.eye(3), t0=np.zeros(3),
            max_iter=60, max_correspondence_dist=0.3)
        rel = pose_to_matrix(R, t)
        T_world = T_world @ rel
        poses.append(T_world.copy())
        rels.append(rel)
        prev_cloud, prev_tree = cl, build_kdtree(cl)

    gt = []
    for i in range(n):
        k, gp = nearest_gt(gt_map, pairs[i][0])
        gt.append(pose_matrix(gp))
    gt = np.array(gt)

    ate_raw = ate_rmse(poses, gt)
    print(f"\n[1] 纯里程计 ATE: {ate_raw*100:.2f} cm")

    # 2. 找回环候选并验证
    print(f"\n[2] 回环检测（min_gap={min_gap}, max_dist={max_dist}）...")
    cands = find_loop_candidates(poses, min_gap, max_dist)
    print(f"  候选对: {len(cands)}")
    if cands:
        pds = [c[3] for c in cands]
        print(f"  路程间隔中位: {np.median(pds):.2f} m")

    # 限制验证数量（每个候选都要跑一次 ICP，成本高）
    MAX_VERIFY = 60
    if len(cands) > MAX_VERIFY:
        print(f"  候选过多，只验证最接近的 {MAX_VERIFY} 对")
        cands = cands[:MAX_VERIFY]

    loops = []
    for k, item in enumerate(cands):
        i, j, d, path_d = item
        if clouds[i] is None or clouds[j] is None:
            continue
        ok, Z, rmse, ratio = verify_loop_with_icp(clouds[i], clouds[j])
        if ok:
            loops.append((i, j, Z, rmse, ratio))
            if verbose:
                print(f"  ✓ ({i},{j}) 位姿距{d*100:.1f}cm "
                      f"路程{path_d:.2f}m rmse={rmse*100:.1f}cm "
                      f"内点率={ratio*100:.0f}%")
        elif k % 20 == 0:
            print(f"  ...已验证 {k+1}/{len(cands)}", flush=True)

    print(f"  通过验证: {len(loops)}")
    if not loops:
        print("  （没有回环，无法测试回环效果）")
        print(f"\n最终 ATE: {ate_raw*100:.2f} cm（纯里程计）")
        return

    # 3. 位姿图优化
    print(f"\n[3] 位姿图优化（加 {len(loops)} 条回环边）...")
    g = PoseGraph()
    ids = list(range(n))
    g.add_nodes(ids, poses)

    for a in range(n - 1):
        g.add_edge(a, a + 1, rels[a + 1], weight=1.0, kind="odom")

    for (i, j, Z, rmse, ratio) in loops:
        # Z 是 cloud_i 在 cloud_j 坐标系 -> 转成位姿图要的形式
        # 位姿图边 Z_ij 满足 T_j = T_i @ Z_ij，这里需要仔细对应
        Z_edge = np.linalg.inv(Z)   # 尝试：i->j 的相对位姿
        g.add_edge(i, j, Z_edge, weight=1.0, kind="loop", rot_weight=1.0)

    hist = g.optimize(iterations=100, damping=1e-8, huber=0.20,
                      fix_first=True, verbose=False)
    opt = g.poses
    ate_opt = ate_rmse(opt, gt)

    print(f"  优化迭代: {len(hist)}")
    print(f"\n[4] 优化后 ATE: {ate_opt*100:.2f} cm")

    print()
    print("=" * 72)
    print("结果")
    print("=" * 72)
    print(f"{'配置':<32}{'ATE (cm)':>12}")
    print("-" * 72)
    print(f"{'LiDAR 纯里程计':<32}{ate_raw*100:>12.2f}")
    print(f"{'LiDAR + 回环 + 位姿图':<32}{ate_opt*100:>12.2f}")
    delta = (ate_opt - ate_raw) * 100
    print(f"{'变化':<32}{delta:>+12.2f}")

    os.makedirs(_RES, exist_ok=True)
    with open(os.path.join(_RES, "lidar_loop_posegraph.txt"), "w") as f:
        f.write(f"frames={n}\n")
        f.write(f"loop_candidates={len(cands)}\n")
        f.write(f"loops_verified={len(loops)}\n")
        f.write(f"raw_ATE_cm={ate_raw*100:.2f}\n")
        f.write(f"optimized_ATE_cm={ate_opt*100:.2f}\n")
    print("\n已保存 results/lidar_loop_posegraph.txt")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    main(n)
