"""final_experiment.py — 最终实验：完整对比

跑三个配置：
  A. LiDAR 纯里程计
  B. LiDAR + 回环 + 位姿图
  C. 视觉纯里程计（基准）

用完整 TUM fr1/desk 序列（613 帧，有真回环）。
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
from registration import (icp_point_to_point, pose_to_matrix,  # noqa: E402
                          transform_points)

from paths import TUM_DESK as D, RESULTS as _RES, imread_any
FX, FY, CX, CY = 517.3, 516.5, 318.6, 255.3


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


# ─────────────────────────────────────────────────────────────
def run_lidar(pairs, n):
    clouds, poses, rels = [], [], []
    T = np.eye(4)
    pc, pt = None, None
    for i in range(n):
        rt, rp, dt, dp = pairs[i]
        dep = imread_any(dp, cv2.IMREAD_UNCHANGED)
        if dep is None:
            clouds.append(None); poses.append(T.copy()); rels.append(np.eye(4))
            continue
        if dep.ndim == 3:
            dep = dep[:, :, 0]
        cl = voxel_downsample(depth_to_pointcloud(dep, max_depth=4.0), 0.02)
        clouds.append(cl)
        if pc is None:
            poses.append(np.eye(4)); rels.append(np.eye(4))
            pc, pt = cl, build_kdtree(cl)
            continue
        R, t, _ = icp_point_to_point(
            cl[:, :3], pc[:, :3], target_tree=pt,
            R0=np.eye(3), t0=np.zeros(3),
            max_iter=60, max_correspondence_dist=0.3)
        rel = pose_to_matrix(R, t)
        T = T @ rel
        poses.append(T.copy()); rels.append(rel)
        pc, pt = cl, build_kdtree(cl)
    return clouds, poses, rels


def run_visual(pairs, n):
    orb = cv2.ORB_create(nfeatures=1500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]], np.float64)
    poses = []
    T = np.eye(4)
    prev = None
    for i in range(n):
        rt, rp, dt, dp = pairs[i]
        img = imread_any(rp, cv2.IMREAD_GRAYSCALE)
        if img is None:
            poses.append(T.copy()); continue
        kp, des = orb.detectAndCompute(img, None)
        if prev is None or des is None:
            poses.append(np.eye(4)); prev = (kp, des); continue
        ms = sorted(bf.match(des, prev[1]), key=lambda m: m.distance)[:200]
        if len(ms) < 12:
            poses.append(T.copy()); prev = (kp, des); continue
        dep = imread_any(dp, cv2.IMREAD_UNCHANGED)
        if dep.ndim == 3:
            dep = dep[:, :, 0]
        dep = dep.astype(np.float64) / 5000.0
        p3, p2 = [], []
        for m in ms:
            u, v = kp[m.queryIdx].pt
            ui, vi = int(round(u)), int(round(v))
            if 0 <= vi < dep.shape[0] and 0 <= ui < dep.shape[1]:
                z = dep[vi, ui]
                if 0.3 < z < 4.0:
                    p3.append([(u - CX) * z / FX, (v - CY) * z / FY, z])
                    p2.append(prev[0][m.trainIdx].pt)
        if len(p3) < 8:
            poses.append(T.copy()); prev = (kp, des); continue
        ok, rvec, tvec, inl = cv2.solvePnPRansac(
            np.array(p3, np.float64), np.array(p2, np.float64), K, None,
            iterationsCount=100, reprojectionError=3.0,
            flags=cv2.SOLVEPNP_EPNP)
        if ok and inl is not None and len(inl) >= 8:
            Rm, _ = cv2.Rodrigues(rvec)
            T_pc = np.eye(4); T_pc[:3, :3] = Rm; T_pc[:3, 3] = tvec.ravel()
            T = T @ np.linalg.inv(T_pc)
        poses.append(T.copy()); prev = (kp, des)
    return poses


def main():
    pairs = load_pairs()
    gt_map = read_tum_groundtruth(os.path.join(D, "groundtruth.txt"))
    n = len(pairs)

    print("=" * 72)
    print(f"最终实验：TUM fr1/desk 完整序列（{n} 帧）")
    print("=" * 72)

    gt = []
    for i in range(n):
        k, gp = nearest_gt(gt_map, pairs[i][0])
        gt.append(pose_matrix(gp))
    gt = np.array(gt)
    path = np.sum(np.linalg.norm(np.diff(np.array([g[:3, 3] for g in gt]),
                                         axis=0), axis=1))
    print(f"真值轨迹长度: {path:.2f} m")

    # A. 视觉
    print("\n[A] 视觉里程计（ORB+PnP）...")
    vp = run_visual(pairs, n)
    ate_v = ate_rmse(vp, gt)
    print(f"    ATE: {ate_v*100:.2f} cm")

    # B. LiDAR
    print("\n[B] LiDAR 里程计（深度点云 ICP）...")
    clouds, lp, lrels = run_lidar(pairs, n)
    ate_l = ate_rmse(lp, gt)
    print(f"    ATE: {ate_l*100:.2f} cm")

    # C. LiDAR + 回环
    print("\n[C] LiDAR + 回环 + 位姿图...")
    from scipy.spatial import cKDTree
    P = np.array([p[:3, 3] for p in lp])
    tree = cKDTree(P)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])

    cands = []
    for i in range(n):
        for j in tree.query_ball_point(P[i], r=0.30):
            if j >= i + 90 and abs(cum[j] - cum[i]) >= 1.5:
                cands.append((i, j, float(np.linalg.norm(P[i] - P[j]))))
    cands = sorted(set(cands), key=lambda x: x[2])[:80]
    print(f"    候选: {len(cands)}")

    loops = []
    for i, j, d in cands:
        if clouds[i] is None or clouds[j] is None:
            continue
        tj = build_kdtree(clouds[j])
        R, t, info = icp_point_to_point(
            clouds[i][:, :3], clouds[j][:, :3], target_tree=tj,
            R0=np.eye(3), t0=np.zeros(3),
            max_iter=80, max_correspondence_dist=0.3)
        moved = transform_points(clouds[i][:, :3], R, t)
        dd, _ = tj.query(moved, k=1, workers=-1)
        ratio = float((dd < 0.3).mean())
        if ratio >= 0.5 and info["final_rmse"] < 0.3:
            loops.append((i, j, pose_to_matrix(R, t), ratio))
    print(f"    通过: {len(loops)}")

    ate_loop = None
    if loops:
        g = PoseGraph()
        g.add_nodes(list(range(n)), lp)
        for a in range(n - 1):
            g.add_edge(a, a + 1, lrels[a + 1], weight=1.0, kind="odom")
        for i, j, Z, ratio in loops:
            g.add_edge(i, j, np.linalg.inv(Z), weight=1.0,
                       kind="loop", rot_weight=1.0)
        hist = g.optimize(iterations=100, damping=1e-8, huber=0.20,
                          fix_first=True, verbose=False)
        ate_loop = ate_rmse(g.poses, gt)
        print(f"    优化 {len(hist)} 次, ATE: {ate_loop*100:.2f} cm")

    print()
    print("=" * 72)
    print("最终结果")
    print("=" * 72)
    print(f"{'配置':<32}{'ATE (cm)':>12}")
    print("-" * 72)
    print(f"{'视觉纯里程计 (ORB+PnP)':<32}{ate_v*100:>12.2f}")
    print(f"{'LiDAR 纯里程计':<32}{ate_l*100:>12.2f}")
    if ate_loop:
        print(f"{'LiDAR + 回环 + 位姿图':<32}{ate_loop*100:>12.2f}")
        print(f"{'  相对纯里程计改善':<32}{(ate_loop-ate_l)*100:>+12.2f}")

    os.makedirs(_RES, exist_ok=True)
    with open(os.path.join(_RES, "final_results.txt"), "w") as f:
        f.write(f"TUM fr1/desk full, {n} frames, path {path:.2f}m\n")
        f.write(f"visual_ATE_cm={ate_v*100:.2f}\n")
        f.write(f"lidar_ATE_cm={ate_l*100:.2f}\n")
        if ate_loop:
            f.write(f"lidar_loop_ATE_cm={ate_loop*100:.2f}\n")
            f.write(f"loops={len(loops)}\n")
    print("\n已保存 results/final_results.txt")


if __name__ == "__main__":
    main()
