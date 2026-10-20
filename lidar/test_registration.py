"""
test_registration.py — 用已知真值验证点云配准算法正确性

测试 1、2 是纯算法测试（合成数据，不需要数据集）；
测试 3-6 需要 KITTI 点云（设置 KITTI_DIR 环境变量，或放到 lidar/data/kitti/ 下）。

用法：python test_registration.py
缺少 KITTI 数据时自动跳过 3-6。
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

from pointcloud import (load_kitti_bin, filter_range, filter_ground,
                        voxel_downsample, build_kdtree, estimate_normals)
from registration import (best_fit_transform, transform_points, pose_to_matrix,
                          icp_point_to_point, icp_point_to_plane, NDT,
                          _exp_so3, _log_so3)
from paths import KITTI as BASE

VELO = os.path.join(BASE, "velodyne_points", "data")
HAS_KITTI = os.path.isdir(VELO) and len(
    [f for f in os.listdir(VELO) if f.endswith(".bin")]) > 1


def load_frame(i, voxel=0.5):
    p = load_kitti_bin(os.path.join(VELO, f"{i:010d}.bin"))
    return voxel_downsample(filter_ground(filter_range(p, 50.0)), voxel)


def sep(title):
    print()
    print("=" * 68)
    print(title)
    print("=" * 68)


def test_svd():
    sep("测试 1: SVD 求刚体变换（已知答案）")
    rng = np.random.default_rng(0)
    A = rng.normal(size=(500, 3))
    R_gt = _exp_so3(np.array([0.05, -0.02, 0.1]))
    t_gt = np.array([1.5, -0.8, 0.3])
    B = transform_points(A, R_gt, t_gt)

    R_est, t_est = best_fit_transform(A, B)
    err_R = np.linalg.norm(R_est - R_gt)
    err_t = np.linalg.norm(t_est - t_gt)
    print(f"  旋转误差: {err_R:.2e}")
    print(f"  平移误差: {err_t:.2e}")
    assert err_R < 1e-9 and err_t < 1e-9, "SVD 求解失败"
    print("  PASS: SVD 能精确恢复刚体变换")


def test_lie():
    sep("测试 2: 李群指数/对数往返")
    for w in [np.zeros(3), np.array([0.1, 0, 0]), np.array([0.5, -0.3, 0.2])]:
        R = _exp_so3(w)
        w2 = _log_so3(R)
        same = np.allclose(R, _exp_so3(w2), atol=1e-10)
        print(f"  w={w} -> R -> w2={np.round(w2, 6)}  一致={same}")
        assert same, "李群往返失败"
    print("  PASS: exp/log 互为逆运算")


def test_icp_p2p():
    sep("测试 3: ICP point-to-point 恢复已知变换")
    pts = load_frame(0)
    print(f"  点云: {pts.shape}")

    R_true = _exp_so3(np.array([0.0, 0.0, 0.02]))
    t_true = np.array([0.5, 0.0, 0.0])

    src = pts[:, :3]
    moved = transform_points(src, R_true, t_true)
    tree = build_kdtree(moved)

    R_est, t_est, info = icp_point_to_point(
        src, moved, target_tree=tree, R0=np.eye(3),
        t0=np.array([0.4, 0.0, 0.0]),
        max_iter=60, max_correspondence_dist=2.0)

    err_t = np.linalg.norm(t_est - t_true)
    print(f"  真值 t={t_true}, 估计 t={np.round(t_est, 4)}, 误差 {err_t:.4f} m")
    assert err_t < 0.05, f"ICP 恢复不准: {err_t}"
    print("  PASS: ICP 恢复误差 < 5cm")


def test_icp_p2pl():
    sep("测试 4: ICP point-to-plane")
    pts = load_frame(0)
    R_true = _exp_so3(np.array([0.0, 0.0, 0.02]))
    t_true = np.array([0.5, 0.0, 0.0])

    src = pts[:, :3]
    moved = transform_points(src, R_true, t_true)
    normals = estimate_normals(moved)
    tree = build_kdtree(moved)

    R_est, t_est, info = icp_point_to_plane(
        src, moved, normals, target_tree=tree, R0=np.eye(3),
        t0=np.array([0.4, 0.0, 0.0]),
        max_iter=60, max_correspondence_dist=2.0)
    err_t = np.linalg.norm(t_est - t_true)
    print(f"  估计 t={np.round(t_est, 4)}, 误差 {err_t:.4f} m")
    assert err_t < 0.05, f"point-to-plane 恢复不准: {err_t}"
    print("  PASS: point-to-plane 恢复误差 < 5cm")


def test_real_frames():
    sep("测试 5: 两帧真实点云配准（KITTI 相邻帧）")
    p0 = load_frame(0)
    p1 = load_frame(1)
    print(f"  frame0: {p0.shape}  frame1: {p1.shape}")

    tree = build_kdtree(p1[:, :3])
    R, t, info = icp_point_to_point(
        p0[:, :3], p1[:, :3], target_tree=tree,
        R0=np.eye(3), t0=np.zeros(3),
        max_iter=80, max_correspondence_dist=3.0)
    print(f"  配准结果: t={np.round(t, 3)}  rmse={info['final_rmse']:.4f}")
    print(f"  运动距离: {np.linalg.norm(t):.3f} m")
    assert info['final_rmse'] < 0.8, f"rmse 过大: {info['final_rmse']}"
    print(f"  PASS: rmse {info['final_rmse']:.3f}m 在真实帧合理范围"
          "（合成数据是 1e-14，真实帧有采样噪声）")


def test_degenerate():
    sep("测试 6: 退化情况（初值很差时的失败模式）")
    p0 = load_frame(0)
    p1 = load_frame(1)
    tree = build_kdtree(p1[:, :3])
    R, t, info = icp_point_to_point(
        p0[:, :3], p1[:, :3], target_tree=tree,
        R0=np.eye(3), t0=np.array([0.0, 0.0, 5.0]),
        max_iter=50, max_correspondence_dist=1.0)
    print(f"  给 z 偏 5m 的初值 -> t={np.round(t, 3)} rmse={info['final_rmse']:.4f}")
    print("  （观察：ICP 只能收敛到局部最优，这就是为什么需要初值/里程计预测）")


def main():
    test_svd()
    test_lie()

    if not HAS_KITTI:
        print()
        print("=" * 68)
        print("跳过测试 3-6：未找到 KITTI 点云数据")
        print("=" * 68)
        print(f"  查找路径: {VELO}")
        print("  获取数据：")
        print("    wget https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/"
              "2011_09_26_drive_0001/2011_09_26_drive_0001_sync.zip")
        print("    解压到 lidar/data/kitti/ 下，或设 KITTI_DIR 环境变量")
        print()
        print("  （测试 1-2 已通过，验证了算法的数值正确性）")
        return 0

    test_icp_p2p()
    test_icp_p2pl()
    test_real_frames()
    test_degenerate()

    print()
    print("=" * 68)
    print("全部测试通过")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
