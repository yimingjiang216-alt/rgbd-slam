"""verify_tum_icp.py — 决定性验证：TUM 深度点云上 ICP 能否工作

对照 KITTI 的失败：
  KITTI: 帧间 190cm，零初值 ICP 只恢复 9%
  TUM:   帧间 1.6cm，理论上应该没问题

如果零初值就能配准，说明「换成 TUM 数据」这个决策是对的。
"""
import os
import sys


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    import numpy as np

    from tum_data import (read_tum_list, read_tum_groundtruth, load_depth,
                          depth_to_pointcloud, pose_matrix)
    from pointcloud import voxel_downsample, build_kdtree
    from registration import icp_point_to_point, icp_point_to_plane, pose_to_matrix, transform_points
    from pointcloud import estimate_normals

    D = "tum/extract/rgbd_dataset_freiburg1_desk"

    depths = read_tum_list(os.path.join(D, "depth.txt"))
    gt_map = read_tum_groundtruth(os.path.join(D, "groundtruth.txt"))

    # 用实际存在的深度帧
    depths = [(t, p) for t, p in depths if os.path.exists(os.path.join(D, p))]
    print(f"可用深度帧: {len(depths)}")


    def load_cloud(i, voxel=0.02, max_depth=4.0):
        t, p = depths[i]
        img = load_depth(os.path.join(D, p))
        pts = depth_to_pointcloud(img, max_depth=max_depth)
        # 下采样：室内点云密，2cm 体素足够
        return voxel_downsample(pts, voxel), t


    def nearest_gt(t):
        k = min(gt_map.keys(), key=lambda x: abs(x - t))
        return k, gt_map[k]


    print()
    print("=" * 74)
    print("验证：TUM 深度点云上 ICP 的表现")
    print("=" * 74)

    # 取连续帧对
    pairs = [(i, i + 1) for i in range(10, 20)]

    print(f"{'帧对':<10}{'真值位移':>12}{'零初值估计':>14}{'误差':>10}{'rmse':>9}")
    print("-" * 74)

    errors_zero = []
    for i, j in pairs:
        p0, t0 = load_cloud(i)
        p1, t1 = load_cloud(j)

        # 真值相对位姿
        k0, pose0 = nearest_gt(t0)
        k1, pose1 = nearest_gt(t1)
        T0 = pose_matrix(pose0)
        T1 = pose_matrix(pose1)
        T_rel = np.linalg.inv(T0) @ T1        # p_cur = T_rel @ p_prev
        gt_shift = T_rel[:3, 3]

        # 零初值 ICP：把 p0 配准到 p1
        tree = build_kdtree(p1[:, :3])
        R, t, info = icp_point_to_point(
            p0[:, :3], p1[:, :3], target_tree=tree,
            R0=np.eye(3), t0=np.zeros(3),
            max_iter=80, max_correspondence_dist=0.3)

        # ICP 给的是 p0 在 p1 坐标系的位姿，要取逆得「从 p0 到 p1 的运动」
        T_icp = pose_to_matrix(R, t)
        est_shift = np.linalg.inv(T_icp)[:3, 3]

        err = np.linalg.norm(est_shift - gt_shift)
        errors_zero.append(err)

        print(f"{i}->{j:<6}{np.linalg.norm(gt_shift)*100:>11.1f}cm"
              f"{np.linalg.norm(est_shift)*100:>13.1f}cm"
              f"{err*100:>9.1f}cm{info['final_rmse']*100:>8.1f}cm")

    print()
    print("=" * 74)
    print("统计")
    print("=" * 74)
    errors_zero = np.array(errors_zero)
    print(f"  零初值误差: 均值 {errors_zero.mean()*100:.2f}cm, "
          f"中位 {np.median(errors_zero)*100:.2f}cm, "
          f"最大 {errors_zero.max()*100:.2f}cm")

    print()
    print("对照 KITTI:")
    print(f"  KITTI 零初值误差中位: 181cm（真值 190cm）")
    print(f"  TUM   零初值误差中位: {np.median(errors_zero)*100:.2f}cm")
    print()
    if np.median(errors_zero) < 0.05:
        print("  ✅ TUM 上 ICP 零初值可用（误差 < 5cm）")
        print("  => 可以做完整的 LiDAR 里程计")
    else:
        print("  ⚠️ 误差偏大，需要检查参数")


if __name__ == "__main__":
    main()
