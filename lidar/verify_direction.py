"""verify_direction.py — 严格验证 ICP 输出的语义方向

不要凭观察下结论。构造一个「已知相对位姿」的测试：
  取帧 A 和帧 B（真值已知），把 A 变换到 B 坐标系，
  看 ICP 输出 vs 真值的关系。
"""
import os
import sys


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    import numpy as np

    from tum_data import (read_tum_list, read_tum_groundtruth, load_depth,
                          depth_to_pointcloud, pose_matrix)
    from pointcloud import voxel_downsample, build_kdtree
    from registration import icp_point_to_point, pose_to_matrix, transform_points

    D = "tum/extract/rgbd_dataset_freiburg1_desk"
    depths = read_tum_list(os.path.join(D, "depth.txt"))
    depths = [(t, p) for t, p in depths if os.path.exists(os.path.join(D, p))]
    gt_map = read_tum_groundtruth(os.path.join(D, "groundtruth.txt"))


    def load_cloud(i, voxel=0.02, max_depth=4.0):
        t, p = depths[i]
        img = load_depth(os.path.join(D, p))
        pts = depth_to_pointcloud(img, max_depth=max_depth)
        return voxel_downsample(pts, voxel), t


    def nearest_gt(t):
        k = min(gt_map.keys(), key=lambda x: abs(x - t))
        return k, gt_map[k]


    print("=" * 76)
    print("严格验证：ICP 输出 vs 真值相对位姿")
    print("=" * 76)

    print("\n约定说明：")
    print("  T_cur_from_prev: 把「上一帧坐标系的点」变到「当前帧坐标系」")
    print("                   即 p_cur = T @ p_prev")
    print("  真值: T_world_cur = T_world_prev @ T_cur_from_prev")
    print("        => T_cur_from_prev = inv(T_world_prev) @ T_world_cur")
    print()

    print(f"{'帧对':<10}{'ICP 原始平移':>26}{'真值 cur_from_prev':>26}{'ICP取逆平移':>26}")
    print("-" * 90)

    for i in range(1, 6):
        prev_cloud, t_prev = load_cloud(i - 1)
        cur_cloud, t_cur = load_cloud(i)

        kp, pp = nearest_gt(t_prev)
        kc, pc = nearest_gt(t_cur)
        T_wp = pose_matrix(pp)
        T_wc = pose_matrix(pc)

        # 真值：上一帧->当前帧
        T_gt_cur_from_prev = np.linalg.inv(T_wp) @ T_wc
        gt_t = T_gt_cur_from_prev[:3, 3]

        # ICP：cur(source) -> prev(target)
        tree = build_kdtree(prev_cloud)
        R, t, info = icp_point_to_point(
            cur_cloud[:, :3], prev_cloud[:, :3], target_tree=tree,
            R0=np.eye(3), t0=np.zeros(3),
            max_iter=60, max_correspondence_dist=0.3)

        # 假设 1：ICP 直接输出 T_cur_from_prev
        # 假设 2：ICP 输出 T_prev_from_cur（需要取逆）
        T_inv = np.linalg.inv(pose_to_matrix(R, t))

        print(f"{i-1}->{i:<6}"
              f"{np.round(t,4).tolist()!s:>26}"
              f"{np.round(gt_t,4).tolist()!s:>26}"
              f"{np.round(T_inv[:3,3],4).tolist()!s:>26}")

    print()
    print("=" * 76)
    print("判定")
    print("=" * 76)

    # 统计哪个假设更接近
    err_direct = []
    err_inv = []
    for i in range(1, 15):
        prev_cloud, t_prev = load_cloud(i - 1)
        cur_cloud, t_cur = load_cloud(i)
        kp, pp = nearest_gt(t_prev)
        kc, pc = nearest_gt(t_cur)
        T_gt = np.linalg.inv(pose_matrix(pp)) @ pose_matrix(pc)
        gt_t = T_gt[:3, 3]

        tree = build_kdtree(prev_cloud)
        R, t, info = icp_point_to_point(
            cur_cloud[:, :3], prev_cloud[:, :3], target_tree=tree,
            R0=np.eye(3), t0=np.zeros(3),
            max_iter=60, max_correspondence_dist=0.3)

        err_direct.append(np.linalg.norm(t - gt_t))
        T_inv = np.linalg.inv(pose_to_matrix(R, t))
        err_inv.append(np.linalg.norm(T_inv[:3, 3] - gt_t))

    print(f"假设 A（直接使用 ICP 平移）: 平均误差 {np.mean(err_direct)*100:.2f} cm")
    print(f"假设 B（取逆后使用）:       平均误差 {np.mean(err_inv)*100:.2f} cm")
    print()
    if np.mean(err_direct) < np.mean(err_inv):
        print("=> 假设 A 正确：ICP 平移 = T_cur_from_prev 的平移，不需要取逆")
    else:
        print("=> 假设 B 正确：需要取逆")


if __name__ == "__main__":
    main()
