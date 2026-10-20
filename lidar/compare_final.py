"""compare_final.py — 同一序列（fr1/desk）上的视觉 vs 激光 最终公平对比

视觉方案：rgbd-slam 的 front + BA（已跑完，结果在 slam_ba.npz）
激光方案：本次的 LiDAR 里程计 + 回环 + 位姿图（results/final_results.txt）

统一 ATE 口径：去重心 + 只做刚体旋转对齐，不做相似变换。
"""
import os
import sys


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    import numpy as np

    D = os.path.join("tum", "extract", "rgbd_dataset_freiburg1_desk")


    def read_gt(path):
        """groundtruth.txt -> [(ts, 4x4)]"""
        out = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                v = [float(x) for x in line.split()]
                T = np.eye(4)
                qx, qy, qz, qw = v[4], v[5], v[6], v[7]
                xx, yy, zz = qx*qx, qy*qy, qz*qz
                xy, xz, yz = qx*qy, qx*qz, qy*qz
                wx, wy, wz = qw*qx, qw*qy, qw*qz
                T[:3, :3] = np.array([
                    [1-2*(yy+zz), 2*(xy-wz), 2*(xz+wy)],
                    [2*(xy+wz), 1-2*(xx+zz), 2*(yz-wx)],
                    [2*(xz-wy), 2*(yz+wx), 1-2*(xx+yy)]])
                T[:3, 3] = v[1:4]
                out.append((v[0], T))
        return out


    def read_ts(path):
        out = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                p = line.split()
                if len(p) >= 2:
                    out.append(float(p[0]))
        return out


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
        return float(np.sqrt((err ** 2).mean())), err


    print("=" * 74)
    print("TUM fr1/desk 同序列对比：视觉 vs 激光")
    print("=" * 74)

    gt_all = read_gt(os.path.join(D, "groundtruth.txt"))
    gt_ts = np.array([g[0] for g in gt_all])

    # ── 视觉前端 ──
    print("\n[视觉] 读取 slam_front.npz / slam_ba.npz")
    d_front = np.load(os.path.join(D, "slam_front.npz"))
    d_ba = np.load(os.path.join(D, "slam_ba.npz"))
    P_front = d_front["poses"]
    P_ba = d_ba["poses"]
    print(f"  前端关键帧: {len(P_front)}, BA 后: {len(P_ba)}")

    # 前端位姿对应哪些帧？pipeline 用 rgb.txt 顺序
    rgb_ts = read_ts(os.path.join(D, "rgb.txt"))
    n_vis = len(P_front)
    vis_gt = []
    for i in range(n_vis):
        if i >= len(rgb_ts):
            break
        k = int(np.argmin(np.abs(gt_ts - rgb_ts[i])))
        vis_gt.append(gt_all[k][1])
    vis_gt = np.array(vis_gt)

    ate_front, _ = ate_rmse(P_front, vis_gt)
    ate_ba, _ = ate_rmse(P_ba, vis_gt)
    print(f"  视觉前端 ATE: {ate_front*100:.2f} cm")
    print(f"  视觉 BA 后 ATE: {ate_ba*100:.2f} cm")

    # ── 路径长度 ──
    gt_path = np.sum(np.linalg.norm(np.diff(np.array([g[:3,3] for g in vis_gt]), axis=0), axis=1))
    print(f"  真值路径（对应帧）: {gt_path:.2f} m")

    # ── 激光前端（从 final_results.txt 读已有数字）──
    print("\n[激光] 读取 results/final_results.txt")
    res_file = os.path.join("results", "final_results.txt")
    lidar = {}
    if os.path.exists(res_file):
        with open(res_file) as f:
            for line in f:
                line = line.strip()
                if "=" in line:
                    k, v = line.split("=", 1)
                    lidar[k] = v
        print(f"  {lidar}")

    print()
    print("=" * 74)
    print("最终对比（同一序列 fr1/desk）")
    print("=" * 74)
    print(f"{'方案':<36}{'ATE (cm)':>12}")
    print("-" * 74)
    print(f"{'视觉 前端（无 BA）':<36}{ate_front*100:>12.2f}")
    print(f"{'视觉 前端 + BA':<36}{ate_ba*100:>12.2f}")
    if "lidar_ATE_cm" in lidar:
        print(f"{'激光 前端（无 BA）':<36}{float(lidar['lidar_ATE_cm']):>12.2f}")
    if "lidar_loop_ATE_cm" in lidar:
        print(f"{'激光 前端 + 回环 + 位姿图':<36}{float(lidar['lidar_loop_ATE_cm']):>12.2f}")
    if "visual_ATE_cm" in lidar:
        print(f"{'(参考) 本次跑的视觉无BA，596帧':<36}{float(lidar['visual_ATE_cm']):>12.2f}")

    print()
    print("=" * 74)
    print("说明")
    print("=" * 74)
    print(f"""
    帧数差异：
      视觉前端处理了 {n_vis} 帧（pipeline 的所有 RGB 帧）
      激光处理了 596 帧（RGB 与 depth 成功对齐的帧）

    两者不是完全相同的帧集，但都在同一段 desk 序列上，
      ATE 用统一口径（去重心 + 刚体旋转对齐）计算。

    关键对比：
      视觉 + BA: {ate_ba*100:.2f} cm
      激光 + 回环: {lidar.get('lidar_loop_ATE_cm', 'N/A')} cm
    """)

    # 保存
    os.makedirs("results", exist_ok=True)
    with open("results/visual_vs_lidar_desk.txt", "w") as f:
        f.write(f"sequence=fr1/desk\n")
        f.write(f"visual_frames={n_vis}\n")
        f.write(f"visual_front_ATE_cm={ate_front*100:.2f}\n")
        f.write(f"visual_BA_ATE_cm={ate_ba*100:.2f}\n")
        f.write(f"gt_path_m={gt_path:.2f}\n")
        if "lidar_ATE_cm" in lidar:
            f.write(f"lidar_ATE_cm={lidar['lidar_ATE_cm']}\n")
        if "lidar_loop_ATE_cm" in lidar:
            f.write(f"lidar_loop_ATE_cm={lidar['lidar_loop_ATE_cm']}\n")
    print("已保存 results/visual_vs_lidar_desk.txt")


if __name__ == "__main__":
    main()
