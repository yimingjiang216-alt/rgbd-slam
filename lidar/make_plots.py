"""make_plots.py — 生成全部可视化结果

六张图：
  1. TUM LiDAR 里程计：估计轨迹 vs 真值（X-Y 平面）
  2. TUM 逐帧误差曲线
  3. 视觉 vs 激光同序列轨迹对比
  4. 回环效果对比（加回环前后）
  5. KITTI 失败案例：轨迹严重低估
  6. ICP 单帧配准前后点云（证明配准正确）
"""
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

C_EST = "#1f77b4"
C_GT = "#2ca02c"
C_ERR = "#d62728"
C_ALT = "#ff7f0e"


def ate(est, gt):
    n = min(len(est), len(gt))
    e = np.array([p[:3, 3] for p in est[:n]])
    g = np.array([p[:3, 3] for p in gt[:n]])
    ec, gc = e - e.mean(0), g - g.mean(0)
    U, S, Vt = np.linalg.svd(ec.T @ gc)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    al = (R @ ec.T).T + g.mean(0)
    return float(np.sqrt(((np.linalg.norm(al - g, axis=1)) ** 2).mean())), al


# ══════════════════════════════════════════════════════════
# 图 1 + 2：TUM LiDAR 里程计
# ══════════════════════════════════════════════════════════
def plot_tum_lidar():
    pf = "results/tum_lidar_p2pl_poses.npy"
    gf = "results/tum_lidar_p2pl_gt.npy"
    if not (os.path.exists(pf) and os.path.exists(gf)):
        print("跳过图1/2：缺少 tum_lidar 结果")
        return None

    est = np.load(pf)
    gt = np.load(gf)
    rmse, est_al = ate(est, gt)
    gt_p = np.array([p[:3, 3] for p in gt])
    err = np.linalg.norm(est_al - gt_p, axis=1)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))

    ax[0].plot(gt_p[:, 0], gt_p[:, 1], color=C_GT, lw=2.2,
               label=f"真值 (路径 {np.sum(np.linalg.norm(np.diff(gt_p[:, :2], axis=0), axis=1)):.2f} m)")
    ax[0].plot(est_al[:, 0], est_al[:, 1], color=C_EST, lw=1.6,
               ls="--", label=f"LiDAR 里程计 (ATE {rmse*100:.2f} cm)")
    ax[0].scatter(gt_p[0, 0], gt_p[0, 1], s=90, c="k", marker="o", zorder=5, label="起点")
    ax[0].scatter(gt_p[-1, 0], gt_p[-1, 1], s=110, c="k", marker="*", zorder=5, label="终点")
    ax[0].set_xlabel("x (m)")
    ax[0].set_ylabel("y (m)")
    ax[0].set_title(f"TUM fr1/desk：LiDAR 里程计轨迹（{len(est)} 帧）")
    ax[0].legend(loc="best", fontsize=9)
    ax[0].grid(alpha=0.3)
    ax[0].axis("equal")

    ax[1].plot(err * 100, color=C_ERR, lw=1.6)
    ax[1].axhline(rmse * 100, color="k", ls=":", lw=1.2,
                  label=f"RMSE = {rmse*100:.2f} cm")
    ax[1].fill_between(range(len(err)), err * 100, alpha=0.25, color=C_ERR)
    ax[1].set_xlabel("帧号")
    ax[1].set_ylabel("位置误差 (cm)")
    ax[1].set_title("逐帧位置误差")
    ax[1].legend(fontsize=9)
    ax[1].grid(alpha=0.3)

    plt.tight_layout()
    p = os.path.join(OUT, "01_tum_lidar_odometry.png")
    plt.savefig(p, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {p}")
    return rmse


# ══════════════════════════════════════════════════════════
# 图 3：视觉 vs 激光 同序列
# ══════════════════════════════════════════════════════════
def plot_visual_vs_lidar():
    D = os.path.join("tum", "extract", "rgbd_dataset_freiburg1_desk")
    need = [os.path.join(D, "slam_front.npz"), os.path.join(D, "slam_ba.npz")]
    if not all(os.path.exists(f) for f in need):
        print("跳过图3：缺少 rgbd-slam 结果")
        return

    d_front = np.load(need[0])
    d_ba = np.load(need[1])
    P_front = d_front["poses"]
    P_ba = d_ba["poses"]

    # 真值
    gt_all = []
    with open(os.path.join(D, "groundtruth.txt")) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            v = [float(x) for x in line.split()]
            qx, qy, qz, qw = v[4], v[5], v[6], v[7]
            xx, yy, zz = qx*qx, qy*qy, qz*qz
            xy, xz, yz = qx*qy, qx*qz, qy*qz
            wx, wy, wz = qw*qx, qw*qy, qw*qz
            T = np.eye(4)
            T[:3, :3] = np.array([
                [1-2*(yy+zz), 2*(xy-wz), 2*(xz+wy)],
                [2*(xy+wz), 1-2*(xx+zz), 2*(yz-wx)],
                [2*(xz-wy), 2*(yz+wx), 1-2*(xx+yy)]])
            T[:3, 3] = v[1:4]
            gt_all.append((v[0], T))
    gt_ts = np.array([g[0] for g in gt_all])

    rgb_ts = []
    with open(os.path.join(D, "rgb.txt")) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            if len(p) >= 2:
                rgb_ts.append(float(p[0]))

    vis_gt = []
    for i in range(len(P_front)):
        if i >= len(rgb_ts):
            break
        k = int(np.argmin(np.abs(gt_ts - rgb_ts[i])))
        vis_gt.append(gt_all[k][1])
    vis_gt = np.array(vis_gt)

    a_front, _ = ate(P_front, vis_gt)
    a_ba, _ = ate(P_ba, vis_gt)

    # 激光
    lf = "results/tum_lidar_p2pl_poses.npy"
    lg = "results/tum_lidar_p2pl_gt.npy"
    has_lidar = os.path.exists(lf) and os.path.exists(lg)
    if has_lidar:
        lest = np.load(lf)
        lgt = np.load(lg)
        a_li, li_al = ate(lest, lgt)

    fig, ax = plt.subplots(figsize=(9, 8))

    gtxy = np.array([p[:3, 3] for p in vis_gt])
    ax.plot(gtxy[:, 0], gtxy[:, 1], color=C_GT, lw=2.6, label="真值", zorder=2)
    ax.plot(P_ba[:, 0, 3], P_ba[:, 1, 3], color=C_ALT, lw=1.5, alpha=0.85,
            label=f"视觉完整前端 + BA (ATE {a_ba*100:.2f} cm)  613 帧", zorder=3)
    if has_lidar:
        # ★ 只画与激光相同的帧数范围，避免帧数不同造成误导
        n_cmp = min(len(li_al), len(gtxy))
        ax.plot(li_al[:n_cmp, 0], li_al[:n_cmp, 1], color=C_EST, lw=1.5,
                ls="--", alpha=0.9,
                label=f"LiDAR 里程计 (ATE {a_li*100:.2f} cm)  {len(lest)} 帧",
                zorder=4)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("TUM fr1/desk 同序列对比：视觉 vs 激光\n"
                 "（两者帧数不同，仅作趋势参考）")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    ax.axis("equal")

    plt.tight_layout()
    p = os.path.join(OUT, "02_visual_vs_lidar.png")
    plt.savefig(p, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {p}")


# ══════════════════════════════════════════════════════════
# 图 4：回环效果
# ══════════════════════════════════════════════════════════
def plot_loop_effect():
    fig, ax = plt.subplots(figsize=(9, 5))

    configs = ["LiDAR 纯里程计", "LiDAR + 回环 + 位姿图"]
    vals = [45.89, 39.44]

    bars = ax.bar(configs, vals, color=[C_EST, C_GT], width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.2f} cm",
                ha="center", fontsize=12, fontweight="bold")

    ax.annotate("", xy=(1, 39.44), xytext=(1, 45.89),
                arrowprops=dict(arrowstyle="->", color=C_ERR, lw=2.2))
    ax.text(1.16, 42.6, "↓ 14.1%", color=C_ERR, fontsize=12, fontweight="bold")

    ax.set_ylabel("ATE RMSE (cm)")
    ax.set_title("回环闭合效果（TUM fr1/desk，596 帧，9.31 m）")
    ax.set_ylim(0, 54)
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    p = os.path.join(OUT, "03_loop_closure.png")
    plt.savefig(p, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {p}")


# ══════════════════════════════════════════════════════════
# 图 5：KITTI 失败案例
# ══════════════════════════════════════════════════════════
def plot_kitti_failure():
    pf = "slam-lidar/results/lidar_p2p_poses.npy"
    if not os.path.exists(pf):
        pf = "results/lidar_p2p_poses.npy"
    if not os.path.exists(pf):
        print("跳过图5：缺少 KITTI 结果")
        return

    # 从 lidar_odometry 重新算真值
    sys.path.insert(0, "slam-lidar")
    try:
        from lidar_odometry import load_ground_truth
    except Exception:
        print("跳过图5：无法导入")
        return

    BASE = "slam-lidar/data/kitti/2011_09_26/2011_09_26_drive_0001_sync"
    gt = load_ground_truth(os.path.join(BASE, "oxts", "data"))
    est = np.load(pf)

    n = min(len(est), len(gt))
    gt_p = np.array([gt[i][:3, 3] for i in range(n)])
    est_p = np.array([est[i][:3, 3] for i in range(n)])

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))

    ax[0].plot(gt_p[:, 0], gt_p[:, 1], color=C_GT, lw=2.4, label="真值 (162.2 m)")
    ax[0].plot(est_p[:, 0], est_p[:, 1], color=C_ERR, lw=1.8, ls="--",
               label="LiDAR 里程计 (仅 90.3 m)")
    ax[0].scatter(gt_p[0, 0], gt_p[0, 1], s=90, c="k", zorder=5)
    ax[0].scatter(gt_p[-1, 0], gt_p[-1, 1], s=120, c=C_GT, marker="*", zorder=5)
    ax[0].scatter(est_p[-1, 0], est_p[-1, 1], s=120, c=C_ERR, marker="X", zorder=5)
    ax[0].set_xlabel("x (m)")
    ax[0].set_ylabel("y (m)")
    ax[0].set_title("KITTI drive 0001：里程计严重低估（ATE 20.15 m）")
    ax[0].legend(fontsize=9)
    ax[0].grid(alpha=0.3)
    ax[0].axis("equal")

    ax[1].plot(gt_p[:, 0], np.zeros(n), color=C_GT, lw=2.4, label="真值")
    ax[1].plot(est_p[:, 0], np.zeros(n), color=C_ERR, lw=1.8, ls="--", label="估计")
    ax[1].set_yticks([])
    ax[1].set_xlabel("x (m)")
    ax[1].set_title(f"沿 x 轴：估计 {est_p[-1,0]:.1f} m / 真值 {gt_p[-1,0]:.1f} m "
                    f"(仅 {(est_p[-1,0]/gt_p[-1,0])*100:.0f}%)")
    ax[1].legend(fontsize=9)
    ax[1].grid(alpha=0.3)

    plt.tight_layout()
    p = os.path.join(OUT, "04_kitti_failure.png")
    plt.savefig(p, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {p}")


# ══════════════════════════════════════════════════════════
# 图 6：ICP 配准前后点云
# ══════════════════════════════════════════════════════════
def plot_icp_registration():
    sys.path.insert(0, "slam-lidar")
    try:
        from pointcloud import (load_kitti_bin, filter_range, filter_ground,
                                voxel_downsample, build_kdtree)
        from registration import icp_point_to_point, transform_points
    except Exception as e:
        print(f"跳过图6：{e}")
        return

    BASE = "slam-lidar/data/kitti/2011_09_26/2011_09_26_drive_0001_sync"
    VELO = os.path.join(BASE, "velodyne_points", "data")

    def load(i):
        p = load_kitti_bin(os.path.join(VELO, f"{i:010d}.bin"))
        return voxel_downsample(filter_ground(filter_range(p, 50.0)), 0.5)

    p0 = load(10)
    p1 = load(11)
    tree = build_kdtree(p1[:, :3])
    R, t, info = icp_point_to_point(
        p0[:, :3], p1[:, :3], target_tree=tree,
        R0=np.eye(3), t0=np.zeros(3),
        max_iter=80, max_correspondence_dist=3.0)
    moved = transform_points(p0[:, :3], R, t)

    fig, ax = plt.subplots(1, 3, figsize=(16, 5.2))

    ax[0].scatter(p1[:, 0], p1[:, 1], s=1.2, c=C_GT, alpha=0.5, label="参考帧 (t+1)")
    ax[0].scatter(p0[:, 0], p0[:, 1], s=1.2, c=C_EST, alpha=0.5, label="待配准帧 (t)")
    ax[0].set_title("配准前（俯视）")
    ax[0].legend(fontsize=9, markerscale=8)
    ax[0].set_xlabel("x (m)"); ax[0].set_ylabel("y (m)")
    ax[0].grid(alpha=0.3); ax[0].axis("equal")

    ax[1].scatter(p1[:, 0], p1[:, 1], s=1.2, c=C_GT, alpha=0.5, label="参考帧")
    ax[1].scatter(moved[:, 0], moved[:, 1], s=1.2, c=C_ERR, alpha=0.5, label="配准后")
    ax[1].set_title(f"配准后（RMSE {info['final_rmse']*100:.1f} cm）")
    ax[1].legend(fontsize=9, markerscale=8)
    ax[1].set_xlabel("x (m)"); ax[1].set_ylabel("y (m)")
    ax[1].grid(alpha=0.3); ax[1].axis("equal")

    # 侧视
    ax[2].scatter(p1[:, 0], p1[:, 2], s=1.2, c=C_GT, alpha=0.5, label="参考帧")
    ax[2].scatter(moved[:, 0], moved[:, 2], s=1.2, c=C_ERR, alpha=0.5, label="配准后")
    ax[2].set_title("配准后（侧视）")
    ax[2].legend(fontsize=9, markerscale=8)
    ax[2].set_xlabel("x (m)"); ax[2].set_ylabel("z (m)")
    ax[2].grid(alpha=0.3)

    plt.tight_layout()
    p = os.path.join(OUT, "05_icp_registration.png")
    plt.savefig(p, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {p}")


# ══════════════════════════════════════════════════════════
# 图 7：判据 1 的可视化（帧间位移 vs 误差）
# ══════════════════════════════════════════════════════════
def plot_criterion():
    fig, ax = plt.subplots(figsize=(9, 5.5))

    datasets = ["TUM fr1/desk", "KITTI drive 0001"]
    disps = [0.016, 1.90]
    errs = [0.0086, 1.81]

    x = np.arange(len(datasets))
    w = 0.35
    b1 = ax.bar(x - w/2, disps, w, label="帧间位移 (m)", color=C_EST)
    b2 = ax.bar(x + w/2, errs, w, label="零初值 ICP 误差 (m)", color=C_ERR)

    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, h + 0.02,
                    f"{h*100:.1f} cm" if h < 1 else f"{h:.2f} m",
                    ha="center", fontsize=10, fontweight="bold")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=11)
    ax.set_ylabel("距离（对数刻度，m）")
    ax.set_title("判据 1：帧间位移决定 ICP 能否工作\n"
                 "TUM 1.6cm → 误差 0.86cm（成功）；KITTI 190cm → 误差 181cm（失败）")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    p = os.path.join(OUT, "06_criterion.png")
    plt.savefig(p, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {p}")


if __name__ == "__main__":
    print("=" * 60)
    print("生成可视化")
    print("=" * 60)
    plot_tum_lidar()
    plot_visual_vs_lidar()
    plot_loop_effect()
    plot_kitti_failure()
    plot_icp_registration()
    plot_criterion()
    print()
    print(f"全部输出到 {OUT}/")
