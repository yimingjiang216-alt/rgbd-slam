# -*- coding: utf-8 -*-
"""用 evo 的 Python API 出标准图 (适配 evo 1.37 API)"""
import numpy as np, os, sys
sys.path.insert(0, r"C:\Users\r26304\Documents\codex-shop\slam_vo")
from evo.core import sync, metrics
from evo.tools import plot, file_interface

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

seq = r"C:\Users\r26304\Documents\codex-shop\slam_vo\data\rgbd_dataset_rgbd_dataset_freiburg1_xyz"
seq = r"C:\Users\r26304\Documents\codex-shop\slam_vo\data\rgbd_dataset_freiburg1_xyz"

gt = file_interface.read_tum_trajectory_file(os.path.join(seq, "evo_gt.txt"))
est = file_interface.read_tum_trajectory_file(os.path.join(seq, "evo_est.txt"))
gt, est = sync.associate_trajectories(gt, est)
print("关联点数: %d" % len(gt.positions_xyz))

ape = metrics.APE(metrics.PoseRelation.translation_part)
ape.process_data((gt, est))
st = ape.get_all_statistics()
err = np.array(ape.error)

ape_rot = metrics.APE(metrics.PoseRelation.rotation_angle_deg)
ape_rot.process_data((gt, est))
rst = ape_rot.get_all_statistics()
re_ = np.array(ape_rot.error)

ts = np.array(gt.timestamps) - gt.timestamps[0]

fig = plt.figure(figsize=(16, 11))

# 2D 轨迹 (XY 平面)
ax1 = fig.add_subplot(2, 3, 1)
plot.trajectories(ax1, { "ground truth": gt, "RGB-D SLAM (local BA)": est },
                  plot_mode=plot.PlotMode.xy, plot_start_end_markers=True)
ax1.set_title("Trajectory  (XY view)")

# 3D 轨迹
ax2 = fig.add_subplot(2, 3, 2, projection="3d")
plot.trajectories(ax2, { "ground truth": gt, "RGB-D SLAM (local BA)": est },
                  plot_mode=plot.PlotMode.xyz, plot_start_end_markers=True)
ax2.set_title("Trajectory  (3D view)")

# 平移误差随时间
ax3 = fig.add_subplot(2, 3, 3)
plot.error_array(ax3, err, x_array=ts, statistics=st,
                 name="APE (translation)", title="Absolute pose error over time",
                 xlabel="t (s)", ylabel="error (m)")

# 三轴误差
ax4 = fig.add_subplot(2, 3, 4)
for i, nm in enumerate(["x", "y", "z"]):
    ax4.plot(ts, est.positions_xyz[:, i] - gt.positions_xyz[:, i], label="%s error" % nm)
ax4.set_xlabel("t (s)"); ax4.set_ylabel("error (m)")
ax4.set_title("Position error per axis"); ax4.legend(); ax4.grid(True)

# 旋转误差随时间
ax5 = fig.add_subplot(2, 3, 5)
plot.error_array(ax5, re_, x_array=ts, statistics=rst,
                 name="APE (rotation)", title="Rotation error over time",
                 xlabel="t (s)", ylabel="error (deg)")

# 误差直方图
ax6 = fig.add_subplot(2, 3, 6)
ax6.hist(err, bins=40, color="steelblue", edgecolor="k", alpha=0.8)
ax6.axvline(st["rmse"], color="r", ls="--", label="rmse = %.4f m" % st["rmse"])
ax6.set_xlabel("APE (m)"); ax6.set_ylabel("count")
ax6.set_title("APE distribution"); ax6.legend(); ax6.grid(True)

fig.tight_layout()
out = os.path.join(seq, "evo_result.png")
fig.savefig(out, dpi=150)
print("图:", out)

print("\n===== evo 统计 (SE3 Umeyama 对齐, 不含尺度) =====")
print("平移 APE:")
for k in ["rmse", "mean", "median", "std", "min", "max"]:
    print("   %-7s = %.6f m" % (k, st[k]))
print("旋转 APE:")
for k in ["rmse", "mean", "median", "std", "min", "max"]:
    print("   %-7s = %.6f deg" % (k, rst[k]))
