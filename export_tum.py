# -*- coding: utf-8 -*-
"""把 SLAM 结果导出成标准 TUM 格式 (timestamp tx ty tz qx qy qz qw), 供 evo 评估"""
import numpy as np, os, sys, cv2
sys.path.insert(0, r"C:\Users\r26304\Documents\codex-shop\slam_vo")
from slam2 import read_gt, read_list, nearest

seq = r"C:\Users\r26304\Documents\codex-shop\slam_vo\data\rgbd_dataset_freiburg1_xyz"
rgb = read_list(os.path.join(seq, "rgb.txt"))[:798]
ts = np.array([t for t, _ in rgb])

def R2q(R):
    """旋转矩阵 -> 四元数 (qx,qy,qz,qw)"""
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / S; qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S; qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / S; qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S; qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / S; qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S; qz = 0.25 * S
    q = np.array([qx, qy, qz, qw])
    return q / np.linalg.norm(q)

def se3_align(src, dst):
    n = len(src); ms = src.mean(0); md = dst.mean(0)
    Sc = src - ms; Dc = dst - md
    C = Dc.T @ Sc / n
    U, S, Vt = np.linalg.svd(C)
    d = np.ones(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0: d[2] = -1
    R = U @ np.diag(d) @ Vt
    return R, md - R @ ms

# --- 真值 (第一步: 把 TUM groundtruth 转成 evo 格式; 位姿是 T_wc, evo 要的是相机轨迹) ---
def q2R(q):
    x, y, z, w = q
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

def q2T(q):
    T = np.eye(4); T[:3,:3] = q2R(q[3:7]); T[:3,3] = q[0:3]; return T

gt = read_gt(os.path.join(seq, "groundtruth.txt"))
gt_lines = []
for k, t in enumerate(ts):
    q = gt[nearest(t, gt)][1]
    gt_lines.append("%.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f" %
                    (t, q[0], q[1], q[2], q[3], q[4], q[5], q[6]))
open(os.path.join(seq, "evo_gt.txt"), "w").write("\n".join(gt_lines) + "\n")
print("写出 evo_gt.txt  (%d 行)" % len(gt_lines))

# --- 估计轨迹: 需要把 SLAM 的 T_wc 转成 evo 期望的"相机位姿" ---
# evo 的 TUM 格式约定: 存的是 T_wc (相机->世界), 与 TUM groundtruth 一致, 直接用
P = np.load(os.path.join(seq, "map_v2.npz"))["poses"]   # 在线局部BA结果
est = P[:, :3, 3].copy()
# 对齐到真值坐标系(平移+旋转), 这样 evo 可以直接按 --align 再对齐, 这里先对齐便于看
gt_xyz = np.array([gt[nearest(t, gt)][1][0:3] for t in ts])
R_, t_ = se3_align(est, gt_xyz)
lines = []
for k in range(len(P)):
    Rk = R_ @ P[k][:3, :3]
    tk = R_ @ P[k][:3, 3] + t_
    q = R2q(Rk)
    lines.append("%.9f %.9f %.9f %.9f %.9f %.9f %.9f %.9f" % (ts[k], tk[0], tk[1], tk[2], q[0], q[1], q[2], q[3]))
open(os.path.join(seq, "evo_est.txt"), "w").write("\n".join(lines) + "\n")
print("写出 evo_est.txt  (%d 行)  <- 在线局部BA结果" % len(lines))

# 同时导出"无在线BA"版本用于对比(从历史文件重建)
Pb = np.load(os.path.join(seq, "map_v2_ba.npz"))["poses"] if os.path.exists(os.path.join(seq,"map_v2_ba.npz")) else None
print("\n完成. 文件:")
for f in ["evo_gt.txt", "evo_est.txt"]:
    print("  ", os.path.join(seq, f))
