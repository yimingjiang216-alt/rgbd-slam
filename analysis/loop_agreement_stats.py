# -*- coding: utf-8 -*-
"""How many verified loops agree with odometry within odometry's own error?

The acceptance policy in slam3.build_graph accepts a loop edge only if its
measurement agrees with the odometry-predicted relative pose inside the error
bars odometry itself demonstrates (else the edge can only inject error).
This script counts passers at candidate thresholds.

Run:  python analysis/loop_agreement_stats.py <seq_dir>
"""
import os
import sys

import numpy as np

seq = sys.argv[1] if len(sys.argv) > 1 else r"C:\slam_data\rgbd_dataset_freiburg1_xyz"

d = np.load(os.path.join(seq, "map_v2.npz"))
poses_arr = d["poses"]
L = np.load(os.path.join(seq, "loops.npz"))
li, lj, LT = L["i"], L["j"], L["T"]


def diff(T_a, T_b):
    dt = float(np.linalg.norm(T_a[:3, 3] - T_b[:3, 3]))
    c = (np.trace(T_a[:3, :3].T @ T_b[:3, :3]) - 1.0) / 2.0
    return dt, float(np.degrees(np.arccos(np.clip(c, -1, 1))))


agree = []
for t in range(len(li)):
    i, j = int(li[t]), int(lj[t])
    Zvo = np.linalg.inv(poses_arr[i]) @ poses_arr[j]
    agree.append(diff(Zvo, LT[t]))
agree = np.array(agree)          # (N, 2): trans m, rot deg
base = np.linalg.norm(poses_arr[li][:3, 3] - poses_arr[lj][:3, 3], axis=1)

print("n_loops = %d" % len(li))
print("%12s %8s %8s | %6s" % ("agree_trans", "agree_rot", "n_pass", "base_med"))
for at in (0.05, 0.08, 0.10, 0.15, 0.20, 0.35):
    for ar in (3.0, 5.0, 8.0, 15.0, 35.0):
        m = (agree[:, 0] <= at) & (agree[:, 1] <= ar)
        if m.sum():
            print("%12.2f %8.1f | %6d %8.3f" %
                  (at, ar, m.sum(), np.median(base[m])))

# joint scan for the record
print()
for at in (0.05, 0.08, 0.10):
    for ar in (3.0, 5.0, 8.0):
        m = (agree[:, 0] <= at) & (agree[:, 1] <= ar)
        print("agree <= %.2f m / %.0f deg : %3d / %d pass"
              % (at, ar, int(m.sum()), len(li)))
