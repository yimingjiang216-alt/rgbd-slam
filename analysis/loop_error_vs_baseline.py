# -*- coding: utf-8 -*-
"""Per-loop measurement error vs baseline, against ground truth.

Quantifies, per verified loop pair on fr1/xyz, how the PnP measurement error
(rotation and translation) grows as the revisit baseline shrinks -- the
evidence behind MIN_LOOP_BASELINE / LOOP_BASE_REF in slam3.py.

Run:  python analysis/loop_error_vs_baseline.py <seq_dir>
"""
import os
import sys

import numpy as np

seq = sys.argv[1] if len(sys.argv) > 1 else r"C:\slam_data\rgbd_dataset_freiburg1_xyz"

d = np.load(os.path.join(seq, "map_v2.npz"))
poses_arr = d["poses"]
L = np.load(os.path.join(seq, "loops.npz"))
li, lj, LT = L["i"], L["j"], L["T"]

Tq = np.loadtxt(os.path.join(seq, "gt_quat.txt"))
gx = np.loadtxt(os.path.join(seq, "gt_xyz.txt"))


def gtT(k):
    T = np.eye(4)
    x, y, z, w = Tq[k][3:7]
    T[:3, :3] = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    T[:3, 3] = gx[k]
    return T


def rel_err(T_meas, Ti, Tj):
    """Return (translation err m, rotation err deg) of T_meas vs GT Z_ij."""
    Zg = np.linalg.inv(Ti) @ Tj
    E = np.linalg.inv(Zg) @ T_meas
    dt = float(np.linalg.norm(E[:3, 3]))
    c = (np.trace(E[:3, :3]) - 1.0) / 2.0
    dr = float(np.degrees(np.arccos(np.clip(c, -1, 1))))
    return dt, dr


rows = []
for t in range(len(li)):
    i, j = int(li[t]), int(lj[t])
    Ti, Tj = gtT(i), gtT(j)
    base_gt = float(np.linalg.norm(Ti[:3, 3] - Tj[:3, 3]))
    et, er = rel_err(LT[t], Ti, Tj)
    Zvo = np.linalg.inv(poses_arr[i]) @ poses_arr[j]
    vt, vr = rel_err(Zvo, Ti, Tj)
    rows.append((base_gt, et, er, vt, vr))

rows.sort()
out = ["%6s  %8s  %8s  %8s  %8s" % ("base_m", "pnp_dT", "pnp_dR",
                                    "vo_dT", "vo_dR")]
out += ["%6.3f  %8.4f  %8.2f  %8.4f  %8.2f" % r for r in rows]
base = np.array([r[0] for r in rows])
et = np.array([r[1] for r in rows])
er = np.array([r[2] for r in rows])
vt = np.array([r[3] for r in rows])
vr = np.array([r[4] for r in rows])

out.append("")
out.append("=== bucket medians ===")
out.append("%10s %4s | %8s %8s | %8s %8s" %
           ("base", "n", "pnp_dT", "pnp_dR", "vo_dT", "vo_dR"))
for lo, hi in [(0.0, 0.05), (0.05, 0.10), (0.10, 0.15), (0.15, 0.20),
               (0.20, 0.25), (0.25, 0.35)]:
    m = (base >= lo) & (base < hi)
    if m.sum() == 0:
        continue
    out.append("%4.2f-%4.2f %4d | %8.4f %8.2f | %8.4f %8.2f" %
               (lo, hi, m.sum(), np.median(et[m]), np.median(er[m]),
                np.median(vt[m]), np.median(vr[m])))

out.append("")
out.append("PnP rotation beats VO rotation: %d / %d loops"
           % (int((er < vr).sum()), len(er)))
for thr in (0.15, 0.20, 0.25, 0.30):
    m = base >= thr
    if m.sum():
        out.append("baseline >= %.2f : %3d pairs | PnP rot %.2f deg | "
                   "VO rot %.2f deg | PnP trans %.4f m"
                   % (thr, int(m.sum()), np.median(er[m]),
                      np.median(vr[m]), np.median(et[m])))

txt = "\n".join(out)
open("analysis/loop_error_vs_baseline_out.txt", "w", encoding="utf-8").write(txt)
print(txt)
