# -*- coding: utf-8 -*-
"""Why did loop detection find 0 loops on fr1/room?

Stage-by-stage diagnosis against ground truth:
  A. true revisits (GT distance < 0.5 m, gap >= 20 kf) -- does the geometry
     support loops at all?
  B. how many does find_candidates propose (VO estimate + max_dist)?
  C. on the closest GT revisits: how large is the viewpoint change, and how
     far does ORB matching get (descriptor count / matches / 3D-backed)?

Run:  python analysis/room_loop_diagnosis.py
"""
import os
import sys

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import loop_detection as ld

SEQ = r"C:\slam_data\rgbd_dataset_freiburg1_room"

d = np.load(os.path.join(SEQ, "map_v2.npz"))
poses_arr = d["poses"]
kf = [int(x) for x in d["kf"]]

names = []
for line in open(os.path.join(SEQ, "rgb.txt")):
    line = line.strip()
    if line and not line.startswith("#"):
        names.append(line.split()[1])
gray = {}
for k in kf:
    gray[k] = cv2.imread(os.path.join(SEQ, names[k]), cv2.IMREAD_GRAYSCALE)

poses = {k: poses_arr[k] for k in kf}
Tq = np.loadtxt(os.path.join(SEQ, "gt_quat.txt"))
gx = np.loadtxt(os.path.join(SEQ, "gt_xyz.txt"))


def gtT(k):
    T = np.eye(4)
    x, y, z, w = Tq[k][3:7]
    T[:3, :3] = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    T[:3, 3] = gx[k]
    return T


def gt_view_angle(i, j):
    """Relative view-direction change between the two keyframes, in deg."""
    Ti, Tj = gtT(i), gtT(j)
    # camera looks along +Z in camera frame: direction in world
    di = Ti[:3, :3] @ np.array([0.0, 0.0, 1.0])
    dj = Tj[:3, :3] @ np.array([0.0, 0.0, 1.0])
    c = float(np.clip(np.dot(di, dj) / (np.linalg.norm(di) * np.linalg.norm(dj)),
                      -1, 1))
    return float(np.degrees(np.arccos(c)))


# ---- A. true revisits by GT ------------------------------------------------
revisits = []
for a in range(len(kf)):
    for b in range(a + 1, len(kf)):
        i, j = kf[a], kf[b]
        if sum(1 for x in kf if i < x < j) < 20:
            continue
        Ti, Tj = gtT(i), gtT(j)
        dt = float(np.linalg.norm(Ti[:3, 3] - Tj[:3, 3]))
        if dt < 0.5:
            revisits.append((i, j, dt))
print("[A] true revisit pairs (GT dist < 0.5 m, gap >= 20 kf): %d" % len(revisits))
if revisits:
    dd = np.array([r[2] for r in revisits])
    print("    GT revisit distance: med %.3f / p75 %.3f / max %.3f m"
          % (np.median(dd), np.percentile(dd, 75), dd.max()))

# ---- B. does find_candidates propose them? ---------------------------------
proposed = {}
for cur in kf:
    for k, dt, dr in ld.find_candidates(kf, poses, cur, min_gap=20,
                                        max_dist=0.6):
        proposed[(min(k, cur), max(k, cur))] = dt
hit = sum(1 for i, j, _ in revisits if (min(i, j), max(i, j)) in proposed)
print("[B] candidates proposed (max_dist=0.6): %d pairs, covering "
      "%d/%d true revisits" % (len(proposed), hit, len(revisits)))

# ---- C. viewpoint change and ORB matching on the closest revisits ----------
print("[C] closest 20 GT revisits: viewpoint change vs ORB matching")
print("%10s %7s %7s %7s | %6s %6s %6s" %
      ("pair", "gt_d", "vo_d", "viewdg", "desc", "match", "ok>=25"))
revisits.sort(key=lambda r: r[2])
for i, j, gtd in revisits[:20]:
    vo_dt, _ = ld.pose_difference(poses[i], poses[j])
    ang = gt_view_angle(i, j)
    _, des_i = ld.orb_features(gray[i])
    _, des_j = ld.orb_features(gray[j])
    n_i = len(des_i) if des_i is not None else 0
    n_j = len(des_j) if des_j is not None else 0
    ia, ib = ld.match_descriptors(des_i, des_j)
    print("%6d-%4d %7.3f %7.3f %7.1f | %4d/%-4d %6d %6s" %
          (i, j, gtd, vo_dt, ang, n_i, n_j, len(ia),
           "yes" if len(ia) >= 25 else "no"))
