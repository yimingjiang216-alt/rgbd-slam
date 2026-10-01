# -*- coding: utf-8 -*-
"""Stage-by-stage pass counts for loop candidates on fr1/room.

Replicates ld.detect_loops with counters after every stage so we can see
which gate zeroes the yield, and prints the PnP-vs-odometry agreement
distribution of the pairs that survive PnP.

Run:  python analysis/room_stage_counts.py
"""
import os
import sys

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import loop_detection as ld

SEQ = r"C:\slam_data\rgbd_dataset_freiburg1_room"
MAX_DIST, MIN_INLIERS = 1.2, 15

d = np.load(os.path.join(SEQ, "map_v2.npz"))
poses_arr = d["poses"]
xyz = d["xyz"]
kf = [int(x) for x in d["kf"]]
obs = d["obs"]

names = []
for line in open(os.path.join(SEQ, "rgb.txt")):
    line = line.strip()
    if line and not line.startswith("#"):
        names.append(line.split()[1])
gray = {k: cv2.imread(os.path.join(SEQ, names[k]), cv2.IMREAD_GRAYSCALE)
        for k in kf}
poses = {k: poses_arr[k] for k in kf}

uv_at_kf, pts_in_kf = {}, {}
for pid, fidx, u, v in obs:
    fidx = int(fidx)
    if fidx in gray:
        uv_at_kf[(fidx, int(pid))] = (u, v)
        pts_in_kf.setdefault(fidx, {}).setdefault(int(pid), (u, v))

K = np.array([[517.3, 0, 318.6], [0, 516.5, 255.3], [0, 0, 1]], np.float64)

n_cand = n_match = n_3d = n_pnp = n_ratio = n_agree = 0
agrees = []
des_cache = {}
for cur in kf:
    cands = ld.find_candidates(kf, poses, cur, min_gap=20, max_dist=MAX_DIST)
    if not cands:
        continue
    kp_cur, des_cur = ld.orb_features(gray[cur])
    if des_cur is None or len(kp_cur) < 20:
        continue
    for cand, cdist, cang in cands:
        n_cand += 1
        if cand not in des_cache:
            des_cache[cand] = ld.orb_features(gray[cand])[1]
        des_c = des_cache[cand]
        if des_c is None:
            continue
        ia, ib = ld.match_descriptors(des_c, des_cur)
        if len(ia) < MIN_INLIERS:
            continue
        n_match += 1
        cand_pts = pts_in_kf.get(cand, {})
        if not cand_pts:
            continue
        cpu = np.array([uv_at_kf[(cand, p)] for p in cand_pts], np.float64)
        cpx = np.array([xyz[p] for p in cand_pts], np.float64)
        kp_c, _ = ld.orb_features(gray[cand])
        pts3d, pts2d = [], []
        for a, b in zip(ia, ib):
            p_uv = np.array(kp_c[a].pt)
            d2 = np.sum((cpu - p_uv) ** 2, axis=1)
            t = int(np.argmin(d2))
            if d2[t] > 9.0:
                continue
            pts3d.append(cpx[t])
            pts2d.append(np.array(kp_cur[b].pt))
        if len(pts3d) < MIN_INLIERS:
            continue
        n_3d += 1
        r = ld.pose_from_pnp(pts3d, pts2d, K)
        if not r["ok"] or len(r["inliers"]) < MIN_INLIERS:
            continue
        if r["inlier_ratio"] < 0.15:
            continue
        n_pnp += 1
        T_pred = ld.relative_pose(poses[cand], poses[cur])
        dt, dr = ld.pose_difference(T_pred, r["T"])
        agrees.append((dt, dr, cand, cur))
        if dt > 0.35 or dr > 35.0:
            continue
        n_agree += 1

agrees.sort()
print("candidates            : %d" % n_cand)
print("after ORB matching>=%d : %d" % (MIN_INLIERS, n_match))
print("after 3D backing >=%d  : %d" % (MIN_INLIERS, n_3d))
print("after PnP inliers>=%d  : %d" % (MIN_INLIERS, n_pnp))
print("after agree<=0.35/35  : %d" % n_agree)
print()
print("PnP-vs-odometry agreement of the %d PnP-survivors:" % len(agrees))
print("%8s %8s %8s" % ("agreeT", "agreeR", "pair"))
for dt, dr, c, u in agrees[:25]:
    print("%8.3f %8.1f %4d-%4d" % (dt, dr, c, u))
