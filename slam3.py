# -*- coding: utf-8 -*-
"""RGB-D SLAM v3 -- v2 pipeline + loop closure and pose-graph optimisation.

What this adds over slam2.py
----------------------------
slam2.py produced a persistent map and refined it with *online local BA*.  That
keeps the trajectory locally consistent but does nothing about drift: after a
while the estimate has wandered away from where the camera actually is, and
nothing in the system can tell, because every constraint it has is between
temporally adjacent frames.

This stage adds the two missing pieces that make it a SLAM system rather than a
visual odometry system:

  loop closure   - recognise a place the camera has already visited, and add a
                   constraint between the *current* pose and the *old* one.
                   That is a constraint between frames that are far apart in
                   time, which is exactly what breaks the chain of accumulated
                   drift.
  pose graph     - optimise all keyframe poses together so that the odometry
                   constraints and the loop constraints are satisfied as well
                   as possible at the same time.  Drift is distributed over the
                   whole trajectory instead of being left at the end.

Flow:  map_v2_ba.npz  ->  loop detection (PnP + RANSAC)  ->  pose graph
                       ->  map_v3.npz  ->  evaluation against v2

Run:  python slam3.py                 (detect + optimise + evaluate)
      python slam3.py detect          (only write the loop constraints)
      python slam3.py eval            (re-evaluate what is already on disk)
"""
import os
import sys
import json

import numpy as np
import cv2

import loop_detection as ld
from pose_graph import PoseGraph, se3_exp, se3_log

SEQ = r"C:\Users\r26304\Documents\codex-shop\slam_vo\data\rgbd_dataset_freiburg1_xyz"
K = np.array([[517.3, 0.0, 318.6], [0.0, 516.5, 255.3], [0.0, 0.0, 1.0]])

# physical meaning of one unit of translation vs one unit of rotation in the
# residual.  Rotations are weighted up because a small angular error at 1 m
# range already moves a pixel; leaving them at 1:1 makes the optimiser trade
# degrees away for millimetres, which is the wrong trade for a camera.
ROT_WEIGHT = 3.0


def read_rgb_names(seq):
    out = []
    for line in open(os.path.join(seq, "rgb.txt")):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        s = line.split()
        if len(s) >= 2:
            out.append(s[1])
    return out


def rot_angle(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1, 1))))


def se3_align(src, dst):
    """Umeyama alignment with rotation only (no scale), as used in v2."""
    n = len(src)
    ms, md = src.mean(0), dst.mean(0)
    Sc, Dc = src - ms, dst - md
    C = Dc.T @ Sc / n
    U, S, Vt = np.linalg.svd(C)
    d = np.ones(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        d[2] = -1
    R = U @ np.diag(d) @ Vt
    return R, md - R @ ms


# ------------------------------------------------------------------ detection


def detect(seq=SEQ, verbose=True):
    """Run loop detection on the v2 map and cache the constraints."""
    # map_v2.npz is the file whose poses, xyz and obs belong together: the
    # points were back-projected with exactly those poses.  map_v2_ba.npz
    # carries the *refined* poses but reuses the old xyz, so the two are no
    # longer mutually consistent (reprojecting its points through its own poses
    # lands ~6 px off by frame 459).  Loop detection needs a self-consistent
    # map, so it starts from map_v2.npz.
    p = os.path.join(seq, "map_v2.npz")
    if not os.path.exists(p):
        raise RuntimeError("run slam2.py front first to produce map_v2.npz")
    d = np.load(p)
    poses_arr = d["poses"]
    xyz = d["xyz"]
    kf = [int(x) for x in d["kf"]]
    base = os.path.basename(p)
    obs = d["obs"]

    names = read_rgb_names(seq)
    gray = {}
    for k in kf:
        img = cv2.imread(os.path.join(seq, names[k]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError("cannot read " + names[k])
        gray[k] = img

    poses = {k: poses_arr[k] for k in kf}
    print("[1] loop detection on %d keyframes (%s)" % (len(kf), base), flush=True)
    loops = ld.detect_loops(kf, poses, gray, xyz, obs, verbose=verbose)
    print("    %d verified loop closures" % len(loops), flush=True)

    np.savez(os.path.join(seq, "loops.npz"),
             i=np.array([l["i"] for l in loops], int),
             j=np.array([l["j"] for l in loops], int),
             T=np.array([l["T_ij"] for l in loops], np.float64),
             T_wc=np.array([l["T_wc_meas"] for l in loops], np.float64),
             inliers=np.array([l["inliers"] for l in loops], int),
             ratio=np.array([l["inlier_ratio"] for l in loops], np.float64),
             reproj=np.array([l["reproj"] for l in loops], np.float64))
    return loops

# ---------------------------------------------------------------- pose graph


def build_graph(seq=SEQ, min_inliers=40, min_ratio=0.30, max_reproj=1.6,
                verbose=True):
    """Turn keyframe poses + loop constraints into a pose graph and optimise.

    The graph nodes are the keyframes (not all 798 frames): a keyframe every
    ~15 frames already carries all the motion information, and keeping the
    graph small keeps the dense solve fast.  Non-keyframe poses are afterwards
    re-expressed through the keyframe they belong to, so the ATE can still be
    computed over the whole sequence.
    """
    # same file the loop constraints were measured against, so the odometry
    # edges and the loop edges describe the same trajectory estimate.
    d = np.load(os.path.join(seq, "map_v2.npz"))
    poses_arr = d["poses"]
    kf = [int(x) for x in d["kf"]]

    lp = os.path.join(seq, "loops.npz")
    if not os.path.exists(lp):
        raise RuntimeError("run detect first")
    L = np.load(lp)
    li, lj, LT = L["i"], L["j"], L["T"]
    linl, lrat, lrep = L["inliers"], L["ratio"], L["reproj"]

    g = PoseGraph()
    g.add_nodes(kf, [poses_arr[k] for k in kf])
    for a in range(len(kf) - 1):
        Z = np.linalg.inv(poses_arr[kf[a]]) @ poses_arr[kf[a + 1]]
        g.add_edge(kf[a], kf[a + 1], Z, weight=1.0, kind="odom")

    keep = (linl >= min_inliers) & (lrat >= min_ratio) & (lrep <= max_reproj)
    n_add = 0
    for t in range(len(li)):
        if not keep[t]:
            continue
        # a loop edge is one measurement, so it must not out-vote the chain of
        # odometry edges that spans the same motion; weight 1.0 keeps it a peer.
        g.add_edge(int(li[t]), int(lj[t]), LT[t], weight=1.0,
                   kind="loop", rot_weight=ROT_WEIGHT)
        n_add += 1

    if verbose:
        print("[2] pose graph: %d nodes, %d odometry edges, %d loop edges"
              % (len(g.ids), g.n_edges("odom"), n_add), flush=True)
        s = g.stats("loop")
        print("    loop residual before: mean %.4f  max %.4f  (n=%d)"
              % (s["mean"], s["max"], s["n"]), flush=True)

    hist = g.optimize(iterations=200, damping=1e-8, huber=0.20,
                      fix_first=True, verbose=verbose)

    if verbose:
        s = g.stats("loop")
        print("    loop residual after : mean %.4f  max %.4f"
              % (s["mean"], s["max"]), flush=True)
        s = g.stats("odom")
        print("    odom residual after : mean %.4f  max %.4f"
              % (s["mean"], s["max"]), flush=True)

    # ---- write the optimised poses back over the full sequence ------------
    # keyframe k owns the frames from itself up to the next keyframe; each such
    # frame keeps its *relative* motion to its keyframe, which the pose graph
    # has just corrected.
    out = poses_arr.copy()
    fixed = {k: g.poses[g.index[k]] for k in g.ids}
    for a in range(len(kf)):
        k0 = kf[a]
        k1 = kf[a + 1] if a + 1 < len(kf) else len(poses_arr)
        rel = np.linalg.inv(poses_arr[k0])
        for t in range(k0, k1):
            out[t] = fixed[k0] @ (rel @ poses_arr[t])
    # frames before the first keyframe (there are none in practice)
    for t in range(0, kf[0]):
        out[t] = poses_arr[t]

    np.savez(os.path.join(seq, "map_v3.npz"),
             poses=out, xyz=d["xyz"], kf=np.array(kf), K=d["K"],
             kf_poses=np.array([fixed[k] for k in kf]))
    return g, hist, n_add

# ------------------------------------------------------------------- evaluate


def evaluate(seq=SEQ, verbose=True):
    gt = np.loadtxt(os.path.join(seq, "gt_xyz.txt"))
    Tq = np.loadtxt(os.path.join(seq, "gt_quat.txt"))
    Rg = np.array([q2R(Tq[k][3:7]) for k in range(len(Tq))])

    out = {}
    for tag, f in [("VO", "map_v2.npz"),
                   ("VO+localBA", "map_v2_ba.npz"),
                   ("+loopclosure", "map_v3.npz")]:
        p = os.path.join(seq, f)
        if not os.path.exists(p):
            continue
        P = np.load(p)["poses"]
        est = P[:, :3, 3]
        R_, t_ = se3_align(est, gt)
        al = (R_ @ est.T).T + t_
        e = np.linalg.norm(al - gt, axis=1)
        ate = float(np.sqrt((e ** 2).mean()))
        rot = [rot_angle(P[k][:3, :3].T @ (Rg[0].T @ Rg[k]))
               for k in range(0, len(P), 10)]
        acc = np.linalg.norm(np.diff(al, 2, axis=0), axis=1)
        out[tag] = dict(ate=ate, ate_mean=float(e.mean()), ate_max=float(e.max()),
                        rot_med=float(np.median(rot)), jitter=float(acc.mean()))

    if verbose:
        print("[3] evaluation", flush=True)
        for tag, v in out.items():
            print("    %-13s ATE=%.4f m (mean %.4f max %.4f)  rot med %.2f deg"
                  % (tag, v["ate"], v["ate_mean"], v["ate_max"], v["rot_med"]),
                  flush=True)
        if "VO+localBA" in out and "+loopclosure" in out:
            a, b = out["VO+localBA"]["ate"], out["+loopclosure"]["ate"]
            print("    ATE: %.4f -> %.4f  (%+.1f%%)"
                  % (a, b, (b - a) / a * 100.0), flush=True)

    json.dump(out, open(os.path.join(seq, "eval_v3.json"), "w"),
              indent=2, ensure_ascii=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    ax[0].plot(gt[:, 0], gt[:, 2], "g-", lw=2.5, label="ground truth")
    for tag, f, st in [("VO+localBA", "map_v2_ba.npz", "b--"),
                       ("+loopclosure", "map_v3.npz", "r-")]:
        p = os.path.join(seq, f)
        if not os.path.exists(p):
            continue
        P = np.load(p)["poses"]
        est = P[:, :3, 3]
        R_, t_ = se3_align(est, gt)
        al = (R_ @ est.T).T + t_
        e = np.linalg.norm(al - gt, axis=1)
        ax[0].plot(al[:, 0], al[:, 2], st, lw=1.8,
                   label="%s (ATE=%.3f m)" % (tag, np.sqrt((e ** 2).mean())))
        ax[1].plot(e, st, label="%s (mean %.3f)" % (tag, e.mean()))
    ax[0].set_xlabel("X (m)"); ax[0].set_ylabel("Z (m)")
    ax[0].set_title("trajectory (top view)"); ax[0].legend(); ax[0].axis("equal")
    ax[1].set_xlabel("frame"); ax[1].set_ylabel("error (m)")
    ax[1].set_title("position error"); ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(seq, "traj_v3.png"), dpi=110)
    if verbose:
        print("    figure: traj_v3.png", flush=True)
    return out


def q2R(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    seq = sys.argv[2] if len(sys.argv) > 2 else SEQ
    if cmd in ("detect", "all"):
        detect(seq)
    if cmd in ("graph", "all"):
        build_graph(seq)
    if cmd in ("eval", "all"):
        evaluate(seq)


if __name__ == "__main__":
    main()
