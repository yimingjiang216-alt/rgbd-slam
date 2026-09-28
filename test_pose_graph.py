# -*- coding: utf-8 -*-
"""Pose-graph optimiser tests on synthetic data with known ground truth.

Run:  python test_pose_graph.py
"""
import numpy as np
from pose_graph import PoseGraph, se3_exp

rng = np.random.default_rng(7)
FAIL = []


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + ("   " + detail if detail else ""))
    if not ok:
        FAIL.append(name)


def square_loop(side=1.0, n_per_side=25):
    """Walk a closed square.  Returns (N, 4, 4) camera-to-world poses."""
    step = side / n_per_side
    T = np.eye(4)
    poses = [T.copy()]
    for s in range(4):
        d = np.zeros(6)
        d[0 if s in (0, 2) else 1] = step if s in (0, 1) else -step
        for _ in range(n_per_side):
            T = T @ se3_exp(d)
            poses.append(T.copy())
    return np.array(poses)


def perturb(gt, scale, rng):
    """Perturb poses 1..N-1 by a small random SE(3) delta; node 0 is the gauge."""
    out = gt.copy()
    for i in range(1, len(gt)):
        out[i] = gt[i] @ se3_exp(rng.normal(size=6) * scale)
    return out


gt = square_loop()
N = len(gt)
ZL = np.linalg.inv(gt[0]) @ gt[N - 1]          # the true closing constraint

# -- 1. consistent graph: must recover ground truth exactly ------------------
g = PoseGraph()
g.add_nodes(list(range(N)), perturb(gt, 0.05, rng))
for i in range(N - 1):
    g.add_edge(i, i + 1, np.linalg.inv(gt[i]) @ gt[i + 1], weight=1.0)
g.optimize(iterations=200, damping=1e-10, huber=0.0)
err = np.abs(g.poses - gt).max()
check("consistent graph recovers ground truth", err < 1e-8,
      "max|T - T_gt| = %.2e" % err)

# -- 2. noiseless odometry + a loop edge: the loop must close exactly --------
g2 = PoseGraph()
g2.add_nodes(list(range(N)), gt.copy())
for i in range(N - 1):
    g2.add_edge(i, i + 1, np.linalg.inv(gt[i]) @ gt[i + 1], weight=1.0)
g2.add_edge(0, N - 1, ZL, weight=1.0)
for i in range(1, N):                          # break the loop open
    g2.poses[i] = gt[i] @ se3_exp(np.array([0.03, 0.0, 0.0, 0.0, 0.0, 0.02]))
before = np.linalg.norm(g2.residual(g2.edges[-1]))
g2.optimize(iterations=100, damping=1e-12, huber=0.0)
after = np.linalg.norm(g2.residual(g2.edges[-1]))
check("loop-closure residual driven to zero", after < 1e-10,
      "%.4f m -> %.2e m" % (before, after))

# -- 3. drifting odometry: loop closure must reduce the trajectory error -----
drift = 0.004
noisy = [gt[0].copy()]
odom = []
for i in range(N - 1):
    Zn = (np.linalg.inv(gt[i]) @ gt[i + 1]).copy()
    Zn[:3, 3] += rng.normal(scale=drift, size=3)
    odom.append(Zn)
    noisy.append(noisy[-1] @ Zn)
noisy = np.array(noisy)

gp = PoseGraph()
gp.add_nodes(list(range(N)), noisy)
for i in range(N - 1):
    gp.add_edge(i, i + 1, odom[i], weight=1.0)
gp.add_edge(0, N - 1, ZL, weight=1.0)

e_before = np.linalg.norm(gp.poses[:, :3, 3] - gt[:, :3, 3], axis=1).mean()
lb = np.linalg.norm(gp.residual(gp.edges[-1]))
gp.optimize(iterations=150, damping=1e-9, huber=0.0)
e_after = np.linalg.norm(gp.poses[:, :3, 3] - gt[:, :3, 3], axis=1).mean()
la = np.linalg.norm(gp.residual(gp.edges[-1]))

check("trajectory error improves after loop closure", e_after < e_before,
      "mean %.4f m -> %.4f m" % (e_before, e_after))
check("loop edge error shrinks", la < lb * 1e-1,
      "%.4f m -> %.4f m" % (lb, la))

# -- 4. Huber kernel must down-weight one grossly wrong edge -----------------
def run_with_bad(hub):
    gh = PoseGraph()
    gh.add_nodes(list(range(N)), noisy)
    for i in range(N - 1):
        gh.add_edge(i, i + 1, odom[i], weight=1.0)
    bad = ZL.copy()
    bad[:3, 3] += np.array([0.5, -0.4, 0.3])   # 50 cm of nonsense
    gh.add_edge(0, N - 1, bad, weight=1.0)
    gh.optimize(iterations=150, damping=1e-9, huber=hub)
    return np.linalg.norm(gh.poses[:, :3, 3] - gt[:, :3, 3], axis=1).mean()

err_huber = run_with_bad(0.10)
err_plain = run_with_bad(0.0)
check("Huber tolerates an outlier edge", err_huber < err_plain,
      "huber %.4f m vs plain %.4f m (odom only %.4f m)"
      % (err_huber, err_plain, e_before))

# -- 5. the gauge must stay pinned ------------------------------------------
start = noisy[0].copy()
check("fixed gauge (node 0 unmoved)",
      np.abs(gp.poses[0] - start).max() < 1e-12,
      "delta %.2e" % np.abs(gp.poses[0] - start).max())

print()
print("FAILED:", FAIL if FAIL else "none")
