# -*- coding: utf-8 -*-
"""Tests for the loop-edge acceptance policy introduced after the fr1/xyz
failure (ATE 0.0449 -> 0.1975 m with v3.0's flat 3x loop rotation trust).

Final policy (evidence-driven; see slam3.py header and results/room_*.txt):
  baseline <  MIN_LOOP_BASELINE (0.05 m)  rejected -- no parallax at all;
  MIN_LOOP_BASELINE .. LOOP_BASE_REF (0.30 m)  rejected -- recorded negative
      results: down-weighted rotation -> xyz ATE 0.1978 m; translation-only
      -> xyz ATE 0.1685 m with 78 deg rotation;
  baseline >= LOOP_BASE_REF  accepted with full SE(3) trust (ROT_WEIGHT).
  On long-drift sequences, max_loop_edges keeps only the strongest edges
  (fr1/room top-10 by inliers: ATE 0.3447 -> 0.3089 m, -10.4%).

Run:  python test_loop_gating.py
"""
import numpy as np
from slam3 import ROT_WEIGHT, MIN_LOOP_BASELINE, LOOP_BASE_REF, loop_rot_weight

FAIL = []


def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + ("   " + detail if detail else ""))
    if not ok:
        FAIL.append(name)


# -- 1. below either threshold the edge is declined (weight 0 -> drop) -------
check("near-zero-parallax loops are rejected",
      loop_rot_weight(MIN_LOOP_BASELINE - 1e-9) == 0.0
      and loop_rot_weight(0.0) == 0.0)
check("mid-baseline loops are declined outright",
      loop_rot_weight(0.1189) == 0.0 and loop_rot_weight(0.20) == 0.0)

# -- 2. full trust at and above the reference baseline ------------------------
check("long-baseline loops keep full SE(3) trust",
      loop_rot_weight(LOOP_BASE_REF) == ROT_WEIGHT
      and loop_rot_weight(1.0) == ROT_WEIGHT)

# -- 3. monotone: more baseline never means less trust -----------------------
bs = np.linspace(0.0, 1.0, 2001)
ws = np.array([loop_rot_weight(b) for b in bs])
check("weight is monotonically non-decreasing in baseline",
      bool(np.all(np.diff(ws) >= -1e-12)))

# -- 4. sanity on the measured fr1/xyz loop baselines ------------------------
# median 0.1189 m / p25 0.0938 m / max 0.3089 m (results/loop_stats.txt):
# every xyz loop except the two longest is declined; those two carry full
# rotation trust.
declined = sum(1 for b in [0.032, 0.074, 0.118, 0.163, 0.296, 0.309]
               if loop_rot_weight(b) == 0.0)
check("xyz baseline sample: 5 of 6 declined, the 0.309 m one kept",
      declined == 5 and loop_rot_weight(0.309) == ROT_WEIGHT)

print()
if FAIL:
    print("%d test(s) FAILED" % len(FAIL))
    raise SystemExit(1)
print("all loop-gating tests passed")
