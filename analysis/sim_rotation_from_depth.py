# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import least_squares

rng = np.random.default_rng(1)
N = 200
B_true = 0.12
f_list = {"depth 0.5 mm": 0.0005, "depth 2 mm": 0.002, "depth 5 mm": 0.005, "depth 10 mm": 0.010, "depth 25 mm": 0.025}
base_Z = rng.uniform(1.0, 3.0, N)
X = rng.uniform(-0.5, 0.5, N)
Y = rng.uniform(-0.5, 0.5, N)
uv0 = np.stack([517.3 * X / base_Z + 318.6, 516.5 * Y / base_Z + 255.3], 1)

lines = []
for name, sz in f_list.items():
    rot_errs = []
    for k in range(40):
        Z = base_Z + rng.normal(0, sz, N)
        Z = np.clip(Z, 0.3, 6.0)
        # solve for the relative geometry of the second view.
        # parameters: the small rotation (3) and translation (3) of camera1
        # w.r.t. camera0 in camera0 frame.  Points are FIXED at their true
        # position: only the two views' relative pose is unknown (this is what
        # a 3-point/2-view minimal problem does with a rigid scene).
        def res(p):
            r = p[:3]; t = p[3:]
            th = np.linalg.norm(r)
            K = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
            if th < 1e-12:
                R = np.eye(3) + K
            else:
                Kn = K / th
                R = np.eye(3) + np.sin(th) * Kn + (1 - np.cos(th)) * (Kn @ Kn)
            # point in cam0 frame from noisy depth
            P0 = np.stack([X, Y, Z], 1)
            P1 = (P0 - t) @ R.T
            z1 = P1[:, 2]
            bad = z1 < 0.05
            z1 = np.where(bad, 0.05, z1)
            uv1_pred = np.stack([517.3 * P1[:, 0] / z1 + 318.6,
                                 516.5 * P1[:, 1] / z1 + 255.3], 1)
            # observed pixel of the SAME physical point in view 1 (exact)
            Ptrue1 = np.stack([X - B_true, Y, base_Z], 1)
            uv1_obs = np.stack([517.3 * Ptrue1[:, 0] / base_Z + 318.6,
                                516.5 * Ptrue1[:, 1] / base_Z + 255.3], 1)
            return (uv1_pred - uv1_obs).ravel()
        p0 = np.zeros(6)
        o = least_squares(res, p0, method="lm", xtol=1e-14, ftol=1e-14, gtol=1e-14)
        r = o.x[:3]; th = np.degrees(np.linalg.norm(r))
        # translation error against the true baseline vector (-B,0,0)
        terr = np.linalg.norm(o.x[3:] - np.array([-B_true, 0, 0]))
        rot_errs.append((th, terr))
    rot_errs = np.array(rot_errs)
    lines.append("%-11s  rot err med %6.2f deg (p90 %6.2f)  | trans err med %.4f m"
                 % (name, np.median(rot_errs[:, 0]), np.percentile(rot_errs[:, 0], 90), np.median(rot_errs[:, 1])))
open("sim_rot_out.txt", "w", encoding="utf-8").write("\n".join(lines))
print("done")
