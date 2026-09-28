# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import least_squares

rng = np.random.default_rng(2)
N = 60
f = 517.3
cx, cy = 318.6, 255.3
TH = np.deg2rad(3.0)      # true relative pitch between the two views
B_true = 0.12
d = rng.uniform(1.0, 3.0, N)          # true depth
sigma_d = 0.02                         # 20 mm depth noise  (TUM rgbd is noisy)
sigma_px = 0.5                         # 0.5 px keypoint noise

def Rz(a):
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
Ry = lambda a: np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])

lines = []
for sigma_d in (0.002, 0.005, 0.02, 0.05):
    errs = []
    for k in range(40):
        u = rng.uniform(120, 520, N); v = rng.uniform(60, 450, N)
        x = (u - cx) * d / f; y = (v - cy) * d / f
        P0 = np.stack([x, y, d], 1)                       # cam0 frame
        R = Ry(TH) @ Rz(TH * 0.5)
        P1 = (P0 - np.array([-B_true, 0.0, 0.0])) @ R.T   # cam1 frame
        u1 = f * P1[:, 0] / P1[:, 2] + cx
        v1 = f * P1[:, 1] / P1[:, 2] + cy
        # measurements: noisy 2D in view1, noisy depth in view0
        u1n = u1 + rng.normal(0, sigma_px, N)
        v1n = v1 + rng.normal(0, sigma_px, N)
        dn = d + rng.normal(0, sigma_d, N)
        P0n = np.stack([(u - cx) * dn / f, (v - cy) * dn / f, dn], 1)
        def res(p):
            r = p[:3]; t = p[3:]
            th = np.linalg.norm(r)
            if th < 1e-12:
                Rx = np.eye(3) + np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
            else:
                a = r / th
                Kx = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
                Rx = np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * (Kx @ Kx)
            P = (P0n - t) @ Rx.T
            z = np.where(P[:, 2] < 0.05, 0.05, P[:, 2])
            return np.concatenate([f * P[:, 0] / z + cx - u1n, f * P[:, 1] / z + cy - v1n])
        o = least_squares(res, np.zeros(6), method="lm", xtol=1e-12, ftol=1e-12, gtol=1e-12)
        errs.append(np.degrees(np.linalg.norm(o.x[:3] - np.array([0, TH, TH * 0.5]))))
    errs = np.array(errs)
    lines.append("depth_sigma %5.1f mm, 0.5px keypoint -> recovered rotation err  med %6.2f deg  p90 %6.2f deg"
                 % (sigma_d * 1000, np.median(errs), np.percentile(errs, 90)))
    print(lines[-1])
open("sim_rot2_out.txt", "w", encoding="utf-8").write("\n".join(lines))
