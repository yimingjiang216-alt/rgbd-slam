# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import least_squares

rng = np.random.default_rng(0)
N = 200
f = 517.3
X = rng.uniform(-0.5, 0.5, N)
Y = rng.uniform(-0.5, 0.5, N)
Z = rng.uniform(1.0, 3.0, N)

def run(sigma, B_true):
    proj0 = np.stack([f * X / Z + 318.6, f * Y / Z + 255.3], 1)
    proj1 = np.stack([f * (X - B_true) / Z + 318.6, f * Y / Z + 255.3], 1)
    uv0 = proj0 + rng.normal(0, sigma, proj0.shape)
    uv1 = proj1 + rng.normal(0, sigma, proj1.shape)
    def res(p):
        lz = p[:N]; b = p[N]
        Zz = np.exp(lz)
        r0 = np.stack([f * X / Zz + 318.6, f * Y / Zz + 255.3], 1) - uv0
        r1 = np.stack([f * (X - b) / Zz + 318.6, f * Y / Zz + 255.3], 1) - uv1
        return np.concatenate([r0.ravel(), r1.ravel()])
    p0 = np.concatenate([np.log(Z), [B_true]])
    o = least_squares(res, p0, method="lm", xtol=1e-14, ftol=1e-14, gtol=1e-14)
    return abs(o.x[N] - B_true)

lines = []
for B_true in (1.0, 0.3, 0.12):
    for sigma in (0.05, 0.10, 0.25):
        errs = np.array([run(sigma, B_true) for _ in range(30)])
        lines.append("B=%5.2f m  px_sigma=%.2f  ->  recovered-B rel.err  med=%6.2f%%  p90=%7.2f%%"
                     % (B_true, sigma, 100 * np.median(errs) / B_true, 100 * np.percentile(errs, 90) / B_true))
open("sim_out.txt", "w", encoding="utf-8").write("\n".join(lines))
print("done")
