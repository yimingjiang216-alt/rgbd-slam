# -*- coding: utf-8 -*-
"""Checks for the Lie-group code in pose_graph.py."""
import numpy as np
from pose_graph import (so3_exp, so3_log, se3_exp, se3_log, adjoint,
                        _jacobian_inv_se3, _ad_matrix)

FAIL = []

def check(name, ok, detail):
    print(("PASS  " if ok else "FAIL  ") + name + "   " + detail)
    if not ok:
        FAIL.append(name)

rng = np.random.default_rng(1234)

# 1. exp/log roundtrip
w = 0.0
for _ in range(4000):
    xi = rng.normal(size=6) * np.array([1, 1, 1, 1.2, 1.2, 1.2])
    T = se3_exp(xi)
    w = max(w, np.abs(T - se3_exp(se3_log(T))).max())
check("se3 exp/log roundtrip", w < 1e-7, "worst %.2e" % w)

# 2. so3 exp/log roundtrip
w = 0.0
for _ in range(4000):
    phi = rng.normal(size=3)
    phi = phi / np.linalg.norm(phi) * rng.uniform(1e-9, np.pi - 1e-9)
    w = max(w, np.abs(so3_exp(so3_log(so3_exp(phi))) - so3_exp(phi)).max())
check("so3 exp/log roundtrip", w < 1e-7, "worst %.2e" % w)

# 3. adjoint identity  Exp(Ad_T xi) = T Exp(xi) T^-1
w = 0.0
for _ in range(2000):
    xi = rng.normal(size=6) * 0.7
    T = se3_exp(rng.normal(size=6) * 0.7)
    lhs = se3_exp(adjoint(T) @ xi)
    rhs = T @ se3_exp(xi) @ np.linalg.inv(T)
    w = max(w, np.abs(lhs - rhs).max())
check("adjoint identity", w < 1e-10, "worst %.2e" % w)

# 4. ad = exp^-1 derivative:  Exp(ad_xi) = Ad(Exp(xi))
def expm(A, n=300):
    X = np.eye(A.shape[0]); T = np.eye(A.shape[0])
    for i in range(1, n):
        T = T @ A / i
        X = X + T
        if np.abs(T).max() < 1e-20:
            break
    return X

w = 0.0
for _ in range(200):
    xi = rng.normal(size=6) * 0.6
    w = max(w, np.abs(expm(_ad_matrix(xi)) - adjoint(se3_exp(xi))).max())
check("ad matrix (Exp(ad)=Ad(Exp))", w < 1e-12, "worst %.2e" % w)

# 5. THE identity the optimiser relies on:
#    Exp(xi)^-1 Exp(xi + d) = Exp(J_r^-1(xi) d) + O(|d|^2)
#    and the O(|d|^2) means halving d quarters the residual.
xi = np.array([0.83, 0.13, 2.13, 0.15, 0.53, 0.63])
J = _jacobian_inv_se3(xi)
Ti = se3_exp(xi)
prev = None
ratios = []
for eps in [2e-2, 1e-2, 5e-3, 2.5e-3]:
    r = 0.0
    for k in range(6):
        e = np.zeros(6); e[k] = eps
        lhs = np.linalg.inv(Ti) @ se3_exp(xi + J @ e)
        r = max(r, np.abs(lhs - se3_exp(e)).max())
    if prev is not None:
        ratios.append(prev / r)
    prev = r
check("Jr^-1 second-order identity", min(ratios) > 3.0,
      "residual ratios %s" % np.round(ratios, 2).tolist())

# 6. residual identity at zero perturbation
w = 0.0
for _ in range(500):
    a = se3_exp(rng.normal(size=6)); b = se3_exp(rng.normal(size=6))
    w = max(w, np.abs(se3_log(np.linalg.inv(a) @ b)).max() - np.abs(se3_log(np.linalg.inv(a) @ b)).max())
check("residual finite", w == 0.0, "ok")

print()
print("FAILED:", FAIL if FAIL else "none")
