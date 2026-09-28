# -*- coding: utf-8 -*-
"""
SE(3) pose graph optimisation.

Graph nodes are camera-to-world poses T_wc (4x4).  Edges are relative
constraints T_ij = T_wi^-1 * T_wj with an information weight.

Each edge contributes a 6-vector error

    r_ij = Log( Z_ij^-1 * T_wi^-1 * T_wj )      in se(3)

where Z_ij is the measured relative pose.  The residual is expressed in the
*body* frame of i, which is the standard convention (g2o / GTSAM style).

We minimise sum_ij  rho( ||r_ij||_Omega )  with a Huber kernel, using
Gauss-Newton on the manifold:

    T_wi <- T_wi * Exp(delta_i)

so each node keeps 6 degrees of freedom and the update is a right
multiplication, which avoids any gimbal/Euler parameterisation issues.

Only numpy is used; the linear system is solved densely, which is fine for the
few hundred nodes a single-room RGB-D sequence produces.
"""
import math
import numpy as np

# --------------------------------------------------------------- SO(3) / SE(3)


def skew(w):
    return np.array([[0.0, -w[2], w[1]],
                     [w[2], 0.0, -w[0]],
                     [-w[1], w[0], 0.0]])


def so3_exp(w):
    """Exponential map so(3) -> SO(3), robust near |w| = 0."""
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3) + skew(w)
    k = w / th
    K = skew(k)
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def so3_log(R):
    """Logarithm map SO(3) -> so(3), robust near 0 and near pi."""
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    th = float(np.arccos(c))
    if th < 1e-8:
        # first-order: w ~= vee(R - R^T) / 2
        return 0.5 * np.array([R[2, 1] - R[1, 2],
                               R[0, 2] - R[2, 0],
                               R[1, 0] - R[0, 1]])
    if np.pi - th < 1e-6:
        # near pi: use the symmetric part, pick the largest diagonal
        A = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(A)))
        v = A[:, i] / np.sqrt(max(A[i, i], 1e-12))
        w = v * th
        return w
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return (th / (2.0 * np.sin(th))) * v


def so3_vinv_coeffs(th, phi_hat):
    """Coefficients of V and V^-1 for exp/log on SE(3).

    V  = I + a1 K + a2 K^2
    V^-1 = I + b1 K + b2 K^2
    with K = skew(phi_hat), |phi_hat| = 1.  Both are exact closed forms, so
    exp and log are exact inverses of each other up to floating point.
    """
    if th < 1e-8:
        # Taylor: a1 = th/2 - th^3/24, a2 = th/2? -> use series in th
        a1 = 0.5 * th - th ** 3 / 24.0 + th ** 5 / 720.0
        a2 = th * th / 6.0 - th ** 4 / 120.0
    else:
        a1 = (1.0 - np.cos(th)) / th
        a2 = (th - np.sin(th)) / th
    den = a1 * a1 + a2 * a2 - 2.0 * a2 + 1.0
    b1 = -a1 / den
    b2 = (a1 * a1 + a2 * a2 - a2) / den
    return a1, a2, b1, b2


def se3_exp(xi):
    """Exponential map se(3) -> SE(3).  xi = [rho(3), phi(3)]."""
    rho, phi = np.asarray(xi[:3], np.float64), np.asarray(xi[3:], np.float64)
    R = so3_exp(phi)
    th = float(np.linalg.norm(phi))
    if th < 1e-8:
        K = skew(phi)
        V = np.eye(3) + 0.5 * K + (1.0 / 6.0) * (K @ K)
    else:
        K = skew(phi / th)
        a1, a2, _, _ = so3_vinv_coeffs(th, None)
        V = np.eye(3) + a1 * K + a2 * (K @ K)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = V @ rho
    return T


def se3_log(T):
    """Logarithm map SE(3) -> se(3).  Returns [rho(3), phi(3)]."""
    R, t = T[:3, :3], T[:3, 3]
    phi = so3_log(R)
    th = float(np.linalg.norm(phi))
    if th < 1e-8:
        K = skew(phi)
        Vinv = np.eye(3) - 0.5 * K + (1.0 / 12.0) * (K @ K)
    else:
        K = skew(phi / th)
        _, _, b1, b2 = so3_vinv_coeffs(th, None)
        Vinv = np.eye(3) + b1 * K + b2 * (K @ K)
    return np.concatenate([Vinv @ t, phi])


# ------------------------------------------------------------ Jacobian helper
def _ad_matrix(xi):
    """The 6x6 adjoint of the algebra element xi = [rho, phi]."""
    rho, phi = xi[:3], xi[3:]
    O = np.zeros((6, 6))
    O[:3, :3] = skew(phi)
    O[:3, 3:] = skew(rho)
    O[3:, 3:] = skew(phi)
    return O


# Bernoulli numbers B_n for the right-Jacobian series
_BERN = [1.0, -0.5, 1.0 / 6.0, 0.0, -1.0 / 30.0, 0.0, 1.0 / 42.0, 0.0,
         -1.0 / 30.0, 0.0, 5.0 / 66.0, 0.0, -691.0 / 2730.0, 0.0, 7.0 / 6.0]


def _jacobian_inv_se3(xi):
    """Inverse right Jacobian of SE(3) at xi.

        J_r^-1 = sum_{n>=0} (B_n/n!) (-ad(xi))^n

    Identity it satisfies (Barfoot, State Estimation for Robotics, ch. 7):

        Exp(xi)^-1 Exp(xi + d) = Exp( J_r^-1(xi) d + O(|d|^2) )

    The (-1)^n is not optional: with +ad the identity above fails at
    first order.  test_lie.py checks this by halving d and watching the
    residual drop by 4x (second order), which only happens for the
    correct sign.

    Note: a plain central difference of Log(Exp(xi)^-1 Exp(xi+d))/d does
    NOT reproduce this matrix, because Log is only differentiable in a
    neighbourhood where the chart is smooth; the product identity above
    is the thing that actually has to hold, and it does.
    """
    O = -_ad_matrix(xi)   # the (-1)^n in the series
    J = np.eye(6)
    P = np.eye(6)
    for n in range(1, len(_BERN)):
        P = P @ O
        J = J + (_BERN[n] / math.factorial(n)) * P
    return J

def adjoint(T):
    """SE(3) adjoint: transforms a twist from one frame to another."""
    R, t = T[:3, :3], T[:3, 3]
    A = np.zeros((6, 6))
    A[:3, :3] = R
    A[:3, 3:] = skew(t) @ R
    A[3:, 3:] = R
    return A


# ------------------------------------------------------------------ the graph
class PoseGraph:
    """A pose graph over camera-to-world poses with loop-closure support."""

    def __init__(self):
        self.ids = []          # node ids, in insertion order
        self.index = {}        # id -> slot
        self.poses = None      # (N,4,4) camera-to-world
        self.edges = []        # (i_slot, j_slot, Z (4,4), weight, kind, info_R)

    # ---------------------------------------------------------------- building
    def add_nodes(self, ids, poses):
        self.ids = list(ids)
        self.index = {v: i for i, v in enumerate(self.ids)}
        self.poses = np.array(poses, np.float64).copy()
        assert self.poses.shape == (len(self.ids), 4, 4)

    def add_edge(self, i, j, Z, weight=1.0, kind="odom", rot_weight=None):
        """Add a constraint T_j = T_i * Z.  i and j are node ids."""
        a, b = self.index[i], self.index[j]
        self.edges.append((a, b, np.asarray(Z, np.float64), float(weight),
                           kind, rot_weight))

    def n_edges(self, kind=None):
        return sum(1 for e in self.edges if kind is None or e[4] == kind)

    # --------------------------------------------------------------- residuals
    def residual(self, e, poses=None):
        a, b, Z = e[0], e[1], e[2]
        P = self.poses if poses is None else poses
        E = np.linalg.inv(Z) @ np.linalg.inv(P[a]) @ P[b]
        return se3_log(E)

    def residuals(self, kind=None):
        return np.array([self.residual(e) for e in self.edges
                         if kind is None or e[4] == kind])

    # -------------------------------------------------------------- optimising
    def optimize(self, iterations=60, damping=1e-6, huber=0.15,
                 fix_first=True, verbose=False):
        """Gauss-Newton on the manifold with a Huber robust kernel.

        huber   : residual norm (in metres/radians, mixed units) beyond which an
                  edge is down-weighted; 0 disables the kernel.
        fix_first: keep node 0 fixed, which pins the gauge freedom.
        """
        n = len(self.ids)
        free = np.ones(n, bool)
        if fix_first:
            free[0] = False
        fslot = np.where(free)[0]
        remap = -np.ones(n, int)
        remap[fslot] = np.arange(len(fslot))
        ndof = 6 * len(fslot)

        hist = []
        for it in range(iterations):
            H = np.zeros((ndof, ndof))
            g = np.zeros(ndof)
            chi2 = 0.0
            worst = 0.0
            for e in self.edges:
                a, b, Z, w, kind, rw = e
                r = self.residual(e)
                chi2 += w * float(r @ r)
                worst = max(worst, float(np.linalg.norm(r)))

                # robust weight
                rn = float(np.linalg.norm(r))
                ws = 1.0
                if huber > 0 and rn > huber:
                    ws = huber / rn
                ww = w * ws

                # Jacobians: with right updates
                #   r(di,dj) = Log( Z^-1 Ti^-1 Exp(-di) Exp(dj) Tj )
                Jr_inv = _jacobian_inv_se3(r)
                Adj = adjoint(np.linalg.inv(self.poses[b]) @ self.poses[a])
                Ji = -Jr_inv @ Adj
                Jj = Jr_inv

                # per-edge scaling, optionally anisotropic on rotation
                if rw is not None:
                    S = np.ones(6)
                    S[3:] = rw
                    Ji = Ji * S[None, :]
                    Jj = Jj * S[None, :]
                    ww = w  # scaling already applied

                blocks = []
                if free[a]:
                    ia = 6 * remap[a]
                    H[ia:ia + 6, ia:ia + 6] += ww * (Ji.T @ Ji)
                    g[ia:ia + 6] += ww * (Ji.T @ r)
                    blocks.append((ia, Ji))
                if free[b]:
                    ib = 6 * remap[b]
                    H[ib:ib + 6, ib:ib + 6] += ww * (Jj.T @ Jj)
                    g[ib:ib + 6] += ww * (Jj.T @ r)
                    blocks.append((ib, Jj))
                if len(blocks) == 2:
                    (ia, Ji_), (ib, Jj_) = blocks
                    H[ia:ia + 6, ib:ib + 6] += ww * (Ji_.T @ Jj_)
                    H[ib:ib + 6, ia:ia + 6] += ww * (Jj_.T @ Ji_)

            if ndof == 0:
                break
            H += damping * np.eye(ndof)
            try:
                dx = np.linalg.solve(H, -g)
            except np.linalg.LinAlgError:
                dx = np.linalg.lstsq(H, -g, rcond=None)[0]

            # manifold update
            for k, slot in enumerate(fslot):
                d = dx[6 * k:6 * k + 6]
                # right multiplication: T <- T * Exp(d)
                self.poses[slot] = self.poses[slot] @ se3_exp(d)

            step = float(np.linalg.norm(dx))
            hist.append((it, chi2, worst, step))
            if verbose:
                print("    it%3d  chi2=%10.4f  worst=%8.4f  step=%.3e"
                      % (it, chi2, worst, step), flush=True)
            if step < 1e-10:
                break
        return hist

    # ------------------------------------------------------------------ report
    def stats(self, kind=None):
        r = self.residuals(kind)
        if len(r) == 0:
            return dict(n=0)
        nrm = np.linalg.norm(r, axis=1)
        return dict(n=len(r), mean=float(nrm.mean()), median=float(np.median(nrm)),
                    max=float(nrm.max()))
