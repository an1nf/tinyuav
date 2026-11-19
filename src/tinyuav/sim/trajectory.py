"""Trajectories: expert cubic B-splines, student points at t = 0.1..1 s, order-5 polynomial projection
with time scaling, and selection by collision then snap cost.
"""
import numpy as np

T_I = np.arange(1, 11) / 10.0          # s


def _bspline_basis(t, knots, k=3):
    """Cox-de Boor basis (len(t), n_ctrl) of a clamped B-spline of degree k."""
    t = np.clip(np.asarray(t, float), knots[0], knots[-1] - 1e-9)
    n = len(knots) - k - 1
    N = np.array([[(knots[i] <= x < knots[i + 1]) * 1.0 for i in range(len(knots) - 1)] for x in t])
    for d in range(1, k + 1):
        M = np.zeros((len(t), len(knots) - 1 - d))
        for i in range(len(knots) - 1 - d):
            a = knots[i + d] - knots[i]
            b = knots[i + d + 1] - knots[i + 1]
            if a > 0:
                M[:, i] += (t - knots[i]) / a * N[:, i]
            if b > 0:
                M[:, i] += (knots[i + d + 1] - t) / b * N[:, i + 1]
        N = M
    return N[:, :n]


KNOTS = np.array([0, 0, 0, 0, 0.5, 1, 1, 1, 1], float)
_BS = _bspline_basis(T_I, KNOTS)


def bspline(P0, V0, C, t=None):
    """Clamped cubic B-spline over 1 s: first two control points from the state, last three sampled; (K, 10, 3)."""
    B = _BS if t is None else _bspline_basis(t, KNOTS)
    P0, V0 = np.asarray(P0, float), np.asarray(V0, float)
    P1 = P0 + V0 * (KNOTS[4] / 3)
    return (B[None, :, 0, None] * P0[None, None] + B[None, :, 1, None] * P1[None, None]
            + np.einsum('nj,kjd->knd', B[:, 2:], C))


def _basis(t, d=0):
    """Order-5 polynomial basis, or its d-th derivative, at t."""
    t = np.atleast_1d(np.asarray(t, float))
    B = np.zeros((len(t), 6))
    for k in range(d, 6):
        c = 1.0
        for j in range(d):
            c *= (k - j)
        B[:, k] = c * t ** (k - d)
    return B


_KKT = {}


def poly5_fit(points, p0, v0, a0, t=T_I):
    """Least-squares fit of points (N, 3) with p, v, a fixed at t = 0: coefficients (3, 6)."""
    key = tuple(np.round(t, 6))
    if key not in _KKT:
        A = _basis(t)
        Ceq = np.vstack([_basis(0.0), _basis(0.0, 1), _basis(0.0, 2)])
        K = np.block([[2 * A.T @ A, Ceq.T], [Ceq, np.zeros((3, 3))]])
        _KKT[key] = (A, np.linalg.inv(K))
    A, Kinv = _KKT[key]
    out = np.zeros((3, 6))
    for ax in range(3):
        rhs = np.concatenate([2 * A.T @ points[:, ax], [p0[ax], v0[ax], a0[ax]]])
        out[ax] = (Kinv @ rhs)[:6]
    return out


def poly_eval(coef, t, d=0):
    """d-th derivative at t: (len(t), 3)."""
    return _basis(t, d) @ coef.T


def snap_cost(coef, T=1.0):
    """Integral of |d4p/dt4|^2 over [0, T], analytic."""
    A, B = 24 * coef[:, 4], 120 * coef[:, 5]
    return float(np.sum(A * A * T + A * B * T ** 2 + B * B * T ** 3 / 3))


class Plan:
    """Polynomial anchored at t0, played with time scale beta: tau = beta (t - t0)."""

    def __init__(self, coef, t0, beta=1.0, horizon=1.0, source='student'):
        self.coef, self.t0, self.beta, self.horizon, self.source = coef, t0, beta, horizon, source

    def ref(self, t):
        """Position, velocity and acceleration at absolute time t, clamped to the horizon."""
        tau = float(np.clip((t - self.t0) * self.beta, 0.0, self.horizon))
        p = poly_eval(self.coef, [tau])[0]
        v = poly_eval(self.coef, [tau], 1)[0] * self.beta
        a = poly_eval(self.coef, [tau], 2)[0] * self.beta ** 2
        return p, v, a


def time_scale(coef, v_des):
    """beta = v_des / mean speed of the plan."""
    v_mu = float(np.linalg.norm(poly_eval(coef, [1.0])[0] - poly_eval(coef, [0.0])[0]))
    return v_des / max(v_mu, 1e-3)


def select(coefs, costs, ratio=0.95):
    """Index: hypotheses with c*/c_k >= ratio, then the lowest snap cost."""
    costs = np.asarray(costs, float) + 1e-3
    ok = np.where(costs.min() / costs >= ratio)[0]
    return int(ok[np.argmin([snap_cost(coefs[k]) for k in ok])])


def to_heading_frame(P, p0, yaw):
    """World points to the frame at p0 rotated by -yaw."""
    c, s = np.cos(yaw), np.sin(yaw)
    d = np.asarray(P, float) - np.asarray(p0, float)
    return np.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1], d[..., 2]], -1)


def from_heading_frame(Q, p0, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    Q = np.asarray(Q, float)
    return np.stack([c * Q[..., 0] - s * Q[..., 1], s * Q[..., 0] + c * Q[..., 1], Q[..., 2]], -1) + np.asarray(p0, float)


def rotate_to_heading(v, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    v = np.asarray(v, float)
    return np.stack([c * v[..., 0] + s * v[..., 1], -s * v[..., 0] + c * v[..., 1], v[..., 2]], -1)
