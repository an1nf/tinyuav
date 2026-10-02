"""Privileged expert of Loquercio et al. 2021 (Methods A, §S6).

It knows the true state and the whole geometry. Two stages:

1. **Global collision-free trajectory** τ_gbl from the start to 40 m ahead (the paper uses Liu et al.
   2017). Here: Dijkstra on a 0.2 m grid at flight altitude with a clearance penalty, shortcut
   smoothing and constant-speed time parameterization. It biases the sampler toward free space and
   extends its effective horizon (ablation "LocalPlan", Fig. 7a).
2. **Metropolis-Hastings sampling** of the distribution P(τ | τ_ref, C) ∝ exp(−c(τ)) (Eq. 1), with
   c(τ) = ∫ λ_c C_collision(τ(t)) + (τ(t) − τ_ref(t))ᵀ Q (τ(t) − τ_ref(t)) dt (Eq. 2), λ_c = 1000,
   C_collision = 0 if d_c > 2 r_q, else −d_c²/r_q² + 4 (Eq. 3), r_q = 0.2 m. τ is a cubic B-spline with
   three control points sampled in spherical coordinates (two more are fixed by the current position
   and velocity, τ(0) = x); 50 000 samples with a Gaussian proposal of
   variance 2, 5, 10 that increases every 16 000 samples; discretization 0.1 s. Samples in collision
   are removed and the three best are the labels. The reference segment of τ_gbl is added as one more
candidate (it is collision-free by construction).

Not specified in the paper and chosen here: Q, the units of the proposal variances (base σ of 0.15 m in
radius and 0.06 rad in angle, multiplied by √variance), parallel chains (100 × 500 steps instead of one
chain of 50 000) and a minimum distance between the three labels (otherwise the top 3 of an M-H chain
are near-copies of one mode).
"""
import heapq
import math

import numpy as np

from nanouav.agile.geometry import Field
from nanouav.agile.trajectory import T_I, bspline

RQ = 0.2
LAMBDA_C = 1000.0
DT = 0.1


def collision_cost(d, rq=RQ):
    """Eq. 3 (d = distance from the drone center to the nearest obstacle point; inside an obstacle the
    signed distance is negative and the cost saturates at 4)."""
    d = np.maximum(d, 0.0)
    return np.where(d > 2 * rq, 0.0, -d * d / (rq * rq) + 4.0)


class GlobalRef:
    """Collision-free reference at constant speed. pos(s) by arc length s; closest(p) → s."""

    def __init__(self, pts, v_des):
        self.pts = np.asarray(pts, float)
        seg = np.linalg.norm(np.diff(self.pts, axis=0), axis=1)
        self.s = np.r_[0.0, np.cumsum(seg)]
        self.length = float(self.s[-1])
        self.v = v_des

    def pos(self, s):
        s = np.clip(np.asarray(s, float), 0, self.length)
        return np.stack([np.interp(s, self.s, self.pts[:, k]) for k in range(3)], -1)

    def closest(self, p, s_hint=None, window=6.0):
        lo, hi = 0, len(self.pts)
        if s_hint is not None:
            lo = max(int(np.searchsorted(self.s, s_hint - window)) - 1, 0)
            hi = min(int(np.searchsorted(self.s, s_hint + window)) + 1, len(self.pts))
        P = self.pts[lo:hi]
        d = np.linalg.norm(P - np.asarray(p, float)[None], axis=1)
        i = int(np.argmin(d))
        return float(self.s[lo + i]), float(d[i])

    def ahead(self, s0, t=T_I):
        """Reference points at s0 + v·t (the expert's τ_ref over the horizon)."""
        return self.pos(s0 + self.v * np.asarray(t))


def global_plan(env, v_des, rq=RQ, res=0.2, inflate=0.15, w_clear=4.0, straight=None):
    """→ GlobalRef from env.start to the goal line (x ≥ goal.x). straight=True: straight segment.
    Narrow openings (e.g. a 0.8 m gap for a 0.4 m drone) may not exist on the coarse grid with the full
    inflation: the search is retried with a finer grid and a smaller inflation."""
    if not (straight if straight is not None else env.straight_ref):
        err = None
        for r_, inf_ in ((res, inflate), (0.1, inflate), (0.1, 0.05), (0.05, 0.02)):
            try:
                return _global_plan(env, v_des, rq, r_, inf_, w_clear)
            except RuntimeError as e:
                err = e
        raise err
    return _global_plan(env, v_des, rq, res, inflate, w_clear, straight=True)


def _global_plan(env, v_des, rq, res, inflate, w_clear, straight=False):
    A, B = np.asarray(env.start, float), np.asarray(env.goal, float)
    if straight:
        n = int(np.linalg.norm(B - A) / 0.1) + 1
        return GlobalRef(A + np.linspace(0, 1, n)[:, None] * (B - A), v_des)
    field = Field(env.world, ground=False)
    x0, x1 = A[0] - 4, B[0] + 4
    y0, y1 = min(A[1], B[1]) - 15, max(A[1], B[1]) + 15
    xs, ys = np.arange(x0, x1, res), np.arange(y0, y1, res)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    d = np.full(X.shape, np.inf)
    for dz in (-0.5, 0.0, 0.5):
        P = np.stack([X.ravel(), Y.ravel(), np.full(X.size, A[2] + dz)], 1)
        d = np.minimum(d, field(P).reshape(X.shape))
    free = d > rq + inflate
    cost = 1.0 + w_clear * np.clip(1.5 - d, 0, None) ** 2
    nx, ny = X.shape
    si = (int(round((A[0] - x0) / res)), int(round((A[1] - y0) / res)))
    gx = int(round((B[0] - x0) / res))
    dist = np.full((nx, ny), np.inf)
    prev = -np.ones((nx, ny, 2), int)
    dist[si] = 0.0
    pq = [(0.0, si)]
    nb = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    goal = None
    while pq:
        dc, (i, j) = heapq.heappop(pq)
        if dc > dist[i, j]:
            continue
        if i >= gx:
            goal = (i, j)
            break
        for di, dj in nb:
            a, b = i + di, j + dj
            if 0 <= a < nx and 0 <= b < ny and free[a, b]:
                nd = dc + res * math.hypot(di, dj) * 0.5 * (cost[i, j] + cost[a, b])
                if nd < dist[a, b]:
                    dist[a, b] = nd
                    prev[a, b] = (i, j)
                    heapq.heappush(pq, (nd, (a, b)))
    if goal is None:
        raise RuntimeError(f'{env.name}: no collision-free global path')
    path = [goal]
    while path[-1] != si:
        path.append(tuple(prev[path[-1]]))
    path = np.array([[xs[i], ys[j]] for i, j in path[::-1]])
    raw = _shortcut(path, field, A[2], rq + inflate + 0.1)
    d_raw = field(np.c_[_resample(raw, 0.05), np.full(len(_resample(raw, 0.05)), A[2])]).min()
    for n_smooth in (3, 2, 1, 0):                       # corner cutting must not approach obstacles
        path = _chaikin(raw, n_smooth)
        P = _resample(path, 0.05)
        if n_smooth == 0 or field(np.c_[P, np.full(len(P), A[2])]).min() >= d_raw - 0.01:
            break
    pts = np.c_[path, np.full(len(path), A[2])]
    pts[0] = A
    return GlobalRef(_resample(pts, 0.1), v_des)


def _segment_free(a, b, field, z, clr):
    n = max(int(np.linalg.norm(b - a) / 0.1), 2)
    P = a + np.linspace(0, 1, n)[:, None] * (b - a)
    return bool((field(np.c_[P, np.full(n, z)]) > clr).all())


def _shortcut(path, field, z, clr):
    out, i = [path[0]], 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not _segment_free(path[i], path[j], field, z, clr):
            j = max(i + 1, (i + j) // 2 if j - i > 8 else j - 1)
        out.append(path[j])
        i = j
    return np.array(out)


def _chaikin(P, n):
    for _ in range(n):
        Q = [P[0]]
        for a, b in zip(P[:-1], P[1:]):
            Q += [0.75 * a + 0.25 * b, 0.25 * a + 0.75 * b]
        Q.append(P[-1])
        P = np.array(Q)
    return P


def _resample(P, ds):
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    t = np.arange(0, s[-1], ds)
    return np.stack([np.interp(t, s, P[:, k]) for k in range(P.shape[1])], 1)


def _sph(r, az, el):
    return np.stack([r * np.cos(el) * np.cos(az), r * np.cos(el) * np.sin(az), r * np.sin(el)], -1)


class Expert:
    def __init__(self, world, rq=RQ, lam_c=LAMBDA_C, Q=(10.0, 10.0, 20.0), n_samples=50000, chains=100,
                 variances=(2.0, 5.0, 10.0), switch=16000, sigma_r=0.15, sigma_a=0.06, min_sep=0.3,
                 z_min=0.4, seed=0):
        self.field = Field(world, ground=True)
        self.rq, self.lam_c, self.Q = rq, lam_c, np.asarray(Q, float)
        self.n_samples, self.chains, self.variances, self.switch = n_samples, chains, variances, switch
        self.sig = np.array([sigma_r, sigma_a, sigma_a])
        self.min_sep, self.z_min = min_sep, z_min
        self.rng = np.random.default_rng(seed)

    def cost(self, P0, V0, C, ref, field=None):
        """C (K,3,3) sampled control points → (total cost, collision cost per trajectory, min distance, points)."""
        T = bspline(P0, V0, C)                              # (K, 10, 3)
        K = len(T)
        d = (field or self.field)(T.reshape(-1, 3)).reshape(K, -1)
        cc = collision_cost(d, self.rq)
        e = T - ref[None]
        track = (e * e * self.Q).sum(-1)
        c = ((self.lam_c * cc + track) * DT).sum(1)
        return c, (cc * DT).sum(1), d.min(1), T

    def label(self, p0, ref, k=3, v0=None):
        """M-H sampling around the reference points ref (10,3) → dict with the k best collision-free
        trajectories (k,10,3, world frame), their costs and diagnostics."""
        p0 = np.asarray(p0, float)
        v0 = np.zeros(3) if v0 is None else np.asarray(v0, float)
        reach = float(np.linalg.norm(ref - p0, axis=1).max()) * 1.6 + 2 * self.rq + 1.0
        field = self.field.local(p0, reach)
        K = self.chains
        steps = int(math.ceil(self.n_samples / K))
        # initial chain state: control points on the reference at t = 1/3, 2/3, 1 (spherical, around p0)
        init = np.stack([[np.interp(j / 3, np.r_[0.0, T_I], np.r_[p0[a], ref[:, a]]) for a in range(3)]
                         for j in (1, 2, 3)]) - p0
        r0 = np.linalg.norm(init, axis=1)
        th = np.stack([r0, np.arctan2(init[:, 1], init[:, 0]), np.arcsin(np.clip(init[:, 2] / np.maximum(r0, 1e-6), -1, 1))], 1)
        theta = np.repeat(th[None], K, 0) + self.rng.normal(0, 1, (K, 3, 3)) * self.sig * math.sqrt(self.variances[0])
        theta[..., 0] = np.abs(theta[..., 0])
        C = p0 + _sph(theta[..., 0], theta[..., 1], theta[..., 2])
        c, cc, dmin, T = self.cost(p0, v0, C, ref, field)
        keep_T, keep_c, keep_cc, keep_d = [T], [c], [cc], [dmin]
        acc = 0
        for st in range(1, steps):
            n_done = st * K
            var = self.variances[min(n_done // self.switch, len(self.variances) - 1)]
            prop = theta + self.rng.normal(0, 1, theta.shape) * self.sig * math.sqrt(var)
            prop[..., 0] = np.abs(prop[..., 0])
            Cp = p0 + _sph(prop[..., 0], prop[..., 1], prop[..., 2])
            cp, ccp, dp, Tp = self.cost(p0, v0, Cp, ref, field)
            ok = np.log(self.rng.random(K) + 1e-300) < -(cp - c)        # α = min(1, s(τ')/s(τ))
            theta[ok], c[ok] = prop[ok], cp[ok]
            acc += ok.sum()
            keep_T.append(Tp), keep_c.append(cp), keep_cc.append(ccp), keep_d.append(dp)
        # the reference itself is also a candidate: in a gap barely wider than 2·r_q every sample pays
        # collision cost and the sampler may not refine to the centered τ_gbl within 50 000 samples
        d_ref = field(ref)
        cc_ref = (collision_cost(d_ref, self.rq) * DT).sum()
        keep_T.append(ref[None]), keep_c.append(np.array([self.lam_c * cc_ref])), keep_cc.append(np.array([cc_ref]))
        keep_d.append(np.array([d_ref.min()]))
        T = np.concatenate(keep_T)
        c = np.concatenate(keep_c)
        cc = np.concatenate(keep_cc)
        dmin = np.concatenate(keep_d)
        free = (dmin > self.rq) & (T[..., 2].min(1) > self.z_min)
        idx = np.where(free)[0]
        idx = idx[np.argsort(c[idx])]
        chosen = []
        for i in idx:
            if all(np.linalg.norm(T[i] - T[j], axis=1).mean() > self.min_sep for j in chosen):
                chosen.append(i)
            if len(chosen) == k:
                break
        while chosen and len(chosen) < k:                    # fewer distinct modes: repeat the best
            chosen.append(chosen[0])
        return {'trajs': T[chosen] if chosen else np.zeros((0, 10, 3)), 'costs': c[chosen] if chosen else np.zeros(0),
                'coll': cc[chosen] if chosen else np.zeros(0), 'n_free': int(len(idx)), 'n': int(len(c)),
                'accept': acc / max((steps - 1) * K, 1)}
