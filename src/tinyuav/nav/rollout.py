"""Closed loop: 6-DoF quadrotor, stereo camera with SGM, and one pilot.

gbl       tracks the global collision-free path (privileged)
blind     tracks the straight start-goal line, no perception
expert    privileged Metropolis-Hastings expert in receding horizon
student   DAgger network: depth, velocity, attitude, direction to 3 trajectories and costs
dagger    student within xi of the global path, else the path; every frame labeled by the expert
reactive  candidate fan judged on its own SGM, no network
diffphys  recurrent policy trained through differentiable physics: acceleration from depth and state

A mission gives the direction, reference, desired speed and yaw per frame (`GoalMission`, or
`tinyuav.follow.mission.BalloonMission`). Plans are projected on order-5 polynomials, time-scaled to v_des
and tracked by the cascaded controller.
"""
import math
import time
from dataclasses import dataclass, field

import numpy as np

from tinyuav.sim import trajectory as TR
from tinyuav.nav.expert import Expert, global_plan, GlobalRef
from tinyuav.sim.geometry import Field
from tinyuav.sim.vehicle import AGILE, RATE_MAX, TILT_MAX_DEG
from tinyuav.sim.stereo import StereoRig, observe_full
from tinyuav.sim.controller import CascadedController
from tinyuav.sim.quad import QuadSim, euler_from_rot

DT_CTRL = 0.002
KP_POS = 3.0


@dataclass
class Config:
    v_des: float = 5.0
    cam_hz: float = 15.0
    latency: float = 0.05             # s, capture to command
    t_max: float = 0.0                # 0: length / v_des + 6 s
    xi: float = 0.0                   # DAgger threshold (m)
    net_size: tuple = (120, 160)      # network depth resolution (H, W)
    esdf_res: float = 0.25            # local distance map for the collision-cost label
    esdf_box: tuple = (-2.0, 14.0, -8.0, 8.0)
    rig: StereoRig = field(default_factory=StereoRig)
    seed: int = 0
    label_every: int = 1              # DAgger: label every n frames
    # student direction input: 'line' (test-time reference), 'gbl' (global path) or 'goal'
    omega_ref: str = 'line'
    # expert label reference: 'gbl' (global path from the start) or 'ctg' (replanned on the cost-to-go)
    label_ref: str = 'gbl'


def straight_ref(env, v):
    A, B = np.asarray(env.start, float), np.asarray(env.goal, float)
    n = int(np.linalg.norm(B - A) / 0.1) + 1
    return GlobalRef(A + np.linspace(0, 1, n)[:, None] * (B - A), v)


def goal_direction(line, p, s_hint=None):
    """Unit vector to the point 1 s ahead of the closest point of the start-goal line, and that arc length."""
    s, _ = line.closest(p, s_hint)
    q = line.pos(min(s + line.v * 1.0, line.length))
    d = q - p
    n = np.linalg.norm(d)
    return (d / n if n > 1e-6 else np.array([1.0, 0, 0])), s


def downsample_depth(d, size):
    """Nearest-neighbor resize that keeps holes (0)."""
    H, W = d.shape
    h, w = size
    ys = (np.arange(h) + 0.5) * H / h
    xs = (np.arange(w) + 0.5) * W / w
    return d[ys.astype(int)][:, xs.astype(int)]


def local_esdf(field, p, yaw, z, cfg):
    """2D distance without ground on a heading-frame grid, min over z +- 0.4 m: (ny, nx) float16."""
    x0, x1, y0, y1 = cfg.esdf_box
    r = cfg.esdf_res
    xs, ys = np.arange(x0, x1, r) + r / 2, np.arange(y0, y1, r) + r / 2
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    W = TR.from_heading_frame(np.stack([X, Y, np.zeros_like(X)], -1), p, yaw)
    d = np.full(X.shape, np.inf)
    for dz in (-0.4, 0.0, 0.4):
        P = W.reshape(-1, 3).copy()
        P[:, 2] = z + dz
        d = np.minimum(d, field(P).reshape(X.shape))
    return np.clip(d, -1, 30).astype(np.float16)


class GoalMission:
    """Reach a goal 40 m ahead along a straight line; privileged pilots use the global path."""
    needs_camera = False

    def __init__(self, env, v_des, gref=None, omega_ref='line', label_ref='gbl'):
        self.env, self.v, self.omega_ref, self.label_ref = env, v_des, omega_ref, label_ref
        self._ctg = None
        if label_ref == 'ctg' and gref is None:
            # xi is measured from the cost-to-go path planned at the start
            gref = self._cost_to_go().ref_from(np.asarray(env.start, float), v_des)
        self.gref = gref or global_plan(env, v_des, straight=False)
        self.line = straight_ref(env, v_des)
        self.s_g = self.s_l = 0.0

    def start(self):
        A = np.asarray(self.env.start, float)
        d0 = self.gref.pos(0.5) - A
        yaw0 = math.atan2(d0[1], d0[0])
        return A, yaw0, np.array([math.cos(yaw0), math.sin(yaw0), 0.0]) * self.v

    def t_max(self):
        return self.line.length / self.v + 6.0

    def render_world(self):
        return self.env.world

    def step(self, t):
        pass

    def frame(self, t, p, v, yaw, cam=None):
        omega, self.s_l = goal_direction(self.line, p, self.s_l)
        self.s_g, d_g = self.gref.closest(p, self.s_g)
        if self.omega_ref == 'gbl':                      # global path 1 s ahead
            q = self.gref.pos(min(self.s_g + self.v * 1.0, self.gref.length)) - p
            omega = q / max(float(np.linalg.norm(q)), 1e-6)
        elif self.omega_ref == 'goal':                   # straight to the goal
            q = np.asarray(self.env.goal, float) - p
            omega = q / max(float(np.linalg.norm(q)), 1e-6)
        s_b, _ = self.line.closest(p, self.s_l)
        out = {'omega': omega, 'ref': self.gref.ahead(self.s_g), 'd_ref': d_g, 'v_des': self.v,
               'blind': self.line.ahead(s_b), 'yaw': None}
        if self.label_ref == 'ctg':
            out['ref'] = out['label_ref'] = self.ctg_ref(p)
        return out

    def _cost_to_go(self):
        if self._ctg is None:
            from tinyuav.nav.expert import W_LINE, CostToGo
            self._ctg = CostToGo(self.env, w_line=W_LINE)
        return self._ctg

    def ctg_ref(self, p):
        """Path replanned from p down the cost-to-go, 1 s of it."""
        r = self._cost_to_go().ref_from(p, self.v)
        return r.ahead(0.0) if r is not None else self.gref.ahead(self.s_g)

    def check(self, t, p):
        """(success, failure, clearance to dynamic objects)."""
        return self.line.closest(p, self.s_l)[0] >= self.line.length - 0.5, False, np.inf

    def metrics(self):
        return {}


def fit_plan(P, p, v, a, v_des, proj='scaled', tscale='chord', v_cap=None):
    """Projection of points on the flown polynomial and its time scale: (coef, beta)."""
    if proj == 'raw':
        coef = TR.poly5_fit(P, p, v, a)
        return coef, TR.time_scale(coef, v_des)
    P = np.asarray(P, float)
    if tscale == 'arc':
        length = float(np.linalg.norm(np.diff(np.vstack([p, P]), axis=0), axis=1).sum())
    else:
        length = float(np.linalg.norm(P[-1] - p))
    beta = float(np.clip(v_des / max(length / TR.T_I[-1], 1e-3), 0.25, 4.0))
    coef = TR.poly5_fit(P, p, np.asarray(v) / beta, np.asarray(a) / beta ** 2)
    if v_cap:
        cap = max(v_cap * v_des, float(np.linalg.norm(v)) + 0.1)
        for _ in range(3):
            peak = beta * float(np.linalg.norm(TR.poly_eval(coef, np.linspace(0, 1, 21), 1), axis=1).max())
            if peak <= cap * 1.02:
                break
            beta = max(beta * cap / peak, 0.25)
            coef = TR.poly5_fit(P, p, np.asarray(v) / beta, np.asarray(a) / beta ** 2)
    return coef, beta


def _among_free(fits, clear, margin=0.15, costs=None):
    """Index of the least-snap plan (or the cost rule if costs are given) among those with clearance >=
    radius + margin; if none, the one with the most clearance.
    """
    ok = np.where(np.asarray(clear) >= AGILE.radius + margin)[0]
    if not len(ok):
        return int(np.argmax(clear))
    if costs is None:
        return int(ok[np.argmin([TR.snap_cost(fits[k][0]) for k in ok])])
    return int(ok[TR.select([fits[k][0] for k in ok], [costs[k] for k in ok])])


def oracle_clearance(fits, field, margin=0.15, lookahead=0.0):
    """True clearance of each flown polynomial, optionally with its continuation."""
    tt = np.linspace(0, 1, 21)
    clear = np.array([float(field(TR.poly_eval(c, tt)).min()) for c, _ in fits])
    return with_lookahead(clear, fits, lambda X: float(field(X).min()), lookahead, margin)


def remainder(plan, t, n=11):
    """n points of `plan` from time t to its end, and the time left (s)."""
    tau0 = float(np.clip((t - plan.t0) * plan.beta, 0.0, plan.horizon))
    return TR.poly_eval(plan.coef, np.linspace(tau0, plan.horizon, n)), (plan.horizon - tau0) / plan.beta


def keep_previous(clear_new, clear_old, left, margin=0.15, min_left=0.3):
    """Keep the current plan when no new one is free and at least `min_left` s of it still is."""
    free = AGILE.radius + margin
    return bool(left >= min_left and clear_old >= free and float(np.max(clear_new)) < free)


FAN_ANGLES = (0, 10, -10, 20, -20, 30, -30, 45, -45, 60, -60)    # deg around omega
FAN_DZ = (0.0, 0.5, -0.5)                                        # m up/down at 1 s


def fan_points(p, v, omega, v_des, z_lim=None, angles=FAN_ANGLES, dz=FAN_DZ):
    """Quadratic Bezier candidates over 1 s fanned around omega (and up/down): (K, 10, 3)."""
    p, v = np.asarray(p, float), np.asarray(v, float)
    f = np.asarray(omega, float)[:2]
    f = f / max(float(np.linalg.norm(f)), 1e-6)
    left = np.array([-f[1], f[0]])
    L = float(v_des) * 1.0
    sp = float(np.linalg.norm(v))
    u = v / sp if sp > 0.5 else np.r_[f, 0.0]
    C = p + 0.5 * L * u
    t = TR.T_I[:, None]
    out = []
    for dzi in dz:
        for a in angles:
            r = math.radians(a)
            E = p + np.r_[L * (math.cos(r) * f + math.sin(r) * left), dzi]
            if z_lim is not None:
                E[2] = float(np.clip(E[2], *z_lim))
            out.append((1 - t) ** 2 * p + 2 * t * (1 - t) * C + t ** 2 * E)
    return np.array(out)


def fan_cost(W, ref, prev=None, w_prev=0.3, clear=None, w_clear=0.0, sigma=0.3):
    """Candidate cost: mean squared distance to the line reference, plus w_prev times that to the current
    plan, plus a soft clearance term w_clear * exp(-(clearance - 0.35) / sigma).
    """
    W = np.asarray(W, float)
    J = ((W - np.asarray(ref, float)[None]) ** 2).sum(-1).mean(-1)
    if prev is not None:
        J = J + w_prev * ((W - np.asarray(prev, float)[None]) ** 2).sum(-1).mean(-1)
    if w_clear and clear is not None:
        c = np.minimum(np.asarray(clear, float), 10.0)
        J = J + w_clear * np.exp(-(c - (AGILE.radius + 0.15)) / sigma)
    return J


def _min_pool(dep, s):
    """Nearest valid depth of every s x s block and its pixel: (depth (h, w), v, u)."""
    H, W = dep.shape
    h, w = H // s, W // s
    B = dep[:h * s, :w * s].reshape(h, s, w, s).transpose(0, 2, 1, 3).reshape(h, w, s * s)
    B = np.where(B > 0, B, np.inf)
    k = np.argmin(B, axis=2)
    d = np.take_along_axis(B, k[..., None], 2)[..., 0]
    vv = np.arange(h)[:, None] * s + k // s
    uu = np.arange(w)[None, :] * s + k % s
    return np.where(np.isfinite(d), d, 0.0), vv, uu


def edge_filter(depth, tau0=0.3, gamma=0.03):
    """Drop SGM pixels with a 4-neighbour jump above max(tau0, gamma z^2) or an invalid neighbour."""
    d = np.asarray(depth, np.float32)
    valid = d > 0
    bad = np.zeros_like(valid)
    tau = np.maximum(tau0, gamma * d * d)
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        n = np.roll(d, (dy, dx), (0, 1))
        nv = n > 0
        bad |= valid & ((nv & (np.abs(d - n) > tau)) | ~nv)
    return np.where(bad, 0.0, d)


def sgm_points(depth, rig, pose, stride=4, pool='stride', edge=False):
    """World points of the drone's own SGM; pool='stride' samples one pixel per block, 'min' the nearest."""
    from tinyuav.follow.estimate import camera_pose
    if edge:
        depth = edge_filter(depth)
    H, W = depth.shape
    if pool == 'min':
        dep, vv, uu = _min_pool(np.asarray(depth, np.float32), stride)
        ok = dep > 0
        X, v, u = dep[ok], vv[ok].astype(float), uu[ok].astype(float)
    else:
        dep = np.asarray(depth, np.float32)[::stride, ::stride]
        v, u = np.nonzero(dep > 0)
        X = dep[v, u]
        u, v = u * stride, v * stride
    u, v = u * rig.width / W, v * rig.height / H
    O, Rc = camera_pose(rig, pose)
    dirs = np.stack([np.ones_like(X), -(u + 0.5 - rig.width / 2) / rig.focal, -(v + 0.5 - rig.height / 2) / rig.focal], 1)
    return O + (dirs * X[:, None]) @ Rc.T


class PerceivedObstacles:
    """Obstacles from the drone's own SGM: cells hit in >= min_hits of the last n_frames frames.
    carve=True also needs a cell to be hit more often than seen through.
    """

    def __init__(self, n_frames=15, cell=0.15, min_hits=2, carve=False):
        from collections import deque
        self.hist, self.cell, self.min_hits = deque(maxlen=n_frames), cell, min_hits
        self.carve, self.seen_through = carve, deque(maxlen=n_frames)
        self._cells = self._tree = None

    def _carved(self, depth, rig, pose):
        """Keys of the window cells this frame saw through."""
        from tinyuav.follow.estimate import project
        A = [h for h in self.hist if len(h)]
        if not A:
            return np.zeros((0, 3), np.int64)
        K = np.unique(np.vstack(A), axis=0)
        uv, X = project(rig, pose, (K + 0.5) * self.cell)
        import cv2
        d = np.asarray(depth, np.float32)
        # nearest depth of the 3 x 3 neighbourhood: silhouette pixels are not seen through
        d = cv2.erode(np.where(d > 0, d, 1e6).astype(np.float32), np.ones((3, 3), np.uint8))
        d[d >= 1e6] = 0.0
        H, W = d.shape
        u = np.round((uv[:, 0] + 0.5) * W / rig.width - 0.5).astype(int)
        v = np.round((uv[:, 1] + 0.5) * H / rig.height - 0.5).astype(int)
        ok = (X > 0.3) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        z = np.zeros(len(K), np.float32)
        z[ok] = d[v[ok], u[ok]]
        through = ok & (z > 0) & (z > X + np.maximum(0.15, 0.02 * X * X))
        return K[through]

    def add(self, P, depth=None, rig=None, pose=None):
        if self.carve:
            self.seen_through.append(self._carved(depth, rig, pose) if depth is not None
                                     else np.zeros((0, 3), np.int64))
        self.hist.append(np.unique(np.floor(np.asarray(P, float) / self.cell).astype(np.int64), axis=0)
                         if len(P) else np.zeros((0, 3), np.int64))
        self._cells = self._tree = None

    def cells(self):
        if self._cells is None:
            A = [h for h in self.hist if len(h)]
            if not A:
                self._cells = np.zeros((0, 3))
            else:
                c, n = np.unique(np.vstack(A), axis=0, return_counts=True)
                keep = n >= min(self.min_hits, len(self.hist))
                T = [h for h in self.seen_through if len(h)]
                if self.carve and T:
                    ct, nt = np.unique(np.vstack(T), axis=0, return_counts=True)
                    idx = {tuple(r): k for r, k in zip(ct.tolist(), nt.tolist())}
                    through = np.array([idx.get(tuple(r), 0) for r in c.tolist()])
                    keep &= n > through
                self._cells = (c[keep] + 0.5) * self.cell
        return self._cells

    def nearest(self, X):
        """Distance from each point of X (M, 3) to the nearest perceived cell and that cell (inf, NaN if none)."""
        X = np.asarray(X, float).reshape(-1, 3)
        C = self.cells()
        if not len(C):
            return np.full(len(X), np.inf), np.full((len(X), 3), np.nan)
        if self._tree is None:
            try:
                from scipy.spatial import cKDTree
                self._tree = cKDTree(C)
            except ImportError:
                self._tree = False
        if self._tree is not False:
            d, j = self._tree.query(X)
            return d, C[j]
        d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        j = np.argmin(d2, axis=1)
        return np.sqrt(d2[np.arange(len(X)), j]), C[j]

    def clearance(self, X):
        """Distance from X (M, 3) to the nearest perceived obstacle (inf if none)."""
        d, _ = self.nearest(X)
        return float(d.min()) - 0.5 * self.cell if np.isfinite(d).any() else np.inf


def extension(fit, lookahead, n=8):
    """Plan continued at its final velocity for `lookahead` s."""
    coef, beta = fit
    p1 = TR.poly_eval(coef, [1.0])[0]
    v1 = TR.poly_eval(coef, [1.0], 1)[0] * beta
    s = np.linspace(lookahead / n, lookahead, n)
    return p1 + s[:, None] * v1


def with_lookahead(clear, fits, clear_of, lookahead, margin=0.15):
    """Clearance of each plan with its continuation: min(plan, continuation + margin)."""
    if not lookahead:
        return np.asarray(clear, float)
    ext = np.array([clear_of(extension(f, lookahead)) for f in fits])
    return np.minimum(np.asarray(clear, float), ext + margin)


def perceived_clearance(fits, obstacles):
    """Clearance of each flown polynomial to the perceived obstacles."""
    tt = np.linspace(0, 1, 21)
    return np.array([obstacles.clearance(TR.poly_eval(c, tt)) for c, _ in fits])


def perceived_index(fits, costs, obstacles, margin=0.15, clear=None):
    """Choice by the clearance to the perceived obstacles (see `_among_free`)."""
    clear = perceived_clearance(fits, obstacles) if clear is None else clear
    return _among_free(fits, clear, margin, costs)


def choose(fits, costs, W, prev=None, mode='snap'):
    """Index of the hypothesis to fly (see `Rollout.SELECT`)."""
    if mode == 'consistent' and prev is not None:
        c = np.asarray(costs, float) + 1e-3
        ok = np.where(c.min() / c >= 0.8)[0]
        return int(ok[np.argmin([np.linalg.norm(W[j][3:] - prev[3:], axis=1).mean() for j in ok])])
    return TR.select([c for c, _ in fits], costs)


class Rollout:
    A0_MODE = 'plan'             # initial acceleration of a plan: previous plan | zero | measured
    # projection: 'scaled' fits with v/beta and a/beta^2 so the scaled plan starts at the current state;
    # 'raw' fits with v and a
    PROJ = 'scaled'
    # time scale from the chord to the last point ('chord') or the path length ('arc')
    TSCALE = 'chord'
    V_CAP = None                 # lower beta until the peak plan speed <= V_CAP * v_des
    # choice: 'snap' (cost rule then least snap), 'consistent', 'oracle' (true clearance),
    # 'sgm' (own SGM), 'oracle_cost' / 'sgm_cost' (cost rule among the free ones)
    SELECT = 'snap'
    # sgm/oracle: the continuation at the final velocity must also be free
    LOOKAHEAD = 0.0
    # sgm/oracle: keep the current plan when no new one is free
    KEEP = False
    # sgm/oracle: fall back to fan candidates
    FAN = False
    POOL = 'stride'              # SGM sampling for 'sgm': 'stride' or 'min'
    EDGE = False                 # drop SGM pixels on depth edges
    SLOW = 0.0                   # reactive: retry the fan at SLOW * v_des
    CARVE = False                # free-space evidence in the perceived obstacles
    W_CLEAR = 0.0                # fan: weight of the soft clearance cost (m^2)
    DP_GT = False                # diffphys: true depth instead of SGM
    DP_Q = 0.0                   # diffphys: block quantile of the depth input
    DP_PERSIST = False           # diffphys: a near block must persist two frames
    YAW_FWD = False              # yaw follows the velocity only toward the goal
    DP_AMAX = 0.0                # diffphys: clip horizontal acceleration (m/s^2)

    def __init__(self, env, cfg, pilot='gbl', policy=None, label=False, expert_kw=None, gref=None, mission=None):
        self.env, self.cfg, self.pilot, self.policy, self.label = env, cfg, pilot, policy, label
        self.field = Field(env.world, ground=True)
        self.field_ng = Field(env.world, ground=False)
        self.mission = mission or GoalMission(env, cfg.v_des, gref, getattr(cfg, 'omega_ref', 'line'),
                                              getattr(cfg, 'label_ref', 'gbl'))
        self.gref = getattr(self.mission, 'gref', None)
        self.line = getattr(self.mission, 'line', None)
        self.z_ground = float(getattr(env.world, 'floor', 0.0))   # sample height above the floor
        kw = dict(expert_kw or {})
        kw.setdefault('z_min', self.z_ground + 0.4)
        self.expert = Expert(env.world, seed=cfg.seed, **kw) if (label or pilot == 'expert') else None
        self.rng = np.random.default_rng(cfg.seed)
        self._v_des = self._v_ref = cfg.v_des
        self.trace = None            # student decisions: hypotheses, costs, choice, clearance
        self.trace_depth = False     # keep the depth input (videos)
        self.dp_trace = None         # diffphys decisions: t, p, v, attitude, command
        # external renderer: f(p, yaw, pitch, roll) gives
        # (gray, SGM depth, true depth[, rgb]), e.g. Flightmare
        self.observer = None
        self.record_cam = False      # record frames for privileged pilots too

    def _trace_stats(self, cam):
        """SGM valid fraction (all, left, right halves) and median relative error."""
        sgm, gt = np.asarray(cam['depth']), np.asarray(cam['depth_gt'])
        v = sgm > 0
        h = sgm.shape[1] // 2
        d = self.trace[-1]
        d['sgm_valid'] = [round(float(v.mean()), 3), round(float(v[:, :h].mean()), 3), round(float(v[:, h:].mean()), 3)]
        m = v & (gt > 0)
        d['sgm_err'] = round(float(np.median(np.abs(sgm[m] - gt[m]) / gt[m])), 3) if m.sum() > 50 else None

    def _trace_images(self, depth, cam, gray):
        """Network input, full-resolution SGM and camera image of a trace entry."""
        d = self.trace[-1]
        d['depth'] = np.round(depth * 1000).astype(np.uint16)
        d['sgm'] = np.round(np.asarray(cam['depth']) * 1000).astype(np.uint16)
        d['img'] = cam['rgb'] if cam.get('rgb') is not None else gray
        if self.observer is not None:
            d['gray'] = gray                                   # external renderer's left image

    def _fit(self, P, p, v, a, v_des):
        """(coef, beta) of the plan through P at v_des."""
        return fit_plan(P, p, v, a, v_des, self.PROJ, self.TSCALE, self.V_CAP)

    def _plan_points(self, P, p, v, a, t, source, v_des=None):
        coef, beta = self._fit(P, p, v, a, v_des or self._v_des)
        return TR.Plan(coef, t, beta, source=source)

    def plan_ref(self, ref, p, v, a, t, source='gbl'):
        return self._plan_points(ref, p, v, a, t, source, self._v_ref)

    def plan_expert(self, ref, p, v, a, t):
        """Best expert label whose projected polynomial stays collision-free."""
        L = self.expert.label(p, ref, k=3, v0=v)
        if not len(L['trajs']):
            return self.plan_ref(ref, p, v, a, t)
        best, best_d = None, -np.inf
        for P in L['trajs']:
            pl = self._plan_points(P, p, v, a, t, 'expert', self._v_ref)
            d = float(self.field(TR.poly_eval(pl.coef, np.linspace(0, 1, 21))).min())
            if d > AGILE.radius + 0.05:
                return pl
            if d > best_d:
                best, best_d = pl, d
        return best

    def plan_student(self, depth, p, v, a, R, yaw, omega, t):
        trajs, costs = self.policy(depth, TR.rotate_to_heading(v, yaw), R.ravel(), TR.rotate_to_heading(omega, yaw))
        W = TR.from_heading_frame(trajs, p, yaw)
        fits = [self._fit(W[k], p, v, a, self._v_des) for k in range(len(W))]
        eff, clear_of = None, None
        if self.SELECT in ('oracle', 'oracle_cost'):        # privileged
            eff = oracle_clearance(fits, self.field, lookahead=self.LOOKAHEAD)
            k = _among_free(fits, eff, costs=costs if self.SELECT == 'oracle_cost' else None)
            clear_of = lambda X: float(self.field(X).min())
        elif self.SELECT in ('sgm', 'sgm_cost'):             # own SGM
            self._seen = perceived_clearance(fits, self._perceived)
            eff = with_lookahead(self._seen, fits, self._perceived.clearance, self.LOOKAHEAD)
            k = perceived_index(fits, costs if self.SELECT == 'sgm_cost' else None, self._perceived, clear=eff)
            clear_of = self._perceived.clearance
        else:
            k = choose(fits, costs, W, getattr(self, '_prev_W', None), self.SELECT)
        self._prev_W = W[k]
        self._last_sel = (k, fits)
        self._kept, self._old, self._fanned = None, None, None
        old = getattr(self, '_flying', None)
        if eff is not None and old is not None and old.source == 'student' and (self.KEEP or self.trace is not None):
            X, left = remainder(old, t + self.cfg.latency)   # from when the new plan takes over
            c_old = clear_of(X)
            if self.LOOKAHEAD:
                c_old = min(c_old, clear_of(extension((old.coef, old.beta), self.LOOKAHEAD)) + 0.15)
            would = keep_previous(eff, c_old, left)
            # no new plan free while the rest of the current one is
            self._old = {'none_free': bool(float(np.max(eff)) < AGILE.radius + 0.15), 'old_free': would,
                         'old_left': round(float(left), 3), 'X': X}
            if self.KEEP and would:
                self._kept = (X, c_old)
                return old, W, costs
        if self.FAN and eff is not None and float(np.max(eff)) < AGILE.radius + 0.15:
            pick = self._fan_choice(p, v, a, omega, t, clear_of)
            if pick is not None:
                self._fanned = pick[1]
                return pick[0], W, costs
        return TR.Plan(fits[k][0], t, fits[k][1], source='student'), W, costs

    def _fan_choice(self, p, v, a, omega, t, clear_of, source='student', v_des=None):
        """Free fan candidate nearest the line: (Plan, points) or None."""
        z0 = float(self.env.start[2])
        vd = v_des or self._v_des
        Wf = fan_points(p, v, omega, vd, (self.z_ground + 0.7, z0 + 1.0))
        fits = [self._fit(P, p, v, a, vd) for P in Wf]
        tt = np.linspace(0, 1, 21)
        eff = np.array([clear_of(TR.poly_eval(c, tt)) for c, _ in fits])
        eff = with_lookahead(eff, fits, clear_of, self.LOOKAHEAD)
        ok = np.where(eff >= AGILE.radius + 0.15)[0]
        if not len(ok):
            return None
        old = getattr(self, '_flying', None)
        prev = np.array([old.ref(t + ti)[0] for ti in TR.T_I]) if old is not None else None
        ref = self._blind
        if v_des:
            ref = p + (np.asarray(self._blind) - p) * (vd / max(self._v_des, 1e-6))
        J = fan_cost(Wf[ok], ref, prev, clear=eff[ok], w_clear=self.W_CLEAR)
        k = int(ok[np.argmin(J)])
        return TR.Plan(fits[k][0], t, fits[k][1], source=source), Wf[k]

    def plan_reactive(self, p, v, a, omega, t):
        """Reactive pilot: fan candidates judged by the clearance to its own SGM of the last second; the free one
        nearest the line, else keep the current plan if still free (KEEP), else the most clearance.
        """
        clear_of = self._perceived.clearance if self._perceived is not None else (lambda X: np.inf)
        pick = self._fan_choice(p, v, a, omega, t, clear_of, source='reactive')
        if pick is not None:
            return pick
        old = getattr(self, '_flying', None)
        if self.KEEP and old is not None and old.source == 'reactive':
            X, left = remainder(old, t + self.cfg.latency)
            if left >= 0.3 and clear_of(X) >= AGILE.radius + 0.15:
                return old, X[1:]
        if self.SLOW:                                      # nothing free at speed: slower fan
            pick = self._fan_choice(p, v, a, omega, t, clear_of, source='reactive', v_des=self.SLOW * self._v_des)
            if pick is not None:
                return pick
        z0 = float(self.env.start[2])
        Wf = fan_points(p, v, omega, self._v_des, (self.z_ground + 0.7, z0 + 1.0))
        fits = [self._fit(P, p, v, a, self._v_des) for P in Wf]
        tt = np.linspace(0, 1, 21)
        k = int(np.argmax([clear_of(TR.poly_eval(c, tt)) for c, _ in fits]))
        return TR.Plan(fits[k][0], t, fits[k][1], source='reactive'), Wf[k]

    def run(self):
        cfg, M = self.cfg, self.mission
        self._perceived = None                        # SELECT = 'sgm': perceived obstacles
        self._flying = self._kept = self._old = self._fanned = None   # KEEP / FAN: plan being flown
        self._blind, sel_ms = None, []                                   # ms
        self._dp = None                                                  # recurrent policy, fresh state
        A, yaw0, v0 = M.start()
        sim = QuadSim(AGILE, pos=A, yaw=yaw0, seed=cfg.seed)
        sim.v = np.asarray(v0, float)
        ctrl = CascadedController(AGILE, tilt_max_deg=TILT_MAX_DEG)
        ctrl.K['rate_max'] = RATE_MAX
        t_max = cfg.t_max or M.t_max()
        t, t_cam = 0.0, 0.0
        plan, pending, yaw_hold = None, [], None
        psi_last = None
        samples, log = [], []
        n_student = n_frames = 0
        min_clear, crashed, success, failed = np.inf, False, False, False
        while t < t_max:
            st = sim.state()
            p, v, R = st['p'], st['v'], st['R']
            roll, pitch, yaw = euler_from_rot(R)
            if t >= t_cam - 1e-9:
                t_cam += 1.0 / cfg.cam_hz
                n_frames += 1
                M.step(t)
                # initial acceleration: reference acceleration of the current plan; the measured one feeds back
                # vibration and tracking error
                a0 = {'plan': plan.ref(t)[2] if plan is not None else np.zeros(3), 'zero': np.zeros(3),
                      'meas': sim.acc}[self.A0_MODE]
                depth = cam = None
                need_depth = self.label or self.pilot in ('student', 'dagger', 'reactive', 'diffphys') or \
                    (self.record_cam and self.trace is not None)
                if need_depth or M.needs_camera:
                    if self.observer is not None:
                        obs = self.observer(p, yaw, pitch, roll)
                    else:
                        obs = observe_full(M.render_world(), cfg.rig, p, yaw, pitch, roll, self.rng,
                                           rgb=getattr(M, 'needs_rgb', False))
                    gray, depth_full, depth_gt = obs[:3]
                    depth = downsample_depth(depth_full, cfg.net_size)
                    cam = {'gray': gray, 'depth': depth_full, 'depth_gt': depth_gt, 'pose': (p, yaw, pitch, roll),
                           'rig': cfg.rig, 'rgb': obs[3] if len(obs) > 3 else None}
                if (self.SELECT in ('sgm', 'sgm_cost') or self.pilot == 'reactive') and cam is not None:
                    if self._perceived is None:
                        self._perceived = PerceivedObstacles(n_frames=max(1, int(round(cfg.cam_hz))),
                                                             carve=self.CARVE)
                    self._perceived.add(sgm_points(cam['depth'], cfg.rig, (p, yaw, pitch, roll), pool=self.POOL,
                                                   edge=self.EDGE), cam['depth'], cfg.rig, (p, yaw, pitch, roll))
                F = M.frame(t, p, v, yaw, cam)
                omega, ref, self._v_des, yaw_hold = F['omega'], F['ref'], F['v_des'], F.get('yaw')
                lref = F.get('label_ref', ref)                  # label reference
                self._v_ref = F.get('v_ref', self._v_des)     # speed of the privileged reference
                use_student = self.pilot == 'student' or (self.pilot == 'dagger' and self.policy is not None
                                                          and F['d_ref'] <= cfg.xi)
                if self.label and ref is not None and (n_frames - 1) % cfg.label_every == 0:
                    L = self.expert.label(p, lref, v0=v)
                    if len(L['trajs']):
                        samples.append({
                            'depth': np.round(depth * 1000).astype(np.uint16),
                            'v': TR.rotate_to_heading(v, yaw).astype(np.float32),
                            'R': R.ravel().astype(np.float32),
                            'omega': TR.rotate_to_heading(omega, yaw).astype(np.float32),
                            'trajs': TR.to_heading_frame(L['trajs'], p, yaw).astype(np.float32),
                            'costs': L['costs'].astype(np.float32),
                            'esdf': local_esdf(self.field_ng, p, yaw, p[2], cfg), 'z': np.float32(p[2] - self.z_ground)})
                if F.get('hold') and self.pilot in ('student', 'direct', 'blind', 'diffphys'):   # target lost: hover
                    new = TR.Plan(TR.poly5_fit(np.repeat(p[None], 10, 0), p, v, a0), t, 1.0, source='hold')
                elif self.pilot == 'expert':
                    new = self.plan_expert(ref, p, v, a0, t)
                elif self.pilot in ('blind', 'direct'):        # straight to the goal
                    new = self._plan_points(F['blind'], p, v, a0, t, self.pilot)
                elif self.pilot == 'diffphys':                 # acceleration from the recurrent policy
                    if getattr(self, '_dp', None) is None:
                        from tinyuav.nav.diffphys import DiffPhysPilot
                        self._dp = DiffPhysPilot(self.policy)
                        self._dp.q, self._dp.edge, self._dp.persist = self.DP_Q, self.EDGE, self.DP_PERSIST
                    if getattr(M, 'follows', False):          # follow law: direction and speed
                        gdir = np.asarray(omega, float)
                    else:                                      # straight to the goal
                        gd = np.asarray(self.env.goal, float) - p
                        gdir = gd / max(float(np.linalg.norm(gd)), 1e-6)
                    t_d = time.perf_counter()
                    acc = self._dp(cam['depth_gt'] if self.DP_GT else cam['depth'], p, v, R, gdir, yaw, self._v_des)
                    sel_ms.append(1000 * (time.perf_counter() - t_d))
                    if self.DP_AMAX > 0:
                        ah = float(np.hypot(acc[0], acc[1]))
                        if ah > self.DP_AMAX:
                            acc = np.array([acc[0] * self.DP_AMAX / ah, acc[1] * self.DP_AMAX / ah, acc[2]])
                    if self.dp_trace is not None:
                        x = getattr(self._dp, '_last_x', None)
                        self.dp_trace.append({'t': round(float(t), 3), 'p': np.round(p, 3).tolist(), 'v': np.round(v, 3).tolist(),
                                              'att_deg': [round(math.degrees(float(a_)), 1) for a_ in (yaw, pitch, roll)],
                                              'acc': np.round(acc, 3).tolist(),
                                              'near_m': round(1.0 / float(x.max()), 2) if x is not None else None,
                                              'near_rc': [int(i) for i in np.unravel_index(int(x.argmax()), x.shape)] if x is not None else None})
                    coef = np.zeros((3, 6))
                    coef[:, 0], coef[:, 1], coef[:, 2] = p, v, 0.5 * acc
                    new = TR.Plan(coef, t, 1.0, source='diffphys')
                elif self.pilot == 'reactive':
                    self._flying, self._blind = (pending[-1][1] if pending else plan), F['blind']
                    t_d = time.perf_counter()
                    new, _ = self.plan_reactive(p, v, a0, omega, t)
                    sel_ms.append(1000 * (time.perf_counter() - t_d))
                elif use_student:
                    self._flying = pending[-1][1] if pending else plan   # plan the new one would replace
                    self._blind = F['blind']
                    t_d = time.perf_counter()
                    new, W, costs = self.plan_student(depth, p, v, a0, R, yaw, omega, t)
                    sel_ms.append(1000 * (time.perf_counter() - t_d))
                    n_student += 1
                    if self.trace is not None:
                        k_sel, fits = self._last_sel
                        tt = np.linspace(0, 1, 21)
                        clear = [float(self.field(TR.poly_eval(c, tt)).min()) for c, _ in fits]
                        lat = TR.to_heading_frame(W[:, -1], p, yaw)[:, 1]
                        self.trace.append({'t': round(t, 3), 'p': np.round(p, 3).tolist(), 'v': np.round(v, 3).tolist(),
                                           'k': int(k_sel), 'costs': np.round(costs, 4).tolist(),
                                           'clear': np.round(clear, 3).tolist(), 'lat': np.round(lat, 2).tolist(),
                                           'beta': round(float(fits[k_sel][1]), 3),
                                           'W': np.round(W, 2).tolist(), 'omega': np.round(omega, 3).tolist(),
                                           'att': [round(float(yaw), 4), round(float(pitch), 4), round(float(roll), 4)]})
                        if self.SELECT in ('sgm', 'sgm_cost'):
                            self.trace[-1]['clear_seen'] = [round(float(min(c, 99.0)), 3) for c in self._seen]
                        if self._old is not None:
                            o = self._old
                            self.trace[-1].update(none_free=o['none_free'], old_free=o['old_free'],
                                                  old_left=o['old_left'],
                                                  old_clear=round(float(self.field(o['X']).min()), 3))
                        if self._fanned is not None:
                            Xf = np.vstack([p, self._fanned])
                            self.trace[-1].update(fanned=True, W_kept=np.round(self._fanned, 2).tolist(),
                                                  clear_kept=round(float(self.field(Xf).min()), 3),
                                                  seen_kept=round(float(min(self._perceived.clearance(Xf)
                                                                            if self._perceived is not None else 99.0, 99.0)), 3))
                        if self._kept is not None:
                            X, c_old = self._kept
                            self.trace[-1].update(kept=True, W_kept=np.round(X[1:], 2).tolist(),
                                                  clear_kept=round(float(self.field(X).min()), 3),
                                                  seen_kept=round(float(min(c_old, 99.0)), 3))
                        self._trace_stats(cam)
                        if self.trace_depth:
                            self._trace_images(depth, cam, gray)
                else:
                    new = self.plan_ref(ref, p, v, a0, t)
                if self.record_cam and self.trace is not None and not use_student and cam is not None:
                    W1 = TR.poly_eval(new.coef, np.linspace(0.1, 1.0, 10))      # plan sent, 1 s
                    self.trace.append({'t': round(t, 3), 'p': np.round(p, 3).tolist(), 'v': np.round(v, 3).tolist(),
                                       'k': 0, 'costs': [], 'clear': [round(float(self.field(W1).min()), 3)],
                                       'W': [np.round(W1, 2).tolist()], 'beta': round(float(new.beta), 3),
                                       'omega': np.round(omega, 3).tolist(),
                                       'att': [round(float(yaw), 4), round(float(pitch), 4), round(float(roll), 4)],
                                       'pilot': self.pilot})
                    if self.trace_depth:
                        self._trace_images(depth, cam, gray)
                pending.append((t + cfg.latency, new, yaw_hold))   # from the state at capture time
            while pending and pending[0][0] <= t:
                _, plan, yaw_cmd = pending.pop(0)
                yaw_hold = yaw_cmd
            if plan is None:
                p_ref, v_ref, a_ref = p + v * DT_CTRL, v.copy(), np.zeros(3)
            else:
                p_ref, v_ref, a_ref = plan.ref(t)
            v_cmd = v_ref + KP_POS * (p_ref - p)
            if yaw_hold is not None:
                psi = yaw_hold
            elif self.YAW_FWD and np.dot(v_ref[:2], np.asarray(self.env.goal, float)[:2] - sim.p[:2]) <= 0:
                psi = psi_last if psi_last is not None else yaw   # backing away: keep the yaw
            else:
                psi = math.atan2(v_ref[1], v_ref[0]) if np.hypot(v_ref[0], v_ref[1]) > 0.5 else yaw
            psi_last = psi
            w_cmd, _ = ctrl.update(st, v_cmd, a_ref, psi, 0.0, DT_CTRL)
            sim.step(w_cmd, DT_CTRL)
            t += DT_CTRL
            if int(round(t / DT_CTRL)) % 5 == 0:
                ok, bad, c_dyn = M.check(t, sim.p)
                c = min(float(self.field(sim.p[None])[0]), c_dyn)
                min_clear = min(min_clear, c)
                if len(log) == 0 or t - log[-1][0] >= 0.05:
                    log.append((round(t, 3), *np.round(sim.p, 3), *np.round(sim.v, 3), plan.source if plan else '-'))
                if c < AGILE.radius:
                    crashed = True
                    break
                if bad:
                    failed = True
                    break
                if ok:
                    success = True
                    break
        speeds = np.array([np.linalg.norm(r[4:7]) for r in log]) if log else np.zeros(1)
        out = {'success': bool(success), 'crashed': bool(crashed), 'failed': bool(failed),
               'timeout': bool(not success and not crashed and not failed),
               'time_s': round(t, 2), 'min_clearance_m': round(float(min_clear), 3),
               'mean_speed': round(float(speeds.mean()), 2), 'frames': n_frames,
               'student_frac': round(n_student / max(n_frames, 1), 3), 'env': self.env.name,
               'decide_ms': round(float(np.median(sel_ms)), 2) if sel_ms else None,
               'decide_ms_p95': round(float(np.percentile(sel_ms, 95)), 2) if sel_ms else None,
               'v_des': self.cfg.v_des, 'pilot': self.pilot}
        out.update(M.metrics())
        return out, samples, log
