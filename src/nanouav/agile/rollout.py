"""Closed loop for the recreation of Loquercio et al. 2021: 6-DoF quadrotor (``platform.AGILE``), stereo
camera + SGM, and one of these pilots:

* ``gbl``     — track the global collision-free trajectory τ_gbl (what the paper's MPC does while the
                student is off-track during DAgger);
* ``blind``   — track the straight start→goal reference without perception (paper baseline);
* ``expert``  — the privileged M-H expert replanning in receding horizon (upper bound; not real time);
* ``student`` — the learned policy (depth, velocity, attitude, goal direction → 3 trajectories + costs);
* ``dagger``  — student if the drone is within ξ of τ_gbl, otherwise τ_gbl; every frame is labeled
                by the expert (Methods C).

What to fly toward comes from a *mission*: `GoalMission` (reach a goal 40 m ahead, the paper's task) or
`nanouav.follow.mission.FollowMission` (follow a person). The mission gives, per camera frame, the goal
direction ω for the student, the reference for the expert / τ_gbl pilots, the desired speed and,
optionally, a yaw to hold; it also moves dynamic objects and decides success.

Every plan is executed the same way: projection on an order-5 polynomial from the current state (Eq. 8),
time scaling to v_des and tracking by the cascaded controller with position feedback (the paper uses an
MPC [Falanga 2018]; here the controller tracks p_ref, v_ref and a_ref of the polynomial).
"""
import math
from dataclasses import dataclass, field

import numpy as np

from nanouav.agile import trajectory as TR
from nanouav.agile.expert import DT, Expert, collision_cost, global_plan, GlobalRef
from nanouav.agile.geometry import Field
from nanouav.agile.platform import AGILE, RATE_MAX, TILT_MAX_DEG
from nanouav.agile.stereo import StereoRig, observe_full
from nanouav.control.controller import CascadedController
from nanouav.control.quad_sim import QuadSim, euler_from_rot

DT_CTRL = 0.002
KP_POS = 3.0


@dataclass
class Config:
    v_des: float = 5.0
    cam_hz: float = 15.0              # paper: ~24 Hz decision rate on the TX2
    latency: float = 0.05             # s, capture → plan applied (paper onboard: 41.6 ms)
    t_max: float = 0.0                # 0 → length / v_des + 6 s
    xi: float = 0.0                   # DAgger threshold (m)
    net_size: tuple = (120, 160)      # stored / network depth resolution (H, W); paper: 480 × 640
    esdf_res: float = 0.25            # local 2D distance map for the collision-cost target
    esdf_box: tuple = (-2.0, 14.0, -8.0, 8.0)
    rig: StereoRig = field(default_factory=StereoRig)
    seed: int = 0
    label_every: int = 1              # DAgger: expert label every n camera frames (long follow episodes)
    # where the student's direction input ω points: 'line' (the straight start→goal reference, the only one
    # that exists at test time) or 'gbl' (τ_gbl, the reference the expert's labels follow). R23: with 'line'
    # during training the labels are not a function of what the student sees (they follow τ_gbl, a global
    # plan); with 'gbl' they are, but the student then learns to trust an ω that at test points through
    # obstacles. 'goal': straight to the goal point, identical in training and test.
    omega_ref: str = 'line'
    # reference of the expert's labels: 'gbl' (τ_gbl, planned once from the start: R18) or 'ctg' (replanned
    # from the current position down the cost-to-go to the goal, keeping to the line: `expert.CostToGo`, R23).
    # With 'ctg' the DAgger takeover flies the replanned reference too, and ξ is measured from the cost-to-go
    # path planned at the start instead of τ_gbl.
    label_ref: str = 'gbl'


def straight_ref(env, v):
    A, B = np.asarray(env.start, float), np.asarray(env.goal, float)
    n = int(np.linalg.norm(B - A) / 0.1) + 1
    return GlobalRef(A + np.linspace(0, 1, n)[:, None] * (B - A), v)


def goal_direction(line, p, s_hint=None):
    """Unit vector toward the point 1 s ahead of the closest point of the straight start→goal line."""
    s, _ = line.closest(p, s_hint)
    q = line.pos(min(s + line.v * 1.0, line.length))
    d = q - p
    n = np.linalg.norm(d)
    return (d / n if n > 1e-6 else np.array([1.0, 0, 0])), s


def downsample_depth(d, size):
    """Nearest-neighbor resize keeping holes (0 = invalid)."""
    H, W = d.shape
    h, w = size
    ys = (np.arange(h) + 0.5) * H / h
    xs = (np.arange(w) + 0.5) * W / w
    return d[ys.astype(int)][:, xs.astype(int)]


def local_esdf(field, p, yaw, z, cfg):
    """2D distance (no ground) on a grid in the heading frame, min over z ± 0.4 m. → (ny, nx) float16."""
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
    """Paper task: fly to a goal 40 m ahead. ω points to the straight start→goal line 1 s ahead; the expert
    and the τ_gbl pilot use the global collision-free trajectory."""
    needs_camera = False

    def __init__(self, env, v_des, gref=None, omega_ref='line', label_ref='gbl'):
        self.env, self.v, self.omega_ref, self.label_ref = env, v_des, omega_ref, label_ref
        self._ctg = None
        if label_ref == 'ctg' and gref is None:
            # R23: the nominal route is the cost-to-go path from the start (it keeps to the line); DAgger's ξ
            # is measured from it, and the expert, the takeover and the labels use the path replanned from p
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
        if self.omega_ref == 'gbl':                      # toward τ_gbl 1 s ahead of its closest state (R23)
            q = self.gref.pos(min(self.s_g + self.v * 1.0, self.gref.length)) - p
            omega = q / max(float(np.linalg.norm(q)), 1e-6)
        elif self.omega_ref == 'goal':                   # straight to the goal point (train and test)
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
            from nanouav.agile.expert import W_LINE, CostToGo
            self._ctg = CostToGo(self.env, w_line=W_LINE)
        return self._ctg

    def ctg_ref(self, p):
        """Reference replanned from p down the cost-to-go (built once per environment), 1 s of it."""
        r = self._cost_to_go().ref_from(p, self.v)
        return r.ahead(0.0) if r is not None else self.gref.ahead(self.s_g)

    def check(self, t, p):
        """→ (success, failure, clearance to dynamic objects)."""
        return self.line.closest(p, self.s_l)[0] >= self.line.length - 0.5, False, np.inf

    def metrics(self):
        return {}


def fit_plan(P, p, v, a, v_des, proj='scaled', tscale='chord', v_cap=None):
    """Projection of predicted points onto the polynomial that is flown (Eq. 8) and its time scale β
    (see the options of `Rollout`). → (coef, β)."""
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
    """Among the plans with clearance ≥ radius + margin: the least snap (costs=None) or the paper's rule on
    the predicted costs (c*/c_k ≥ 0.95, then the least snap). None free: the one with the most clearance."""
    ok = np.where(np.asarray(clear) >= AGILE.radius + margin)[0]
    if not len(ok):
        return int(np.argmax(clear))
    if costs is None:
        return int(ok[np.argmin([TR.snap_cost(fits[k][0]) for k in ok])])
    return int(ok[TR.select([fits[k][0] for k in ok], [costs[k] for k in ok])])


def oracle_clearance(fits, field, margin=0.15, lookahead=0.0):
    """True clearance of each flown polynomial (privileged), with its continuation if lookahead > 0."""
    tt = np.linspace(0, 1, 21)
    clear = np.array([float(field(TR.poly_eval(c, tt)).min()) for c, _ in fits])
    return with_lookahead(clear, fits, lambda X: float(field(X).min()), lookahead, margin)


def oracle_index(fits, field, margin=0.15, costs=None, lookahead=0.0):
    """Privileged choice (diagnosis only) by the true clearance of each flown polynomial (see `_among_free`).
    With the student's own hypotheses it gives the ceiling of a perfect cost head and selection."""
    return _among_free(fits, oracle_clearance(fits, field, margin, lookahead), margin, costs)


def remainder(plan, t, n=11):
    """The part of `plan` still ahead at time t: n points from there to its end, and its duration (s)."""
    tau0 = float(np.clip((t - plan.t0) * plan.beta, 0.0, plan.horizon))
    return TR.poly_eval(plan.coef, np.linspace(tau0, plan.horizon, n)), (plan.horizon - tau0) / plan.beta


def keep_previous(clear_new, clear_old, left, margin=0.15, min_left=0.3):
    """--keep: go on flying the plan being flown when none of the new hypotheses is free (clearance ≥ radius
    + margin) but what is left of it (≥ min_left s) still is. Never swap a free plan for one that is not."""
    free = AGILE.radius + margin
    return bool(left >= min_left and clear_old >= free and float(np.max(clear_new)) < free)


def sgm_points(depth, rig, pose, stride=4):
    """The drone's own SGM depth (m, 0 = none; full resolution of `rig`) → world points, from its pose
    estimate (p, yaw, pitch, roll)."""
    from nanouav.follow.estimate import camera_pose
    H, W = depth.shape
    dep = np.asarray(depth, np.float32)[::stride, ::stride]
    v, u = np.nonzero(dep > 0)
    X = dep[v, u]
    u, v = u * stride * rig.width / W, v * stride * rig.height / H
    O, Rc = camera_pose(rig, pose)
    dirs = np.stack([np.ones_like(X), -(u + 0.5 - rig.width / 2) / rig.focal, -(v + 0.5 - rig.height / 2) / rig.focal], 1)
    return O + (dirs * X[:, None]) @ Rc.T


class PerceivedObstacles:
    """Obstacles as the drone itself perceives them: cells of `cell` m hit by its SGM in ≥ `min_hits` of the
    last `n_frames` frames (stereo speckle does not repeat). Nothing privileged: own depth, own pose."""

    def __init__(self, n_frames=15, cell=0.15, min_hits=2):
        from collections import deque
        self.hist, self.cell, self.min_hits = deque(maxlen=n_frames), cell, min_hits
        self._cells = None

    def add(self, P):
        self.hist.append(np.unique(np.floor(np.asarray(P, float) / self.cell).astype(np.int64), axis=0)
                         if len(P) else np.zeros((0, 3), np.int64))
        self._cells = None

    def cells(self):
        if self._cells is None:
            A = [h for h in self.hist if len(h)]
            if not A:
                self._cells = np.zeros((0, 3))
            else:
                c, n = np.unique(np.vstack(A), axis=0, return_counts=True)
                self._cells = (c[n >= min(self.min_hits, len(self.hist))] + 0.5) * self.cell
        return self._cells

    def clearance(self, X):
        """Distance from the points X (M, 3) to the nearest perceived obstacle (∞ if none)."""
        X = np.asarray(X, float)
        C = self.cells()
        C = C[((C >= X.min(0) - 1.5) & (C <= X.max(0) + 1.5)).all(1)] if len(C) else C
        if not len(C):
            return np.inf
        d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        return float(np.sqrt(d2.min())) - 0.5 * self.cell


def extension(fit, lookahead, n=8):
    """Points of the plan continued beyond its end at its final velocity for `lookahead` s (real time)."""
    coef, beta = fit
    p1 = TR.poly_eval(coef, [1.0])[0]
    v1 = TR.poly_eval(coef, [1.0], 1)[0] * beta
    s = np.linspace(lookahead / n, lookahead, n)
    return p1 + s[:, None] * v1


def with_lookahead(clear, fits, clear_of, lookahead, margin=0.15):
    """Effective clearance of each plan when its continuation must also be free (≥ the drone's radius, without
    the margin): min(plan clearance, continuation clearance + margin). lookahead = 0: unchanged."""
    if not lookahead:
        return np.asarray(clear, float)
    ext = np.array([clear_of(extension(f, lookahead)) for f in fits])
    return np.minimum(np.asarray(clear, float), ext + margin)


def perceived_clearance(fits, obstacles):
    """Clearance of each flown polynomial (1 s) to the obstacles the drone perceives."""
    tt = np.linspace(0, 1, 21)
    return np.array([obstacles.clearance(TR.poly_eval(c, tt)) for c, _ in fits])


def perceived_index(fits, costs, obstacles, margin=0.15, clear=None):
    """Choice with the drone's own perception, by the clearance of each flown polynomial to the obstacles in
    its SGM (see `_among_free`; costs=None: the least snap among the free ones, like the oracle)."""
    clear = perceived_clearance(fits, obstacles) if clear is None else clear
    return _among_free(fits, clear, margin, costs)


def choose(fits, costs, W, prev=None, mode='snap'):
    """Index of the hypothesis to fly (see `Rollout.SELECT`; 'oracle' is chosen in `Rollout.plan_student`)."""
    if mode == 'consistent' and prev is not None:
        c = np.asarray(costs, float) + 1e-3
        ok = np.where(c.min() / c >= 0.8)[0]
        return int(ok[np.argmin([np.linalg.norm(W[j][3:] - prev[3:], axis=1).mean() for j in ok])])
    return TR.select([c for c, _ in fits], costs)


class Rollout:
    A0_MODE = 'plan'             # initial acceleration of each plan: previous plan | zero | measured
    # Projection of the predicted points onto the polynomial that is flown (Eq. 8 + time scale β):
    # 'scaled' fits the polynomial with v/β and a/β², so that after the time scaling the plan starts at the
    # current velocity and acceleration; 'raw' (≤ 0.16.1) fitted with v and a, so the commanded velocity
    # jumped to β·v at every new plan. The student's predictions regress toward the mean training speed
    # (at 3 m/s it predicts ~4.3 m per second: β ≈ 0.7, a 30 % velocity drop 15 times per second).
    PROJ = 'scaled'
    # Time scale β = v_des / v_μ. 'chord' (≤ 0.17.1): v_μ from the straight distance to the last point;
    # 'arc': from the length of the path through the points. With the chord, a curved or hooked
    # prediction (short chord, long path) is flown much faster than v_des: crashes of the 850-environment
    # student at v_des = 10 happened at 12–19 m/s.
    TSCALE = 'chord'
    V_CAP = None                 # e.g. 1.3: lower β until the plan's peak speed ≤ V_CAP · v_des (or the current speed)
    # Choice among the M hypotheses: 'snap' (paper: c*/c_k ≥ 0.95, then the least snap), 'consistent',
    # 'oracle' (privileged, diagnosis only: the least snap among the plans whose flown polynomial is truly free),
    # 'sgm' (the same with the obstacles of the drone's own SGM of the last second: not privileged), and
    # 'oracle_cost' / 'sgm_cost' (the paper's cost rule instead of the least snap among the free ones)
    # (c*/c_k ≥ 0.8, then the one nearest to the previous choice: no switching sides around a tree unless
    # the predicted cost says so)
    SELECT = 'snap'
    # 'sgm' / 'oracle': a plan is free only if also its continuation at its final velocity for LOOKAHEAD s is
    # (≥ the drone's radius): against plans that are free for their 1 s but end pointing into a trunk
    LOOKAHEAD = 0.0
    # 'sgm' / 'oracle': when none of the M new hypotheses is free, go on flying the current plan if what is
    # left of it (≥ 0.3 s) still is, instead of the new one with the most clearance (`keep_previous`)
    KEEP = False

    def __init__(self, env, cfg, pilot='gbl', policy=None, label=False, expert_kw=None, gref=None, mission=None):
        self.env, self.cfg, self.pilot, self.policy, self.label = env, cfg, pilot, policy, label
        self.field = Field(env.world, ground=True)
        self.field_ng = Field(env.world, ground=False)
        self.mission = mission or GoalMission(env, cfg.v_des, gref, getattr(cfg, 'omega_ref', 'line'),
                                              getattr(cfg, 'label_ref', 'gbl'))
        self.gref = getattr(self.mission, 'gref', None)
        self.line = getattr(self.mission, 'line', None)
        self.z_ground = float(getattr(env.world, 'floor', 0.0))   # 'z' of the samples: height above the floor
        kw = dict(expert_kw or {})
        kw.setdefault('z_min', self.z_ground + 0.4)                 # the expert's lowest point, above the floor
        self.expert = Expert(env.world, seed=cfg.seed, **kw) if (label or pilot == 'expert') else None
        self.rng = np.random.default_rng(cfg.seed)
        self._v_des = self._v_ref = cfg.v_des
        self.trace = None            # a list → per student decision: hypotheses, costs, choice, true clearance
        self.trace_depth = False     # also keep the network's depth input in the trace (videos)
        # an external renderer instead of ours: f(p, yaw, pitch, roll) → (gray, SGM depth, true depth[, rgb]),
        # e.g. Flightmare (nanouav.flightmare.run.FlightmareObserver)
        self.observer = None
        self.record_cam = False      # with a trace: also observe and record frames for privileged pilots (videos)
        # a list → labelled states are stored without the expert label (computed later, in parallel:
        # nanouav.flightmare.dagger) — the expert is the slow part when the renderer is a single Unity player
        self.defer = None

    def _trace_stats(self, cam):
        """SGM quality of the frame behind a decision (diagnosis): valid fraction (all, left and right halves)
        and median relative error against the true depth where both exist."""
        sgm, gt = np.asarray(cam['depth']), np.asarray(cam['depth_gt'])
        v = sgm > 0
        h = sgm.shape[1] // 2
        d = self.trace[-1]
        d['sgm_valid'] = [round(float(v.mean()), 3), round(float(v[:, :h].mean()), 3), round(float(v[:, h:].mean()), 3)]
        m = v & (gt > 0)
        d['sgm_err'] = round(float(np.median(np.abs(sgm[m] - gt[m]) / gt[m])), 3) if m.sum() > 50 else None

    def _trace_images(self, depth, cam, gray):
        """Images of a trace entry (videos): the network input, the full-resolution SGM and the camera image."""
        d = self.trace[-1]
        d['depth'] = np.round(depth * 1000).astype(np.uint16)
        d['sgm'] = np.round(np.asarray(cam['depth']) * 1000).astype(np.uint16)
        d['img'] = cam['rgb'] if cam.get('rgb') is not None else gray
        if self.observer is not None:
            d['gray'] = gray                                   # the external renderer's left image

    # ── plans ─────────────────────────────────────────────────────────────
    def _fit(self, P, p, v, a, v_des):
        """→ (coef, β) of the plan through the points P at speed v_des (see PROJ, TSCALE, V_CAP)."""
        return fit_plan(P, p, v, a, v_des, self.PROJ, self.TSCALE, self.V_CAP)

    def _plan_points(self, P, p, v, a, t, source, v_des=None):
        coef, beta = self._fit(P, p, v, a, v_des or self._v_des)
        return TR.Plan(coef, t, beta, source=source)

    def plan_ref(self, ref, p, v, a, t, source='gbl'):
        return self._plan_points(ref, p, v, a, t, source, self._v_ref)

    def plan_expert(self, ref, p, v, a, t):
        """Best expert label whose *executed* polynomial (Eq. 8 projection from the current p, v, a) stays
        collision-free; the projection can cut ~0.1–0.2 m into an obstacle the label just clears."""
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
        if self.SELECT in ('oracle', 'oracle_cost'):        # privileged: true clearance (diagnosis only)
            eff = oracle_clearance(fits, self.field, lookahead=self.LOOKAHEAD)
            k = _among_free(fits, eff, costs=costs if self.SELECT == 'oracle_cost' else None)
            clear_of = lambda X: float(self.field(X).min())
        elif self.SELECT in ('sgm', 'sgm_cost'):             # own perception: clearance in its own SGM
            self._seen = perceived_clearance(fits, self._perceived)
            eff = with_lookahead(self._seen, fits, self._perceived.clearance, self.LOOKAHEAD)
            k = perceived_index(fits, costs if self.SELECT == 'sgm_cost' else None, self._perceived, clear=eff)
            clear_of = self._perceived.clearance
        else:
            k = choose(fits, costs, W, getattr(self, '_prev_W', None), self.SELECT)
        self._prev_W = W[k]
        self._last_sel = (k, fits)
        self._kept, self._old = None, None
        old = getattr(self, '_flying', None)
        if eff is not None and old is not None and old.source == 'student' and (self.KEEP or self.trace is not None):
            X, left = remainder(old, t + self.cfg.latency)   # from when the new plan would take over
            c_old = clear_of(X)
            if self.LOOKAHEAD:
                c_old = min(c_old, clear_of(extension((old.coef, old.beta), self.LOOKAHEAD)) + 0.15)
            would = keep_previous(eff, c_old, left)
            # diagnosis, with or without --keep: none of the new plans free while what is left of the one
            # being flown still is (the case --keep acts on)
            self._old = {'none_free': bool(float(np.max(eff)) < AGILE.radius + 0.15), 'old_free': would,
                         'old_left': round(float(left), 3), 'X': X}
            if self.KEEP and would:
                self._kept = (X, c_old)
                return old, W, costs
        return TR.Plan(fits[k][0], t, fits[k][1], source='student'), W, costs

    # ── episode ───────────────────────────────────────────────────────────
    def run(self):
        cfg, M = self.cfg, self.mission
        self._perceived = None                        # SELECT = 'sgm': the episode's own perceived obstacles
        self._flying = self._kept = self._old = None  # KEEP: the plan being flown, whether it was kept
        A, yaw0, v0 = M.start()
        sim = QuadSim(AGILE, pos=A, yaw=yaw0, seed=cfg.seed)
        sim.v = np.asarray(v0, float)
        ctrl = CascadedController(AGILE, tilt_max_deg=TILT_MAX_DEG)
        ctrl.K['rate_max'] = RATE_MAX
        t_max = cfg.t_max or M.t_max()
        t, t_cam = 0.0, 0.0
        plan, pending, yaw_hold = None, [], None
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
                # initial acceleration of the new plan: the reference acceleration of the plan being flown.
                # The measured one closes a loop through vibration and tracking error (Eq. 8 uses the
                # state estimate; with an MPC that is smooth, with this controller it is not).
                a0 = {'plan': plan.ref(t)[2] if plan is not None else np.zeros(3), 'zero': np.zeros(3),
                      'meas': sim.acc}[self.A0_MODE]
                depth = cam = None
                need_depth = self.label or self.pilot in ('student', 'dagger') or \
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
                if self.SELECT in ('sgm', 'sgm_cost') and cam is not None:   # its own SGM, every camera frame
                    if self._perceived is None:
                        self._perceived = PerceivedObstacles(n_frames=max(1, int(round(cfg.cam_hz))))
                    self._perceived.add(sgm_points(cam['depth'], cfg.rig, (p, yaw, pitch, roll)))
                F = M.frame(t, p, v, yaw, cam)
                omega, ref, self._v_des, yaw_hold = F['omega'], F['ref'], F['v_des'], F.get('yaw')
                lref = F.get('label_ref', ref)                  # the expert's labels (R23: may differ from ref)
                self._v_ref = F.get('v_ref', self._v_des)     # speed of the privileged reference
                use_student = self.pilot == 'student' or (self.pilot == 'dagger' and self.policy is not None
                                                          and F['d_ref'] <= cfg.xi)
                if self.label and ref is not None and (n_frames - 1) % cfg.label_every == 0 and self.defer is not None:
                    self.defer.append({
                        'depth': np.round(depth * 1000).astype(np.uint16),
                        'v': TR.rotate_to_heading(v, yaw).astype(np.float32),
                        'R': R.ravel().astype(np.float32),
                        'omega': TR.rotate_to_heading(omega, yaw).astype(np.float32),
                        'z': np.float32(p[2] - self.z_ground),
                        '_p': p.copy(), '_ref': np.asarray(lref, float).copy(), '_v': v.copy(), '_yaw': float(yaw)})
                elif self.label and ref is not None and (n_frames - 1) % cfg.label_every == 0:
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
                if F.get('hold') and self.pilot in ('student', 'direct', 'blind'):   # target lost: hover
                    new = TR.Plan(TR.poly5_fit(np.repeat(p[None], 10, 0), p, v, a0), t, 1.0, source='hold')
                elif self.pilot == 'expert':
                    new = self.plan_expert(ref, p, v, a0, t)
                elif self.pilot in ('blind', 'direct'):        # straight toward the goal, no perception
                    new = self._plan_points(F['blind'], p, v, a0, t, self.pilot)
                elif use_student:
                    self._flying = pending[-1][1] if pending else plan   # the plan the new one would replace
                    new, W, costs = self.plan_student(depth, p, v, a0, R, yaw, omega, t)
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
                        if self.SELECT in ('sgm', 'sgm_cost'):   # what its own SGM said about each plan
                            self.trace[-1]['clear_seen'] = [round(float(min(c, 99.0)), 3) for c in self._seen]
                        if self._old is not None:             # the plan it would replace (see KEEP)
                            o = self._old
                            self.trace[-1].update(none_free=o['none_free'], old_free=o['old_free'],
                                                  old_left=o['old_left'],
                                                  old_clear=round(float(self.field(o['X']).min()), 3))
                        if self._kept is not None:            # --keep: it flew on the previous plan
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
                    W1 = TR.poly_eval(new.coef, np.linspace(0.1, 1.0, 10))      # the plan being sent, 1 s
                    self.trace.append({'t': round(t, 3), 'p': np.round(p, 3).tolist(), 'v': np.round(v, 3).tolist(),
                                       'k': 0, 'costs': [], 'clear': [round(float(self.field(W1).min()), 3)],
                                       'W': [np.round(W1, 2).tolist()], 'beta': round(float(new.beta), 3),
                                       'omega': np.round(omega, 3).tolist(),
                                       'att': [round(float(yaw), 4), round(float(pitch), 4), round(float(roll), 4)],
                                       'pilot': self.pilot})
                    if self.trace_depth:
                        self._trace_images(depth, cam, gray)
                pending.append((t + cfg.latency, new, yaw_hold))   # computed from the state at capture time t
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
            else:
                psi = math.atan2(v_ref[1], v_ref[0]) if np.hypot(v_ref[0], v_ref[1]) > 0.5 else yaw
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
               'v_des': self.cfg.v_des, 'pilot': self.pilot}
        out.update(M.metrics())
        return out, samples, log
