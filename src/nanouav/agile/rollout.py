"""Closed loop for the recreation of Loquercio et al. 2021: 6-DoF quadrotor (``platform.AGILE``), stereo
camera + SGM, and one of these pilots:

* ``gbl``     — track the global collision-free trajectory τ_gbl (what the paper's MPC does while the
                student is off-track during DAgger);
* ``blind``   — track the straight start→goal reference without perception (paper baseline);
* ``expert``  — the privileged M-H expert replanning in receding horizon (upper bound; not real time);
* ``student`` — the learned policy (depth, velocity, attitude, goal direction → 3 trajectories + costs);
* ``dagger``  — student if the drone is within ξ of τ_gbl, otherwise τ_gbl; every frame is labeled
                by the expert (Methods C).

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
from nanouav.agile.stereo import StereoRig, observe
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


class Rollout:
    def __init__(self, env, cfg, pilot='gbl', policy=None, label=False, expert_kw=None, gref=None):
        self.env, self.cfg, self.pilot, self.policy, self.label = env, cfg, pilot, policy, label
        self.field = Field(env.world, ground=True)
        self.field_ng = Field(env.world, ground=False)
        self.gref = gref or global_plan(env, cfg.v_des, straight=False)   # collision-free (expert, DAgger)
        self.line = straight_ref(env, cfg.v_des)
        self.expert = Expert(env.world, seed=cfg.seed, **(expert_kw or {})) if (label or pilot == 'expert') else None
        self.rng = np.random.default_rng(cfg.seed)

    # ── plans ─────────────────────────────────────────────────────────────
    def _plan_points(self, P, p, v, a, t, source):
        coef = TR.poly5_fit(P, p, v, a)
        return TR.Plan(coef, t, TR.time_scale(coef, self.cfg.v_des), source=source)

    def plan_gbl(self, p, v, a, t, s_hint):
        s, _ = self.gref.closest(p, s_hint)
        return self._plan_points(self.gref.ahead(s), p, v, a, t, 'gbl'), s

    def plan_expert(self, p, v, a, t, s_g):
        """Best expert label whose *executed* polynomial (Eq. 8 projection from the current p, v, a) stays
        collision-free; the projection can cut ~0.1–0.2 m into an obstacle the label just clears."""
        L = self.expert.label(p, self.gref.ahead(s_g), k=3, v0=v)
        if not len(L['trajs']):
            return self.plan_gbl(p, v, a, t, s_g)[0]
        best, best_d = None, -np.inf
        for P in L['trajs']:
            pl = self._plan_points(P, p, v, a, t, 'expert')
            d = float(self.field(TR.poly_eval(pl.coef, np.linspace(0, 1, 21))).min())
            if d > AGILE.radius + 0.05:
                return pl
            if d > best_d:
                best, best_d = pl, d
        return best

    def plan_student(self, depth, p, v, a, R, yaw, omega, t):
        trajs, costs = self.policy(depth, TR.rotate_to_heading(v, yaw), R.ravel(), TR.rotate_to_heading(omega, yaw))
        W = TR.from_heading_frame(trajs, p, yaw)
        coefs = [TR.poly5_fit(W[k], p, v, a) for k in range(len(W))]
        k = TR.select(coefs, costs)
        return TR.Plan(coefs[k], t, TR.time_scale(coefs[k], self.cfg.v_des), source='student'), W, costs

    # ── episode ───────────────────────────────────────────────────────────
    def run(self):
        cfg, env = self.cfg, self.env
        A = np.asarray(env.start, float)
        d0 = self.gref.pos(0.5) - A
        yaw0 = math.atan2(d0[1], d0[0])
        sim = QuadSim(AGILE, pos=A, yaw=yaw0, seed=cfg.seed)
        sim.v = np.array([math.cos(yaw0), math.sin(yaw0), 0.0]) * cfg.v_des
        ctrl = CascadedController(AGILE, tilt_max_deg=TILT_MAX_DEG)
        ctrl.K['rate_max'] = RATE_MAX
        t_max = cfg.t_max or self.line.length / cfg.v_des + 6.0
        t, t_cam = 0.0, 0.0
        s_g = s_l = 0.0
        plan, pending = None, []
        samples, log = [], []
        n_student = n_frames = 0
        min_clear, crashed, success = np.inf, False, False
        while t < t_max:
            st = sim.state()
            p, v, R = st['p'], st['v'], st['R']
            roll, pitch, yaw = euler_from_rot(R)
            if t >= t_cam - 1e-9:
                t_cam += 1.0 / cfg.cam_hz
                n_frames += 1
                omega, s_l = goal_direction(self.line, p, s_l)
                s_g, d_g = self.gref.closest(p, s_g)
                depth = None
                need_depth = self.label or self.pilot in ('student', 'dagger')
                if need_depth:
                    depth_full, _ = observe(env.world, cfg.rig, p, yaw, pitch, roll, self.rng)
                    depth = downsample_depth(depth_full, cfg.net_size)
                use_student = self.pilot == 'student' or (self.pilot == 'dagger' and self.policy is not None
                                                          and d_g <= cfg.xi)
                if self.label:
                    L = self.expert.label(p, self.gref.ahead(s_g), v0=v)
                    if len(L['trajs']):
                        samples.append({
                            'depth': np.round(depth * 1000).astype(np.uint16),
                            'v': TR.rotate_to_heading(v, yaw).astype(np.float32),
                            'R': R.ravel().astype(np.float32),
                            'omega': TR.rotate_to_heading(omega, yaw).astype(np.float32),
                            'trajs': TR.to_heading_frame(L['trajs'], p, yaw).astype(np.float32),
                            'costs': L['costs'].astype(np.float32),
                            'esdf': local_esdf(self.field_ng, p, yaw, p[2], cfg), 'z': np.float32(p[2])})
                if self.pilot == 'expert':
                    new = self.plan_expert(p, v, sim.acc, t, s_g)
                elif self.pilot == 'blind':                   # baseline: straight reference, no perception
                    s_b, _ = self.line.closest(p, s_l)
                    new = self._plan_points(self.line.ahead(s_b), p, v, sim.acc, t, 'blind')
                elif use_student:
                    new, _, _ = self.plan_student(depth, p, v, sim.acc, R, yaw, omega, t)
                    n_student += 1
                else:
                    new, _ = self.plan_gbl(p, v, sim.acc, t, s_g)
                pending.append((t + cfg.latency, new))      # computed from the state at capture time t
            while pending and pending[0][0] <= t:
                plan = pending.pop(0)[1]
            if plan is None:
                p_ref, v_ref, a_ref = p + v * DT_CTRL, v.copy(), np.zeros(3)
            else:
                p_ref, v_ref, a_ref = plan.ref(t)
            v_cmd = v_ref + KP_POS * (p_ref - p)
            psi = math.atan2(v_ref[1], v_ref[0]) if np.hypot(v_ref[0], v_ref[1]) > 0.5 else yaw
            w_cmd, _ = ctrl.update(st, v_cmd, a_ref, psi, 0.0, DT_CTRL)
            sim.step(w_cmd, DT_CTRL)
            t += DT_CTRL
            if int(round(t / DT_CTRL)) % 5 == 0:
                c = float(self.field(sim.p[None])[0])
                min_clear = min(min_clear, c)
                if len(log) == 0 or t - log[-1][0] >= 0.05:
                    log.append((round(t, 3), *np.round(sim.p, 3), *np.round(sim.v, 3), plan.source if plan else '-'))
                if c < AGILE.radius:
                    crashed = True
                    break
                if self.line.closest(sim.p, s_l)[0] >= self.line.length - 0.5:
                    success = True
                    break
        speeds = np.array([np.linalg.norm(r[4:7]) for r in log]) if log else np.zeros(1)
        return {'success': bool(success), 'crashed': bool(crashed), 'timeout': bool(not success and not crashed),
                'time_s': round(t, 2), 'min_clearance_m': round(float(min_clear), 3),
                'mean_speed': round(float(speeds.mean()), 2), 'frames': n_frames,
                'student_frac': round(n_student / max(n_frames, 1), 3), 'env': env.name,
                'v_des': self.cfg.v_des, 'pilot': self.pilot}, samples, log
