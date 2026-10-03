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

    def __init__(self, env, v_des, gref=None):
        self.env, self.v = env, v_des
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
        s_b, _ = self.line.closest(p, self.s_l)
        return {'omega': omega, 'ref': self.gref.ahead(self.s_g), 'd_ref': d_g, 'v_des': self.v,
                'blind': self.line.ahead(s_b), 'yaw': None}

    def check(self, t, p):
        """→ (success, failure, clearance to dynamic objects)."""
        return self.line.closest(p, self.s_l)[0] >= self.line.length - 0.5, False, np.inf

    def metrics(self):
        return {}


class Rollout:
    A0_MODE = 'plan'             # initial acceleration of each plan: previous plan | zero | measured

    def __init__(self, env, cfg, pilot='gbl', policy=None, label=False, expert_kw=None, gref=None, mission=None):
        self.env, self.cfg, self.pilot, self.policy, self.label = env, cfg, pilot, policy, label
        self.field = Field(env.world, ground=True)
        self.field_ng = Field(env.world, ground=False)
        self.mission = mission or GoalMission(env, cfg.v_des, gref)
        self.gref = getattr(self.mission, 'gref', None)
        self.line = getattr(self.mission, 'line', None)
        self.expert = Expert(env.world, seed=cfg.seed, **(expert_kw or {})) if (label or pilot == 'expert') else None
        self.rng = np.random.default_rng(cfg.seed)
        self._v_des = self._v_ref = cfg.v_des

    # ── plans ─────────────────────────────────────────────────────────────
    def _plan_points(self, P, p, v, a, t, source, v_des=None):
        coef = TR.poly5_fit(P, p, v, a)
        return TR.Plan(coef, t, TR.time_scale(coef, v_des or self._v_des), source=source)

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
        coefs = [TR.poly5_fit(W[k], p, v, a) for k in range(len(W))]
        k = TR.select(coefs, costs)
        return TR.Plan(coefs[k], t, TR.time_scale(coefs[k], self._v_des), source='student'), W, costs

    # ── episode ───────────────────────────────────────────────────────────
    def run(self):
        cfg, M = self.cfg, self.mission
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
                need_depth = self.label or self.pilot in ('student', 'dagger')
                if need_depth or M.needs_camera:
                    obs = observe_full(M.render_world(), cfg.rig, p, yaw, pitch, roll, self.rng,
                                       rgb=getattr(M, 'needs_rgb', False))
                    gray, depth_full, depth_gt = obs[:3]
                    depth = downsample_depth(depth_full, cfg.net_size)
                    cam = {'gray': gray, 'depth': depth_full, 'depth_gt': depth_gt, 'pose': (p, yaw, pitch, roll),
                           'rig': cfg.rig, 'rgb': obs[3] if len(obs) > 3 else None}
                F = M.frame(t, p, v, yaw, cam)
                omega, ref, self._v_des, yaw_hold = F['omega'], F['ref'], F['v_des'], F.get('yaw')
                self._v_ref = F.get('v_ref', self._v_des)     # speed of the privileged reference
                use_student = self.pilot == 'student' or (self.pilot == 'dagger' and self.policy is not None
                                                          and F['d_ref'] <= cfg.xi)
                if self.label and ref is not None and (n_frames - 1) % cfg.label_every == 0:
                    L = self.expert.label(p, ref, v0=v)
                    if len(L['trajs']):
                        samples.append({
                            'depth': np.round(depth * 1000).astype(np.uint16),
                            'v': TR.rotate_to_heading(v, yaw).astype(np.float32),
                            'R': R.ravel().astype(np.float32),
                            'omega': TR.rotate_to_heading(omega, yaw).astype(np.float32),
                            'trajs': TR.to_heading_frame(L['trajs'], p, yaw).astype(np.float32),
                            'costs': L['costs'].astype(np.float32),
                            'esdf': local_esdf(self.field_ng, p, yaw, p[2], cfg), 'z': np.float32(p[2])})
                if F.get('hold') and self.pilot in ('student', 'direct', 'blind'):   # target lost: hover
                    new = TR.Plan(TR.poly5_fit(np.repeat(p[None], 10, 0), p, v, a0), t, 1.0, source='hold')
                elif self.pilot == 'expert':
                    new = self.plan_expert(ref, p, v, a0, t)
                elif self.pilot in ('blind', 'direct'):        # straight toward the goal, no perception
                    new = self._plan_points(F['blind'], p, v, a0, t, self.pilot)
                elif use_student:
                    new, _, _ = self.plan_student(depth, p, v, a0, R, yaw, omega, t)
                    n_student += 1
                else:
                    new = self.plan_ref(ref, p, v, a0, t)
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
