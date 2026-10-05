"""The R18 student outside our simulator: the decision and control logic of the Gazebo node (R21), without ROS.

`AgileCore` receives what a real drone has — a stereo pair, the state estimate (odometry) and the
time — and gives the velocity / yaw command for a velocity-controlled multicopter:

* each stereo pair → SGM depth (the same `stereo.sgm_depth` and parameters as in training) → the
  network input; ω toward the straight start→goal line; the student's M hypotheses → the chosen plan
  (the same `fit_plan` / `choose` as `Rollout`, with its options);
* at the control rate → v_cmd = v_ref + τ·a_ref + k_p·(p_ref − p) and the yaw along the velocity. The
  τ·a_ref lead compensates the first-order response of a velocity controller (Gazebo's
  MulticopterVelocityControl: a ≈ gain · (v_cmd − v), τ = 1 / gain);
* the episode status from the known world (crash = clearance < drone radius, success = end of the line).

The trace has the same fields as `Rollout.trace`, so `r18 video` and `diagnose`-style analysis work on
Gazebo flights. `LagDrone` is a velocity-controlled point mass with that first-order response: it lets
the whole loop be tested offline (`tests/test_gz_agile.py`).
"""
import math

import numpy as np

from nanouav.agile import trajectory as TR
from nanouav.agile.geometry import Field
from nanouav.agile.rollout import choose, downsample_depth, fit_plan, goal_direction, straight_ref
from nanouav.agile.stereo import StereoRig, sgm_depth


class AgileCore:
    def __init__(self, env, policy, v_des, rig=None, net_size=(120, 160), opts=None, tau=0.2, kp=1.0,
                 radius=0.2, t_max=None, keep_depth=False, ramp=1.5):
        self.env, self.policy, self.v = env, policy, float(v_des)
        self.rig, self.net_size = rig or StereoRig(), net_size
        o = opts or {}
        self.proj, self.tscale = o.get('proj', 'scaled'), o.get('tscale', 'chord')
        self.v_cap, self.select = o.get('vcap') or None, o.get('select', 'snap')
        self.tau, self.kp, self.radius = tau, kp, radius
        # a real flight starts from hover, the training rollouts at v_des: the planned speed is at most the
        # current one + ramp (otherwise the first 1 s polynomial from v = 0 peaks at ~2 · v_des)
        self.ramp = ramp
        self.line = straight_ref(env, self.v)
        self.field = Field(env.world, ground=True)
        self.t_max = t_max or self.line.length / self.v + 6.0
        self.plan, self.prev_W, self.s_l = None, None, 0.0
        self.trace, self.log, self.keep_depth = [], [], keep_depth
        self.t0, self.min_clear, self.result = None, np.inf, None

    # ── perception + decision (each stereo pair) ──────────────────────────────
    def frame(self, t, left, right, st):
        """left, right: uint8 images of the rig; st: {'p', 'v', 'R', 'yaw', 'pitch', 'roll'} at capture."""
        if self.t0 is None:
            self.t0 = t
        p, v, R, yaw = np.asarray(st['p'], float), np.asarray(st['v'], float), np.asarray(st['R'], float), st['yaw']
        depth = downsample_depth(sgm_depth(left, right, self.rig), self.net_size)
        omega, self.s_l = goal_direction(self.line, p, self.s_l)
        a0 = self.plan.ref(t)[2] if self.plan is not None else np.zeros(3)
        trajs, costs = self.policy(depth, TR.rotate_to_heading(v, yaw), R.ravel(), TR.rotate_to_heading(omega, yaw))
        W = TR.from_heading_frame(trajs, p, yaw)
        v_eff = min(self.v, max(float(np.linalg.norm(v[:2])), 1.0) + self.ramp) if self.ramp else self.v
        fits = [fit_plan(W[k], p, v, a0, v_eff, self.proj, self.tscale, self.v_cap) for k in range(len(W))]
        k = choose(fits, costs, W, self.prev_W, self.select)
        self.prev_W = W[k]
        self.plan = TR.Plan(fits[k][0], t, fits[k][1], source='student')
        tt = np.linspace(0, 1, 21)
        clear = [float(self.field(TR.poly_eval(c, tt)).min()) for c, _ in fits]
        d = {'t': round(t - self.t0, 3), 'p': np.round(p, 3).tolist(), 'v': np.round(v, 3).tolist(), 'k': int(k),
             'costs': np.round(costs, 4).tolist(), 'clear': np.round(clear, 3).tolist(),
             'lat': np.round(TR.to_heading_frame(W[:, -1], p, yaw)[:, 1], 2).tolist(), 'beta': round(float(fits[k][1]), 3),
             'W': np.round(W, 2).tolist(), 'att': [round(float(yaw), 4), round(float(st.get('pitch', 0.0)), 4),
                                                   round(float(st.get('roll', 0.0)), 4)]}
        if self.keep_depth:
            d['depth'] = np.round(depth * 1000).astype(np.uint16)
        self.trace.append(d)
        return self.plan

    # ── control (each control step) ───────────────────────────────────────────
    def command(self, t, st):
        """→ (v_cmd world (3,), yaw_cmd). Before the first plan: hold the current position."""
        p, v = np.asarray(st['p'], float), np.asarray(st['v'], float)
        if self.plan is None:
            return np.zeros(3), float(st['yaw'])
        p_ref, v_ref, a_ref = self.plan.ref(t)
        v_cmd = v_ref + self.tau * a_ref + self.kp * (p_ref - p)
        yaw_cmd = math.atan2(v_ref[1], v_ref[0]) if math.hypot(v_ref[0], v_ref[1]) > 0.5 else float(st['yaw'])
        return v_cmd, yaw_cmd

    # ── episode ───────────────────────────────────────────────────────────────
    def status(self, t, st):
        """'flying' | 'success' | 'crash' | 'timeout'; logs the state (call at ≥ 20 Hz)."""
        if self.result is not None:
            return self.result['status']
        p, v = np.asarray(st['p'], float), np.asarray(st['v'], float)
        t_rel = t - (self.t0 if self.t0 is not None else t)
        c = float(self.field(p[None])[0])
        self.min_clear = min(self.min_clear, c)
        if not self.log or t_rel - self.log[-1][0] >= 0.05:
            self.log.append((round(t_rel, 3), *np.round(p, 3), *np.round(v, 3), 'student'))
        s, _ = self.line.closest(p, self.s_l)
        status = 'flying'
        if c < self.radius:
            status = 'crash'
        elif s >= self.line.length - 0.5:
            status = 'success'
        elif t_rel > self.t_max:
            status = 'timeout'
        if status != 'flying':
            sp = np.array([np.linalg.norm(r[4:7]) for r in self.log]) if self.log else np.zeros(1)
            self.result = {'status': status, 'success': status == 'success', 'crashed': status == 'crash',
                           'time_s': round(t_rel, 2), 'min_clearance_m': round(self.min_clear, 3),
                           'mean_speed': round(float(sp.mean()), 2), 'max_speed': round(float(sp.max()), 2),
                           'frames': len(self.trace), 'v_des': self.v, 'env': self.env.name}
        return status


class LagDrone:
    """Velocity-controlled point mass: a = clip(gain · (v_cmd − v), a_max), |v| ≤ v_max; yaw rate-limited.
    The response of Gazebo's MulticopterVelocityControl to a Twist, without attitude (level camera)."""

    def __init__(self, p, yaw=0.0, gain=5.0, a_max=12.0, v_max=12.0, yaw_rate=3.0):
        self.p, self.v, self.yaw = np.asarray(p, float).copy(), np.zeros(3), float(yaw)
        self.gain, self.a_max, self.v_max, self.yaw_rate = gain, a_max, v_max, yaw_rate

    def step(self, v_cmd, yaw_cmd, dt):
        a = self.gain * (np.asarray(v_cmd, float) - self.v)
        n = np.linalg.norm(a)
        if n > self.a_max:
            a *= self.a_max / n
        self.v = self.v + a * dt
        n = np.linalg.norm(self.v)
        if n > self.v_max:
            self.v *= self.v_max / n
        self.p = self.p + self.v * dt
        e = (yaw_cmd - self.yaw + math.pi) % (2 * math.pi) - math.pi
        self.yaw += float(np.clip(e, -self.yaw_rate * dt, self.yaw_rate * dt))

    def state(self):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        return {'p': self.p.copy(), 'v': self.v.copy(), 'R': R, 'yaw': self.yaw, 'pitch': 0.0, 'roll': 0.0}


class GlobalRefPolicy:
    """A 'perfect student' for tests: its single hypothesis is the global collision-free path 1 s ahead (in
    the heading frame). With it, any failure of the loop is the plumbing's, not the network's."""

    def __init__(self, env, v, M=3):
        from nanouav.agile.expert import global_plan
        self.g, self.v, self.M, self.s = global_plan(env, v, straight=False), v, M, 0.0
        self._p = self._yaw = None

    def locate(self, p, yaw):
        self._p, self._yaw = np.asarray(p, float), yaw

    def __call__(self, depth, v_h, R, omega_h):
        self.s, _ = self.g.closest(self._p, self.s)
        P = self.g.ahead(self.s)
        H = TR.to_heading_frame(P, self._p, self._yaw)
        return np.repeat(H[None], self.M, 0), np.array([0.1] + [0.5] * (self.M - 1))


def fly_offline(env, policy, v_des, seed=0, cam_hz=15.0, ctrl_hz=100.0, gain=5.0, opts=None, tau=None, render=True):
    """The Gazebo loop with our renderer and a `LagDrone`: stereo pairs at cam_hz, commands at ctrl_hz.
    → (result, core)."""
    from nanouav.agile.stereo import render_stereo
    rng = np.random.default_rng(seed)
    core = AgileCore(env, policy, v_des, opts=opts, tau=1.0 / gain if tau is None else tau)
    A = np.asarray(env.start, float)
    d0 = core.line.pos(1.0) - A
    drone = LagDrone(A, math.atan2(d0[1], d0[0]), gain=gain)
    t, t_cam, dt = 0.0, 0.0, 1.0 / ctrl_hz
    while True:
        st = drone.state()
        if t >= t_cam - 1e-9:
            t_cam += 1.0 / cam_hz
            if hasattr(policy, 'locate'):
                policy.locate(st['p'], st['yaw'])
            if render:
                gl, gr, _ = render_stereo(env.world, core.rig, st['p'], st['yaw'], rng=rng)
            else:
                gl = gr = np.zeros(core.rig.size, np.uint8)
            core.frame(t, gl, gr, st)
        v_cmd, yaw_cmd = core.command(t, st)
        drone.step(v_cmd, yaw_cmd, dt)
        t += dt
        if core.status(t, drone.state()) != 'flying':
            return core.result, core
