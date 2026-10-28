"""Follow-a-person mission for `nanouav.agile.rollout.Rollout`.

Each camera frame: detector → tracker → target lock → 3D position from the stereo depth → world filter →
follow goal = the point at `standoff` m from the target on the drone side, at the follow altitude. For the
student, ω points to that goal and v_des = target speed + a gain on the distance error (capped); the yaw is
held toward the target so it stays in view. If the target is lost the drone keeps flying toward the
predicted position for `t_coast` s, then hovers and turns toward it; after `t_fail` s it is a failure.

The privileged pilots (expert, τ_gbl) and DAgger's expert labels use the true target: the drone's
reference is the person's own path lifted to the follow altitude (people walk collision-free paths, so
their trail is a collision-free route), from the drone's closest point, at the speed that closes the gap
to the point `standoff` m behind the person.

Success: the target walks its whole path, the drone does not crash, and the lock was on the true target
in ≥ 80 % of the frames once acquired.
"""
import math

import numpy as np

from nanouav.agile.expert import GlobalRef
from nanouav.agile.trajectory import T_I
from nanouav.follow.detector import SimDetector
from nanouav.follow.estimate import TargetFilter, box_to_world, rotation_warp
from nanouav.follow.tracker import TargetLock, Tracker


def turn_predict(q, v, w, dt, cap=math.pi / 2):
    """Where a target last seen at q with velocity v is after dt s if it keeps turning at w rad/s on the ground
    plane (after a total turn of `cap` it goes straight on); its height moves with v_z. w = 0: constant velocity."""
    q, v = np.asarray(q, float), np.asarray(v, float)
    s, th0 = math.hypot(v[0], v[1]), math.atan2(v[1], v[0])
    if abs(w) < 1e-3 or s < 1e-6:
        return q + v * dt
    tc = min(dt, cap / abs(w))
    th1 = th0 + w * tc
    x = q[0] + s / w * (math.sin(th1) - math.sin(th0)) + s * math.cos(th1) * (dt - tc)
    y = q[1] - s / w * (math.cos(th1) - math.cos(th0)) + s * math.sin(th1) * (dt - tc)
    return np.array([x, y, q[2] + v[2] * dt])


class FollowMission:
    needs_camera = True
    follows = True                                # the diffphys pilot flies omega × v_des instead of the goal
    cmc = False                                   # camera-motion compensation in the tracker (on for balloons)
    innov_gate = None                             # n σ: skip outlier measurements (≤ 5 in a row) in the filter
    # 0.27.5, fast targets (None / False: as before)
    yaw_ext = None          # s: target lost → the camera looks at the last estimate moved on at most yaw_ext s
    gate_v = None           # m/s: the re-lock gate grows at max(1.5, |v_target| + gate_v) m/s while lost
    relock_free = False     # the first measurement after a re-lock skips the innovation gate
    turn_pred = False       # 0.27.6, target lost → predict it turning at the rate it turned before (≤ 90°/s, ≤ 90° in all)
    turn_win = 0.6          # s of filter velocities for that turn rate

    def __init__(self, env, standoff=4.0, z_follow=2.0, v_max=6.0, k_dist=0.8, t_coast=1.5, t_fail=6.0,
                 detector=None, seed=0):
        self.env, self.walkers, self.target = env, env.walkers, env.walkers[0]
        self.standoff, self.z, self.v_max, self.k = standoff, z_follow, v_max, k_dist
        self.t_coast, self.t_fail = t_coast, t_fail
        self.det = detector or SimDetector(seed=seed)
        self.tracker, self.lock, self.flt = self._make_tracker(), TargetLock(), self._make_filter()
        self._pose_prev, self._n_rej, self._seen, self._hist = None, 0, None, []
        P = self.target.path.pts.copy()
        d0 = P[min(10, len(P) - 1)] - P[0]
        d0 /= max(np.linalg.norm(d0), 1e-6)
        self.L_back = standoff + 3.0                          # the trail starts behind the person's start
        back = P[0] - np.outer(np.arange(int(self.L_back / 0.1), 0, -1) * 0.1, d0)
        P = np.vstack([back, P])
        P = self._trail_z(P)
        self.trail = GlobalRef(P, self.target.speed)          # privileged reference (expert, DAgger)
        self.gref, self.line = self.trail, None
        self.t_last_seen, self.t = None, 0.0
        self.n_frames = self.n_locked = self.n_true = self.n_wrong = 0
        self.dist_err, self.max_dist, self.t_lost_max = [], 0.0, 0.0
        self._lost_since = None

    # ── rollout interface ─────────────────────────────────────────────────
    def start(self):
        p0 = self.target.pos(0.0)
        A = np.array([p0[0] - self.standoff - 1.0, p0[1], self.trail.pts[0, 2]])
        return A, 0.0, np.zeros(3)

    def t_max(self):
        dur = self.target.duration() if hasattr(self.target, 'duration') else self.target.path.length / self.target.speed
        return self.target.t0 + dur + 6.0

    def render_world(self):
        return self.env.render_world()

    def step(self, t):
        self.t = t
        for w in self.walkers:
            w.move(t)

    def _privileged(self, t, p):
        """Reference along the person's trail from the drone's closest point, advancing at the speed that
        closes the gap to the follow point (standoff m behind the person) and never passing it.
        → (10 reference points, distance to the trail, reference speed)."""
        s_f = self.target.s(t) + self.L_back - self.standoff
        s_c, d_ref = self.trail.closest(p, getattr(self, '_s_c', None))
        self._s_c = s_c
        v_t = (self.target.speed_at(t) if hasattr(self.target, 'speed_at') else self.target.speed) \
            if not self.target.done(t) and t >= self.target.t0 else 0.0
        v_ref = float(np.clip(v_t + self.k * (s_f - s_c), 0.0, self.v_max))
        ref = self.trail.pos(np.minimum(s_c + max(v_ref, 0.3) * T_I, s_f + v_t * T_I + 0.3))   # never past the moving follow point
        return ref, d_ref, max(v_ref, 0.3)

    # ── target-specific parts (a person here; `balloon_mission.BalloonMission` overrides them) ──
    def _make_filter(self):
        return TargetFilter()

    def _make_tracker(self):
        return Tracker()

    def _trail_z(self, P):
        P[:, 2] = self.z
        return P

    def _detect(self, cam):
        return self.det.detect(cam, self.walkers) if isinstance(self.det, SimDetector) else self.det.detect(cam['gray'])

    def _image(self, cam):
        return cam['gray']

    def _pos_of(self, cam, tr):
        return box_to_world(cam['rig'], cam['pose'], cam['depth'], tr.box)[0]

    def _goal_z(self, tgt):
        return self.z

    def frame(self, t, p, v, yaw, cam):
        self.n_frames += 1
        dets = self._detect(cam)
        warp = rotation_warp(cam['rig'], self._pose_prev, cam['pose']) if self.cmc and self._pose_prev else None
        self._pose_prev = cam['pose']
        tracks = self.tracker.update(dets, self._image(cam), warp=warp)
        pos_of = lambda tr: self._pos_of(cam, tr)
        if self.lock.track_id is None:
            tr = self._initial(tracks, cam)
            how = 'locked' if tr else 'search'
        else:
            if self.gate_v is not None and self.flt.vel is not None:
                self.lock.gate_growth = max(1.5, float(np.hypot(*self.flt.vel[:2])) + self.gate_v)
            tr, how = self.lock.update(tracks, pos_of, self.flt, t)
            if self.lock.banned:                          # identity switch: end that track
                self.tracker.drop(self.lock.banned)
                self.lock.banned.clear()
        self.flt.predict(t)
        if tr is not None:
            q = pos_of(tr)
            if q is not None:
                d = float(np.hypot(*(q[:2] - p[:2])))
                sig = 0.15 + 0.05 * d
                if self.innov_gate is not None and self._n_rej < 5 and not (self.relock_free and how == 'relocked') \
                        and not self.flt.gate(q, t, n_sigma=self.innov_gate, floor=3 * sig):
                    self._n_rej += 1          # outlier (e.g. two balloons merged into one blob): coast
                else:
                    self._n_rej = 0
                    self.flt.update(t, q, sig)
                    self._seen = (t, self.flt.pos.copy(), self.flt.vel.copy())
                    self._hist = [h for h in self._hist if h[0] > t - self.turn_win] + [(t, self.flt.vel[:2].copy())]
                self.t_last_seen = t
            self.n_locked += 1
            if tr.gt == 0:
                self.n_true += 1
            elif tr.gt > 0 or tr.gt == -1:
                self.n_wrong += 1
        # metrics on the true target
        true = self.target.pos(t)
        dt_true = float(np.hypot(*(true[:2] - p[:2])))
        if self.t_last_seen is not None:
            self.dist_err.append(abs(dt_true - self.standoff))
            self.max_dist = max(self.max_dist, dt_true)
        lost_for = (t - self.t_last_seen) if self.t_last_seen is not None else t
        self.t_lost_max = max(self.t_lost_max, lost_for if self.t_last_seen is not None else 0.0)
        ref, d_ref, v_ref = self._privileged(t, p)
        common = {'ref': ref, 'd_ref': d_ref, 'v_ref': v_ref}
        tgt = self.flt.pos
        if tgt is None:                                     # not acquired yet: hover, look ahead
            return dict(common, omega=np.array([1.0, 0, 0]), v_des=0.5, blind=np.repeat(p[None], 10, 0), yaw=0.0,
                        hold=True)
        vt = self.flt.vel
        pred = None
        if self.turn_pred and self._seen is not None and lost_for > 0:   # 0.27.6: it keeps turning
            t_s, q_s, v_s = self._seen
            w = self._turn_rate()
            pred = lambda dt: turn_predict(q_s, v_s, w, dt)
            tgt = pred(t - t_s)
        if lost_for > self.t_coast:
            vt = np.zeros(3)
        horiz = p[:2] - tgt[:2]
        dist = float(np.hypot(*horiz))
        u = horiz / max(dist, 1e-6)
        goal = np.r_[tgt[:2] + u * self.standoff, self._goal_z(tgt)]
        e = goal - p
        n = float(np.linalg.norm(e))
        omega = e / n if n > 1e-3 else np.r_[-u, 0.0]
        v_t = float(np.hypot(*vt[:2]))
        v_des = float(np.clip(v_t + self.k * (dist - self.standoff), 0.3, self.v_max))
        look = tgt
        if self.yaw_ext is not None and self._seen is not None and lost_for > 0:   # look where it was last seen
            t_s, q_s, v_s = self._seen
            look = pred(min(t - t_s, self.yaw_ext)) if pred else q_s + v_s * min(t - t_s, self.yaw_ext)
        yaw_t = math.atan2(look[1] - p[1], look[0] - p[0])
        hold = lost_for > self.t_coast or (dist < self.standoff + 0.3 and v_t < 0.2)
        blind = p + np.outer(T_I, omega * v_des)
        return dict(common, omega=omega, v_des=v_des, blind=blind, yaw=yaw_t, hold=hold, how=how)

    def _turn_rate(self, w_max=math.pi / 2):
        """rad/s: slope of the heading of the filter velocity over the last turn_win s (0 if too few or slow)."""
        H = [(t, math.atan2(v[1], v[0])) for t, v in self._hist if math.hypot(v[0], v[1]) > 1.0]
        if len(H) < 3 or H[-1][0] - H[0][0] < 0.2:
            return 0.0
        T, A = np.array(H).T
        return float(np.clip(np.polyfit(T - T[-1], np.unwrap(A), 1)[0], -w_max, w_max))

    def _initial(self, tracks, cam):
        """The operator designates the person to follow (in simulation: the track that is the target, like
        tapping on it in the video)."""
        cand = [tr for tr in tracks if tr.gt == 0] or None
        if cand:
            self.lock.lock(cand[0])
            return cand[0]
        return None

    def check(self, t, p):
        c_dyn = min(w.clearance(p) for w in self.walkers)
        lost_for = (t - self.t_last_seen) if self.t_last_seen is not None else (t if t > 5.0 else 0.0)
        failed = lost_for > self.t_fail
        done = self.target.done(t) and t > self.target.t0 + 1.0
        ok = done and self.n_true >= 0.8 * max(self.n_locked, 1) and self.max_dist < 3 * self.standoff + 4
        return ok, failed or (done and not ok), c_dyn

    def metrics(self):
        return {'locked_frac': round(self.n_locked / max(self.n_frames, 1), 3),
                'true_frac': round(self.n_true / max(self.n_locked, 1), 3),
                'wrong_frames': int(self.n_wrong), 'relocks': int(self.lock.relocks), 'switches': int(self.lock.switches),
                'dist_err_med': round(float(np.median(self.dist_err)), 2) if self.dist_err else None,
                'max_dist': round(self.max_dist, 2), 'max_lost_s': round(self.t_lost_max, 2),
                'speed': round(self.target.speed, 2), 'n_people': len(self.walkers),
                'density': round(self.env.meta.get('density', 0), 4)}
