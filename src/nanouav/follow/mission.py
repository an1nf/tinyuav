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
from nanouav.follow.estimate import TargetFilter, box_to_world
from nanouav.follow.tracker import TargetLock, Tracker


class FollowMission:
    needs_camera = True

    def __init__(self, env, standoff=4.0, z_follow=2.0, v_max=6.0, k_dist=0.8, t_coast=1.5, t_fail=6.0,
                 detector=None, seed=0):
        self.env, self.walkers, self.target = env, env.walkers, env.walkers[0]
        self.standoff, self.z, self.v_max, self.k = standoff, z_follow, v_max, k_dist
        self.t_coast, self.t_fail = t_coast, t_fail
        self.det = detector or SimDetector(seed=seed)
        self.tracker, self.lock, self.flt = Tracker(), TargetLock(), TargetFilter()
        P = self.target.path.pts.copy()
        d0 = P[min(10, len(P) - 1)] - P[0]
        d0 /= max(np.linalg.norm(d0), 1e-6)
        self.L_back = standoff + 3.0                          # the trail starts behind the person's start
        back = P[0] - np.outer(np.arange(int(self.L_back / 0.1), 0, -1) * 0.1, d0)
        P = np.vstack([back, P])
        P[:, 2] = z_follow
        self.trail = GlobalRef(P, self.target.speed)          # privileged reference (expert, DAgger)
        self.gref, self.line = self.trail, None
        self.t_last_seen, self.t = None, 0.0
        self.n_frames = self.n_locked = self.n_true = self.n_wrong = 0
        self.dist_err, self.max_dist, self.t_lost_max = [], 0.0, 0.0
        self._lost_since = None

    # ── rollout interface ─────────────────────────────────────────────────
    def start(self):
        p0 = self.target.pos(0.0)
        A = np.array([p0[0] - self.standoff - 1.0, p0[1], self.z])
        return A, 0.0, np.zeros(3)

    def t_max(self):
        return self.target.t0 + self.target.path.length / self.target.speed + 6.0

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
        v_t = self.target.speed if not self.target.done(t) and t >= self.target.t0 else 0.0
        v_ref = float(np.clip(v_t + self.k * (s_f - s_c), 0.0, self.v_max))
        ref = self.trail.pos(np.minimum(s_c + max(v_ref, 0.3) * T_I, s_f + v_t * T_I + 0.3))   # never past the moving follow point
        return ref, d_ref, max(v_ref, 0.3)

    def frame(self, t, p, v, yaw, cam):
        self.n_frames += 1
        dets = self.det.detect(cam, self.walkers) if isinstance(self.det, SimDetector) else self.det.detect(cam['gray'])
        tracks = self.tracker.update(dets, cam['gray'])
        pos_of = lambda tr: box_to_world(cam['rig'], cam['pose'], cam['depth'], tr.box)[0]
        if self.lock.track_id is None:
            tr = self._initial(tracks, cam)
            how = 'locked' if tr else 'search'
        else:
            tr, how = self.lock.update(tracks, pos_of, self.flt, t)
        self.flt.predict(t)
        if tr is not None:
            q = pos_of(tr)
            if q is not None:
                d = float(np.hypot(*(q[:2] - p[:2])))
                self.flt.update(t, q, 0.15 + 0.05 * d)
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
        if lost_for > self.t_coast:
            vt = np.zeros(3)
        horiz = p[:2] - tgt[:2]
        dist = float(np.hypot(*horiz))
        u = horiz / max(dist, 1e-6)
        goal = np.r_[tgt[:2] + u * self.standoff, self.z]
        e = goal - p
        n = float(np.linalg.norm(e))
        omega = e / n if n > 1e-3 else np.r_[-u, 0.0]
        v_t = float(np.hypot(*vt[:2]))
        v_des = float(np.clip(v_t + self.k * (dist - self.standoff), 0.3, self.v_max))
        yaw_t = math.atan2(tgt[1] - p[1], tgt[0] - p[0])
        hold = lost_for > self.t_coast or (dist < self.standoff + 0.3 and v_t < 0.2)
        blind = p + np.outer(T_I, omega * v_des)
        return dict(common, omega=omega, v_des=v_des, blind=blind, yaw=yaw_t, hold=hold, how=how)

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
                'wrong_frames': int(self.n_wrong), 'relocks': int(self.lock.relocks),
                'dist_err_med': round(float(np.median(self.dist_err)), 2) if self.dist_err else None,
                'max_dist': round(self.max_dist, 2), 'max_lost_s': round(self.t_lost_max, 2),
                'speed': round(self.target.speed, 2), 'n_people': len(self.walkers),
                'density': round(self.env.meta.get('density', 0), 4)}
