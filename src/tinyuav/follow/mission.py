"""Balloon-following mission for `tinyuav.nav.rollout.Rollout`.

Per camera frame: color blobs, tracker, target lock, 3D position from the apparent size, Kalman filter,
then the follow command: direction and speed toward the point `standoff` m from the target, yaw on the
target. Privileged pilots fly the target's own path from the drone's closest point.
"""
import math

import numpy as np

from tinyuav.follow.blob import BlobDetector
from tinyuav.follow.estimate import TargetFilter, blob_to_world, project, rotation_warp
from tinyuav.follow.tracker import TargetLock, Tracker
from tinyuav.nav.expert import GlobalRef
from tinyuav.sim.trajectory import T_I

FAST_TARGET = {'q_acc': 5.0, 'yaw_ext': 0.5, 'gate_v': 1.0, 'relock_free': True, 'turn_pred': True}


def turn_predict(q, v, w, dt, cap=math.pi / 2):
    """Position after dt s of a target at q with velocity v turning at w rad/s in the ground plane, then
    straight after a total turn of `cap`."""
    q, v = np.asarray(q, float), np.asarray(v, float)
    s, th0 = math.hypot(v[0], v[1]), math.atan2(v[1], v[0])
    if abs(w) < 1e-3 or s < 1e-6:
        return q + v * dt
    tc = min(dt, cap / abs(w))
    th1 = th0 + w * tc
    x = q[0] + s / w * (math.sin(th1) - math.sin(th0)) + s * math.cos(th1) * (dt - tc)
    y = q[1] - s / w * (math.cos(th1) - math.cos(th0)) + s * math.sin(th1) * (dt - tc)
    return np.array([x, y, q[2] + v[2] * dt])


class BalloonMission:
    needs_camera = True
    needs_rgb = True
    follows = True              # diffphys flies omega * v_des
    innov_gate = 3.0            # n sigma; up to 5 outliers in a row are skipped
    q_acc = 2.0                 # m/s^2, filter process noise
    yaw_ext = None              # s: while lost, look at the last estimate extrapolated at most this long
    gate_v = None               # m/s: while lost, the re-lock gate grows at |v_target| + gate_v
    relock_free = False         # first measurement after a re-lock skips the innovation gate
    turn_pred = False           # while lost, extrapolate the last turn rate
    turn_win = 0.6              # s of filter velocities for the turn rate

    def __init__(self, env, standoff=4.0, v_max=8.0, k_dist=0.8, t_coast=1.5, t_fail=6.0, z_min=1.0, z_max=3.0,
                 detector=None, seed=0, opts=None):
        for k, v in (opts or {}).items():
            setattr(self, k, v)
        self.env, self.balloons, self.target = env, env.balloons, env.balloons[0]
        self.z_min, self.z_max = z_min, z_max
        self.z = float(np.clip(self.target.pos(0.0)[2], z_min, z_max))
        self.standoff, self.v_max, self.k = standoff, v_max, k_dist
        self.t_coast, self.t_fail = t_coast, t_fail
        self.det = detector or BlobDetector(self.target.color)
        self.tracker = Tracker(min_gate_px=12.0)
        self.lock = TargetLock(gate_locked=True, w_dist=1.0, gate_growth=1.5, jump_m=0.8, jump_v=5.0)
        self.flt = TargetFilter(q_acc=self.q_acc, dims=3)
        self._pose_prev, self._n_rej, self._seen, self._hist = None, 0, None, []
        P = self.target.path.pts.copy()
        d0 = P[min(10, len(P) - 1)] - P[0]
        d0 /= max(np.linalg.norm(d0), 1e-6)
        self.L_back = standoff + 3.0
        nb = int(self.L_back / 0.1)
        P = np.vstack([P[0] - np.outer(np.arange(nb, 0, -1) * 0.1, d0), P])
        P[:nb, 2] = P[nb, 2]
        P[:, 2] = np.clip(P[:, 2], z_min, z_max)
        self.trail = GlobalRef(P, self.target.speed)
        self.gref, self.line = self.trail, None
        self.t_last_seen, self.t = None, 0.0
        self.n_frames = self.n_locked = self.n_true = self.n_wrong = 0
        self.dist_err, self.max_dist, self.t_lost_max = [], 0.0, 0.0
        self.locks = {}                              # t -> (locked box or None, lock state), for videos

    def start(self):
        p0 = self.target.pos(0.0)
        return np.array([p0[0] - self.standoff - 1.0, p0[1], self.trail.pts[0, 2]]), 0.0, np.zeros(3)

    def t_max(self):
        return self.target.t0 + self.target.duration() + 6.0

    def render_world(self):
        return self.env.render_world()

    def step(self, t):
        self.t = t
        for b in self.balloons:
            b.move(t)

    def _privileged(self, t, p):
        """Reference on the target's trail, advancing at the speed that closes the gap to the follow point
        without passing it: (10 points, distance to the trail, speed)."""
        s_f = self.target.s(t) + self.L_back - self.standoff
        s_c, d_ref = self.trail.closest(p, getattr(self, '_s_c', None))
        self._s_c = s_c
        v_t = self.target.speed_at(t) if not self.target.done(t) and t >= self.target.t0 else 0.0
        v_ref = float(np.clip(v_t + self.k * (s_f - s_c), 0.0, self.v_max))
        ref = self.trail.pos(np.minimum(s_c + max(v_ref, 0.3) * T_I, s_f + v_t * T_I + 0.3))
        return ref, d_ref, max(v_ref, 0.3)

    def _detect(self, cam):
        dets = self.det.detect(cam['rgb'])
        if dets:                                     # ground-truth labels, for metrics only
            C = np.array([b.body.center for b in self.balloons])
            uv, X = project(cam['rig'], cam['pose'], C)
            for d in dets:
                x1, y1, x2, y2 = d['box']
                m = max(x2 - x1, y2 - y1) * 0.5 + 2
                inside = [(np.hypot(uv[i, 0] - (x1 + x2) / 2, uv[i, 1] - (y1 + y2) / 2), X[i], i)
                          for i in range(len(C))
                          if X[i] > 0.3 and x1 - m <= uv[i, 0] <= x2 + m and y1 - m <= uv[i, 1] <= y2 + m]
                d['gt'] = min(inside)[2] if inside else -1
        return dets

    def _pos_of(self, cam, tr):
        return blob_to_world(cam['rig'], cam['pose'], cam['depth'], tr.box, self.target.radius)[0]

    def frame(self, t, p, v, yaw, cam):
        self.n_frames += 1
        dets = self._detect(cam)
        warp = rotation_warp(cam['rig'], self._pose_prev, cam['pose']) if self._pose_prev else None
        self._pose_prev = cam['pose']
        tracks = self.tracker.update(dets, cam.get('rgb'), warp=warp)
        pos_of = lambda tr: self._pos_of(cam, tr)
        if self.lock.track_id is None:
            tr = next((k for k in tracks if k.gt == 0), None)      # the operator designates the target
            if tr is not None:
                self.lock.lock(tr)
            how = 'locked' if tr else 'search'
        else:
            if self.gate_v is not None and self.flt.vel is not None:
                self.lock.gate_growth = max(1.5, float(np.hypot(*self.flt.vel[:2])) + self.gate_v)
            tr, how = self.lock.update(tracks, pos_of, self.flt, t)
            if self.lock.banned:
                self.tracker.drop(self.lock.banned)
                self.lock.banned.clear()
        self.flt.predict(t)
        if tr is not None:
            q = pos_of(tr)
            if q is not None:
                sig = 0.15 + 0.05 * float(np.hypot(*(q[:2] - p[:2])))
                if self._n_rej < 5 and not (self.relock_free and how == 'relocked') \
                        and not self.flt.gate(q, t, n_sigma=self.innov_gate, floor=3 * sig):
                    self._n_rej += 1
                else:
                    self._n_rej = 0
                    self.flt.update(t, q, sig)
                    self._seen = (t, self.flt.pos.copy(), self.flt.vel.copy())
                    self._hist = [h for h in self._hist if h[0] > t - self.turn_win] + [(t, self.flt.vel[:2].copy())]
                self.t_last_seen = t
            self.n_locked += 1
            if tr.gt == 0:
                self.n_true += 1
            else:
                self.n_wrong += 1
        d_true = float(np.hypot(*(self.target.pos(t)[:2] - p[:2])))
        if self.t_last_seen is not None:
            self.dist_err.append(abs(d_true - self.standoff))
            self.max_dist = max(self.max_dist, d_true)
        lost_for = (t - self.t_last_seen) if self.t_last_seen is not None else t
        self.t_lost_max = max(self.t_lost_max, lost_for if self.t_last_seen is not None else 0.0)
        ref, d_ref, v_ref = self._privileged(t, p)
        common = {'ref': ref, 'd_ref': d_ref, 'v_ref': v_ref}
        tgt = self.flt.pos
        self._record_lock(t, how)
        if tgt is None:
            return dict(common, omega=np.array([1.0, 0, 0]), v_des=0.5, blind=np.repeat(p[None], 10, 0), yaw=0.0,
                        hold=True)
        vt, pred = self.flt.vel, None
        if self.turn_pred and self._seen is not None and lost_for > 0:
            t_s, q_s, v_s = self._seen
            w = self._turn_rate()
            pred = lambda dt: turn_predict(q_s, v_s, w, dt)
            tgt = pred(t - t_s)
        if lost_for > self.t_coast:
            vt = np.zeros(3)
        horiz = p[:2] - tgt[:2]
        dist = float(np.hypot(*horiz))
        u = horiz / max(dist, 1e-6)
        goal = np.r_[tgt[:2] + u * self.standoff, np.clip(tgt[2], self.z_min, self.z_max)]
        e = goal - p
        n = float(np.linalg.norm(e))
        omega = e / n if n > 1e-3 else np.r_[-u, 0.0]
        v_t = float(np.hypot(*vt[:2]))
        v_des = float(np.clip(v_t + self.k * (dist - self.standoff), 0.3, self.v_max))
        look = tgt
        if self.yaw_ext is not None and self._seen is not None and lost_for > 0:
            t_s, q_s, v_s = self._seen
            dt = min(t - t_s, self.yaw_ext)
            look = pred(dt) if pred else q_s + v_s * dt
        hold = lost_for > self.t_coast or (dist < self.standoff + 0.3 and v_t < 0.2)
        return dict(common, omega=omega, v_des=v_des, blind=p + np.outer(T_I, omega * v_des),
                    yaw=math.atan2(look[1] - p[1], look[0] - p[0]), hold=hold, how=how)

    def _record_lock(self, t, how):
        tr = next((k for k in self.tracker.tracks if k.id == self.lock.track_id and k.lost == 0), None)
        box = tuple(float(x) for x in tr.box) if tr is not None and how in ('locked', 'relocked') else None
        self.locks[round(t, 3)] = (box, how)

    def label_trace(self, trace):
        for d in trace or []:
            d['target_box'], d['target_how'] = self.locks.get(round(d['t'], 3), (None, None))

    def _turn_rate(self, w_max=math.pi / 2):
        """Slope (rad/s) of the filter-velocity heading over the last turn_win s."""
        H = [(t, math.atan2(v[1], v[0])) for t, v in self._hist if math.hypot(v[0], v[1]) > 1.0]
        if len(H) < 3 or H[-1][0] - H[0][0] < 0.2:
            return 0.0
        T, A = np.array(H).T
        return float(np.clip(np.polyfit(T - T[-1], np.unwrap(A), 1)[0], -w_max, w_max))

    def check(self, t, p):
        c_dyn = min(b.clearance(p) for b in self.balloons)
        lost_for = (t - self.t_last_seen) if self.t_last_seen is not None else (t if t > 5.0 else 0.0)
        done = self.target.done(t) and t > self.target.t0 + 1.0
        ok = done and self.n_true >= 0.8 * max(self.n_locked, 1) and self.max_dist < 3 * self.standoff + 4
        return ok, lost_for > self.t_fail or (done and not ok), c_dyn

    def metrics(self):
        return {'locked_frac': round(self.n_locked / max(self.n_frames, 1), 3),
                'true_frac': round(self.n_true / max(self.n_locked, 1), 3),
                'wrong_frames': int(self.n_wrong), 'relocks': int(self.lock.relocks), 'switches': int(self.lock.switches),
                'dist_err_med': round(float(np.median(self.dist_err)), 2) if self.dist_err else None,
                'max_dist': round(self.max_dist, 2), 'max_lost_s': round(self.t_lost_max, 2),
                'speed': round(self.target.speed, 2), 'n_balloons': len(self.balloons),
                'same_color': self.env.meta.get('same_color', 0), 'density': round(self.env.meta.get('density', 0), 4)}
