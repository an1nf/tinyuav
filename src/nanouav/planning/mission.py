"""
nanouav/planning/mission.py — Mission layer on top of EvasionPlanner.

  WaypointMission : route A → B → C … (straight segments; evasion deviates and
                    returns to the segment line). Switches segment on the acceptance
                    radius or when passing the waypoint; brakes to 0 at the last one.
  TargetTracker   : α-β filter (position + velocity) over noisy target
                    measurements (UWB / GNSS-RTK / visual detector), tolerates dropouts.
  TargetFollower  : follows a moving target at distance d_follow with feed-forward
                    of its velocity, pointing the camera at it.

All of them only set the planner's segment (A, B), cruise speed and yaw mode;
evasion, the safety filter and smoothness are unchanged.
"""
import math
import numpy as np


class WaypointMission:
    def __init__(self, waypoints, v_cruise=1.0, accept=0.35, z=None, yaw_mode='velocity'):
        self.wps = [np.asarray(w, float) for w in waypoints]     # includes the start A
        self.i = 1
        self.v_cruise, self.accept, self.z, self.yaw_mode = v_cruise, accept, z, yaw_mode
        self.done = False
        self.t_reach = []

    @property
    def segment(self):
        return self.wps[self.i - 1], self.wps[self.i]

    def update(self, planner, p, t=None):
        A, B = self.segment
        last = self.i == len(self.wps) - 1
        d = np.linalg.norm(p[:2] - B[:2])
        seg = B[:2] - A[:2]
        passed = np.dot(p[:2] - A[:2], seg) > np.dot(seg, seg)       # passed B
        if not last and (d < self.accept or passed):
            self.t_reach.append(t)
            self.i += 1
            A, B = self.segment
            last = self.i == len(self.wps) - 1
        elif last and d < planner.ACCEPT + 0.05 and not self.done:
            self.done = True
            self.t_reach.append(t)
        planner.set_segment(A, B, arrive=last, v_cruise=self.v_cruise)
        planner.yaw_mode = self.yaw_mode
        if self.z is not None:
            planner.z_cruise = float(self.z)
        return self.done


class TargetTracker:
    """2D α-β filter. update(meas|None, dt) → (pos, vel)."""

    def __init__(self, alpha=0.45, beta=0.12, t_coast=0.6):
        self.alpha, self.beta = alpha, beta
        self.x = None
        self.v = np.zeros(2)
        self.t_since = 0.0
        self.t_coast = t_coast            # extrapolate for at most t_coast s after losing the target
        self.last_meas = None

    def update(self, meas, dt):
        if self.x is None:
            if meas is None:
                return None, None
            self.x = np.asarray(meas, float)[:2].copy()
            return self.x, self.v
        xp = self.x + self.v * dt
        if meas is None:                     # dropout: bounded prediction (do not chase a phantom)
            self.t_since += dt
            if self.t_since <= self.t_coast:
                self.x = xp
                self.v *= 0.98
            else:
                self.v *= 0.5
            return self.x, self.v
        self.last_meas = np.asarray(meas, float)[:2].copy()
        r = np.asarray(meas, float)[:2] - xp
        self.x = xp + self.alpha * r
        self.v = self.v + (self.beta / dt) * r
        self.t_since = 0.0
        return self.x, self.v

    def predict(self, dt):
        return None if self.x is None else self.x + self.v * dt


class VisualTargetSensor:
    """
    Target measurement as an onboard detector (bbox) would deliver it, not an oracle.
    Model taken from Su et al., EAAI 177:114890 (2026) — FeatherYOLO + IBVS on Crazyflie:
      * a measurement exists only if the target is inside the camera FOV and not occluded;
      * bearing ← X_offset of the bbox center (pixel noise ∝ bbox width);
      * range   ← apparent size (their A_scale): d = f·W/w_px, with a per-episode bias
                 in the assumed target size (tall/short person, clothing) → error ∝ d;
      * false negatives growing with distance + dropouts due to lighting.
    range_src='tof' (our own proposal): if the bearing falls inside the ToF FOV (40°),
    the range comes from the ToF/CNN-calibrated zone covering the bbox → ~cm error, without
    depending on the assumed size. Outside the 40° the apparent size is used.
    """

    def __init__(self, rng, range_src='bbox', hfov=1.047, width=96, tof_hfov=0.698,
                 d_max=5.0, p_det=0.97, size_bias_sd=0.10, px_sd=0.5, rel_sd=0.06):
        self.rng, self.range_src = rng, range_src
        self.hfov, self.tof_half = hfov, tof_hfov / 2
        self.f = (width / 2) / math.tan(hfov / 2)
        self.d_max, self.p_det = d_max, p_det
        self.size_bias = float(np.exp(rng.normal(0, size_bias_sd)))   # r_assumed / r_true
        self.px_sd, self.rel_sd = px_sd, rel_sd
        self.n, self.n_det, self.t_last, self.max_gap = 0, 0, None, 0.0

    def _occluded(self, world, cam, tgt_xy, target_obj, d):
        D = np.array([tgt_xy[0] - cam[0], tgt_xy[1] - cam[1], 0.0])
        D /= max(np.linalg.norm(D), 1e-9)
        for o in world.objects:
            if o is target_obj:
                continue
            t, _ = o.intersect(cam, D[None, :])
            if float(t[0]) < d - 0.05:
                return True
        return False

    def measure(self, t, p, yaw, tgt_xy, tgt_r, world=None, target_obj=None):
        self.n += 1
        if self.t_last is None:
            self.t_last = t
        dv = np.asarray(tgt_xy[:2], float) - p[:2]
        d = float(np.linalg.norm(dv))
        brg = math.atan2(dv[1], dv[0])
        off = (brg - yaw + math.pi) % (2 * math.pi) - math.pi
        half_ang = math.atan2(tgt_r, max(d, 1e-3))
        ok = abs(off) + 0.5 * half_ang < self.hfov / 2 and 0.3 < d < self.d_max
        if ok:
            p_d = self.p_det * float(np.clip(1.0 - (d - 3.0) / 3.0, 0.3, 1.0))
            ok = self.rng.random() < p_d
        if ok and world is not None:
            cam = np.array([p[0], p[1], p[2] if len(p) > 2 else 0.5])
            ok = not self._occluded(world, cam, tgt_xy, target_obj, d)
        if not ok:
            self.max_gap = max(self.max_gap, t - self.t_last)
            return None
        self.n_det += 1
        self.max_gap = max(self.max_gap, t - self.t_last)
        self.t_last = t
        w_px = 2 * self.f * tgt_r / d
        w_meas = max(w_px + self.rng.normal(0, self.rel_sd * w_px + self.px_sd), 1.0)
        x_px = self.f * math.tan(off) + self.rng.normal(0, 0.5 * self.rel_sd * w_px + self.px_sd)
        off_m = math.atan2(x_px, self.f)
        if self.range_src == 'tof' and abs(off_m) < self.tof_half:
            d_m = d + self.rng.normal(0, 0.02 + 0.02 * d)          # calibrated ToF/CNN zone + assumed r
        else:
            d_m = 2 * self.f * tgt_r * self.size_bias / w_meas     # range from apparent size
        a = yaw + off_m
        return p[:2] + d_m * np.array([math.cos(a), math.sin(a)])

    @property
    def stats(self):
        return {'det_rate': self.n_det / max(self.n, 1), 'max_gap_s': self.max_gap,
                'size_bias': self.size_bias}


class TargetFollower:
    """
    Distance-keeping follow: v_des = v_obj·u + k·(d − d_follow), bounded.
    The lane is the drone → target line (re-anchored every frame); the camera
    points at the target (yaw_mode='target') so it does not leave the field of view.

    guidance='ibvs' adds the guidance layer of Su et al. (2026), ablation G0→G6:
    dead zone with hysteresis on the distance error (enters < dz_in, exits > dz_out),
    α=0.3 low-pass filter on the command, and bounded backing-up (v_back_max): their only
    collision happened while backing up into what the camera cannot see.
    """

    def __init__(self, d_follow=1.3, v_max=1.2, k=0.8, latency=0.15, target_r=0.25,
                 guidance='base', dz_in=0.10, dz_out=0.20, lp_alpha=0.3, v_back_max=0.35):
        self.d_follow, self.v_max, self.k, self.latency = d_follow, v_max, k, latency
        self.target_r = target_r
        self.guidance, self.dz_in, self.dz_out = guidance, dz_in, dz_out
        self.lp_alpha, self.v_back_max = lp_alpha, v_back_max
        self._hold, self._v_f = False, 0.0
        # reacquisition (loss > t_lost): hover and sweep yaw toward the side the target exited
        self.reacquire, self.t_lost, self.w_sweep = True, 0.6, 1.0
        self._brg, self._brg_rate = None, 0.0

    def _shape(self, v_ff, err):
        if self.guidance != 'ibvs':
            return v_ff + self.k * err
        if self._hold and abs(err) > self.dz_out:
            self._hold = False
        elif not self._hold and abs(err) < self.dz_in:
            self._hold = True
        v = v_ff + (0.0 if self._hold else self.k * err)
        self._v_f = self.lp_alpha * self._v_f + (1 - self.lp_alpha) * v
        return max(self._v_f, -self.v_back_max)

    def update(self, planner, p, tracker):
        tp = tracker.predict(self.latency)
        if tp is None:
            planner.set_segment(p[:2], p[:2] + [1.0, 0.0], arrive=True, v_cruise=0.0)
            return None
        dvec = tp - p[:2]
        d = float(np.linalg.norm(dvec))
        u = dvec / max(d, 1e-6)
        if tracker.t_since == 0.0:                      # bearing and its trend at the last sighting
            b = math.atan2(dvec[1], dvec[0])
            if self._brg is not None:
                db = (b - self._brg + math.pi) % (2 * math.pi) - math.pi
                self._brg_rate = 0.7 * self._brg_rate + 0.3 * db
            self._brg = b
        elif self.reacquire and tracker.t_since > self.t_lost and self._brg is not None:
            sgn = 1.0 if self._brg_rate >= 0 else -1.0
            a = self._brg + sgn * min(self.w_sweep * (tracker.t_since - self.t_lost) + 0.35, math.pi)
            look = p[:2] + 2.0 * np.array([math.cos(a), math.sin(a)])
            planner.set_segment(p[:2], p[:2] + 0.5 * u, arrive=True, v_cruise=0.0)
            planner.yaw_mode, planner.look_point = 'target', look
            planner.exclude = (tp, self.target_r + 0.35)
            return d
        v_des = float(np.clip(self._shape(tracker.v @ u, d - self.d_follow), -self.v_max, self.v_max))
        planner.set_segment(p[:2], tp, arrive=False, v_cruise=v_des)
        planner.yaw_mode = 'target'
        planner.look_point = tp
        planner.exclude = (tp, self.target_r + 0.35)
        return d
