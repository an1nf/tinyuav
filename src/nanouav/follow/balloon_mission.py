"""Follow a balloon (R20): `FollowMission` with a color-blob detector on the RGB image, the 3D position of a
sphere of known size, a 3D target filter and the follow point at the balloon height.

Everything else (tracker, target lock with the 3D gate, ω / v_des / yaw for the student, the privileged
trail reference for the expert and DAgger, metrics) is the person-following mission of R19.
"""
import numpy as np

from nanouav.follow.blob import BlobDetector
from nanouav.follow.estimate import TargetFilter, blob_to_world, project
from nanouav.follow.mission import FollowMission
from nanouav.follow.tracker import TargetLock, Tracker


class BalloonMission(FollowMission):
    needs_rgb = True
    cmc = True
    innov_gate = 3.0

    def __init__(self, env, standoff=4.0, v_max=8.0, k_dist=0.8, t_coast=1.5, t_fail=6.0, z_min=1.0, z_max=3.0,
                 detector=None, seed=0):
        self.z_min, self.z_max = z_min, z_max
        color = env.walkers[0].color
        super().__init__(env, standoff=standoff, z_follow=float(np.clip(env.walkers[0].pos(0.0)[2], z_min, z_max)),
                         v_max=v_max, k_dist=k_dist, t_coast=t_coast, t_fail=t_fail,
                         detector=detector or BlobDetector(color), seed=seed)
        # balloons of one color look the same: the 3D prediction decides (see TargetLock)
        self.lock = TargetLock(gate_locked=True, w_dist=1.0, gate_growth=1.5, jump_m=0.8, jump_v=5.0)

    def _make_tracker(self):
        return Tracker(min_gate_px=12.0)

    def _make_filter(self):
        return TargetFilter(q_acc=2.0, dims=3)

    def _trail_z(self, P):
        nb = int(self.L_back / 0.1)
        P[:nb, 2] = P[nb, 2]                    # the backward extension stays at the start height
        P[:, 2] = np.clip(P[:, 2], self.z_min, self.z_max)
        return P

    def _detect(self, cam):
        dets = self.det.detect(cam['rgb'])
        self._label(cam, dets)
        return dets

    def _label(self, cam, dets):
        """Which balloon is each detection? (metrics only: the projection of the true centers)"""
        if not dets:
            return
        C = np.array([b.body.center for b in self.walkers])
        uv, X = project(cam['rig'], cam['pose'], C)
        for d in dets:
            x1, y1, x2, y2 = d['box']
            m = max(x2 - x1, y2 - y1) * 0.5 + 2
            inside = [(np.hypot(uv[i, 0] - (x1 + x2) / 2, uv[i, 1] - (y1 + y2) / 2), X[i], i) for i in range(len(C))
                      if X[i] > 0.3 and x1 - m <= uv[i, 0] <= x2 + m and y1 - m <= uv[i, 1] <= y2 + m]
            d['gt'] = min(inside)[2] if inside else -1

    def _image(self, cam):
        return cam['rgb']

    def _pos_of(self, cam, tr):
        return blob_to_world(cam['rig'], cam['pose'], cam['depth'], tr.box, self.target.radius)[0]

    def _goal_z(self, tgt):
        return float(np.clip(tgt[2], self.z_min, self.z_max))

    def metrics(self):
        m = super().metrics()
        m.pop('n_people', None)
        m.update(n_balloons=len(self.walkers), same_color=self.env.meta.get('same_color', 0))
        return m
