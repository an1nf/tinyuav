"""Multi-person tracker and target lock.

`Tracker` follows the ByteTrack recipe (Zhang et al. 2022): a constant-velocity Kalman filter per box,
association of the confident detections first and of the low-score ones second (they recover occluded
or small people), IoU + appearance cost solved with the Hungarian method.

`TargetLock` turns that into "follow *this* person": it locks one track and keeps an appearance model of
it (color/gray histograms of the upper and lower body, adapted slowly while the lock is confident). If the
locked track is lost (occlusion, out of view) it predicts the person in the world and only re-locks a
track that both looks like the target and is where the target can be — another person crossing in front
does not steal the lock. The appearance model is the replaceable part: a deep re-identification
embedding (OSNet, or the ultralytics BoT-SORT ReID features) can be plugged into `appearance()`.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(a, b):
    """a (N,4), b (M,4) → (N,M)."""
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar = lambda c: (c[:, 2] - c[:, 0]) * (c[:, 3] - c[:, 1])
    return inter / (ar(a)[:, None] + ar(b)[None, :] - inter + 1e-9)


def appearance(image, box, bins=16):
    """Histograms of the upper and lower half of the box (gray or each HSV-like channel), L1-normalized."""
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    H, W = image.shape[:2]
    x1, x2 = max(x1, 0), min(x2, W)
    y1, y2 = max(y1, 0), min(y2, H)
    if x2 - x1 < 2 or y2 - y1 < 4:
        return None
    cx1, cx2 = x1 + (x2 - x1) // 4, x2 - (x2 - x1) // 4
    ym = (y1 + y2) // 2
    feats = []
    for ya, yb in ((y1 + (y2 - y1) // 8, ym), (ym, y2)):
        patch = image[ya:yb, cx1:max(cx2, cx1 + 1)]
        chans = [patch] if patch.ndim == 2 else [patch[..., c] for c in range(patch.shape[-1])]
        for ch in chans:
            h, _ = np.histogram(ch, bins=bins, range=(0, 256))
            feats.append(h / max(h.sum(), 1))
    return np.concatenate(feats)


def similarity(f, g):
    """Bhattacharyya coefficient averaged over the histograms (1 = identical)."""
    if f is None or g is None:
        return 0.0
    return float(np.sum(np.sqrt(f * g)) / max(len(f) / 16, 1))


class _KBox:
    """Kalman filter on (cx, cy, w, h) with constant velocity."""

    def __init__(self, box):
        x1, y1, x2, y2 = box
        self.x = np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, 0, 0, 0, 0], float)
        h = self.x[3]
        self.P = np.diag([h * 0.1, h * 0.1, h * 0.1, h * 0.1, h, h, h * 0.1, h * 0.1]) ** 2

    def predict(self):
        F = np.eye(8)
        F[:4, 4:] = np.eye(4)
        h = max(self.x[3], 4.0)
        Q = np.diag([0.05 * h, 0.05 * h, 0.05 * h, 0.05 * h, 0.1 * h, 0.1 * h, 0.02 * h, 0.02 * h]) ** 2
        self.x = F @ self.x
        self.x[2:4] = np.maximum(self.x[2:4], 2.0)
        self.P = F @ self.P @ F.T + Q

    def update(self, box):
        x1, y1, x2, y2 = box
        z = np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])
        H = np.hstack([np.eye(4), np.zeros((4, 4))])
        h = max(z[3], 4.0)
        R = np.diag([0.05 * h, 0.05 * h, 0.08 * h, 0.08 * h]) ** 2
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - H @ self.x)
        self.P = (np.eye(8) - K @ H) @ self.P

    @property
    def box(self):
        cx, cy, w, h = self.x[:4]
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


class Track:
    _next = 1

    def __init__(self, det, feat):
        self.id = Track._next
        Track._next += 1
        self.kf = _KBox(det['box'])
        self.feat, self.score, self.hits, self.lost, self.age = feat, det['score'], 1, 0, 0
        self.gt = det.get('gt', -1)
        self.det = det

    @property
    def box(self):
        return self.kf.box


class Tracker:
    def __init__(self, high=0.5, low=0.1, iou_min=0.2, max_lost=30, min_hits=2, w_app=0.3, min_gate_px=1.0):
        self.high, self.low, self.iou_min, self.max_lost = high, low, iou_min, max_lost
        self.min_hits, self.w_app = min_hits, w_app
        self.min_gate_px = min_gate_px            # small boxes (a far balloon): gate on at least this many px
        self.tracks = []

    def _match(self, tracks, dets, feats, use_app):
        if not tracks or not dets:
            return [], list(range(len(tracks))), list(range(len(dets)))
        I = iou([t.box for t in tracks], [d['box'] for d in dets])
        C = 1 - I
        if use_app:
            A = np.array([[similarity(t.feat, f) for f in feats] for t in tracks])
            C = (1 - self.w_app) * C + self.w_app * (1 - A)
        # boxes jump when the camera rotates (yaw toward the person, pitch while accelerating): besides
        # IoU, accept pairs whose centers are within 0.6 box heights and whose sizes agree
        tb = np.array([t.box for t in tracks], float)
        db = np.array([d['box'] for d in dets], float)
        tc, dc = (tb[:, :2] + tb[:, 2:]) / 2, (db[:, :2] + db[:, 2:]) / 2
        th, dh = tb[:, 3] - tb[:, 1], db[:, 3] - db[:, 1]
        hh = np.maximum(np.maximum(th[:, None], dh[None, :]), self.min_gate_px)
        cd = np.hypot(tc[:, None, 0] - dc[None, :, 0], tc[:, None, 1] - dc[None, :, 1]) / np.maximum(hh, 1)
        ratio = np.maximum(th[:, None], dh[None, :]) / np.maximum(np.minimum(th[:, None], dh[None, :]), 1)
        near = (cd < 0.6) & (ratio < (1.6 if self.min_gate_px <= 1 else 2.2))
        C = C + 0.5 * np.minimum(cd, 2.0)
        r, c = linear_sum_assignment(C)
        m = [(i, j) for i, j in zip(r, c) if I[i, j] >= self.iou_min or near[i, j]]
        mi, mj = {i for i, _ in m}, {j for _, j in m}
        return m, [i for i in range(len(tracks)) if i not in mi], [j for j in range(len(dets)) if j not in mj]

    def drop(self, ids):
        """End these tracks (their next detections start new tracks)."""
        self.tracks = [t for t in self.tracks if t.id not in ids]

    def update(self, dets, image=None, warp=None):
        """warp: optional camera-motion compensation, a function (u, v) → (u', v') from the previous image
        to this one (e.g. `estimate.rotation_warp` from the attitude change); applied to every track before
        predicting."""
        for t in self.tracks:
            if warp is not None:
                t.kf.x[0], t.kf.x[1] = warp(t.kf.x[0], t.kf.x[1])
            t.kf.predict()
            t.age += 1
        feats = [appearance(image, d['box']) if image is not None else None for d in dets]
        hi = [j for j, d in enumerate(dets) if d['score'] >= self.high]
        lo = [j for j, d in enumerate(dets) if self.low <= d['score'] < self.high]
        m1, un_t, un_d = self._match(self.tracks, [dets[j] for j in hi], [feats[j] for j in hi], image is not None)
        matched = [(self.tracks[i], hi[j]) for i, j in m1]
        rest = [self.tracks[i] for i in un_t]
        m2, un_t2, _ = self._match(rest, [dets[j] for j in lo], [feats[j] for j in lo], False)
        matched += [(rest[i], lo[j]) for i, j in m2]
        for t, j in matched:
            t.kf.update(dets[j]['box'])
            t.score, t.hits, t.lost, t.det, t.gt = dets[j]['score'], t.hits + 1, 0, dets[j], dets[j].get('gt', -1)
            if feats[j] is not None:
                t.feat = feats[j] if t.feat is None else 0.9 * t.feat + 0.1 * feats[j]
        for i in un_t2:
            rest[i].lost += 1
            rest[i].det = None
        for j in (hi[k] for k in un_d):
            self.tracks.append(Track(dets[j], feats[j]))
        self.tracks = [t for t in self.tracks if t.lost <= self.max_lost]
        return [t for t in self.tracks if t.lost == 0 and t.hits >= self.min_hits]


class TargetLock:
    """Keep following one person. State: 'search' (no lock yet), 'locked', 'lost' (predicting)."""

    def __init__(self, sim_min=0.75, lost_relock_after=3, gate_floor=1.5, gate_locked=False, w_dist=0.0,
                 jump_m=1.5, jump_v=10.0, gate_growth=None):
        """gate_locked: also check the 3D position of the locked track every frame — a track that jumps
        (more than jump_m + jump_v · Δt from its previous position) has switched to another object (two
        balloons of the same color crossing: the tracker can swap their identities); it is dropped and
        never re-locked. w_dist: when re-locking,
        prefer the candidate nearest to the prediction (score = similarity − w_dist · distance / gate)
        instead of the most similar one; for identical-looking targets."""
        self.track_id, self.feat, self.state = None, None, 'search'
        self.sim_min, self.relock_after, self.gate_floor = sim_min, lost_relock_after, gate_floor
        self.gate_locked, self.w_dist, self.banned = gate_locked, w_dist, set()
        self.jump_m, self.jump_v, self._last = jump_m, jump_v, None
        self.gate_growth, self._t_lost = gate_growth, None      # m per second lost (None: 0.5 m per 15 frames)
        self.frames_lost, self.switches, self.relocks = 0, 0, 0

    def lock(self, track):
        self.track_id, self.state, self.frames_lost, self._last = track.id, 'locked', 0, None
        self.feat = None if track.feat is None else track.feat.copy()

    def select_initial(self, tracks, image_size, prefer=None):
        """First lock: the track nearest to `prefer` (a pixel, e.g. where the operator taps) or to the
        image center, among confident ones."""
        if not tracks:
            return None
        W, H = image_size
        c = np.array(prefer if prefer is not None else (W / 2, H / 2))
        t = min(tracks, key=lambda tr: np.hypot(*((np.array(tr.box[:2]) + tr.box[2:]) / 2 - c)))
        self.lock(t)
        return t

    def update(self, tracks, world_pos_of, flt, t):
        """tracks: confirmed tracks of this frame; world_pos_of(track) → ground position or None;
        flt: TargetFilter (for the gate). → (track or None, how) with how ∈ {'locked', 'relocked', 'lost'}."""
        if self.track_id is None:
            return None, 'search'
        cur = next((tr for tr in tracks if tr.id == self.track_id and tr.id not in self.banned), None)
        if cur is not None and self.gate_locked:
            q = world_pos_of(cur)
            if q is not None and self._last is not None and \
                    np.linalg.norm(q - self._last[0]) > self.jump_m + self.jump_v * (t - self._last[1]):
                self.banned.add(cur.id)            # that track is now another object
                self.switches += 1
                self._last = None
                cur = None
            elif q is not None:
                self._last = (q, t)
        if cur is not None:
            self.frames_lost, self._t_lost = 0, None
            self.state = 'locked'
            if cur.feat is not None and self.feat is not None and similarity(cur.feat, self.feat) > self.sim_min:
                self.feat = 0.95 * self.feat + 0.05 * cur.feat      # slow adaptation while confident
            return cur, 'locked'
        self.frames_lost += 1
        self.state = 'lost'
        if self._t_lost is None:
            self._t_lost = t
        if self.frames_lost < self.relock_after:
            return None, 'lost'
        best, best_s = None, self.sim_min
        floor = self.gate_floor + (0.5 * self.frames_lost / 15 if self.gate_growth is None
                                   else self.gate_growth * (t - self._t_lost))
        for tr in tracks:
            if tr.id in self.banned:
                continue
            s = similarity(tr.feat, self.feat) if self.feat is not None else 0.0
            q = world_pos_of(tr)
            if q is None or not flt.gate(q, t, floor=floor):
                continue
            if s <= self.sim_min:
                continue
            score = s - self.w_dist * flt.distance(q, t) / floor if self.w_dist else s
            if best is None or score > best_s:
                best, best_s = tr, score
        if best is not None:
            self.track_id, self.state, self.frames_lost, self._t_lost = best.id, 'locked', 0, None
            q = world_pos_of(best) if self.gate_locked else None
            self._last = (q, t) if q is not None else None
            self.relocks += 1
            return best, 'relocked'
        return None, 'lost'
