"""Color-blob detector: finds balloons (or any object of one saturated color) in an RGB image.

No network: HSV threshold around the hue of the target color → small closing → connected components →
pieces split by a thin occluder (a trunk in front of the balloon) are merged → one box per blob, scored
by saturation and by how round and full it is. The same code runs on the simulated RGB image and on a
real camera (``nanouav run r20 video``); the thresholds are the only thing to tune for real light
(``nanouav run r20 pick-color`` measures them on a frame).

Detections have the format of the person detectors (``{'box', 'score'}``) plus ``'area'`` (pixels) and
``'radius_px'``.
"""
import colorsys

import cv2
import numpy as np

COLORS = {                                    # RGB reflectance of the simulated balloons
    'red': (0.85, 0.07, 0.07),
    'orange': (0.95, 0.42, 0.04),
    'yellow': (0.92, 0.85, 0.08),
    'green': (0.08, 0.72, 0.18),
    'blue': (0.08, 0.25, 0.88),
    'magenta': (0.85, 0.08, 0.70),
}


def hue_of(color):
    """RGB in [0, 1] → OpenCV hue (0–180)."""
    return colorsys.rgb_to_hsv(*color)[0] * 180.0


class BlobDetector:
    def __init__(self, color='red', hue_tol=10.0, s_min=0.65, v_min=0.12, min_area=6, max_aspect=3.0,
                 close=3, merge_gap=0.6, max_pieces=40):
        rgb = COLORS[color] if isinstance(color, str) else tuple(color)
        self.color, self.h0 = color, hue_of(rgb)
        self.hue_tol, self.s_min, self.v_min = hue_tol, s_min, v_min
        self.min_area, self.max_aspect, self.close, self.merge_gap = min_area, max_aspect, close, merge_gap
        self.max_pieces = max_pieces

    def mask(self, rgb):
        hsv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV)
        h, s, v = (hsv[..., k].astype(np.float32) for k in range(3))
        dh = np.abs(h - self.h0)
        dh = np.minimum(dh, 180.0 - dh)                    # the hue wraps around (red is at 0 and 180)
        m = (dh <= self.hue_tol) & (s >= 255 * self.s_min) & (v >= 255 * self.v_min)
        m = m.astype(np.uint8)
        if self.close > 1:
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((self.close, self.close), np.uint8))
        return m, s

    def _merge(self, boxes):
        """Merge pieces of one blob cut by a thin occluder: overlapping rows, small horizontal gap (or the
        transpose for a horizontal occluder). Pieces: [x1, y1, x2, y2, area, component]."""
        boxes = [list(b) for b in boxes]
        changed = True
        while changed:
            changed = False
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    a, b = boxes[i], boxes[j]
                    if a[5] == b[5] and a[5] >= 0:      # halves of a blob that was split on purpose
                        continue
                    size = max(a[3] - a[1], b[3] - b[1], a[2] - a[0], b[2] - b[0])
                    ov_y = min(a[3], b[3]) - max(a[1], b[1])
                    ov_x = min(a[2], b[2]) - max(a[0], b[0])
                    gap_x = max(a[0], b[0]) - min(a[2], b[2])
                    gap_y = max(a[1], b[1]) - min(a[3], b[3])
                    hy = min(a[3] - a[1], b[3] - b[1])
                    hx = min(a[2] - a[0], b[2] - b[0])
                    side = ov_y > 0.5 * hy and gap_x < self.merge_gap * size
                    vert = ov_x > 0.5 * hx and gap_y < self.merge_gap * size
                    W = max(a[2], b[2]) - min(a[0], b[0])
                    H = max(a[3], b[3]) - min(a[1], b[1])
                    round_ = max(W, H) <= 1.35 * min(W, H)    # the pieces of one disc make a square box
                    if (side or vert) and round_:
                        boxes[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]), a[4] + b[4],
                                    a[5] if a[5] == b[5] else -1]
                        boxes.pop(j)
                        changed = True
                        break
                if changed:
                    break
        return boxes

    @staticmethod
    def _split(comp):
        """Two balloons of one color that overlap in the image make one blob: split it where the distance
        transform has two separate peaks (the two centers), assigning each pixel to the nearer disc.
        → list of (mask, offset) or None."""
        dt = cv2.distanceTransform(comp, cv2.DIST_L2, 3)
        m = float(dt.max())
        if m < 2.5:
            return None
        peak = (dt >= cv2.dilate(dt, np.ones((5, 5), np.uint8))) & (dt >= 0.5 * m)
        ys, xs = np.nonzero(peak)
        order = np.argsort(-dt[ys, xs])
        P = []
        for i in order:                                   # strongest peaks, at least their radius apart
            q = (xs[i], ys[i], dt[ys[i], xs[i]])
            if all(np.hypot(q[0] - p[0], q[1] - p[1]) > 0.9 * max(q[2], p[2]) for p in P):
                P.append(q)
            if len(P) == 2:
                break
        if len(P) < 2:
            return None
        (xa, ya, ra), (xb, yb, rb) = P
        u = np.linspace(0, 1, 25)
        neck = dt[np.round(ya + u * (yb - ya)).astype(int), np.round(xa + u * (xb - xa)).astype(int)].min()
        if neck > 0.8 * min(ra, rb):                       # no waist between the peaks: one blob, not two discs
            return None
        yy, xx = np.mgrid[:comp.shape[0], :comp.shape[1]]
        near_a = np.hypot(xx - xa, yy - ya) - ra <= np.hypot(xx - xb, yy - yb) - rb
        return [(comp & near_a).astype(np.uint8), (comp & ~near_a).astype(np.uint8)]

    def detect(self, rgb, targets=None):
        m, sat = self.mask(rgb)
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        pieces = []
        keep = np.argsort(-st[1:, 4])[:self.max_pieces] + 1      # the largest blobs (noise makes thousands of specks)
        for k in sorted(keep):
            x, y, w, h, a = st[k]
            if a < max(2, self.min_area // 2):
                continue
            parts = self._split((lab[y:y + h, x:x + w] == k).astype(np.uint8)) if a >= 30 else None
            if parts is None:
                pieces.append([x, y, x + w, y + h, a, k])
                continue
            for pm in parts:                              # same `k`: the halves are never merged again
                py, px = np.nonzero(pm)
                pieces.append([x + px.min(), y + py.min(), x + px.max() + 1, y + py.max() + 1, int(pm.sum()), k])
        out = []
        for x1, y1, x2, y2, area, _ in self._merge(pieces):
            w, h = x2 - x1, y2 - y1
            if area < self.min_area or max(w, h) / max(min(w, h), 1) > self.max_aspect:
                continue
            fill = area / max(np.pi / 4 * max(w, h) ** 2, 1.0)      # 1 = a full disc
            s_mean = float(sat[y1:y2, x1:x2][m[y1:y2, x1:x2] > 0].mean()) / 255.0
            score = float(np.clip(0.25 + 0.75 * min(fill / 0.75, 1.0) * s_mean, 0.05, 0.99))
            out.append({'box': (float(x1), float(y1), float(x2), float(y2)), 'score': score, 'area': int(area),
                        'radius_px': max(w, h) / 2, 'gt': -1})
        return out
