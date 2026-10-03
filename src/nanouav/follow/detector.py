"""Person detectors. Every detector returns a list of detections ``{'box': (x1, y1, x2, y2), 'score': s}``
(pixels of the left camera image); the simulated one also gives ``'gt'`` (index of the walker, -1 for a
false positive) for the metrics.

* `YoloDetector` — ultralytics YOLO (COCO class 0 = person, or a model fine-tuned on VisDrone person,
  see `nanouav run r19 train-det`). For real images / video and for the hardware.
* `SimDetector` — in our renderer people are cylinders + spheres, which a real detector would not
  recognize; this one projects the true person into the image and degrades it like a real detector:
  misses that grow as the person gets small or occluded, box jitter, and false positives.
  The default rates are of the order of YOLO11n on VisDrone-like views; `calibrate()` can fit them to
  measured precision/recall vs. box height.
"""
import math

import numpy as np

from nanouav.follow.estimate import project
from nanouav.follow.people import BODY_R, HEAD_R, HEAD_Z, HEIGHT


class SimDetector:
    def __init__(self, h50=14.0, h_scale=4.0, p_max=0.97, vis_min=0.35, jitter=0.05, fp_rate=0.05, seed=0):
        self.h50, self.h_scale, self.p_max = h50, h_scale, p_max     # recall = p_max·σ((h − h50)/h_scale)
        self.vis_min, self.jitter, self.fp_rate = vis_min, jitter, fp_rate
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _points(w):
        a = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        ring = np.stack([w.body.cx + BODY_R * np.cos(a), w.body.cy + BODY_R * np.sin(a)], 1)
        P = [np.c_[ring, np.full(12, z)] for z in (0.0, 0.7, 1.4)]
        P.append(np.array([[w.body.cx, w.body.cy, HEIGHT], [w.body.cx + HEAD_R, w.body.cy, HEAD_Z],
                           [w.body.cx - HEAD_R, w.body.cy, HEAD_Z], [w.body.cx, w.body.cy + HEAD_R, HEAD_Z],
                           [w.body.cx, w.body.cy - HEAD_R, HEAD_Z]]))
        return np.concatenate(P)

    def visible_box(self, rig, pose, depth_gt, w):
        """True box and visible fraction of a walker (None if behind / outside the image)."""
        uv, X = project(rig, pose, self._points(w))
        if (X < 0.3).any():
            return None, 0.0
        x1, y1 = uv.min(0)
        x2, y2 = uv.max(0)
        W, H = rig.width, rig.height
        cx1, cy1, cx2, cy2 = max(x1, 0), max(y1, 0), min(x2, W - 1), min(y2, H - 1)
        if cx2 <= cx1 or cy2 <= cy1:
            return None, 0.0
        inside = (cx2 - cx1) * (cy2 - cy1) / max((x2 - x1) * (y2 - y1), 1e-6)
        dist = float(np.median(X))
        xs = np.linspace(cx1 + 0.3 * (cx2 - cx1), cx2 - 0.3 * (cx2 - cx1), 5).astype(int)
        ys = np.linspace(cy1 + 0.1 * (cy2 - cy1), cy2 - 0.1 * (cy2 - cy1), 7).astype(int)
        d = depth_gt[np.ix_(ys, xs)]
        unocc = ((d == 0) | (d > dist - 0.4)).mean()          # 0 = beyond the range: not an occluder
        return (cx1, cy1, cx2, cy2), float(inside * unocc)

    def detect(self, cam, walkers):
        rig, pose = cam['rig'], cam['pose']
        dets = []
        for i, w in enumerate(walkers):
            box, vis = self.visible_box(rig, pose, cam['depth_gt'], w)
            if box is None or vis < self.vis_min:
                continue
            h = box[3] - box[1]
            p = self.p_max / (1 + math.exp(-(h - self.h50) / self.h_scale)) * min(1.0, vis / 0.7)
            if self.rng.random() > p:
                continue
            bw, bh = box[2] - box[0], h
            j = self.rng.normal(0, self.jitter, 4) * np.array([bw, bh, bw, bh])
            b = np.array(box) + j
            dets.append({'box': tuple(np.clip(b, 0, [rig.width - 1, rig.height - 1] * 2)),
                         'score': float(np.clip(0.35 + 0.6 * p + self.rng.normal(0, 0.05), 0.05, 0.99)), 'gt': i})
        for _ in range(self.rng.poisson(self.fp_rate)):
            h = self.rng.uniform(10, 40)
            x, y = self.rng.uniform(0, rig.width - h / 2), self.rng.uniform(0, rig.height - h)
            dets.append({'box': (x, y, x + h / 2.5, y + h), 'score': float(self.rng.uniform(0.2, 0.6)), 'gt': -1})
        return dets


class YoloDetector:
    """ultralytics YOLO restricted to people. `model` can be a .pt, an exported TensorRT .engine or ONNX."""

    def __init__(self, model='yolo11n.pt', conf=0.15, imgsz=640, device=None, person_class=0):
        from ultralytics import YOLO
        self.model = YOLO(model)
        self.conf, self.imgsz, self.device = conf, imgsz, device
        names = getattr(self.model, 'names', {0: 'person'}) or {0: 'person'}
        # COCO models: class 0 = person; person-only fine-tuned models: their only class
        self.cls = person_class if len(names) > 1 else 0

    def detect(self, image, walkers=None):
        img = image if image.ndim == 3 else np.repeat(image[..., None], 3, -1)
        r = self.model.predict(img, conf=self.conf, imgsz=self.imgsz, classes=[self.cls], verbose=False,
                               device=self.device)[0]
        out = []
        for b, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            out.append({'box': tuple(float(x) for x in b), 'score': float(s), 'gt': -1})
        return out
