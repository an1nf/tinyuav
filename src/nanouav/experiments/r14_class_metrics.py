"""
R14 · Per-class/per-region depth metrics (inspired by DDOS, Kolbeinsson & Mikolajczyk 2024,
AbsRel_class) adapted to OUR regime: 96×96, 0.2–4 m, flight at 0.65 m.

DDOS shows that models with a good global AbsRel (DepthFormer 0.136) fail completely on ultra-thin
structures (AbsRel ≈ 1.0). The question here: what does our global AbsRel (M ≈ 0.08–0.15) hide?

Regions (exact masks by re-rendering with/without each element):
  wire   horizontal bar Ø 2 cm (cable)    pole   vertical pole r 1.5–5 cm (thinner than in training)
  gap    pixels seen THROUGH an opening   edge   2 px band at depth discontinuities
  obst   remaining obstacles              ground ground
Besides AbsRel and δ1, two asymmetric SAFETY metrics (what matters to the planner):
  over  = P(pred > 1.25·gt)  → the obstacle looks farther than it is (dangerous: collision)
  under = P(pred < 0.8·gt)   → something appears closer (phantom; in 'gap' = the gap closes)
Usage: nanouav run r14 --n 600 --variants M,S,XS
"""
import os, sys, json, time, argparse
import numpy as np
import cv2
from nanouav import paths
ROOT = str(paths.repo_root())
from nanouav.config import norm_to_m, D_MAX, D_MIN
from nanouav.scene.renderer import sample_world, render_view, Box, Cylinder, Texture, World
from nanouav.sensors.tof import tof_8x8
from nanouav.perception.cnn import predict_depth_norm
from nanouav.perception.tof_calibrator import ToFCalibrator, fuse_tof
from nanouav.config import ZONE_BOXES, TOF_GRID


def fuse_tof2(depth_m, tof, pct=25.0, lo=0.5, hi=1.0):
    """ToF↔CNN consistency in BOTH directions (R14). The ToF reports the low percentile of its zone: if the
    same CNN percentile in that zone lies FARTHER than the reading, the CNN 'does not see' the object that the
    ToF does see (over-estimation = dangerous) → the zone is rescaled towards the reading (factor ≥ lo).
    The opposite direction (nearby phantoms) is handled by fuse_tof (lower bound 0.8·tof)."""
    out = fuse_tof(depth_m, tof)
    for gy in range(TOF_GRID):
        for gx in range(TOF_GRID):
            t = float(tof[gy, gx])
            if t >= D_MAX - 0.1:
                continue
            y0, y1, x0, x1 = ZONE_BOXES[gy][gx]
            z = out[y0:y1, x0:x1]
            q = float(np.percentile(z, pct))
            if q > 1.1 * t:
                z *= max(lo, min(hi, t / q))
    return out

CLASSES = ['wire', 'pole', 'gap', 'edge', 'obst', 'ground']
BINS = [(0.2, 1.0), (1.0, 2.0), (2.0, 4.0)]
POSE = ((0.0, 0.0, 0.65), 0.0)


def _tex(rng):
    return Texture(str(rng.choice(['uniform', 'noise'])), albedo=rng.uniform(0.15, 0.9),
                   albedo2=rng.uniform(0.1, 0.5), scale=rng.uniform(0.1, 0.4), seed=int(rng.integers(1 << 30)))


def make_sample(rng):
    w = sample_world(rng)
    base = list(w.objects)
    extra = {'wire': [], 'pole': [], 'gapwall': [], 'gapfill': []}
    if rng.random() < 0.5:
        for _ in range(int(rng.integers(1, 3))):
            x = rng.uniform(0.6, 3.0)
            extra['pole'].append(Cylinder(x, rng.uniform(-0.5, 0.5) * x, rng.uniform(0.015, 0.05), 0.0, 2.0, _tex(rng)))
    if rng.random() < 0.4:
        x = rng.uniform(0.8, 3.0)
        extra['wire'].append(Box((x, 0.0, rng.uniform(0.45, 1.1)), (0.01, 3.0, 0.01), rng.uniform(-0.4, 0.4), _tex(rng)))
    if rng.random() < 0.45:
        x, g, yc = rng.uniform(1.0, 3.0), rng.uniform(0.6, 1.4), rng.uniform(-0.3, 0.3)
        t = _tex(rng)
        lo, hi = yc - g / 2, yc + g / 2
        extra['gapwall'] = [Box((x, (lo - 4) / 2, 1.0), (0.08, (lo + 4) / 2, 1.0), 0.0, t),
                            Box((x, (hi + 4) / 2, 1.0), (0.08, (4 - hi) / 2, 1.0), 0.0, t)]
        extra['gapfill'] = [Box((x, yc, 1.0), (0.08, g / 2, 1.0), 0.0, t)]
        # remove base obstacles that fall inside the wall
        base = [o for o in base if not (abs(getattr(o, 'center', (getattr(o, 'cx', 0),))[0] - x) < 0.4)]
    return w, base, extra


def render_depth(w, objs, rng_seed):
    W = World(objs, w.ground, w.sun, w.ambient, w.diffuse, w.sky)
    g, d = render_view(W, POSE[0], POSE[1], 0, 0, rng=np.random.default_rng(rng_seed))
    return g, d


def masks_for(w, base, extra, seed):
    full = base + extra['pole'] + extra['wire'] + extra['gapwall']
    gray, d = render_depth(w, full, seed)
    m = {}
    eps = 0.02
    _, d_nopole = render_depth(w, base + extra['wire'] + extra['gapwall'], seed)
    m['pole'] = d < d_nopole - eps
    _, d_nowire = render_depth(w, base + extra['pole'] + extra['gapwall'], seed)
    m['wire'] = d < d_nowire - eps
    if extra['gapwall']:
        _, d_fill = render_depth(w, full + extra['gapfill'], seed)
        m['gap'] = d > d_fill + eps                    # seen through the opening
    else:
        m['gap'] = np.zeros_like(d, bool)
    _, d_empty = render_depth(w, [], seed)
    obj = d < d_empty - eps
    ld = np.log(np.clip(d, D_MIN, D_MAX))
    gx = np.abs(cv2.Sobel(ld, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(ld, cv2.CV_32F, 0, 1, ksize=3))
    disc = (gx + gy) > 0.6
    edge = cv2.dilate(disc.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    taken = m['pole'] | m['wire'] | m['gap']
    m['edge'] = edge & ~taken
    m['obst'] = obj & ~taken & ~m['edge']
    m['ground'] = ~obj & ~taken & ~m['edge']
    valid = d < D_MAX - 0.05
    m = {k: v & valid for k, v in m.items()}
    return gray, d, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=600)
    ap.add_argument('--variants', default='M,S,XS')
    ap.add_argument('--seed', type=int, default=1400)
    a = ap.parse_args()
    variants = a.variants.split(',')
    rng = np.random.default_rng(a.seed)
    acc = {v: {mode: {c: {b: [] for b in range(len(BINS))} for c in CLASSES} for mode in ('raw', 'flight', 'flight2')}
           for v in variants}
    glob = {v: {'raw': [], 'flight': [], 'flight2': []} for v in variants}
    t0 = time.time()
    for i in range(a.n):
        w, base, extra = make_sample(rng)
        gray, gt, m = masks_for(w, base, extra, a.seed + i)
        tof = tof_8x8(gt, rng=rng)
        for v in variants:
            dn = predict_depth_norm(gray, v, tof=tof)
            raw = norm_to_m(dn)
            cal = ToFCalibrator()
            for _ in range(4):
                fl = cal.update_and_scale(dn, tof)
            fl2 = fuse_tof2(fl, tof)
            fl = fuse_tof(fl, tof)
            for mode, pred in (('raw', raw), ('flight', fl), ('flight2', fl2)):
                vm = gt < D_MAX - 0.05
                glob[v][mode].append(np.abs(pred[vm] - gt[vm]) / gt[vm])
                for c in CLASSES:
                    for bi, (lo, hi) in enumerate(BINS):
                        mm = m[c] & (gt >= lo) & (gt < hi)
                        if mm.any():
                            acc[v][mode][c][bi].append(np.stack([gt[mm], pred[mm]], 1))
        if (i + 1) % 100 == 0:
            print(f'{i + 1}/{a.n} ({time.time() - t0:.0f}s)', flush=True)
    out = {}
    for v in variants:
        out[v] = {}
        for mode in ('raw', 'flight', 'flight2'):
            g = np.concatenate(glob[v][mode])
            out[v][mode] = {'global_absrel': float(g.mean())}
            for c in CLASSES:
                out[v][mode][c] = {}
                for bi, (lo, hi) in enumerate(BINS):
                    L = acc[v][mode][c][bi]
                    if not L:
                        continue
                    X = np.concatenate(L)
                    gt_, p = X[:, 0], X[:, 1]
                    r = p / gt_
                    out[v][mode][c][f'{lo:g}-{hi:g}'] = {
                        'n_px': int(len(gt_)), 'absrel': float(np.mean(np.abs(p - gt_) / gt_)),
                        'd1': float(np.mean(np.maximum(r, 1 / r) < 1.25)),
                        'over': float(np.mean(r > 1.25)), 'under': float(np.mean(r < 0.8))}
    os.makedirs(str(paths.results_dir()), exist_ok=True)
    json.dump(out, open(os.path.join(str(paths.results_dir()), 'r14_class_metrics.json'), 'w'), indent=1)
    for v in variants:
        for mode in ('raw', 'flight', 'flight2'):
            print(f'\n== {v} · {mode} · global AbsRel {out[v][mode]["global_absrel"]:.3f}')
            print(f'{"class":7s} {"range":7s} {"px":>7s} {"AbsRel":>7s} {"δ1":>6s} {"over":>6s} {"under":>6s}')
            for c in CLASSES:
                for k, s in out[v][mode][c].items():
                    print(f'{c:7s} {k:7s} {s["n_px"]:7d} {s["absrel"]:7.3f} {s["d1"]:6.2f} {s["over"]:6.2f} {s["under"]:6.2f}')


if __name__ == '__main__':
    main()
