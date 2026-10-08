"""Flightmare arena: the scene's static geometry (scanned once) + obstacles we spawn with exact geometry.

This is how the paper's simulation is set up (agile_autonomy: an empty forest scene with trees spawned at
known positions, whose point cloud is the expert's map). The public 2022 player cannot spawn trees, but it
moves and scales "dynamic objects" every frame; the wooden crate prefab rpg_box02 (BoxCollider 0.81 m,
centred) scaled to 0.3–0.8 × 0.3–0.8 × 6 m is a trunk whose geometry we know exactly. So:

  · the static scene (agile_flight's warehouse by default: floor, walls, roof) is mapped once with
    `scan` (reliable there: no foliage, no water) → an occupancy grid → distance grid (`GridSDF`);
  · every environment spawns its own Poisson forest of trunks in the free part of the scene; the map is
    the static grid ∪ the boxes, both exact enough to label (expert) and to referee (crashes);
  · the floor is kept apart, like the ground plane of our simulator: Field(ground=False) — the global
    path and the local distance map of the labels — sees obstacles only.

`Arena` is accepted by geometry.Field like a CloudWorld (split_ground, local).
"""
import json
import math

import numpy as np

from nanouav.scene.renderer import Box


class GridSDF:
    """Distance (m) to the nearest occupied voxel of a point set, on a regular grid (scipy EDT).
    Outside the grid: 0 (unknown space is not flyable)."""

    def __init__(self, points, res, lo, hi):
        from scipy.ndimage import distance_transform_edt
        self.res, self.lo = float(res), np.asarray(lo, float)
        self.shape = tuple(np.maximum(np.ceil((np.asarray(hi, float) - self.lo) / res).astype(int), 1))
        occ = np.zeros(self.shape, bool)
        if len(points):
            i = np.floor((np.asarray(points, float) - self.lo) / res).astype(int)
            k = np.all((i >= 0) & (i < self.shape), 1)
            occ[tuple(i[k].T)] = True
        if occ.any():
            d = distance_transform_edt(~occ) * res
        else:
            d = np.full(self.shape, 1e3)
        self.d = d.astype(np.float32)

    def __call__(self, P):
        P = np.asarray(P, float).reshape(-1, 3)
        i = np.floor((P - self.lo) / self.res).astype(int)
        inside = np.all((i >= 0) & (i < self.shape), 1)
        out = np.zeros(len(P))
        j = i[inside]
        out[inside] = self.d[j[:, 0], j[:, 1], j[:, 2]]
        return out


def _boxes_arrays(boxes):
    C = np.array([b.center for b in boxes], float).reshape(-1, 3)
    H = np.array([b.half for b in boxes], float).reshape(-1, 3)
    return C, H


def box_distance(P, C, H):
    """Signed distance of points P (N, 3) to axis-aligned boxes (centres C, half extents H) → (N,)."""
    if len(C) == 0:
        return np.full(len(P), np.inf)
    out = np.full(len(P), np.inf)
    for s in range(0, len(P), 20000):
        Q = np.abs(P[s:s + 20000, None, :] - C[None]) - H[None]
        o = np.linalg.norm(np.maximum(Q, 0), axis=-1) + np.minimum(Q.max(-1), 0)
        out[s:s + 20000] = o.min(1)
    return out


class Arena:
    split_ground = True

    def __init__(self, sdf, floor, boxes=(), points=None, name='arena', meta=None):
        self.sdf, self.floor, self.name = sdf, float(floor), name
        self.objects = list(boxes)
        self.C, self.H = _boxes_arrays(self.objects)
        self.points = np.zeros((0, 3)) if points is None else points       # static voxels (drawing)
        self.meta = meta or {}
        self.ground = 'floor'                       # not None: Field(ground=True) adds the floor
        self.res, self.inflate, self.hits = sdf.res, 0.0, None

    def with_boxes(self, boxes, name=None):
        return Arena(self.sdf, self.floor, boxes, self.points, name or self.name, self.meta)

    def local(self, center, radius):
        """View with only the boxes that can be within `radius` of `center` (faster expert queries)."""
        c = np.asarray(center, float)
        if len(self.C):
            near = np.linalg.norm(np.maximum(np.abs(self.C - c) - self.H, 0), axis=1) < radius
            boxes = [b for b, k in zip(self.objects, near) if k]
        else:
            boxes = []
        a = object.__new__(Arena)
        a.__dict__.update(self.__dict__)
        a.objects = boxes
        a.C, a.H = _boxes_arrays(boxes)
        return a

    def distance(self, P, ground=True):
        P = np.asarray(P, float).reshape(-1, 3)
        d = np.minimum(self.sdf(P), box_distance(P, self.C, self.H))
        if ground:
            d = np.minimum(d, P[:, 2] - self.floor)
        return d

    def clearance(self, p):
        return float(self.distance(np.asarray(p, float)[None])[0])

    def ground_z(self, x, y):
        return np.full(np.shape(x), self.floor)


# ── building the arena from a scan ─────────────────────────────────────────
def floor_height(points, res):
    """Most populated low horizontal layer of the scanned voxels (the floor; ± half a voxel)."""
    z = points[:, 2]
    lo = np.percentile(z, 1)
    zz = z[z < lo + 3.0]
    k = np.round((zz - zz.min()) / res).astype(int)
    mode = int(np.bincount(k).argmax())
    return float(zz.min() + mode * res)


def largest_free_rect(free, cell):
    """Largest axis-aligned rectangle of True cells (histogram method) → (i0, i1, j0, j1) inclusive."""
    nx, ny = free.shape
    best, out = 0, (0, 0, 0, 0)
    h = np.zeros(ny, int)
    for i in range(nx):
        h = np.where(free[i], h + 1, 0)
        stack = []
        for j in range(ny + 1):
            cur = h[j] if j < ny else 0
            start = j
            while stack and stack[-1][1] >= cur:
                s, hh = stack.pop()
                area = hh * (j - s)
                if area > best:
                    best, out = area, (i - hh + 1, i, s, j - 1)
                start = s
            stack.append((start, cur))
    return out


def build(points, res, box_lo, box_hi, clear=1.2, cell=0.5, z_band=(0.5, 4.0)):
    """Scanned voxels of the empty scene → (GridSDF of the obstacles without the floor, floor height,
    free rectangle [x0, x1, y0, y1] where everything between floor + z_band is ≥ clear from the scene)."""
    floor = floor_height(points, res)
    obst = points[points[:, 2] > floor + 2.5 * res]
    lo = np.asarray(box_lo, float) - 1.0
    hi = np.asarray(box_hi, float) + 1.0
    sdf = GridSDF(obst, res, lo, hi)
    xs = np.arange(box_lo[0] + cell / 2, box_hi[0], cell)
    ys = np.arange(box_lo[1] + cell / 2, box_hi[1], cell)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    free = np.ones(X.shape, bool)
    for z in np.arange(floor + z_band[0], floor + z_band[1] + 1e-6, 0.5):
        free &= sdf(np.c_[X.ravel(), Y.ravel(), np.full(X.size, z)]).reshape(X.shape) >= clear
    i0, i1, j0, j1 = largest_free_rect(free, cell)
    rect = [float(xs[i0] - cell / 2), float(xs[i1] + cell / 2), float(ys[j0] - cell / 2), float(ys[j1] + cell / 2)]
    return sdf, floor, rect


def save(fn, points, res, floor, rect, box_lo, box_hi, meta):
    np.savez_compressed(fn, points=points.astype(np.float32), res=res, floor=floor, rect=np.asarray(rect),
                        box_lo=np.asarray(box_lo), box_hi=np.asarray(box_hi), meta=json.dumps(meta))


def load(fn):
    z = np.load(fn)
    P, res = z['points'].astype(float), float(z['res'])
    floor = float(z['floor'])
    obst = P[P[:, 2] > floor + 2.5 * res]
    sdf = GridSDF(obst, res, z['box_lo'] - 1.0, z['box_hi'] + 1.0)
    meta = json.loads(str(z['meta']))
    meta['rect'] = z['rect'].tolist()
    k = (P[:, 2] > floor + 0.5) & (P[:, 2] < floor + 3.0)
    return Arena(sdf, floor, (), P[k], meta.get('scene', 'arena'), meta)


# ── environments: a Poisson forest of trunks in the free rectangle ─────────
def arena_env(arena, i, base_seed=0, kind='train', length=40.0, n_max=160, test_density=1 / 25):
    """Environment i: trunks (boxes) with spacing δ ∈ {4, 5, 6, 7} m (train) or density 1/25 m⁻² (test),
    0.3–0.8 m wide (test 0.5–0.7 m, the 0.6 m trees of the paper), from the floor to 6 m; start 3 m inside
    the free rectangle, goal `length` m along +x. The global path must exist (else the next sub-seed)."""
    from nanouav.agile.envs import Env
    from nanouav.agile.expert import global_plan
    x0, x1, y0, y1 = arena.meta['rect']
    L = min(length, x1 - x0 - 8.0)
    if L < 15:
        raise RuntimeError(f'the free part of the arena is too short ({x1 - x0:.1f} m)')
    for sub in range(50):
        rng = np.random.default_rng([base_seed, i, sub, 0xF1A])
        if kind == 'train':
            delta = int(rng.integers(4, 8))
            dens, wmin, wmax = 1 / delta ** 2, 0.3, 0.8
            y_s = float(rng.uniform(y0 + 2.5, y1 - 2.5)) if y1 - y0 > 5 else (y0 + y1) / 2
            z_s = arena.floor + float(rng.uniform(1.2, 2.5))
        else:
            delta, dens, wmin, wmax = 5, test_density, 0.5, 0.7
            y_s, z_s = (y0 + y1) / 2, arena.floor + 1.5
        start = np.array([x0 + 3.0, y_s, z_s])
        goal = start + [L, 0.0, 0.0]
        area = (x1 - x0 - 2) * (y1 - y0 - 1)
        n = rng.poisson(dens * area)
        xy = np.c_[rng.uniform(x0 + 1, x1 - 1, n), rng.uniform(y0 + 0.5, y1 - 0.5, n)]
        xy = xy[np.hypot(xy[:, 0] - start[0], xy[:, 1] - start[1]) > 2.5]
        if len(xy) > n_max:                                # keep the trunks nearest to the flight line
            xy = xy[np.argsort(np.abs(xy[:, 1] - start[1]))[:n_max]]
        boxes = []
        for x, y in xy:
            w = float(rng.uniform(wmin, wmax))
            h = 6.0
            boxes.append(Box((float(x), float(y), arena.floor + h / 2), (w / 2, w / 2, h / 2), 0.0))
        world = arena.with_boxes(boxes, f'{arena.name}_{kind}{i}')
        e = Env(world.name, world, start, goal, {'delta': delta, 'n': len(boxes), 'sub': sub}, kind != 'train')
        try:
            global_plan(e, 3.0, straight=False)
        except RuntimeError:
            continue
        return e
    raise RuntimeError(f'no environment with a global path for {kind} {i}')
