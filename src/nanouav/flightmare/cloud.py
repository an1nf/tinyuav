"""The Unity scene as the privileged map: point cloud → distance field, ground height, start/goal lines.

The paper gives the expert the environment as a point cloud exported from Unity (Flightmare's
`PointCloud` request). Here `CloudWorld` wraps those points so that every place that builds a
`geometry.Field(world)` — the collision check, the global path, the M-H expert, the DAgger labels — works
unchanged (k-d tree distance to the nearest occupied voxel).

What Unity writes (SavePointCloud.cs): the centres of the voxels of the requested box for which
Physics.OverlapSphere(centre, resolution) touches a collider, absolute coordinates, ROS order (x, y, z up).
A voxel is occupied up to one resolution away from a surface, so the distance to the nearest occupied
centre is already a lower bound of the true clearance (it underestimates it by less than a resolution).
`to_ros` still checks the up axis and the offset against the requested box (the file carries no frame).
"""
import json
import os

import numpy as np


def read_ply(fn):
    """Vertices (N, 3) of an ascii or binary (little/big endian) .ply."""
    with open(fn, 'rb') as f:
        header, props, n, fmt = [], [], 0, 'ascii'
        while True:
            line = f.readline().decode('ascii', 'ignore').strip()
            header.append(line)
            if line.startswith('format'):
                fmt = line.split()[1]
            elif line.startswith('element vertex'):
                n = int(line.split()[2])
                props = []
                in_vertex = True
            elif line.startswith('element'):
                in_vertex = False
            elif line.startswith('property') and in_vertex:
                props.append(line.split()[1:])
            if line == 'end_header':
                break
        names = [p[-1] for p in props]
        if fmt == 'ascii':
            data = np.loadtxt(f, max_rows=n, ndmin=2)
            return data[:, [names.index(a) for a in 'xyz']].astype(np.float64)
        tmap = {'float': 'f4', 'float32': 'f4', 'double': 'f8', 'float64': 'f8', 'uchar': 'u1', 'uint8': 'u1',
                'char': 'i1', 'int8': 'i1', 'ushort': 'u2', 'short': 'i2', 'int': 'i4', 'uint': 'u4', 'int32': 'i4'}
        end = '<' if 'little' in fmt else '>'
        dt = np.dtype([(p[-1], end + tmap[p[0]]) for p in props])
        arr = np.frombuffer(f.read(n * dt.itemsize), dt, count=n)
        return np.stack([arr['x'], arr['y'], arr['z']], 1).astype(np.float64)


def to_ros(P, origin, size):
    """Points as written by Unity → ROS world frame, by matching the requested box. → (points, info)."""
    P = np.asarray(P, float)
    origin, size = np.asarray(origin, float), np.asarray(size, float)
    spread = np.percentile(P, 99.5, axis=0) - np.percentile(P, 0.5, axis=0)
    # which column is up (ROS z): the one whose spread best matches size[2], the vertical extent asked for
    up = int(np.argmin(np.abs(spread - size[2]) / size[2]))
    if up == 1:                                            # Unity order (x, y_up, z) → ROS (x, z, y)
        Q, frame = P[:, [0, 2, 1]], 'unity'
    else:
        Q, frame = P.copy(), 'ros'
    lo, hi = Q.min(0), Q.max(0)
    want_lo = origin - size / 2
    shift = np.zeros(3)
    for k in range(3):                                     # relative to the request origin? → shift it back
        if not (want_lo[k] - 1.0 <= lo[k] and hi[k] <= want_lo[k] + size[k] + 1.0):
            shift[k] = (origin[k] - (lo[k] + hi[k]) / 2) if k < 2 else (want_lo[k] - lo[k])
    Q = Q + shift
    return Q, {'frame': frame, 'up_column': up, 'shift': shift.round(3).tolist(), 'spread': spread.round(2).tolist()}


class CloudWorld:
    """Stand-in for a renderer World: no primitives, a point cloud (geometry.Field reads `points`)."""

    def __init__(self, points, resolution=0.15, name='flightmare', inflate=0.0):
        """inflate: subtracted from the nearest-point distance (0 for Unity's PointCloud, whose voxels are
        already inflated one resolution; res / 2 for surface points voxelised by `scan`)."""
        from scipy.spatial import cKDTree
        self.points = np.asarray(points, float)
        self.res, self.name, self.inflate = float(resolution), name, float(inflate)
        self.tree = cKDTree(self.points)
        # outside the mapped box nothing is known: treat its vertical sides as walls (no plan leaves the map)
        self.lo, self.hi = self.points[:, :2].min(0), self.points[:, :2].max(0)
        self.objects, self.ground = [], None
        self.sun, self.ambient, self.diffuse, self.sky = (0.5, -0.1, 0.9), 0.35, 0.75, (0.72, 0.6)
        self._ground = None

    def distance(self, P):
        P = np.asarray(P, float).reshape(-1, 3)
        d, _ = self.tree.query(P, k=1, distance_upper_bound=30.0)
        d = np.where(np.isfinite(d), d, 30.0) - self.inflate
        wall = np.minimum(P[:, :2] - self.lo, self.hi - P[:, :2]).min(1)
        return np.minimum(d, wall)

    def clearance(self, p):
        return float(self.distance(np.asarray(p, float)[None])[0])

    # ground height per cell: the top of the run of voxels that starts at the lowest one (a surface-only terrain
    # gives a thin sheet, a solid one a block: both end at the surface), minus the occupancy radius; a 3 × 3
    # median removes the cells where a trunk continues the run upwards
    def ground_map(self, cell=1.0):
        if self._ground is None:
            P, r = self.points, self.res
            x0, y0 = P[:, 0].min(), P[:, 1].min()
            ix, iy = ((P[:, 0] - x0) / cell).astype(int), ((P[:, 1] - y0) / cell).astype(int)
            nx, ny = ix.max() + 1, iy.max() + 1
            c = ix * ny + iy
            k = np.round((P[:, 2] - P[:, 2].min()) / r).astype(int)
            o = np.lexsort((k, c))
            c, k = c[o], k[o]
            first = np.r_[True, c[1:] != c[:-1]]
            gap = np.r_[False, (k[1:] - k[:-1] > 2) & ~first[1:]]
            run = np.cumsum(first | gap)                       # run id; a cell's first run is its first id
            start = np.flatnonzero(first)
            first_run = np.repeat(run[start], np.diff(np.r_[start, len(c)]))
            keep = run == first_run
            top = np.full(nx * ny, -np.inf)
            np.maximum.at(top, c[keep], k[keep])
            G = np.where(np.isfinite(top), top * r + P[:, 2].min() - r, np.nan).reshape(nx, ny)
            Gp = np.pad(G, 1, constant_values=np.nan)
            stack = np.stack([Gp[i:i + nx, j:j + ny] for i in range(3) for j in range(3)])
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                M = np.nanmedian(stack, 0)
            self._ground = (x0, y0, cell, np.where(np.isfinite(G), M, np.nan))
        return self._ground

    def ground_z(self, x, y):
        x0, y0, cell, G = self.ground_map()
        i = np.clip(((np.asarray(x) - x0) / cell).astype(int), 0, G.shape[0] - 1)
        j = np.clip(((np.asarray(y) - y0) / cell).astype(int), 0, G.shape[1] - 1)
        return G[i, j]

    def save(self, fn, meta=None):
        np.savez_compressed(fn, points=self.points.astype(np.float32), res=self.res, inflate=self.inflate,
                            meta=json.dumps(meta or {}))

    @staticmethod
    def load(fn):
        z = np.load(fn)
        w = CloudWorld(z['points'], float(z['res']), os.path.splitext(os.path.basename(fn))[0],
                       float(z['inflate']) if 'inflate' in z.files else 0.0)
        w.meta = json.loads(str(z['meta']))
        return w


def find_lines(world, n=10, length=40.0, z_above=1.5, clear=1.2, max_terrain=1.0, seed=0, heading=0.0,
               margin=5.0, check_v=3.0, max_tries=None, log=None):
    """Start → goal segments of `length` m inside the cloud: clear start and goal, terrain variation along
    the line ≤ max_terrain, a global path with the full inflation (no fine-grid fallback, which accepted
    paths 2 cm from the cloud), and (check_v > 0) the privileged global pilot τ_gbl flying it at check_v
    without crashing. Only heading ≈ 0 is safe: the global planner's goal is the plane x ≥ goal.x.
    → list of dicts (start, goal, ground, heading)."""
    from nanouav.agile.envs import Env
    from nanouav.agile.expert import _global_plan, RQ
    rng = np.random.default_rng(seed)
    P = world.points
    lo, hi = P.min(0) + margin, P.max(0) - margin
    out, tries, stats = [], 0, {'terrain': 0, 'clear': 0, 'plan': 0, 'gbl': 0}
    max_tries = max_tries or 400 * n
    while len(out) < n and tries < max_tries:
        tries += 1
        h = heading
        d = np.array([np.cos(h), np.sin(h)])
        a = rng.uniform(lo[:2], hi[:2])
        b = a + d * length
        if not (lo[0] <= b[0] <= hi[0] and lo[1] <= b[1] <= hi[1]):
            continue
        if any(np.linalg.norm(np.asarray(o['start'][:2]) - a) < 4.0 for o in out):
            continue
        s = np.linspace(0, 1, 41)[:, None]
        xy = a + s * (b - a)
        g = world.ground_z(xy[:, 0], xy[:, 1])
        if np.isnan(g).any() or np.ptp(g) > max_terrain:
            stats['terrain'] += 1
            continue
        z = float(np.median(g) + z_above)
        A, B = np.r_[a, z], np.r_[b, z]
        if world.clearance(A) < clear or world.clearance(B) < clear:
            stats['clear'] += 1
            continue
        line = {'start': A.round(3).tolist(), 'goal': B.round(3).tolist(), 'ground': round(float(np.median(g)), 3),
                'heading': round(float(h), 4)}
        env = line_env(world, line, len(out))
        try:                                           # the full inflation only (no fine-grid fallback)
            _global_plan(env, 3.0, RQ, 0.2, 0.15, 4.0)
        except RuntimeError:
            stats['plan'] += 1
            continue
        if check_v:
            from nanouav.agile.rollout import Config, Rollout
            r = Rollout(env, Config(v_des=check_v, seed=0), 'gbl').run()[0]
            if not r['success']:
                stats['gbl'] += 1
                if log:
                    log(f"line candidate rejected: gbl {'crash' if r['crashed'] else 'fail'} at {r['time_s']} s")
                continue
            line['gbl_min_clearance'] = r['min_clearance_m']
        out.append(line)
        if log:
            log(f'line {len(out)}/{n} after {tries} tries')
    if log:
        log(f'rejected: {stats} ({tries} tries)')
    return out


def line_env(world, line, i=0):
    """Env of a start → goal line. Its goal 'line' is the plane through the goal normal to the segment
    (Env measures progress along x: lines with another heading are flown in a frame turned by it)."""
    from nanouav.agile.envs import Env
    return Env(f'{world.name}_l{i}', world, np.asarray(line['start'], float), np.asarray(line['goal'], float),
               {'ground': line.get('ground', 0.0)}, True)
