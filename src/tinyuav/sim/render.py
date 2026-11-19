"""Ray-cast renderer in the world frame (FLU: x forward, y left, z up): gray image, z-depth, optional color."""
import math
from dataclasses import dataclass, field

import numpy as np

_EPS = 1e-9
_INF = np.inf
Z_MIN = 0.2


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], np.float64)


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float64)


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float64)


def body_rot(yaw, pitch, roll):
    """Body-to-world rotation, ZYX; pitch > 0 is nose down."""
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


_RAY_CACHE = {}


def camera_rays(hfov, size):
    """Unit rays in the body frame (H, W, 3) and the cosine to the optical axis; size = (H, W)."""
    h, w = size
    key = (hfov, h, w)
    if key in _RAY_CACHE:
        return _RAY_CACHE[key]
    f = (w / 2.0) / math.tan(hfov / 2.0)
    u = (np.arange(w) + 0.5 - w / 2) / f
    v = (np.arange(h) + 0.5 - h / 2) / f
    U, V = np.meshgrid(u, v)
    d = np.stack([np.ones_like(U), -U, -V], axis=-1)
    n = np.linalg.norm(d, axis=-1, keepdims=True)
    d = d / n
    cos_ax = (1.0 / n[..., 0])
    _RAY_CACHE[key] = (d, cos_ax)
    return d, cos_ax


@dataclass
class Texture:
    kind: str = 'uniform'          # 'uniform' | 'checker' | 'noise' | 'stripes'
    albedo: float = 0.6
    albedo2: float = 0.3
    scale: float = 1.0             # cell size (m)
    extent: float = 1e9            # checker only inside |x|,|y| < extent
    seed: int = 0
    color: tuple = None            # RGB reflectance, used only for color output
    _grid: np.ndarray = field(default=None, repr=False)

    def sample(self, P):
        if self.kind == 'uniform':
            return np.full(P.shape[:-1], self.albedo, np.float32)
        if self.kind == 'checker':
            ix = np.floor(P[..., 0] / self.scale + 0.5)
            iy = np.floor(P[..., 1] / self.scale + 0.5)
            dark = ((ix + iy) % 2 == 0)
            inside = (np.abs(P[..., 0]) < self.extent) & (np.abs(P[..., 1]) < self.extent)
            return np.where(dark & inside, self.albedo2, self.albedo).astype(np.float32)
        if self.kind == 'stripes':
            s = np.sin(2 * np.pi * (P[..., 0] + P[..., 1] + P[..., 2]) / self.scale)
            return np.where(s > 0, self.albedo, self.albedo2).astype(np.float32)
        # 'noise': bilinear value noise over a periodic random grid
        if self._grid is None:
            self._grid = np.random.default_rng(self.seed).random((32, 32)).astype(np.float32)
        g = self._grid
        x = (P[..., 0] + P[..., 2]) / self.scale
        y = (P[..., 1] + P[..., 2] * 0.7) / self.scale
        x0, y0 = np.floor(x), np.floor(y)
        fx, fy = x - x0, y - y0
        x0 = x0.astype(np.int64) % 32
        y0 = y0.astype(np.int64) % 32
        x1, y1 = (x0 + 1) % 32, (y0 + 1) % 32
        v = (g[y0, x0] * (1 - fx) * (1 - fy) + g[y0, x1] * fx * (1 - fy) +
             g[y1, x0] * (1 - fx) * fy + g[y1, x1] * fx * fy)
        return (self.albedo2 + (self.albedo - self.albedo2) * v).astype(np.float32)


def _safe(d):
    return np.where(np.abs(d) < _EPS, _EPS, d)


@dataclass
class Plane:
    """Ground plane z = 0."""
    tex: Texture = field(default_factory=Texture)
    casts_shadow = False

    def intersect(self, O, D):
        O = np.broadcast_to(np.asarray(O, np.float64), D.shape)
        dz = D[..., 2]
        t = np.where(dz < -1e-6, -O[..., 2] / _safe(dz), _INF)
        t = np.where(t > 1e-4, t, _INF)
        n = np.zeros(D.shape, np.float64)
        n[..., 2] = 1.0
        return t, n

    def sdf(self, p):
        return p[2]


@dataclass
class Box:
    center: tuple
    half: tuple
    yaw: float = 0.0
    tex: Texture = field(default_factory=Texture)
    casts_shadow = True

    def intersect(self, O, D):
        R = rot_z(-self.yaw)
        c = np.asarray(self.center, np.float64)
        h = np.asarray(self.half, np.float64)
        Ol = (np.asarray(O, np.float64) - c) @ R.T
        Dl = D @ R.T
        inv = 1.0 / _safe(Dl)
        t1 = (-h - Ol) * inv
        t2 = (h - Ol) * inv
        tmin = np.minimum(t1, t2).max(axis=-1)
        tmax = np.maximum(t1, t2).min(axis=-1)
        hit = (tmax >= tmin) & (tmin > 1e-4)
        t = np.where(hit, tmin, _INF)
        ts = np.where(hit, tmin, 0.0)[..., None]
        Pl = Ol + Dl * ts
        rel = np.abs(Pl) / h
        face = rel.argmax(axis=-1)
        nl = np.zeros(Pl.shape)
        for ax in range(3):
            nl[..., ax] = np.where(face == ax, np.sign(Pl[..., ax]), 0.0)
        n = nl @ R
        return t, n

    def sdf(self, p):
        R = rot_z(-self.yaw)
        q = np.abs(R @ (np.asarray(p, np.float64) - np.asarray(self.center))) - np.asarray(self.half)
        return float(np.linalg.norm(np.maximum(q, 0)) + min(q.max(), 0.0))


@dataclass
class Cylinder:
    """Vertical cylinder: center (cx, cy), radius r, from z0 to z1."""
    cx: float
    cy: float
    r: float
    z0: float
    z1: float
    tex: Texture = field(default_factory=Texture)
    casts_shadow = True

    def intersect(self, O, D):
        O = np.broadcast_to(np.asarray(O, np.float64), D.shape)
        dx, dy = O[..., 0] - self.cx, O[..., 1] - self.cy
        a = D[..., 0] ** 2 + D[..., 1] ** 2
        b = 2 * (dx * D[..., 0] + dy * D[..., 1])
        c = dx ** 2 + dy ** 2 - self.r ** 2
        disc = b * b - 4 * a * c
        sq = np.sqrt(np.maximum(disc, 0))
        ts = (-b - sq) / (2 * _safe(a))
        zs = O[..., 2] + ts * D[..., 2]
        side = (disc >= 0) & (ts > 1e-4) & (zs >= self.z0) & (zs <= self.z1)
        t_side = np.where(side, ts, _INF)
        tc = (self.z1 - O[..., 2]) / _safe(D[..., 2])
        px, py = dx + tc * D[..., 0], dy + tc * D[..., 1]
        cap = (D[..., 2] < -1e-6) & (tc > 1e-4) & (px ** 2 + py ** 2 <= self.r ** 2)
        t_cap = np.where(cap, tc, _INF)
        t = np.minimum(t_side, t_cap)
        n = np.zeros(D.shape)
        tt = np.where(np.isfinite(t_side), t_side, 0.0)
        n[..., 0] = (dx + tt * D[..., 0]) / self.r
        n[..., 1] = (dy + tt * D[..., 1]) / self.r
        n[t_cap < t_side] = (0.0, 0.0, 1.0)
        return t, n

    def sdf(self, p):
        dxy = math.hypot(p[0] - self.cx, p[1] - self.cy) - self.r
        dz = max(self.z0 - p[2], p[2] - self.z1)
        return float(math.hypot(max(dxy, 0), max(dz, 0)) + min(max(dxy, dz), 0.0))


@dataclass
class Sphere:
    center: tuple
    r: float
    tex: Texture = field(default_factory=Texture)
    casts_shadow = True

    def intersect(self, O, D):
        oc = np.asarray(O, np.float64) - np.asarray(self.center)
        b = 2 * (D * oc).sum(-1)
        c = (oc * oc).sum(-1) - self.r ** 2
        disc = b * b - 4 * c
        t = (-b - np.sqrt(np.maximum(disc, 0))) / 2
        t = np.where((disc >= 0) & (t > 1e-4), t, _INF)
        tt = np.where(np.isfinite(t), t, 0.0)[..., None]
        n = (oc + D * tt) / self.r
        return t, n

    def sdf(self, p):
        return float(np.linalg.norm(np.asarray(p) - np.asarray(self.center)) - self.r)


@dataclass
class World:
    objects: list
    ground: Plane = None
    sun: tuple = (-0.5, 0.1, 0.9)       # direction to the sun
    ambient: float = 0.35
    diffuse: float = 0.75
    sky: tuple = (0.72, 0.60)           # luminance at horizon and zenith

    def all_prims(self):
        return ([self.ground] if self.ground is not None else []) + list(self.objects)

    def clearance(self, p):
        """Distance (m) to the nearest object, ground excluded."""
        if not self.objects:
            return _INF
        return min(o.sdf(p) for o in self.objects)


def render_view(world: World, pos, yaw=0.0, pitch=0.0, roll=0.0, cam_pitch=0.0, cam_offset=(0.0, 0.0, 0.0),
                hfov=math.pi / 2, size=(240, 320), shadows=True, d_max=30.0, chroma=False):
    """Camera view at the body pose: (gray uint8, z-depth float32 in [Z_MIN, d_max]).
    chroma=True also returns a per-pixel color of unit luminance, RGB = gray * chroma."""
    rays_b, cos_ax = camera_rays(hfov, size)
    H, W = rays_b.shape[:2]
    Rb = body_rot(yaw, pitch, roll)
    Rc = Rb @ rot_y(cam_pitch)
    D = rays_b @ Rc.T
    O = np.asarray(pos, np.float64) + Rb @ np.asarray(cam_offset, np.float64)

    prims = world.all_prims()
    t_best = np.full((H, W), _INF)
    n_best = np.zeros((H, W, 3))
    idx = np.full((H, W), -1, np.int32)
    for k, pr in enumerate(prims):
        t, n = pr.intersect(O, D)
        closer = t < t_best
        t_best = np.where(closer, t, t_best)
        n_best = np.where(closer[..., None], n, n_best)
        idx = np.where(closer, k, idx)

    hit = np.isfinite(t_best)
    P = O + D * np.where(hit, t_best, 0.0)[..., None]

    albedo = np.zeros((H, W), np.float32)
    for k, pr in enumerate(prims):
        m = idx == k
        if m.any():
            albedo[m] = pr.tex.sample(P[m])

    L = np.asarray(world.sun, np.float64)
    L = L / np.linalg.norm(L)
    ndl = np.clip((n_best * L).sum(-1), 0, 1)
    shadow = np.ones((H, W))
    if shadows and hit.any():
        Os = P + n_best * 2e-3
        Ls = np.broadcast_to(L, Os.shape)
        for pr in world.objects:
            if pr.casts_shadow:
                ts, _ = pr.intersect(Os, Ls)
                shadow = np.where(np.isfinite(ts) & hit, 0.0, shadow)
    lum = albedo * (world.ambient + world.diffuse * ndl * shadow)

    elev = np.clip(D[..., 2], 0, 1)
    sky = world.sky[0] + (world.sky[1] - world.sky[0]) * elev
    lum = np.where(hit, lum, sky).astype(np.float32)

    depth = np.where(hit, t_best * cos_ax, d_max)
    depth = np.clip(depth, Z_MIN, d_max).astype(np.float32)
    gray = np.clip(lum * 255.0 + 0.5, 0, 255).astype(np.uint8)
    if not chroma:
        return gray, depth
    C = np.empty((H, W, 3), np.float32)
    C[:] = SKY_CHROMA
    for k, pr in enumerate(prims):
        col = getattr(pr.tex, 'color', None)
        m = idx == k
        if m.any():
            C[m] = chroma_of(col) if col is not None else 1.0
    return gray, depth, C


SKY_COLOR = (0.70, 0.82, 1.0)


def chroma_of(color):
    """RGB reflectance as a factor of unit luminance."""
    c = np.asarray(color, np.float32)
    return c / max(float(0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]), 1e-3)


SKY_CHROMA = chroma_of(SKY_COLOR)


def random_lighting(rng):
    hz = rng.uniform(0.45, 0.85)
    return dict(sun=(rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8), rng.uniform(0.4, 1.2)),
                ambient=rng.uniform(0.25, 0.5), diffuse=rng.uniform(0.5, 0.9),
                sky=(hz, hz * rng.uniform(0.7, 1.05)))
