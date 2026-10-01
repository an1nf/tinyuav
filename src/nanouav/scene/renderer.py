"""
nanouav/scene/renderer.py — Synthetic renderer v2 (raycasting in the world frame).

Changes with respect to E0-D:
  * Camera identical to Gazebo (96×96, HFOV 60°, pitch 0.15 rad, X3 offset).
  * Scene in world coordinates (FLU: X forward, Y left, Z up) and arbitrary
    camera pose → the same renderer serves both the dataset and the
    closed-loop simulation (`nanouav run e3`).
  * z depth (along the optical axis), same as the Gazebo depth_camera — previously
    the distance along the ray was returned.
  * Textured ground (world-SDF-style checker / noise), boxes with yaw, cylinders,
    spheres, walls, sun shadows, gradient sky, blur, gamma and noise.
  * Per-primitive SDF (signed distance) for collision detection.

API:
  World, Plane, Box, Cylinder, Sphere, Texture
  render_view(world, pos, yaw, pitch, roll, rng)   → (gray uint8, depth_m float32)
  sample_world(rng) / gazebo_world()
  render_scene(seed)  → (gray, depth_m)       (compatibility)
  render_batch(n, seed) → (X [0,1], Y norm)    (compatibility)
"""
import math
from dataclasses import dataclass, field
import numpy as np
import cv2

from nanouav.config import (IMG_H as H, IMG_W as W, CAM_HFOV, CAM_PITCH, CAM_OFFSET,
                        D_MIN, D_MAX, focal_px, m_to_norm)

_EPS = 1e-9
_INF = np.inf


# ── Rotations ──────────────────────────────────────────────────────────────
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
    """R body→world, ZYX. pitch>0 = nose down (FLU)."""
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


_RAY_CACHE = {}


def camera_rays(hfov=CAM_HFOV):
    """Unit rays in the body frame (H,W,3) and cos(angle to the optical axis)."""
    if hfov in _RAY_CACHE:
        return _RAY_CACHE[hfov]
    f = focal_px(hfov, W)
    u = (np.arange(W) + 0.5 - W / 2) / f
    v = (np.arange(H) + 0.5 - H / 2) / f
    U, V = np.meshgrid(u, v)
    d = np.stack([np.ones_like(U), -U, -V], axis=-1)      # camera→body
    n = np.linalg.norm(d, axis=-1, keepdims=True)
    d = d / n
    cos_ax = (1.0 / n[..., 0])
    _RAY_CACHE[hfov] = (d, cos_ax)
    return d, cos_ax


# ── Textures ───────────────────────────────────────────────────────────────
@dataclass
class Texture:
    kind: str = 'uniform'          # 'uniform' | 'checker' | 'noise' | 'stripes'
    albedo: float = 0.6
    albedo2: float = 0.3
    scale: float = 1.0             # cell size (m)
    extent: float = 1e9            # checker only inside |x|,|y| < extent
    seed: int = 0
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


# ── Primitives ─────────────────────────────────────────────────────────────
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
        n = nl @ R          # (R^T)^T : local→world
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
    sun: tuple = (-0.5, 0.1, 0.9)       # vector TOWARDS the sun
    ambient: float = 0.35
    diffuse: float = 0.75
    sky: tuple = (0.72, 0.60)           # luminance at horizon, zenith

    def all_prims(self):
        return ([self.ground] if self.ground is not None else []) + list(self.objects)

    def clearance(self, p):
        """Minimum distance (m) from a point to any obstacle (ground excluded)."""
        if not self.objects:
            return _INF
        return min(o.sdf(p) for o in self.objects)


# ── Render ─────────────────────────────────────────────────────────────────
def render_view(world: World, pos, yaw=0.0, pitch=0.0, roll=0.0,
                rng=None, cam_pitch=CAM_PITCH, cam_offset=CAM_OFFSET,
                hfov=CAM_HFOV, post=True, shadows=True):
    """
    Render the drone camera view at pose (pos, yaw, pitch, roll).
    Returns (gray uint8 96×96, depth_m float32 96×96 in [D_MIN, D_MAX]).
    """
    rays_b, cos_ax = camera_rays(hfov)
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

    depth = np.where(hit, t_best * cos_ax, D_MAX)
    depth = np.clip(depth, D_MIN, D_MAX).astype(np.float32)

    if post:
        lum = _camera_post(lum, rng if rng is not None else np.random.default_rng())
    gray = np.clip(lum * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return gray, depth


def _camera_post(lum, rng):
    """Exposure, gamma, vignetting, blur and sensor noise."""
    lum = lum * rng.uniform(0.75, 1.25)
    lum = np.clip(lum, 0, 1) ** rng.uniform(0.8, 1.25)
    if rng.random() < 0.5:
        yy, xx = np.mgrid[0:H, 0:W]
        r2 = ((xx - W / 2) ** 2 + (yy - H / 2) ** 2) / (W / 2) ** 2
        lum = lum * (1 - rng.uniform(0, 0.25) * r2)
    s = rng.uniform(0, 0.9)
    if s > 0.3:
        lum = cv2.GaussianBlur(lum.astype(np.float32), (0, 0), s)
    lum = lum + rng.normal(0, rng.uniform(0.004, 0.025), lum.shape)
    return np.clip(lum, 0, 1).astype(np.float32)


# ── Worlds ─────────────────────────────────────────────────────────────────
def _lum(r, g, b):
    return 0.299 * r + 0.587 * g + 0.114 * b


def gazebo_world() -> World:
    """Replica of the SDF world (obstacles, colors→gray, 1 m checker)."""
    ground = Plane(Texture('checker', albedo=0.9, albedo2=0.3, scale=1.0, extent=3.5))
    objs = [
        Box((1.8, 0.0, 0.3), (0.2, 0.2, 0.3), 0.0, Texture(albedo=_lum(0.9, 0.4, 0.05))),
        Box((1.2, -1.0, 0.25), (0.175, 0.175, 0.25), 0.0, Texture(albedo=_lum(0.1, 0.7, 0.2))),
        Box((1.0, 1.1, 0.4), (0.15, 0.15, 0.4), 0.0, Texture(albedo=_lum(0.2, 0.3, 0.9))),
        Cylinder(3.0, 0.4, 0.18, 0.0, 0.7, Texture(albedo=_lum(0.85, 0.75, 0.05))),
    ]
    return World(objs, ground, sun=(0.5, -0.1, 0.9))


def _rand_tex(rng, base=None):
    a = base if base is not None else rng.uniform(0.15, 0.95)
    k = rng.choice(['uniform', 'uniform', 'uniform', 'noise', 'stripes'])
    return Texture(str(k), albedo=a, albedo2=float(np.clip(a * rng.uniform(0.4, 0.8), 0, 1)),
                   scale=rng.uniform(0.1, 0.6), seed=int(rng.integers(1 << 30)))


def random_obstacle(rng, x, y):
    kind = rng.choice(['box', 'cyl', 'wall', 'sphere'], p=[0.45, 0.35, 0.12, 0.08])
    tex = _rand_tex(rng)
    if kind == 'box':
        hx, hy = rng.uniform(0.08, 0.35, 2)
        hz = rng.uniform(0.15, 0.9)
        return Box((x, y, hz), (hx, hy, hz), rng.uniform(-np.pi, np.pi), tex)
    if kind == 'cyl':
        return Cylinder(x, y, rng.uniform(0.05, 0.3), 0.0, rng.uniform(0.3, 1.8), tex)
    if kind == 'wall':
        hz = rng.uniform(0.5, 1.4)
        return Box((x, y, hz), (rng.uniform(0.05, 0.15), rng.uniform(0.6, 2.2), hz),
                   rng.uniform(-0.6, 0.6), tex)
    r = rng.uniform(0.1, 0.35)
    return Sphere((x, y, rng.uniform(r, 1.4)), r, tex)


def random_ground(rng):
    gk = str(rng.choice(['checker', 'noise', 'uniform'], p=[0.5, 0.35, 0.15]))
    a = rng.uniform(0.35, 0.95)
    return Plane(Texture(gk, albedo=a, albedo2=a * rng.uniform(0.25, 0.7),
                         scale=rng.uniform(0.3, 1.5), extent=rng.uniform(2.5, 30),
                         seed=int(rng.integers(1 << 30))))


def random_lighting(rng):
    hz = rng.uniform(0.45, 0.85)
    return dict(sun=(rng.uniform(-0.8, 0.8), rng.uniform(-0.8, 0.8), rng.uniform(0.4, 1.2)),
                ambient=rng.uniform(0.25, 0.5), diffuse=rng.uniform(0.5, 0.9),
                sky=(hz, hz * rng.uniform(0.7, 1.05)))


def sample_world(rng, n_min=1, n_max=6, near_prob=0.5):
    """Random world with the camera at the origin looking towards +X."""
    objs = []
    n = int(rng.integers(n_min, n_max + 1))
    if rng.random() < near_prob:          # nearby, centered obstacle (critical case)
        objs.append(random_obstacle(rng, rng.uniform(0.45, 1.5), rng.uniform(-0.4, 0.4)))
        n -= 1
    for _ in range(max(n, 0)):
        x = rng.uniform(0.6, 5.5)
        y = rng.uniform(-1, 1) * (0.2 + x * 0.65)
        objs.append(random_obstacle(rng, x, y))
    return World(objs, random_ground(rng), **random_lighting(rng))


def _safe_pose(world, rng, pos, tries=20):
    for _ in range(tries):
        if world.clearance(pos) > 0.3:
            return pos
        pos = pos + np.array([-0.3, rng.uniform(-0.3, 0.3), 0.0])
    return pos


def render_scene(seed=None, gazebo_prob=0.15):
    """Random scene. Returns (gray uint8 96×96, depth_m float32 96×96)."""
    rng = np.random.default_rng(seed)
    if rng.random() < gazebo_prob:
        world = gazebo_world()
        pos = np.array([rng.uniform(-2.0, 2.6), rng.uniform(-1.3, 1.3), rng.uniform(0.35, 1.1)])
        yaw = rng.uniform(-0.7, 0.7)
    else:
        world = sample_world(rng)
        pos = np.array([0.0, 0.0, rng.uniform(0.3, 1.3)])
        yaw = rng.normal(0, 0.1)
    pos = _safe_pose(world, rng, pos)
    return render_view(world, pos, yaw, rng.normal(0, 0.08), rng.normal(0, 0.07),
                       rng=rng, cam_pitch=CAM_PITCH + rng.normal(0, 0.03))


def render_batch(n, seed=0):
    """(X float32 N×96×96×1 in [0,1], Y float32 N×96×96×1 normalized)."""
    X = np.zeros((n, H, W, 1), np.float32)
    Y = np.zeros((n, H, W, 1), np.float32)
    for i in range(n):
        g, d = render_scene(seed + i)
        X[i, ..., 0] = g / 255.0
        Y[i, ..., 0] = m_to_norm(d)
    return X, Y
