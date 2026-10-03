"""Balloons to follow through a forest (R20).

A `Balloon` is a colored sphere (r = 0.3 m by default) that floats along a collision-free path between the
trees: the 2D route of the global planner (with a margin wide enough for the drone to pass where the
balloon passes) and a height that rises and falls between `z_lo` and `z_hi` along the way. It has the
interface of `people.Walker`, so `FollowMission` and the rollout treat it the same way.

`balloon_env` gives a forest in color (bark, ground and a few bushes with their own colors, some of them
close to the balloon hue but less saturated: realistic false positives for the color detector), the
target balloon and 0–2 others that cross its path; with probability `p_same` one of them has the target
color, so only the target lock (3D gate) keeps the right one.
"""
import colorsys
from dataclasses import dataclass, field

import numpy as np

from nanouav.agile.envs import Z_FLY, forest
from nanouav.agile.expert import GlobalRef
from nanouav.follow.blob import COLORS
from nanouav.follow.people import PeopleEnv, _walk_path
from nanouav.scene.renderer import Box, Sphere, Texture


def _lum(c):
    return float(0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2])


@dataclass
class Balloon:
    path: GlobalRef                           # 3D path of the center
    speed: float
    t0: float = 0.0
    radius: float = 0.3
    color: str = 'red'
    name: str = 'balloon'
    tex: Texture = field(default=None)
    accel: float = 1.0                        # m/s², from rest at t0 up to `speed` (a balloon does not jump)

    def __post_init__(self):
        rgb = COLORS[self.color]
        if self.tex is None:                  # a little shading texture: latex is not perfectly uniform
            a = _lum(rgb)
            self.tex = Texture('noise', albedo=a, albedo2=a * 0.85, scale=0.15, seed=sum(map(ord, self.name)), color=rgb)
        p = self.path.pos(0.0)
        self.body = Sphere(tuple(float(x) for x in p), self.radius, self.tex)
        self.t = 0.0

    def s(self, t):
        tau, v, a = max(t - self.t0, 0.0), self.speed, self.accel
        s = 0.5 * a * tau * tau if tau < v / a else v * v / (2 * a) + v * (tau - v / a)
        return float(min(s, self.path.length))

    def speed_at(self, t):
        """ds/dt along the path."""
        return float(min(self.speed, self.accel * max(t - self.t0, 0.0))) if not self.done(t) else 0.0

    def duration(self):
        """Time from t0 to the end of the path."""
        v, a, L = self.speed, self.accel, self.path.length
        return float(np.sqrt(2 * L / a)) if L < v * v / (2 * a) else v / a + (L - v * v / (2 * a)) / v

    def pos(self, t):
        return self.path.pos(self.s(t))

    def vel(self, t, dt=0.05):
        return (self.pos(t + dt) - self.pos(t - dt)) / (2 * dt)

    def done(self, t):
        return self.s(t) >= self.path.length - 1e-6

    def move(self, t):
        self.body.center = tuple(float(x) for x in self.pos(t))
        self.t = t

    def prims(self):
        return [self.body]

    def clearance(self, q):
        return float(np.linalg.norm(np.asarray(q, float) - np.asarray(self.body.center)) - self.radius)


def _round_corners(world, P, step, rq, sigma=1.2):
    """Smooth the route (the global plan joins its segments with corners at the waypoints, and a balloon
    drifts in curves): Gaussian filter of the x, y coordinates (σ = 1.2 m), kept only where the smoothed
    point stays `rq` from the trees, then a light pass to remove the kinks between kept and smoothed parts."""
    import cv2
    def blur(Q, sig):
        k = int(3 * sig / step) | 1
        out = Q.copy()
        for c in (0, 1):
            x = np.pad(Q[:, c], k, mode='edge').astype(np.float64)
            out[:, c] = cv2.GaussianBlur(x[:, None], (1, 2 * k + 1), sig / step).ravel()[k:-k]
        out[0], out[-1] = Q[0], Q[-1]
        return out
    S = blur(P, sigma)
    ok = np.array([world.clearance(np.r_[q[:2], 1.5]) > rq for q in S]) if world.objects else np.ones(len(S), bool)
    Q = np.where(ok[:, None], S, P)
    Q2 = blur(Q, 0.3)
    ok2 = np.array([world.clearance(np.r_[q[:2], 1.5]) > rq - 0.05 for q in Q2]) if world.objects else np.ones(len(Q2), bool)
    return np.where(ok2[:, None], Q2, Q)


def fly_path(world, waypoints, speed, rng, z_lo=1.2, z_hi=2.6, rq=0.5, step=0.1):
    """Collision-free 2D route through the waypoints (margin rq), resampled every `step` m, with a smooth
    height profile in [z_lo, z_hi] (two sines of random wavelength and phase)."""
    g = _walk_path(world, waypoints, speed, rq=rq)
    s = np.arange(0.0, g.length, step)
    P = _round_corners(world, g.pos(np.r_[s, g.length]), step, rq)
    lam = rng.uniform(8, 20, 2)
    ph = rng.uniform(0, 2 * np.pi, 2)
    S = np.r_[s, g.length]
    u = 0.5 + 0.35 * np.sin(2 * np.pi * S / lam[0] + ph[0]) + 0.15 * np.sin(2 * np.pi * S / lam[1] + ph[1])
    P[:, 2] = z_lo + (z_hi - z_lo) * np.clip(u, 0, 1)
    return GlobalRef(P, speed)


BARK = [(0.45, 0.32, 0.22), (0.38, 0.30, 0.25), (0.55, 0.50, 0.45), (0.42, 0.40, 0.30), (0.50, 0.36, 0.30)]
GROUND = [(0.35, 0.45, 0.20), (0.45, 0.40, 0.28), (0.30, 0.38, 0.22), (0.55, 0.48, 0.35)]


def _seg_dist(o, a, b):
    """Horizontal distance from an object's footprint (bounding circle) to the segment a–b."""
    if hasattr(o, 'cx'):
        c, r = np.array([o.cx, o.cy]), o.r
    else:
        c = np.asarray(o.center, float)[:2]
        r = float(np.linalg.norm(o.half[:2])) if hasattr(o, 'half') else o.r
    a, b = np.asarray(a, float), np.asarray(b, float)
    u = np.clip(np.dot(c - a, b - a) / np.dot(b - a, b - a), 0, 1)
    return float(np.linalg.norm(c - (a + u * (b - a))) - r)


def colorize(world, rng, n_bushes=None, near=(5.0, 45.0), keep_free=None):
    """Give the forest colors (in place) and add bushes: low boxes / spheres of foliage green, dry
    brown, and some of muted red/orange/yellow (low saturation, like dead leaves or flowers)."""
    for o in world.objects:
        c = np.array(BARK[int(rng.integers(len(BARK)))]) * rng.uniform(0.9, 1.1, 3)
        o.tex.color = tuple(np.clip(c, 0, 1))
    if world.ground is not None:
        world.ground.tex.color = tuple(np.clip(np.array(GROUND[int(rng.integers(len(GROUND)))]) * rng.uniform(0.9, 1.1, 3), 0, 1))
    nb = int(rng.integers(3, 10) if n_bushes is None else n_bushes)
    for _ in range(nb):
        x, y = rng.uniform(*near), rng.uniform(-10, 10)
        if world.clearance(np.array([x, y, 0.3])) < 0.8:
            continue
        if keep_free is not None and _seg_dist(Sphere((x, y, 0.0), 0.6), keep_free[0], keep_free[1]) < keep_free[2]:
            continue
        kind = rng.choice(['green', 'brown', 'muted'], p=[0.5, 0.25, 0.25])
        base = {'green': (0.2, 0.5, 0.15), 'brown': (0.45, 0.33, 0.18)}.get(kind)
        if base is None:                      # muted warm color: hue near red/orange/yellow, saturation < 0.5
            h, sat, val = rng.choice([0.0, 0.05, 0.12]), rng.uniform(0.25, 0.45), rng.uniform(0.5, 0.8)
            base = colorsys.hsv_to_rgb(h, sat, val)
        a = _lum(base)
        tex = Texture('noise', albedo=a, albedo2=a * 0.6, scale=rng.uniform(0.05, 0.2), seed=int(rng.integers(1 << 30)),
                      color=tuple(base))
        r = rng.uniform(0.25, 0.6)
        world.objects.append(Sphere((x, y, r * 0.8), r, tex) if rng.random() < 0.5 else
                             Box((x, y, r * 0.6), (r, r * rng.uniform(0.6, 1.4), r * 0.6), rng.uniform(0, np.pi), tex))
    return world


def balloon_env(seed, density=None, n_distractors=None, speed=None, length=40.0, color='red', p_same=0.25,
                z_lo=1.2, z_hi=2.6, radius=0.3):
    """The target balloon floats ~40 m through a colored forest (or an open field) at 1–4 m/s, rising and
    falling between z_lo and z_hi; 0–2 other balloons cross its path. The drone starts behind it."""
    rng = np.random.default_rng([seed, 0xBA])
    if density is None:
        density = float(rng.choice([0.0, 1 / 49, 1 / 36, 1 / 25]))
    base = forest(int(rng.integers(1 << 30)), density=density if density > 0 else 1e-9, r=(0.15, 0.35), clear=4.0)
    # keep a corridor free from the drone start to the balloon start (the trail starts behind the balloon)
    base.world.objects = [o for o in base.world.objects if _seg_dist(o, (-3.0, 0.0), (6.0, 0.0)) > 1.3]
    colorize(base.world, rng, keep_free=((-3.0, 0.0), (6.0, 0.0), 1.3))
    v = float(speed if speed is not None else rng.uniform(1.0, 4.0))
    for k in range(8):                        # a waypoint inside a tree or a closed passage: draw others
        wp = [(5.0, 0.0), (5 + length / 2, float(rng.uniform(-6, 6))), (5 + length, float(rng.uniform(-6, 6)))]
        try:
            path = fly_path(base.world, wp, v, rng, z_lo, z_hi, rq=0.5 if k < 4 else 0.4)
            break
        except RuntimeError:
            if k == 7:
                raise
    target = Balloon(path, v, t0=0.5, radius=radius, color=color, name='target')
    balloons = [target]
    others = [c for c in COLORS if c != color]
    nd = int(n_distractors if n_distractors is not None else rng.integers(0, 3))
    for k in range(nd):
        xc = float(rng.uniform(12, 35))
        side = 1 if rng.random() < 0.5 else -1
        a, b = (xc + rng.uniform(-3, 3), side * 9.0), (xc + rng.uniform(-3, 3), -side * 9.0)
        vd = float(rng.uniform(0.8, 2.0))
        t_meet = (xc - 5) / v + 0.5
        col = color if rng.random() < p_same else str(rng.choice(others))
        try:   # above the drone's band: they cross the image (and the target) but not the drone's path
            path = fly_path(base.world, [a, b], vd, rng, z_hi + 0.7, z_hi + 1.4)
        except RuntimeError:
            continue
        balloons.append(Balloon(path, vd, t0=max(0.0, t_meet - 9.0 / vd + rng.uniform(-1.5, 1.5)), radius=radius,
                                color=col, name=f'balloon{k + 1}'))
    for b in balloons:
        b.move(0.0)
    start = np.array([0.0, 0.0, Z_FLY])
    meta = {'density': density, 'speed': v, 'n_distractors': len(balloons) - 1,
            'same_color': sum(b.color == color for b in balloons[1:]), 'color': color}
    return PeopleEnv(f'balloon_s{seed}', base.world, start, start + [length + 5, 0, 0], meta, False, balloons)
