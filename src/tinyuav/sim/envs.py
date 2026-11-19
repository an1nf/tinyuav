"""Poisson forests, convex-shape fields and narrow gaps (Loquercio et al. 2021).

Training: trees or convex shapes with spacing delta in {4, 5, 6, 7} m. Test: 30 x 60 m forest of ~0.6 m
trunks at 1/25 m^-2 with a straight 40 m reference, and a wall with one opening. Every texture is noisy so
that SGM can match.
"""
from dataclasses import dataclass, field

import numpy as np

from tinyuav.sim.render import Box, Cylinder, Plane, Sphere, Texture, World, random_lighting

Z_FLY = 1.5           # m, reference altitude


@dataclass
class Env:
    name: str
    world: World
    start: np.ndarray
    goal: np.ndarray                  # 40 m ahead
    meta: dict = field(default_factory=dict)
    straight_ref: bool = True         # straight reference at test time


def _tex(rng, kind=None):
    a = rng.uniform(0.2, 0.9)
    return Texture(kind or str(rng.choice(['noise', 'noise', 'stripes'])), albedo=a,
                   albedo2=float(np.clip(a * rng.uniform(0.3, 0.7), 0, 1)), scale=rng.uniform(0.05, 0.4),
                   seed=int(rng.integers(1 << 30)))


def _ground(rng):
    a = rng.uniform(0.35, 0.8)
    return Plane(Texture('noise', albedo=a, albedo2=a * rng.uniform(0.3, 0.6), scale=rng.uniform(0.1, 0.5),
                         seed=int(rng.integers(1 << 30))))


def _poisson_xy(rng, intensity, x_rng, y_rng, start, clear):
    area = (x_rng[1] - x_rng[0]) * (y_rng[1] - y_rng[0])
    n = rng.poisson(intensity * area)
    xy = np.stack([rng.uniform(*x_rng, n), rng.uniform(*y_rng, n)], 1)
    return xy[np.hypot(xy[:, 0] - start[0], xy[:, 1] - start[1]) > clear]


def forest(seed, density=1 / 25, r=(0.25, 0.35), length=60.0, width=30.0, clear=2.5):
    """Poisson forest; start at the origin, goal 40 m along +x."""
    rng = np.random.default_rng(seed)
    start = np.array([0.0, 0.0, Z_FLY])
    xy = _poisson_xy(rng, density, (-5.0, length - 5.0), (-width / 2, width / 2), start, clear)
    objs = [Cylinder(float(x), float(y), float(rng.uniform(*r)), 0.0, float(rng.uniform(6, 12)), _tex(rng, 'noise'))
            for x, y in xy]
    w = World(objs, _ground(rng), **random_lighting(rng))
    return Env(f'forest_s{seed}', w, start, start + [40.0, 0, 0], {'density': density, 'n': len(objs)})


def shapes(seed, density=1 / 25, length=60.0, width=30.0, clear=3.0):
    """Poisson field of cuboids, cylinders and spheres."""
    rng = np.random.default_rng(seed)
    start = np.array([0.0, 0.0, Z_FLY])
    xy = _poisson_xy(rng, density, (-5.0, length - 5.0), (-width / 2, width / 2), start, clear + 2.0)
    objs = []
    for x, y in xy:
        sx, sy, sz = rng.uniform(0.5, 4), rng.uniform(0.5, 4), rng.uniform(0.5, 8)
        k = rng.integers(3)
        if k == 0:
            objs.append(Box((float(x), float(y), sz / 2), (sx / 2, sy / 2, sz / 2), float(rng.uniform(0, np.pi)), _tex(rng)))
        elif k == 1:
            objs.append(Cylinder(float(x), float(y), min(sx, sy) / 2, 0.0, sz, _tex(rng)))
        else:
            rr = min(sx, sy, sz) / 2
            objs.append(Sphere((float(x), float(y), float(rng.uniform(rr * 0.5, max(sz - rr, rr)))), rr, _tex(rng)))
    w = World(objs, _ground(rng), **random_lighting(rng))
    return Env(f'shapes_s{seed}', w, start, start + [40.0, 0, 0], {'density': density, 'n': len(objs)})


def narrow_gap(seed, gap=(0.8, 1.0), wall_len=40.0, dist=10.0, offset=5.0, height=8.0):
    """Wall `dist` m ahead with one opening at a random lateral offset."""
    rng = np.random.default_rng(seed)
    g, y0 = rng.uniform(*gap), rng.uniform(-offset, offset)
    tex = _tex(rng, 'noise')
    lo, hi = y0 - g / 2, y0 + g / 2
    L = wall_len / 2
    objs = [Box((dist, (lo - L) / 2, height / 2), (0.15, (lo + L) / 2, height / 2), 0.0, tex),
            Box((dist, (hi + L) / 2, height / 2), (0.15, (L - hi) / 2, height / 2), 0.0, tex)]
    start = np.array([0.0, 0.0, Z_FLY])
    w = World(objs, _ground(rng), **random_lighting(rng))
    return Env(f'gap_s{seed}', w, start, start + [40.0, 0, 0], {'gap': g, 'offset': y0})


def training_env(i, base_seed=0):
    """Training environment i: trees or shapes, spacing in {4, 5, 6, 7} m."""
    rng = np.random.default_rng([base_seed, i, 0xA6])
    delta = int(rng.integers(4, 8))
    seed = int(rng.integers(1 << 30))
    if rng.random() < 0.5:
        e = forest(seed, density=1 / delta ** 2, r=(0.15, 0.4))
    else:
        e = shapes(seed, density=1 / delta ** 2)
    e.meta['delta'] = delta
    e.straight_ref = False            # training: the expert follows the global collision-free path
    return e


def gap_finetune_env(i, base_seed=0):
    rng = np.random.default_rng([base_seed, i, 0x6A])
    e = narrow_gap(int(rng.integers(1 << 30)), gap=(0.7, 1.2), wall_len=50.0, dist=10.0)
    e.start[1] = rng.uniform(-5, 5)
    e.goal = e.start + [40.0, 0, 0]
    e.straight_ref = False
    return e


TEST = {'forest': forest, 'gap': narrow_gap}
