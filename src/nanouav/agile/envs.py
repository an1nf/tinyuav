"""Environments of Loquercio et al. 2021.

Training (Methods C): items spawned on the ground by a homogeneous Poisson point process. Two categories:
trees, and convex shapes (ellipsoids → spheres here, cuboids, cylinders) with size x, y ~ U(0.5, 4),
z ~ U(0.5, 8). 850 environments with category and δ ∈ {4, 5, 6, 7} uniformly random; the intensity is
read as 1/δ² items per m², consistent with the test forests of §S2 (δ = 1/49, 1/36, 1/25 m⁻²).
A global collision-free trajectory goes from the start to a point 40 m ahead.

Test (Results): forest in a 30 × 60 m region with trees of ~0.6 m diameter and intensity 1/25 m⁻²,
straight 40 m reference; narrow gap: 40 m wall 10 m ahead with one opening of U(0.8, 1.0) m at a lateral
offset U(−5, 5) m. Fine-tuning for the gap: 50 m wall, opening U(0.7, 1.2) m, start 10 m away with a
lateral offset U(−5, 5) m.

Every texture is noisy: SGM needs texture to match (the paper's Unity assets are textured too).
"""
from dataclasses import dataclass, field

import numpy as np

from nanouav.scene.renderer import Box, Cylinder, Plane, Sphere, Texture, World, random_lighting

Z_FLY = 1.5           # m, flight altitude of the reference


@dataclass
class Env:
    name: str
    world: World
    start: np.ndarray                 # (3,)
    goal: np.ndarray                  # (3,), 40 m ahead
    meta: dict = field(default_factory=dict)
    straight_ref: bool = True         # test protocol: straight reference (the student only gets the goal direction)


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
    """Poisson forest (test: ~0.6 m diameter, δ = 1/25 m⁻²). Start at the origin, goal 40 m along +x."""
    rng = np.random.default_rng(seed)
    start = np.array([0.0, 0.0, Z_FLY])
    xy = _poisson_xy(rng, density, (-5.0, length - 5.0), (-width / 2, width / 2), start, clear)
    objs = [Cylinder(float(x), float(y), float(rng.uniform(*r)), 0.0, float(rng.uniform(6, 12)), _tex(rng, 'noise'))
            for x, y in xy]
    w = World(objs, _ground(rng), **random_lighting(rng))
    return Env(f'forest_s{seed}', w, start, start + [40.0, 0, 0], {'density': density, 'n': len(objs)})


def shapes(seed, density=1 / 25, length=60.0, width=30.0, clear=3.0):
    """Poisson field of convex shapes: cuboids, cylinders and spheres, x, y ~ U(0.5, 4), z ~ U(0.5, 8)."""
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
    """Wall with one vertical opening, `dist` m ahead; the opening is at a lateral offset U(−offset, offset)."""
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
    """Environment i of the training set (850 in the paper): trees or shapes, δ ∈ {4, 5, 6, 7}."""
    rng = np.random.default_rng([base_seed, i, 0xA6])
    delta = int(rng.integers(4, 8))
    seed = int(rng.integers(1 << 30))
    if rng.random() < 0.5:
        e = forest(seed, density=1 / delta ** 2, r=(0.15, 0.4))
    else:
        e = shapes(seed, density=1 / delta ** 2)
    e.meta['delta'] = delta
    e.straight_ref = False            # training: the expert uses the global collision-free trajectory
    return e


def gap_finetune_env(i, base_seed=0):
    rng = np.random.default_rng([base_seed, i, 0x6A])
    e = narrow_gap(int(rng.integers(1 << 30)), gap=(0.7, 1.2), wall_len=50.0, dist=10.0)
    e.start[1] = rng.uniform(-5, 5)
    e.goal = e.start + [40.0, 0, 0]
    e.straight_ref = False
    return e


TEST = {'forest': forest, 'gap': narrow_gap}
