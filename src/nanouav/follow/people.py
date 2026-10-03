"""Walking people for the simulated worlds.

A person is a vertical cylinder (body, r = 0.2 m, 0–1.42 m) and a sphere (head, r = 0.12 m at 1.57 m),
with its own clothing texture (the appearance model of the tracker uses it). A `Walker` moves at constant
speed along a collision-free path planned on the static world (the same global planner as the expert,
with the radius of a person); the primitives are moved in place, so the renderer sees them.
"""
from dataclasses import dataclass, field

import numpy as np

from nanouav.agile.envs import Env, Z_FLY, forest
from nanouav.agile.expert import GlobalRef, global_plan
from nanouav.scene.renderer import Cylinder, Sphere, Texture, World

BODY_R, BODY_H, HEAD_R, HEAD_Z = 0.2, 1.42, 0.12, 1.57
HEIGHT = HEAD_Z + HEAD_R                      # 1.69 m


@dataclass
class Walker:
    path: GlobalRef                           # ground path (z = 0) at the walking speed
    speed: float
    t0: float = 0.0                           # starts walking at t0
    tex: Texture = field(default_factory=Texture)
    name: str = 'person'

    def __post_init__(self):
        p = self.path.pos(0.0)
        self.body = Cylinder(float(p[0]), float(p[1]), BODY_R, 0.0, BODY_H, self.tex)
        self.head = Sphere((float(p[0]), float(p[1]), HEAD_Z), HEAD_R, self.tex)
        self.t = 0.0

    def s(self, t):
        return float(np.clip((t - self.t0) * self.speed, 0.0, self.path.length))

    def pos(self, t):
        return self.path.pos(self.s(t))

    def vel(self, t, dt=0.05):
        return (self.pos(t + dt) - self.pos(t - dt)) / (2 * dt)

    def done(self, t):
        return self.s(t) >= self.path.length - 1e-6

    def move(self, t):
        p = self.pos(t)
        self.body.cx, self.body.cy = float(p[0]), float(p[1])
        self.head.center = (float(p[0]), float(p[1]), HEAD_Z)
        self.t = t

    def prims(self):
        return [self.body, self.head]

    def clearance(self, q):
        """Distance from a point to this person's body (cylinder up to the top of the head)."""
        dxy = np.hypot(q[0] - self.body.cx, q[1] - self.body.cy) - BODY_R
        dz = max(-q[2], q[2] - HEIGHT)
        return float(np.hypot(max(dxy, 0), max(dz, 0)) + min(max(dxy, dz), 0.0))


class _PathEnv:
    """Minimal env for the global planner (start, goal, world)."""

    def __init__(self, world, a, b, name):
        self.world, self.start, self.goal, self.name, self.straight_ref = world, a, b, name, False


def _walk_path(world, waypoints, speed, rq=0.35):
    pts = []
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        g = global_plan(_PathEnv(world, np.r_[a, 0.0], np.r_[b, 0.0], 'walk'), speed, rq=rq, inflate=0.1,
                        point_goal=True)
        pts.append(g.pts if not pts else g.pts[1:])
    P = np.concatenate(pts)
    P[:, 2] = 0.0
    return GlobalRef(P, speed)


def _clothes(rng):
    a = rng.uniform(0.1, 0.95)
    return Texture(str(rng.choice(['noise', 'stripes'])), albedo=a, albedo2=float(np.clip(a * rng.uniform(0.3, 0.8), 0, 1)),
                   scale=rng.uniform(0.04, 0.15), seed=int(rng.integers(1 << 30)))


@dataclass
class PeopleEnv(Env):
    walkers: list = field(default_factory=list)  # walkers[0] is the target

    def render_world(self):
        w = self.world
        return World(list(w.objects) + [p for wk in self.walkers for p in wk.prims()], w.ground,
                     sun=w.sun, ambient=w.ambient, diffuse=w.diffuse, sky=w.sky)


def people_env(seed, density=None, n_distractors=None, speed=None, length=40.0):
    """A person walks ~40 m through a forest (or an open field) with 0–2 other people crossing. The drone
    starts 5 m behind the target at flight altitude. Goal of the drone: keep following that person."""
    rng = np.random.default_rng([seed, 0xF0])
    if density is None:
        density = float(rng.choice([0.0, 1 / 64, 1 / 49, 1 / 36]))
    base = forest(int(rng.integers(1 << 30)), density=density, r=(0.15, 0.35), clear=4.0) if density > 0 else \
        forest(int(rng.integers(1 << 30)), density=1e-9)
    v = float(speed if speed is not None else rng.uniform(1.0, 3.0))
    t_start = (5.0, 0.0)
    mid = (5 + length / 2, float(rng.uniform(-6, 6)))
    end = (5 + length, float(rng.uniform(-6, 6)))
    target = Walker(_walk_path(base.world, [t_start, mid, end], v), v, t0=0.5, tex=_clothes(rng), name='target')
    walkers = [target]
    nd = int(n_distractors if n_distractors is not None else rng.integers(0, 3))
    for k in range(nd):
        # crossing the target's path somewhere ahead, timed to meet it
        xc = float(rng.uniform(12, 35))
        side = 1 if rng.random() < 0.5 else -1
        a, b = (xc + rng.uniform(-3, 3), side * 9.0), (xc + rng.uniform(-3, 3), -side * 9.0)
        vd = float(rng.uniform(0.9, 1.8))
        t_meet = (xc - 5) / v + 0.5
        try:
            path = _walk_path(base.world, [a, b], vd)
        except RuntimeError:
            continue
        walkers.append(Walker(path, vd, t0=max(0.0, t_meet - 9.0 / vd + rng.uniform(-1.5, 1.5)), tex=_clothes(rng),
                              name=f'person{k + 1}'))
    for wk in walkers:
        wk.move(0.0)
    start = np.array([0.0, 0.0, Z_FLY])
    env = PeopleEnv(f'people_s{seed}', base.world, start, start + [length + 5, 0, 0],
                    {'density': density, 'speed': v, 'n_distractors': len(walkers) - 1}, False, walkers)
    return env
