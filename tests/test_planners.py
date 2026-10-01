"""Planners in a minimal kinematic loop (rendered ground-truth depth, perfect state).

The world is a thin wall across the path with one door. The drone must cross the wall plane only
through the door (or around the wall end), never through the wall.
"""
import math

import numpy as np
import pytest

from nanouav.planning.evasion import EvasionPlanner
from nanouav.planning.mh_planner import MHPlanner
from nanouav.scene.renderer import Box, Plane, Texture, World, render_view
from nanouav.sensors.tof import tof_8x8

WALL_X, WALL_L = 1.5, 3.0


def door_world(center, width):
    tex = Texture('noise', albedo=0.6, albedo2=0.3, scale=0.3, seed=1)
    lo, hi = center - width / 2, center + width / 2
    objs = [Box((WALL_X, (lo - WALL_L) / 2, 0.6), (0.05, (lo + WALL_L) / 2, 0.6), 0, tex),
            Box((WALL_X, (hi + WALL_L) / 2, 0.6), (0.05, (WALL_L - hi) / 2, 0.6), 0, tex)]
    return World(objs, Plane(Texture('checker', albedo=0.7, albedo2=0.3, scale=0.8)))


def fly(cls, world, steps=80, dt=0.1, goal=(5.0, 0.0)):
    pl = cls(radius=0.36, v_cruise=0.5)
    pl.CONSISTENT, pl.speed_law = True, 'brake'
    pl.set_segment((-0.5, 0.0), goal)
    p, yaw, v = np.array([-0.5, 0.0, 0.65]), 0.0, np.zeros(3)
    traj = [p.copy()]
    for i in range(steps):
        _, d = render_view(world, p, yaw, 0, 0, rng=np.random.default_rng(i))
        tof = tof_8x8(d, rng=np.random.default_rng(100 + i))
        vc, yc, info = pl.update({'p': p, 'v': v, 'yaw': yaw, 't': dt * i}, d, tof, (p, yaw, 0, 0))
        v = np.asarray(vc, float)
        assert v.shape == (3,) and np.isfinite(v).all() and np.isfinite(yc)
        assert np.linalg.norm(v[:2]) <= 0.5 * 1.3 + 1e-6          # never far above the cruise speed
        p, yaw = p + v * dt, float(yc)
        traj.append(p.copy())
    return np.array(traj)


def wall_crossings(traj):
    """y at every crossing of the wall plane x = WALL_X."""
    ys = []
    for a, b in zip(traj[:-1], traj[1:]):
        if (a[0] - WALL_X) * (b[0] - WALL_X) < 0:
            s = (WALL_X - a[0]) / (b[0] - a[0])
            ys.append(a[1] + s * (b[1] - a[1]))
    return ys


@pytest.mark.parametrize('side', [+1, -1])
def test_mh_goes_through_the_door(side):
    c, w = 0.9 * side, 1.4
    traj = fly(MHPlanner, door_world(c, w))
    ys = wall_crossings(traj)
    assert ys, f'never crossed the wall (end {traj[-1].round(2)})'
    assert all(abs(y - c) < w / 2 for y in ys), ys
    assert traj[-1][0] > 2.5                                          # kept going towards the goal


def test_mh_never_crosses_the_wall_at_a_narrow_door():
    c, w = 0.9, 1.0                                                   # tight door: may go around instead
    traj = fly(MHPlanner, door_world(c, w), steps=100)
    for y in wall_crossings(traj):
        assert abs(y - c) < w / 2 or abs(y) > WALL_L, y
    assert np.hypot(*(traj[-1][:2] - traj[0][:2])) > 1.5               # it did not stall


def test_lane_planner_outputs_are_sane():
    traj = fly(EvasionPlanner, door_world(0.9, 1.4), steps=40)
    for y in wall_crossings(traj):
        assert abs(y - 0.9) < 0.7, y


def test_mh_free_space_heads_to_goal():
    traj = fly(MHPlanner, World([], Plane(Texture('checker', albedo=0.7, albedo2=0.3, scale=0.8))), steps=30)
    d = traj[-1][:2] - traj[0][:2]
    assert d[0] > 1.0 and abs(math.atan2(d[1], d[0])) < math.radians(15)
