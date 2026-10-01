"""R17 mapping options for thin obstacles (all opt-in; the defaults keep the published behaviour).

* ``FLY_KEEP_NEAR``: the flying-pixel filter keeps the near side of a depth edge, so a trunk a few
  pixels wide is not removed entirely.
* ``MAP_DECAY_FOV``: only cells inside the camera sector forget (an obstacle that left the field of
  view keeps its evidence instead of turning into known free space).
* ``MAP_HIT_WINS``: free-space rays grazing a cell that is hit in the same frame do not erase it.
"""
import math

import numpy as np

from nanouav.planning.evasion import EvasionPlanner
from nanouav.planning.mh_planner import MHGrid, MHPlanner


def stripe_depth(width_px=4, near=3.0, far=4.0):
    d = np.full((96, 96), far, np.float32)
    c = 48
    d[:, c - width_px // 2:c + (width_px + 1) // 2] = near
    return d, slice(c - width_px // 2, c + (width_px + 1) // 2)


def test_flying_pixels_default_drops_thin_foreground():
    d, cols = stripe_depth(4)
    assert EvasionPlanner.flying_pixels(d)[:, cols].all()             # the whole trunk disappears


def test_flying_pixels_keep_near_keeps_the_object_drops_the_halo():
    d, cols = stripe_depth(4)
    fly = EvasionPlanner.flying_pixels(d, keep_near=True)
    assert not fly[:, cols].any()                                       # trunk kept
    assert fly[:, cols.start - 1].all() and fly[:, cols.stop].all()      # background next to it dropped
    # CNN-like smoothed edge: the intermediate depths (phantom trail) are still removed
    ramp = np.tile(np.r_[np.full(40, 2.0), np.linspace(2.0, 4.0, 8), np.full(48, 4.0)], (96, 1)).astype(np.float32)
    fly = EvasionPlanner.flying_pixels(ramp, keep_near=True)
    assert fly[:, 43:46].all() and not fly[:, :38].any()


def _grid():
    g = MHGrid(res=0.1, xr=(-2.0, 6.0), yr=(-4.0, 4.0))
    g.origin = np.zeros(2)
    return g


def _cell(g, x, y):
    return int((x - g.x0) / g.res), int((y - g.y0) / g.res)


def test_decay_only_inside_camera_sector():
    g = _grid()
    ahead, side = _cell(g, 2.05, 0.05), _cell(g, 0.05, 1.55)          # 2 m ahead / 1.5 m to the left
    g.g[ahead] = g.g[side] = 2.0
    g.view = (0.0, 0.0, 0.0, math.radians(30), 4.0)                    # camera looking along +x
    for _ in range(100):
        g.update(np.zeros((0, 3)), np.zeros((0, 3)))
    assert g.g[ahead] < 1.5                                            # observed and not confirmed → forgotten
    assert g.g[side] == 2.0                                            # outside the FOV → kept
    g.view = None                                                      # default: global decay
    for _ in range(100):
        g.update(np.zeros((0, 3)), np.zeros((0, 3)))
    assert g.g[side] < 1.5


def test_hit_beats_grazing_ray_in_the_same_frame():
    pt = np.array([[2.05, 0.05, 0.65]])
    for hit_wins, expect in ((False, 2.0), (True, 3.0)):           # 2·0.5 + 1 vs 2 + 1
        g = _grid()
        g.hit_wins = hit_wins
        g.g[_cell(g, 2.05, 0.05)] = 2.0
        g.update(pt, pt.copy(), decay=False)                           # same cell: hit and free sample
        assert g.g[_cell(g, 2.05, 0.05)] == expect


def test_planner_flags_reach_the_grid_and_default_off():
    pl = MHPlanner(radius=0.36, v_cruise=0.5)
    assert not (pl.MAP_DECAY_FOV or pl.MAP_HIT_WINS or pl.FLY_KEEP_NEAR)
    pl.set_segment((0.0, 0.0), (4.0, 0.0))
    st = {'p': np.array([0.0, 0.0, 0.65]), 'v': np.zeros(3), 'yaw': 0.3, 't': 0.0}
    pl.update(st, None, None, None)
    assert pl.grid.view is None and not pl.grid.hit_wins
    pl.MAP_DECAY_FOV = pl.MAP_HIT_WINS = True
    pl.update(st, None, None, None)
    assert pl.grid.view[2] == 0.3 and pl.grid.hit_wins
