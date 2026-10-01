"""Shared scenarios (single source of truth for the offline simulator and Gazebo)."""
import json

import numpy as np
import pytest

from nanouav.sim import scenarios as SC

NAMES = sorted(SC.SCENARIOS)


@pytest.mark.parametrize('name', NAMES)
def test_deterministic_and_serializable(name):
    a, b = SC.make(name, 7), SC.make(name, 7)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert json.loads(json.dumps(a)) == a                     # pure JSON, no numpy types
    assert SC.make(name, 8) != a or name in ('obstaculos',)    # the seed matters (textures at least)


@pytest.mark.parametrize('name', NAMES)
def test_schema(name):
    s = SC.make(name, 0)
    for k in ('name', 'seed', 'A', 'B', 'yaw0', 'z_cruise', 'ground', 'objects', 'dynamic', 't_max'):
        assert k in s
    assert s['name'] == name and len(s['A']) == 2 and len(s['B']) == 2
    assert s['t_max'] > 0 and s['objects']
    for o in s['objects']:
        assert o['type'] in ('box', 'cyl')
        assert o['tex']['kind'] in ('uniform', 'checker', 'noise', 'stripes')


@pytest.mark.parametrize('name', NAMES)
@pytest.mark.parametrize('seed', [0, 1, 2])
def test_start_and_goal_are_free(name, seed):
    s = SC.make(name, seed)
    z = s['z_cruise']
    assert SC.clearance(s, (*s['A'], z)) > 0.25
    assert SC.clearance(s, (*s['B'], z)) > 0.25


def test_clearance_box_and_cylinder():
    s = {'objects': [SC._box((0, 0, 0.5), (0.5, 0.5, 0.5)), SC._cyl(3, 0, 0.2, 1.0)]}
    assert SC.clearance(s, (1.5, 0, 0.5)) == pytest.approx(1.0)
    assert SC.clearance(s, (0, 0, 0.5)) == pytest.approx(-0.5)            # inside → negative
    assert SC.clearance(s, (3, 1.2, 0.5)) == pytest.approx(1.0)
    assert SC.clearance(s, (4.0, 0, 0.5)) == pytest.approx(0.8)
    assert SC.clearance(s, (4.0, 0, 0.5), dyn_pos=[((4.5, 0, 0.5), 0.1)]) == pytest.approx(0.4)


def test_to_world_renders_finite_depth():
    from nanouav.scene import renderer
    s = SC.make('interior', 0)
    w = SC.to_world(s)
    assert len(w.objects) == len(s['objects'])
    g, d = renderer.render_view(w, (*s['A'], s['z_cruise']), s['yaw0'], 0, 0, rng=np.random.default_rng(0))
    assert g.shape == d.shape == (96, 96)
    assert np.isfinite(d).all() and d.min() > 0


def test_narrow_passage_width_parameter():
    wide, narrow = SC.make('estrechos', 0, w=1.4), SC.make('estrechos', 0, w=0.8)
    assert wide != narrow
