"""Numerical regression: the package must keep reproducing the outputs of the original project.

``tests/data/reference.json`` holds reference values of scenarios, rendering, ToF, fusion, optical
flow, both planners and a short closed-loop episode. They were generated with this package right
after the restructure, once it had been verified to give bit-identical results to the old flat
layout (``uav_v3``, SHA-1 hashes of every output). Values are compared with a small tolerance so the
test is stable across numpy / OpenCV builds; scenario specs are compared exactly.

A change here means a behavioural change. If it is intentional, regenerate on purpose::

    python tests/test_regression.py --update      # and explain it in CHANGELOG.md
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REF = Path(__file__).parent / 'data' / 'reference.json'


def summary(a, stride=7):
    """Compact fingerprint of an array: shape, mean, std and a strided sample."""
    a = np.asarray(a, np.float64)
    s = a[::stride, ::stride] if a.ndim >= 2 else a[::stride]
    return {'shape': list(a.shape), 'mean': round(float(a.mean()), 6), 'std': round(float(a.std()), 6),
            'sample': np.round(s, 5).tolist()}


def _world():
    from nanouav.sim import scenarios
    s = scenarios.make('bosque', 1)
    return s, scenarios.to_world(s)


def compute_static():
    from nanouav import config
    from nanouav.perception import tof_calibrator as fusion
    from nanouav.scene import renderer
    from nanouav.sensors import optical_flow, tof as tofmod
    from nanouav.sim import scenarios

    G = {'scen': {f'{n}_{s}': hashlib.sha1(json.dumps(scenarios.make(n, s), sort_keys=True).encode()).hexdigest()[:16]
                  for n in sorted(scenarios.SCENARIOS) for s in (0, 1)}}
    s, w = _world()
    g, d = renderer.render_view(w, (s['A'][0], s['A'][1], 0.65), s['yaw0'], 0, 0, rng=np.random.default_rng(0))
    G['render_gray'], G['render_depth'] = summary(g), summary(d)
    tof = tofmod.tof_8x8(d, rng=np.random.default_rng(1))
    G['tof'] = summary(tof, stride=1)
    cal = fusion.ToFCalibrator()
    dn = config.m_to_norm(d)
    G['fuse'] = summary(fusion.fuse_tof(cal.update_and_scale(dn, tof), tof))
    G['fuse2'] = summary(fusion.fuse_tof2(cal.update_and_scale(dn, tof), tof))
    f0 = (np.random.default_rng(5).random((96, 96)) * 255).astype(np.uint8)
    fp, dv = optical_flow.optical_flow(f0, np.roll(f0, 2, axis=1))
    G['flow'] = [len(fp), round(float(dv), 6)]
    return G


def planner_sequence(cls):
    """25 steps of a planner fed with rendered ground-truth depth while drifting through the forest."""
    from nanouav.scene import renderer
    from nanouav.sensors import tof as tofmod
    _, w = _world()
    pl = cls(radius=0.36, v_cruise=0.5)
    pl.CONSISTENT, pl.speed_law = True, 'brake'
    pl.set_segment((-1.5, 0), (10, 0))
    outs = []
    for i in range(25):
        p = np.array([-1.5 + 0.12 * i, 0.04 * np.sin(i), 0.65])
        yaw = 0.03 * i
        _, dd = renderer.render_view(w, p, yaw, 0, 0, rng=np.random.default_rng(i))
        tt = tofmod.tof_8x8(dd, rng=np.random.default_rng(100 + i))
        v, y, _ = pl.update({'p': p, 'v': np.array([0.4, 0, 0]), 'yaw': yaw, 't': 0.1 * i}, dd, tt, (p, yaw, 0, 0))
        outs.append([float(x) for x in v] + [float(y)])
    return np.round(outs, 6).tolist()


def compute_episode():
    from nanouav.control import vehicle
    from nanouav.perception import cnn
    from nanouav.sim import closed_loop
    cnn._interps.clear()
    np.random.seed(0)
    wc = closed_loop.random_corridor_world(np.random.default_rng(3))
    summ, R, _, _ = closed_loop.run_episode(wc, vehicle.PROFILES['nano'], seed=3, t_max=4.0)
    cnn._interps.clear()
    return {'R': summary(R[::50], stride=1), 'min_clearance_m': round(float(summ['min_clearance_m']), 6),
            'collided': bool(summ['collided'])}


def close(a, b, tol):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            close(a[k], b[k], tol)
    elif isinstance(a, (bool, str)) or a is None:
        assert a == b
    else:
        np.testing.assert_allclose(np.asarray(a, float), np.asarray(b, float), rtol=tol, atol=tol)


@pytest.fixture(scope='module')
def ref():
    return json.loads(REF.read_text())


@pytest.fixture(scope='module')
def static():
    return compute_static()


def test_scenarios_exact(static, ref):
    assert static['scen'] == ref['scen']


@pytest.mark.parametrize('key', ['render_gray', 'render_depth', 'tof', 'fuse', 'fuse2', 'flow'])
def test_static_outputs(key, static, ref):
    close(static[key], ref[key], 1e-4)


def test_lane_planner_sequence(ref):
    from nanouav.planning.evasion import EvasionPlanner
    close(planner_sequence(EvasionPlanner), ref['evasion'], 1e-3)


def test_mh_planner_sequence(ref):
    from nanouav.planning.mh_planner import MHPlanner
    close(planner_sequence(MHPlanner), ref['mh'], 1e-3)


@pytest.mark.slow
def test_closed_loop_episode(ref, monkeypatch):
    monkeypatch.setenv('NANOUAV_FAKE_CNN', '1')
    close(compute_episode(), ref['episode'], 1e-3)


@pytest.mark.skipif(not __import__('os').environ.get('NANOUAV_EXACT'),
                    reason='bit-exact check vs the old layout: set NANOUAV_EXACT=1 (same numpy/OpenCV builds only)')
def test_bit_exact_vs_uav_v3(static, monkeypatch):
    """SHA-1 of the outputs, exactly as computed on the old flat project (tests/data/golden_uav_v3_sha1.json)."""
    from nanouav.planning.evasion import EvasionPlanner
    from nanouav.planning.mh_planner import MHPlanner
    from nanouav.scene import renderer
    from nanouav.sensors import tof as tofmod
    from nanouav import config
    from nanouav.perception import tof_calibrator as fusion
    gold = json.loads((REF.parent / 'golden_uav_v3_sha1.json').read_text())
    H = lambda a: hashlib.sha1(np.ascontiguousarray(np.round(np.asarray(a, np.float64), 6)).tobytes()).hexdigest()[:16]
    s, w = _world()
    g, d = renderer.render_view(w, (s['A'][0], s['A'][1], 0.65), s['yaw0'], 0, 0, rng=np.random.default_rng(0))
    tof = tofmod.tof_8x8(d, rng=np.random.default_rng(1))
    cal, dn = fusion.ToFCalibrator(), config.m_to_norm(d)
    got = {'scen': static['scen'], 'render': [H(g), H(d)], 'tof': H(tof),
           'fuse': H(fusion.fuse_tof(cal.update_and_scale(dn, tof), tof)),
           'fuse2': H(fusion.fuse_tof2(cal.update_and_scale(dn, tof), tof)), 'flow': static['flow'],
           'evasion': H(planner_sequence(EvasionPlanner)), 'mh': H(planner_sequence(MHPlanner))}
    monkeypatch.setenv('NANOUAV_FAKE_CNN', '1')
    e = compute_episode()
    got['episode'] = [None, e['min_clearance_m'], e['collided']]
    gold['episode'][0] = None                          # the old hash covered R[::50] as a whole
    assert got == gold


if __name__ == '__main__' and '--update' in sys.argv:
    import os
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    for v in ('NANOUAV_FUSE2', 'UAV_FUSE2', 'NANOUAV_MH_GRID_RES', 'MH_GRID_RES'):
        os.environ.pop(v, None)
    from nanouav.planning.evasion import EvasionPlanner
    from nanouav.planning.mh_planner import MHPlanner
    G = compute_static()
    G['evasion'] = planner_sequence(EvasionPlanner)
    G['mh'] = planner_sequence(MHPlanner)
    os.environ['NANOUAV_FAKE_CNN'] = '1'
    G['episode'] = compute_episode()
    REF.write_text(json.dumps(G, indent=1, sort_keys=True))
    print(f'updated {REF} ({REF.stat().st_size // 1024} KB)')
