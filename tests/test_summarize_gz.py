"""`nanouav summarize --gazebo`: one row per world × planner × CNN × profile × speed × planner_cfg."""
import json
import sys

from nanouav import paths


def _ep(world, success, v=None, cfg=None, t=30.0):
    r = {'result': 'success' if success else 'collision', 'success': success, 'time_s': t, 'path_m': 12.0,
         'min_clearance_m': 0.5, 'collisions': 0 if success else 1, 'planner': 'mh', 'variant': 'M_ToF',
         'profile': 'x3', 'world': world, 'seed': 2, 'rtf': [1.0]}
    if v is not None:                       # files written before v0.13.3 have no v_cruise / planner_cfg
        r['v_cruise'], r['planner_cfg'] = v, cfg or ''
    return r


def test_rows_split_by_speed_and_cfg(monkeypatch, capsys):
    d = paths.results_dir() / 'gz_runs'
    d.mkdir(parents=True)
    eps = [_ep('bosque', True), _ep('bosque', True, 0.5), _ep('bosque', True, 1.0, t=20.0),
           _ep('bosque', False, 1.0), _ep('bosque', True, 1.0, 'MAP_DECAY_FOV=1', t=21.0)]
    for i, r in enumerate(eps):
        (d / f'bosque_s2_mh_M_ToF_t{i}.json').write_text(json.dumps(r))
    from nanouav.results import summarize_gz
    monkeypatch.setattr(sys, 'argv', ['summarize_gz'])
    summarize_gz.main()
    rows = [L for L in capsys.readouterr().out.splitlines() if L.startswith('bosque')]
    assert len(rows) == 3                                     # old files count as v = 0.5, no cfg
    assert '2/2' in rows[0] and ' 0.5 ' in rows[0]
    assert '1/2' in rows[1] and ' 1 ' in rows[1] and rows[1].rstrip().endswith('│         —')
    assert '1/1' in rows[2] and rows[2].rstrip().endswith('MAP_DECAY_FOV=1')
