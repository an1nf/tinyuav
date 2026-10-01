import json
from pathlib import Path

import pytest

from nanouav import paths
from nanouav.runner import load_suite, run_suite

SUITES = sorted((paths.repo_root() / 'configs' / 'suites').glob('*.yaml'))


@pytest.mark.parametrize('suite', SUITES, ids=lambda p: p.stem)
def test_suites_parse_and_reference_known_experiments(suite):
    spec, jobs = load_suite(suite)
    assert jobs and len({j.id for j in jobs}) == len(jobs)


def test_matrix_expansion(tmp_path):
    f = tmp_path / 's.yaml'
    f.write_text('name: t\nmatrix: {v: [M, M_ToF]}\njobs:\n  - id: a-{v}\n    run: r05 --variant {v}\n'
                 '  - id: b\n    run: r04\n    matrix: {}\n')
    _, jobs = load_suite(f)
    assert [j.id for j in jobs] == ['a-M', 'a-M_ToF', 'b']
    assert jobs[1].argv == ['r05', '--variant', 'M_ToF']


def test_unknown_experiment_is_rejected(tmp_path):
    f = tmp_path / 's.yaml'
    f.write_text('jobs:\n  - run: nope --n 1\n')
    with pytest.raises(KeyError):
        load_suite(f)


def test_run_tiny_suite_and_resume(tmp_path):
    f = tmp_path / 'tiny.yaml'
    f.write_text('name: tiny\njobs:\n  - id: analytic\n    run: r04\n')
    out = tmp_path / 'out'
    assert run_suite(f, workers=1, out=str(out)) == 0
    state = json.loads((out / 'jobs.json').read_text())
    assert state['analytic']['rc'] == 0 and (out / 'r4_speed_limit.json').exists()
    assert run_suite(f, workers=1, out=str(out), resume=True) == 0        # nothing left to do
