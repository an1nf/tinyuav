import importlib.util

import pytest

from nanouav import cli
from nanouav.experiments import REGISTRY, get


def test_every_registered_module_exists():
    missing = [e.id for e in REGISTRY.values() if importlib.util.find_spec(e.module) is None]
    assert not missing, missing


def test_lookup_by_id_legacy_and_module():
    assert get('r05').module.endswith('r05_pole')
    assert get('r5_pole.py').id == 'r05'
    assert get('g1_gazebo_world').id == 'r12'
    with pytest.raises(KeyError):
        get('does-not-exist')


def test_cli_list_and_doctor(capsys):
    assert cli.main(['list']) == 0
    assert 'r15' in capsys.readouterr().out
    assert cli.main(['doctor']) == 0


@pytest.mark.parametrize('exp', ['r05', 'r06', 'r12', 'r14', 'r15', 'e3'])
def test_run_help_reaches_experiment_argparse(exp, capsys):
    assert cli.main(['run', exp, '--help']) == 0
    assert 'usage' in capsys.readouterr().out.lower()


def test_run_analytic_experiment_writes_result(tmp_path):
    assert cli.main(['run', '--results', str(tmp_path), 'r04']) == 0
    assert (tmp_path / 'r4_speed_limit.json').exists()
