import pytest

from tinyuav import cli, paths


def test_results_dir_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv('TINYUAV_RESULTS', str(tmp_path / 'r'))
    assert paths.results_dir() == tmp_path / 'r' and (tmp_path / 'r').is_dir()


def test_help_lists_groups(capsys):
    assert cli.main([]) == 0
    out = capsys.readouterr().out
    assert all(g in out for g in cli.GROUPS)


@pytest.mark.parametrize('group', list(cli.GROUPS))
def test_group_help(group):
    with pytest.raises(SystemExit) as e:
        cli.main([group, '-h'])
    assert e.value.code == 0


def test_unknown_group():
    assert cli.main(['nope']) == 2
