from pathlib import Path

from nanouav import paths, settings


def test_repo_root_is_checkout():
    assert (paths.repo_root() / 'pyproject.toml').exists()


def test_env_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv('NANOUAV_MODELS', str(tmp_path / 'm'))
    monkeypatch.setenv('NANOUAV_DATA', str(tmp_path / 'd'))
    assert paths.models_dir() == tmp_path / 'm'
    assert paths.data_dir() == tmp_path / 'd' and (tmp_path / 'd').is_dir()
    assert Path(paths.results_dir()).is_dir()


def test_settings_new_and_legacy_names(monkeypatch):
    assert not settings.flag('FAKE_CNN')
    monkeypatch.setenv('UAV_FAKE_CNN', '1')
    assert settings.flag('FAKE_CNN')                      # legacy name still honoured
    monkeypatch.setenv('NANOUAV_FAKE_CNN', '0')
    assert not settings.flag('FAKE_CNN')                  # new name wins
    monkeypatch.setenv('NANOUAV_MH_GRID_RES', '0.05')
    assert settings.number('MH_GRID_RES', 0.1) == 0.05
