import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


@pytest.fixture(autouse=True)
def _isolated_results(tmp_path, monkeypatch):
    monkeypatch.setenv('TINYUAV_RESULTS', str(tmp_path / 'results'))
    monkeypatch.setenv('MPLBACKEND', 'Agg')
