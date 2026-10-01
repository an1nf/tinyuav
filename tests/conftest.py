import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))          # works without `pip install -e .` too


@pytest.fixture(autouse=True)
def _isolated_results(tmp_path, monkeypatch):
    """Every test writes results into its own temporary directory, never into the checkout."""
    monkeypatch.setenv('NANOUAV_RESULTS', str(tmp_path / 'results'))
    monkeypatch.setenv('MPLBACKEND', 'Agg')
    for v in ('NANOUAV_FAKE_CNN', 'UAV_FAKE_CNN', 'NANOUAV_FUSE2', 'UAV_FUSE2', 'NANOUAV_MH_GRID_RES', 'MH_GRID_RES'):
        monkeypatch.delenv(v, raising=False)
    yield


def has_tflite_model():
    try:
        import ai_edge_litert  # noqa: F401
    except ImportError:
        try:
            import tflite_runtime  # noqa: F401
        except ImportError:
            return False
    from nanouav import paths
    return (paths.models_dir() / 'model_M.tflite').exists()
