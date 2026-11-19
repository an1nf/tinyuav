"""Output location: $TINYUAV_RESULTS, else results/ in the checkout (or the working directory)."""
import os
from pathlib import Path


def results_dir(create=True):
    env = os.environ.get('TINYUAV_RESULTS')
    if env:
        p = Path(env).expanduser()
    else:
        root = Path(__file__).resolve().parents[2]
        p = (root if (root / 'pyproject.toml').exists() else Path.cwd()) / 'results'
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p
