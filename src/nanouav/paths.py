"""Filesystem layout of a nanouav checkout.

All artefacts that are *not* source code live outside the package and outside git:

    models/    trained networks (.tflite, .keras, metrics.json)   NANOUAV_MODELS
    data/      generated datasets and caches (.npz)               NANOUAV_DATA
    results/   experiment outputs (json, png, logs)               NANOUAV_RESULTS

Each location can be overridden with the environment variable shown. ``NANOUAV_HOME`` overrides
the repository root itself (useful when the package is installed elsewhere, e.g. inside Docker).
"""
from __future__ import annotations

import os
from pathlib import Path

_PKG = Path(__file__).resolve().parent


def repo_root() -> Path:
    """Repository root: ``$NANOUAV_HOME``, else the checkout containing this package, else the cwd."""
    env = os.environ.get('NANOUAV_HOME')
    if env:
        return Path(env).expanduser().resolve()
    cand = _PKG.parents[1]                       # <root>/src/nanouav → <root>
    if (cand / 'pyproject.toml').exists():
        return cand
    return Path.cwd()


def _dir(env: str, default: str, create: bool) -> Path:
    p = Path(os.environ[env]).expanduser() if os.environ.get(env) else repo_root() / default
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def models_dir(create: bool = False) -> Path:
    return _dir('NANOUAV_MODELS', 'models', create)


def data_dir(create: bool = True) -> Path:
    return _dir('NANOUAV_DATA', 'data', create)


def results_dir(create: bool = True) -> Path:
    return _dir('NANOUAV_RESULTS', 'results', create)


def gazebo_dir() -> Path:
    return repo_root() / 'gazebo'
