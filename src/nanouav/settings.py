"""Runtime switches read from the environment (one place, documented).

| Variable                | Legacy name    | Effect                                                             |
|-------------------------|----------------|--------------------------------------------------------------------|
| ``NANOUAV_FAKE_CNN=1``  | ``UAV_FAKE_CNN`` | depth = ground truth + Gaussian noise instead of the .tflite model |
| ``NANOUAV_FUSE2=1``     | ``UAV_FUSE2``    | two-way ToF↔CNN consistency (R14) in the flight fusion            |
| ``NANOUAV_MH_GRID_RES`` | ``MH_GRID_RES``  | cell size (m) of the MH planner map, default 0.1                   |

Legacy names are still accepted so old job files keep working.
"""
from __future__ import annotations

import os

_LEGACY = {'FAKE_CNN': 'UAV_FAKE_CNN', 'FUSE2': 'UAV_FUSE2', 'MH_GRID_RES': 'MH_GRID_RES'}


def get(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(f'NANOUAV_{name}')
    if v is None and name in _LEGACY:
        v = os.environ.get(_LEGACY[name])
    return default if v is None or v == '' else v


def flag(name: str) -> bool:
    return get(name, '0') not in ('0', 'false', 'False', 'no')


def number(name: str, default: float) -> float:
    v = get(name)
    return float(v) if v is not None else default
