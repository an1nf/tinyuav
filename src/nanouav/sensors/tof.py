"""
nanouav/sensors/tof.py — Simulation of the 8×8 ToF (VL53L5CX).

Fix: previously the whole 96×96 image was averaged into 8×8, but the ToF has
HFOV 40° and the camera 60° → the ToF only covers the center of the image. Now each
zone is projected onto its pixel box (nanouav.config.ZONE_BOXES) and a low
percentile is taken ("closest target" mode).
"""
import numpy as np
from nanouav.config import pool_to_tof, TOF_GRID

GRID = TOF_GRID


def _sigma(d: np.ndarray) -> np.ndarray:
    return np.clip(0.015 + d * 0.02, 0.01, 0.12)


def tof_8x8(depth: np.ndarray, add_noise: bool = True, rng=None) -> np.ndarray:
    """depth: float32 (96,96) m → float32 (8,8) m."""
    out = pool_to_tof(depth)
    if add_noise:
        r = rng if rng is not None else np.random
        out = out + r.standard_normal((GRID, GRID)).astype(np.float32) * _sigma(out)
    return np.clip(out, 0.02, 4.0).astype(np.float32)


def tof_min_distance(tof: np.ndarray) -> float:
    return float(tof[2:6, 2:6].min())


def tof_obstacle_sector(tof: np.ndarray, threshold: float = 1.0):
    min_d = tof.min()
    if min_d > threshold:
        return 'libre', min_d
    gy, gx = np.unravel_index(tof.argmin(), tof.shape)
    if gx < 3:
        return 'izquierda', min_d
    elif gx > 4:
        return 'derecha', min_d
    return 'centro', min_d
