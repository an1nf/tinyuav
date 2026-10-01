"""
_01_tof/tof.py — Simulación del ToF 8×8 (VL53L5CX).

Corrige: antes se promediaba toda la imagen 96×96 en 8×8, pero el ToF tiene
HFOV 40° y la cámara 60° → el ToF solo cubre el centro de la imagen. Ahora cada
zona se proyecta a su caja de píxeles (uav_config.ZONE_BOXES) y se toma un
percentil bajo (modo "closest target").
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
