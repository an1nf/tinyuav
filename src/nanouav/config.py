"""
uav_config.py — Única fuente de verdad para geometría de sensores y convención de profundidad.

Convención de profundidad (corrige el doble escalado de la versión anterior):
  * "metros"      : profundidad z (a lo largo del eje óptico), en [D_MIN, D_MAX]
  * "normalizada" : n = (d - D_MIN) / (D_MAX - D_MIN)  ∈ [0, 1]
  La CNN se entrena y exporta en espacio normalizado. Toda conversión pasa por
  norm_to_m / m_to_norm — nunca multiplicar a mano por D_MAX.

Geometría (igual a gazebo/models/X3_UAV/6/model.sdf):
  * Cámara 96×96, HFOV 60° (1.047 rad), inclinada 0.15 rad hacia abajo
  * ToF 8×8 (VL53L5CX en Gazebo), HFOV 40° (0.698 rad), mismo montaje
  El ToF solo ve el centro de la imagen: sus 64 zonas se proyectan a cajas de píxeles.
"""
import math
import numpy as np

IMG_H = IMG_W = 96
CAM_HFOV   = 1.047          # rad
TOF_HFOV   = 0.698          # rad
TOF_GRID   = 8
CAM_PITCH  = 0.15           # rad, positivo = hacia abajo (convención gz/FLU)
CAM_OFFSET = (0.10, 0.0, 0.05)   # m en cuerpo (X3)

D_MIN, D_MAX = 0.2, 4.0
D_RANGE = D_MAX - D_MIN


def focal_px(hfov: float = CAM_HFOV, width: int = IMG_W) -> float:
    return (width / 2.0) / math.tan(hfov / 2.0)


def norm_to_m(n):
    return np.asarray(n, dtype=np.float32) * D_RANGE + D_MIN


def m_to_norm(d):
    return (np.clip(np.asarray(d, dtype=np.float32), D_MIN, D_MAX) - D_MIN) / D_RANGE


def tof_zone_boxes(cam_hfov: float = CAM_HFOV, tof_hfov: float = TOF_HFOV,
                   grid: int = TOF_GRID, width: int = IMG_W):
    """
    Caja de píxeles (y0, y1, x0, x1) de la imagen de cámara que cubre cada zona ToF.
    Ambos sensores comparten centro óptico (misma pose en el SDF).
    Retorna lista grid×grid indexada [gy][gx].
    """
    f = focal_px(cam_hfov, width)
    half = math.tan(tof_hfov / 2.0)
    edges = []
    for k in range(grid + 1):
        ang_t = -half + 2 * half * k / grid          # tan del ángulo del borde
        edges.append(int(round(width / 2.0 + ang_t * f)))
    edges = [int(np.clip(e, 0, width)) for e in edges]
    boxes = [[None] * grid for _ in range(grid)]
    for gy in range(grid):
        for gx in range(grid):
            y0, y1 = edges[gy], max(edges[gy + 1], edges[gy] + 1)
            x0, x1 = edges[gx], max(edges[gx + 1], edges[gx] + 1)
            boxes[gy][gx] = (y0, y1, x0, x1)
    return boxes


ZONE_BOXES = tof_zone_boxes()


def pool_to_tof(depth_m: np.ndarray, pct: float = 25.0) -> np.ndarray:
    """
    Reduce un mapa de profundidad 96×96 (m) a 8×8 zonas ToF.
    Usa un percentil bajo por zona (modo "closest target" del VL53L5CX).
    """
    out = np.empty((TOF_GRID, TOF_GRID), dtype=np.float32)
    for gy in range(TOF_GRID):
        for gx in range(TOF_GRID):
            y0, y1, x0, x1 = ZONE_BOXES[gy][gx]
            out[gy, gx] = np.percentile(depth_m[y0:y1, x0:x1], pct)
    return out
