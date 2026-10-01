"""
nanouav/config.py — Single source of truth for sensor geometry and the depth convention.

Depth convention (fixes the double scaling of the previous version):
  * "meters"     : depth z (along the optical axis), in [D_MIN, D_MAX]
  * "normalized" : n = (d - D_MIN) / (D_MAX - D_MIN)  ∈ [0, 1]
  The CNN is trained and exported in normalized space. Every conversion goes through
  norm_to_m / m_to_norm — never multiply by D_MAX by hand.

Geometry (same as gazebo/models/X3_UAV/6/model.sdf):
  * Camera 96×96, HFOV 60° (1.047 rad), tilted 0.15 rad downward
  * ToF 8×8 (VL53L5CX in Gazebo), HFOV 40° (0.698 rad), same mount
  The ToF only sees the center of the image: its 64 zones project onto pixel boxes.
"""
import math
import numpy as np

IMG_H = IMG_W = 96
CAM_HFOV   = 1.047          # rad
TOF_HFOV   = 0.698          # rad
TOF_GRID   = 8
CAM_PITCH  = 0.15           # rad, positive = downward (gz/FLU convention)
CAM_OFFSET = (0.10, 0.0, 0.05)   # m in body frame (X3)

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
    Pixel box (y0, y1, x0, x1) of the camera image covered by each ToF zone.
    Both sensors share the optical center (same pose in the SDF).
    Returns a grid×grid list indexed [gy][gx].
    """
    f = focal_px(cam_hfov, width)
    half = math.tan(tof_hfov / 2.0)
    edges = []
    for k in range(grid + 1):
        ang_t = -half + 2 * half * k / grid          # tan of the edge angle
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
    Reduces a 96×96 depth map (m) to 8×8 ToF zones.
    Uses a low percentile per zone (the VL53L5CX "closest target" mode).
    """
    out = np.empty((TOF_GRID, TOF_GRID), dtype=np.float32)
    for gy in range(TOF_GRID):
        for gx in range(TOF_GRID):
            y0, y1, x0, x1 = ZONE_BOXES[gy][gx]
            out[gy, gx] = np.percentile(depth_m[y0:y1, x0:x1], pct)
    return out
