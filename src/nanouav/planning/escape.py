"""
nanouav/planning/escape.py  —  Pure reactive evasion

Computes the escape direction and speed from the current perception.

Strategy:
  1. Compute the "danger center of mass" on the CNN depth map
     (closer regions weigh more, 1/d² weighting)
  2. The escape direction is opposite to that center of mass
  3. The escape speed is proportional to the urgency (1 - dist/threshold)
  4. Fusion: direction from the CNN (96×96 resolution), urgency from the ToF (true scale)

Output: EscapeCommand with vx, vy, vz, and the vector visualization.
ESP32-S3 compatible: numpy only, no heavy dependencies.
"""
import numpy as np
from dataclasses import dataclass

EVADE_THRESHOLD  = 0.5   # m — below this, actively evade
CAUTION_THRESHOLD = 1.0  # m — below this, reduce speed
V_MAX_ESCAPE     = 0.5   # m/s — maximum escape speed
V_Z_BOOST        = 0.15  # m/s — vertical component, always positive while evading
DANGER_MIN_DIST  = 0.15  # m — absolute minimum safety distance


@dataclass
class EscapeCommand:
    """Velocity command for the drone PID."""
    vx: float          # m/s lateral (+ = right)
    vy: float          # m/s forward  (+ = forward, - = back up)
    vz: float          # m/s vertical (+ = up)
    urgency: float     # [0, 1] — 0=no urgency, 1=maximum urgency
    danger_cx: float   # normalized danger center x [-1, 1]
    danger_cy: float   # normalized danger center y [-1, 1]
    escape_angle: float  # degrees — escape direction
    mode: str          # 'idle' | 'caution' | 'evade' | 'emergency'

    def __str__(self):
        cols = {'idle': '\033[32m', 'caution': '\033[33m',
                'evade': '\033[31m', 'emergency': '\033[35m'}
        r = '\033[0m'
        c = cols.get(self.mode, '')
        return (f"{c}[{self.mode.upper():9s}]{r} "
                f"vx={self.vx:+.2f} vy={self.vy:+.2f} vz={self.vz:+.2f}  "
                f"urg={self.urgency:.2f}  θ={self.escape_angle:.0f}°")


def _danger_centroid(depth: np.ndarray) -> tuple:
    """
    Computes the danger center of mass on the depth map.

    depth: float32 (H, W) in meters
    Returns: (cx, cy) normalized to [-1, 1]
      cx < 0 → danger on the left
      cy < 0 → danger above
    """
    H, W = depth.shape
    danger = np.clip(1.0 / (depth**2 + 0.01), 0, 1000)
    total  = danger.sum()
    if total < 1e-6:
        return 0.0, 0.0

    dn = danger / total
    y, x = np.mgrid[0:H, 0:W]
    cx = float((dn * (x / W * 2 - 1)).sum())
    cy = float((dn * (y / H * 2 - 1)).sum())
    return cx, cy


def _tof_urgency(tof: np.ndarray) -> tuple:
    """
    Computes urgency and danger direction from the 8×8 ToF.

    Returns: (urgency [0,1], min_dist_m, tof_cx, tof_cy)
    """
    GRID = 8
    min_dist = float(tof.min())
    urgency  = float(np.clip(1.0 - min_dist / EVADE_THRESHOLD, 0.0, 1.0))

    danger = np.clip(1.0 / (tof**2 + 0.01), 0, 1000)
    dn     = danger / (danger.sum() + 1e-9)
    gy, gx = np.mgrid[0:GRID, 0:GRID]
    cx = float((dn * (gx / (GRID-1) * 2 - 1)).sum())
    cy = float((dn * (gy / (GRID-1) * 2 - 1)).sum())
    return urgency, min_dist, cx, cy


def compute_escape(depth_m: np.ndarray,
                   tof: np.ndarray,
                   cnn_weight: float = 0.7) -> EscapeCommand:
    """
    Computes the reactive escape command.

    depth_m:    float32 (96, 96) calibrated depth in meters
    tof:        float32 (8, 8)   ToF reading in meters
    cnn_weight: weight of the CNN direction vs the ToF [0, 1]

    Returns an EscapeCommand.
    """
    # Urgency and danger direction from the ToF (true scale)
    urgency, min_dist, tof_cx, tof_cy = _tof_urgency(tof)

    # Danger direction from the CNN (high resolution)
    cnn_cx, cnn_cy = _danger_centroid(depth_m)

    # Fusion: weighted direction
    w_cnn = cnn_weight
    w_tof = 1.0 - cnn_weight
    danger_cx = w_cnn * cnn_cx + w_tof * tof_cx
    danger_cy = w_cnn * cnn_cy + w_tof * tof_cy

    # Escape direction: opposite to the danger centroid
    esc_x = -danger_cx
    esc_y = -danger_cy
    mag = np.sqrt(esc_x**2 + esc_y**2) + 1e-6
    esc_x /= mag
    esc_y /= mag

    # Escape angle in degrees (0=right, 90=down, 180=left, 270=up)
    angle = float(np.degrees(np.arctan2(esc_y, esc_x))) % 360

    # Mode from the minimum distance
    if min_dist < DANGER_MIN_DIST:
        mode = 'emergency'
        v    = V_MAX_ESCAPE
    elif min_dist < EVADE_THRESHOLD:
        mode = 'evade'
        v    = V_MAX_ESCAPE * urgency
    elif min_dist < CAUTION_THRESHOLD:
        mode = 'caution'
        v    = V_MAX_ESCAPE * urgency * 0.4
        # In caution: only back up gently, no aggressive lateral motion
        esc_x *= 0.3
        esc_y  = -0.7  # back up
    else:
        mode = 'idle'
        v    = 0.0

    # Final command
    vx = float(esc_x * v)              # lateral
    vy = float(esc_y * v)              # forward/backward
    vz = float(V_Z_BOOST * urgency)    # always climb a little while evading

    if mode == 'idle':
        vx = vy = vz = 0.0

    return EscapeCommand(
        vx=round(vx, 4), vy=round(vy, 4), vz=round(vz, 4),
        urgency=round(urgency, 3),
        danger_cx=round(danger_cx, 3),
        danger_cy=round(danger_cy, 3),
        escape_angle=round(angle, 1),
        mode=mode,
    )


def escape_arrow_uv(cmd: EscapeCommand, scale: float = 1.0) -> tuple:
    """
    Returns (u, v) of the escape vector in image coordinates.
    u: horizontal component (+ = right in the image)
    v: vertical component   (+ = down in the image)
    Used to draw the arrow over the camera frame.
    """
    u = cmd.vx / V_MAX_ESCAPE * scale
    v = -cmd.vz / V_Z_BOOST   * scale * 0.3 + cmd.vy / V_MAX_ESCAPE * scale
    return float(u), float(v)
