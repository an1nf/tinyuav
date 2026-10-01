"""
R7 · Camera/ToF geometry vs the ground (Loquercio 2021 tilts the camera 30° for ≥7 m/s).
For each body pitch (nose down when accelerating): what fraction of the 64 ToF zones and of
the camera rows see the GROUND before D_MAX, and at what horizontal distance the central ray
hits the ground. Cruise altitude z = 0.65 m (EvasionPlanner.z_cruise).
"""
import math, json, os, sys
import numpy as np
from nanouav.config import CAM_PITCH, CAM_HFOV, TOF_HFOV, TOF_GRID, D_MAX, IMG_H
from nanouav.control.vehicle import PROFILES, G
from nanouav import paths

Z = 0.65


def ground_frac(down_center, fov, n, z=Z, dmax=D_MAX):
    """Fraction of n rows/zones (angles uniform in tan) whose ray hits the ground at < dmax."""
    half = math.tan(fov / 2)
    t = (np.arange(n) + 0.5) / n * 2 * half - half               # tan of the vertical angle (down +)
    ang = down_center + np.arctan(t)
    hit = ang > 0
    d = np.where(hit, z / np.sin(np.maximum(ang, 1e-6)), np.inf)  # distance along the ray
    return float(np.mean(hit & (d < dmax)))


rows = []
P = PROFILES['nano']
for label, pitch_deg in (('hover', 0.0),
                         ('cruise 2 m/s (drag)', math.degrees(math.atan(P.lin_drag * 2 / (P.mass * G)))),
                         ('accel/brake 2.2 m/s² (fast)', math.degrees(math.atan(2.2 / G))),
                         ('accel 4 m/s²', math.degrees(math.atan(4.0 / G))),
                         ('accel 8 m/s²', math.degrees(math.atan(8.0 / G)))):
    for cam_up in (0.0, 10.0, 20.0):
        down = CAM_PITCH + math.radians(pitch_deg) - math.radians(cam_up)
        rows.append({'case': label, 'body_pitch_deg': round(pitch_deg, 1), 'cam_extra_up_deg': cam_up,
                     'axis_down_deg': round(math.degrees(down), 1),
                     'ground_on_axis_m': round(Z / math.tan(down), 2) if down > 0 else None,
                     'tof_zones_ground_%': round(100 * ground_frac(down, TOF_HFOV, TOF_GRID), 0),
                     'cam_rows_ground_%': round(100 * ground_frac(down, CAM_HFOV, IMG_H), 0)})
for r in rows:
    print(r)
json.dump(rows, open(os.path.join(str(paths.results_dir()), 'r7_camera_tilt.json'), 'w'), indent=1)
