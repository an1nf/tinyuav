"""
R7 · Geometría cámara/ToF vs suelo (Loquercio 2021 inclina la cámara 30° para ≥7 m/s).
Para cada cabeceo del cuerpo (nariz abajo al acelerar), qué fracción de las 64 zonas ToF y de
las filas de la cámara ven el SUELO antes de D_MAX, y a qué distancia horizontal toca el suelo
el rayo central. Altura de crucero z = 0.65 m (EvasionPlanner.z_cruise).
"""
import math, json, os, sys
import numpy as np
from nanouav.config import CAM_PITCH, CAM_HFOV, TOF_HFOV, TOF_GRID, D_MAX, IMG_H
from nanouav.control.vehicle import PROFILES, G
from nanouav import paths

Z = 0.65


def ground_frac(down_center, fov, n, z=Z, dmax=D_MAX):
    """fracción de n filas/zonas (ángulos uniformes en tan) cuyo rayo toca el suelo a < dmax."""
    half = math.tan(fov / 2)
    t = (np.arange(n) + 0.5) / n * 2 * half - half               # tan del ángulo vertical (abajo +)
    ang = down_center + np.arctan(t)
    hit = ang > 0
    d = np.where(hit, z / np.sin(np.maximum(ang, 1e-6)), np.inf)  # distancia a lo largo del rayo
    return float(np.mean(hit & (d < dmax)))


rows = []
P = PROFILES['nano']
for label, pitch_deg in (('hover', 0.0),
                         ('crucero 2 m/s (arrastre)', math.degrees(math.atan(P.lin_drag * 2 / (P.mass * G)))),
                         ('acel/frenado 2.2 m/s² (fast)', math.degrees(math.atan(2.2 / G))),
                         ('acel 4 m/s²', math.degrees(math.atan(4.0 / G))),
                         ('acel 8 m/s²', math.degrees(math.atan(8.0 / G)))):
    for cam_up in (0.0, 10.0, 20.0):
        down = CAM_PITCH + math.radians(pitch_deg) - math.radians(cam_up)
        rows.append({'caso': label, 'pitch_cuerpo_deg': round(pitch_deg, 1), 'cam_extra_arriba_deg': cam_up,
                     'eje_abajo_deg': round(math.degrees(down), 1),
                     'suelo_en_eje_m': round(Z / math.tan(down), 2) if down > 0 else None,
                     'zonas_tof_suelo_%': round(100 * ground_frac(down, TOF_HFOV, TOF_GRID), 0),
                     'filas_cam_suelo_%': round(100 * ground_frac(down, CAM_HFOV, IMG_H), 0)})
for r in rows:
    print(r)
json.dump(rows, open(os.path.join(str(paths.results_dir()), 'r7_camera_tilt.json'), 'w'), indent=1)
