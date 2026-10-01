"""
_04_policy/escape.py  —  Evasión reactiva pura

Calcula la dirección y velocidad de escape dada la percepción actual.

Estrategia:
  1. Calcular "centro de masa del peligro" en el mapa de profundidad CNN
     (zonas más cercanas pesan más, ponderación 1/d²)
  2. La dirección de escape es la opuesta a ese centro de masa
  3. La velocidad de escape es proporcional a la urgencia (1 - dist/umbral)
  4. Fusión: dirección del CNN (resolución 96×96), urgencia del ToF (escala real)

Salida: EscapeCommand con vx, vy, vz, y la visualización del vector.
Compatible con ESP32-S3: solo numpy, sin dependencias pesadas.
"""
import numpy as np
from dataclasses import dataclass

EVADE_THRESHOLD  = 0.5   # m — por debajo de esto, evadir activamente
CAUTION_THRESHOLD = 1.0  # m — por debajo de esto, reducir velocidad
V_MAX_ESCAPE     = 0.5   # m/s — velocidad máxima de escape
V_Z_BOOST        = 0.15  # m/s — componente vertical siempre positiva al evadir
DANGER_MIN_DIST  = 0.15  # m — distancia mínima de seguridad absoluta


@dataclass
class EscapeCommand:
    """Comando de velocidad para el PID del dron."""
    vx: float          # m/s lateral (+ = derecha)
    vy: float          # m/s frontal  (+ = adelante, - = retroceder)
    vz: float          # m/s vertical (+ = subir)
    urgency: float     # [0, 1] — 0=sin urgencia, 1=máxima urgencia
    danger_cx: float   # centro de peligro x normalizado [-1, 1]
    danger_cy: float   # centro de peligro y normalizado [-1, 1]
    escape_angle: float  # grados — dirección de escape
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
    Calcula el centro de masa del peligro en el mapa de profundidad.

    depth: float32 (H, W) en metros
    Retorna: (cx, cy) normalizados en [-1, 1]
      cx < 0 → peligro a la izquierda
      cy < 0 → peligro arriba
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
    Calcula urgencia y dirección de peligro del ToF 8×8.

    Retorna: (urgency [0,1], min_dist_m, tof_cx, tof_cy)
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
    Calcula el comando de escape reactivo.

    depth_m:    float32 (96, 96) profundidad calibrada en metros
    tof:        float32 (8, 8)   lectura ToF en metros
    cnn_weight: peso de la dirección CNN vs ToF [0, 1]

    Retorna EscapeCommand.
    """
    # Urgencia y dirección del peligro desde ToF (escala real)
    urgency, min_dist, tof_cx, tof_cy = _tof_urgency(tof)

    # Dirección del peligro desde CNN (resolución alta)
    cnn_cx, cnn_cy = _danger_centroid(depth_m)

    # Fusión: dirección ponderada
    w_cnn = cnn_weight
    w_tof = 1.0 - cnn_weight
    danger_cx = w_cnn * cnn_cx + w_tof * tof_cx
    danger_cy = w_cnn * cnn_cy + w_tof * tof_cy

    # Dirección de escape: opuesta al centroide de peligro
    esc_x = -danger_cx
    esc_y = -danger_cy
    mag = np.sqrt(esc_x**2 + esc_y**2) + 1e-6
    esc_x /= mag
    esc_y /= mag

    # Ángulo de escape en grados (0=derecha, 90=abajo, 180=izquierda, 270=arriba)
    angle = float(np.degrees(np.arctan2(esc_y, esc_x))) % 360

    # Modo según distancia mínima
    if min_dist < DANGER_MIN_DIST:
        mode = 'emergency'
        v    = V_MAX_ESCAPE
    elif min_dist < EVADE_THRESHOLD:
        mode = 'evade'
        v    = V_MAX_ESCAPE * urgency
    elif min_dist < CAUTION_THRESHOLD:
        mode = 'caution'
        v    = V_MAX_ESCAPE * urgency * 0.4
        # En precaución: solo retroceder suavemente, sin lateral agresivo
        esc_x *= 0.3
        esc_y  = -0.7  # retroceder
    else:
        mode = 'idle'
        v    = 0.0

    # Comando final
    vx = float(esc_x * v)              # lateral
    vy = float(esc_y * v)              # frontal/retroceso
    vz = float(V_Z_BOOST * urgency)    # siempre subir un poco al evadir

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
    Retorna (u, v) del vector de escape en coordenadas de imagen.
    u: componente horizontal (+ = derecha en imagen)
    v: componente vertical   (+ = abajo en imagen)
    Para dibujar la flecha sobre el frame de cámara.
    """
    u = cmd.vx / V_MAX_ESCAPE * scale
    v = -cmd.vz / V_Z_BOOST   * scale * 0.3 + cmd.vy / V_MAX_ESCAPE * scale
    return float(u), float(v)
