"""
E2-B · Calibración de escala CNN ← ToF en tiempo de ejecución (v2).

Cambios:
  * Entrada normalizada [0,1] → metros vía norm_to_m (antes: doble ×D_MAX).
  * Las 64 zonas ToF se comparan con su caja real de píxeles (FOV 40° dentro de 60°),
    no con el 50 % central de la imagen.
  * Estimador robusto: mediana de las razones tof/cnn por zona válida; se rechaza
    la actualización si hay pocas zonas o la dispersión (MAD relativa) es alta.
  * EMA sobre log(escala) (la escala es multiplicativa).

Referencia: Nadalini et al. IoT-J 2025 (GAP9, mismo concepto, distinta HW).
"""
import numpy as np
from nanouav.config import norm_to_m, pool_to_tof, ZONE_BOXES, TOF_GRID, D_MIN, D_MAX

EMA_ALPHA = 0.15
MIN_SCALE, MAX_SCALE = 0.4, 2.5
MIN_ZONES = 8
MAX_REL_MAD = 0.25


class ToFCalibrator:
    def __init__(self, alpha: float = EMA_ALPHA, init_scale: float = 1.0):
        self.alpha = alpha
        self.log_s = float(np.log(init_scale))
        self.n_updates = 0
        self.scale_hist = []
        self.last_quality = 0.0

    @property
    def scale_ema(self):
        return float(np.exp(self.log_s))

    def _estimate_scale(self, depth_m_raw, tof):
        cnn8 = pool_to_tof(depth_m_raw)
        valid = (tof > D_MIN + 0.05) & (tof < D_MAX - 0.3) & (cnn8 > D_MIN + 0.05) & (cnn8 < D_MAX - 0.3)
        if valid.sum() < MIN_ZONES:
            self.last_quality = 0.0
            return None
        r = tof[valid] / cnn8[valid]
        med = float(np.median(r))
        mad = float(np.median(np.abs(r - med))) / max(med, 1e-6)
        self.last_quality = float(valid.mean())
        if mad > MAX_REL_MAD:
            return None
        return float(np.clip(med, MIN_SCALE, MAX_SCALE))

    def update_and_scale(self, depth_norm, tof):
        """depth_norm: (96,96) [0,1] salida CNN · tof: (8,8) m → (96,96) m calibrada."""
        raw = norm_to_m(depth_norm)
        s = self._estimate_scale(raw, tof)
        if s is not None:
            self.log_s = (1 - self.alpha) * self.log_s + self.alpha * np.log(s)
            self.n_updates += 1
            self.scale_hist.append(s)
        return np.clip(raw * self.scale_ema, D_MIN, D_MAX).astype(np.float32)

    def reset(self, init_scale: float = 1.0):
        self.log_s = float(np.log(init_scale))
        self.n_updates = 0
        self.scale_hist = []

    @property
    def calibrated(self) -> bool:
        return self.n_updates >= 5

    def summary(self) -> dict:
        h = np.array(self.scale_hist) if self.scale_hist else np.array([self.scale_ema])
        return {'scale_ema': self.scale_ema, 'n_updates': self.n_updates,
                'scale_mean': float(h.mean()), 'scale_std': float(h.std()) if len(h) > 1 else 0.0,
                'calibrated': self.calibrated}


def fuse_tof(depth_m, tof, near_ratio=0.8, far_floor=3.0):
    """
    Consistencia CNN↔ToF dentro del FOV del ToF (40° centrales).
    El VL53L5CX en modo "closest target" reporta el objeto MÁS cercano de cada
    zona → ningún píxel de la zona puede estar mucho más cerca que esa lectura.
    Los píxeles de la CNN más cercanos que near_ratio·tof son fantasmas y se
    elevan a esa cota; si la zona ve "lejos" (≥ D_MAX−0.1) se elevan a far_floor.
    Fuera del FOV del ToF la profundidad de la CNN queda intacta.
    """
    out = depth_m.copy()
    for gy in range(TOF_GRID):
        for gx in range(TOF_GRID):
            y0, y1, x0, x1 = ZONE_BOXES[gy][gx]
            t = float(tof[gy, gx])
            floor = far_floor if t >= D_MAX - 0.1 else near_ratio * t
            np.maximum(out[y0:y1, x0:x1], floor, out=out[y0:y1, x0:x1])
    return out


def fuse_tof2(depth_m, tof, pct=25.0, lo=0.5):
    """
    R14: consistencia ToF↔CNN en AMBOS sentidos. Además de la cota inferior de fuse_tof (fantasmas),
    si el percentil bajo de la CNN en una zona queda > 1.1× la lectura del ToF, la CNN no está viendo el
    objeto que el ToF sí ve (sobre-estimación = peligrosa) → esa zona se reescala hacia la lectura
    (factor ≥ lo). Offline (600 escenas): AbsRel 0.301 → 0.252; obstáculos < 1 m con pred > 1.25·gt
    35 % → 25 %; postes finos < 1 m 87 % → 61 %. Robusto a la semántica del ToF (percentil 10–40).
    """
    out = fuse_tof(depth_m, tof)
    for gy in range(TOF_GRID):
        for gx in range(TOF_GRID):
            t = float(tof[gy, gx])
            if t >= D_MAX - 0.1:
                continue
            y0, y1, x0, x1 = ZONE_BOXES[gy][gx]
            z = out[y0:y1, x0:x1]
            q = float(np.percentile(z, pct))
            if q > 1.1 * t:
                z *= max(lo, t / q)
    return out


def fuse(depth_m, tof):
    """Fusión usada en vuelo: fuse_tof2 si UAV_FUSE2=1 (R14), si no fuse_tof (resultados publicados)."""
    from nanouav import settings
    return fuse_tof2(depth_m, tof) if settings.flag('FUSE2') else fuse_tof(depth_m, tof)
