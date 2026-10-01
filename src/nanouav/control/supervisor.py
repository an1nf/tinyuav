"""
_06_control/supervisor.py — Monitor de estabilidad y recuperación.

Dispara RECOVER si:
  * inclinación > tilt_trip, o |ω| > rate_trip, o caída vz < vz_trip
  * saturación de roll/pitch del mezclador sostenida > sat_time (la de yaw no:
    el mezclador la resuelve cediendo yaw)
En RECOVER el FlightStack:
  * transfiere la referencia sin salto (bumpless: v_ref ← v medida)
  * lleva la velocidad horizontal a 0 con límites suaves, sostiene altitud (≥ z_min)
  * congela el yaw en el ángulo actual y reduce la inclinación máxima a 15°
Sale a NORMAL tras 'hold' s con inclinación, tasas y velocidad dentro de umbral
(histéresis) — evita oscilar entre modos.
"""
import numpy as np


class StabilityMonitor:
    def __init__(self, tilt_trip=35.0, rate_trip=3.0, vz_trip=-1.2, sat_time=0.25,
                 tilt_ok=5.0, rate_ok=0.3, v_ok=0.15, hold=0.5):
        self.tilt_trip = np.radians(tilt_trip)
        self.rate_trip, self.vz_trip, self.sat_time = rate_trip, vz_trip, sat_time
        self.tilt_ok, self.rate_ok, self.v_ok, self.hold = np.radians(tilt_ok), rate_ok, v_ok, hold
        self.mode = 'NORMAL'
        self.t_sat = 0.0
        self.t_ok = 0.0
        self.t_enter = None
        self.events = []          # (t_inicio, t_fin, causa)
        self.cause = ''

    def update(self, st, saturated, t, dt):
        R, w, v = st['R'], st['w'], st['v']
        tilt = np.arccos(np.clip(R[2, 2], -1, 1))
        rate = np.linalg.norm(w[:2])
        self.t_sat = self.t_sat + dt if saturated else 0.0
        entered = False
        if self.mode == 'NORMAL':
            cause = None
            if tilt > self.tilt_trip:
                cause = f'tilt {np.degrees(tilt):.0f}°'
            elif rate > self.rate_trip:
                cause = f'rate {rate:.1f} rad/s'
            elif v[2] < self.vz_trip:
                cause = f'vz {v[2]:.1f} m/s'
            elif self.t_sat > self.sat_time:
                cause = 'saturación'
            if cause:
                self.mode, self.t_enter, self.t_ok, self.cause = 'RECOVER', t, 0.0, cause
                entered = True
        else:
            ok = (tilt < self.tilt_ok and np.linalg.norm(w) < self.rate_ok and
                  np.linalg.norm(v[:2]) < self.v_ok and abs(v[2]) < 0.2)
            self.t_ok = self.t_ok + dt if ok else 0.0
            if self.t_ok >= self.hold:
                self.events.append((self.t_enter, t, self.cause))
                self.mode = 'NORMAL'
        return self.mode, entered
