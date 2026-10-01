"""
_06_control/flight_stack.py — Integra referencia suave + control + supervisor.

Entrada (desde el planificador de evasión, 10–20 Hz):
    set_target(v_world (3,), yaw)
Salida (a 500 Hz):
    update(state, t, dt) → ω de los 4 motores, info

En el ESP32-S3 esto corre en el núcleo 0 (PID), la percepción en el núcleo 1.
En Gazebo modo 'velocity' solo se usa ReferenceGenerator (el plugin del X3
cierra los lazos internos); en modo 'motors' se usa el FlightStack completo.
"""
import numpy as np
from .reference import JerkLimited, YawShaper, wrap
from .controller import CascadedController
from .supervisor import StabilityMonitor
from .quad_sim import euler_from_rot

# Límites de referencia (suavidad) — normal y recuperación
REF_LIMITS = {
    'normal':  {'a': (1.2, 1.2, 0.8), 'j': (3.0, 3.0, 2.5), 'yaw_r': 0.8, 'yaw_a': 1.5},
    'recover': {'a': (1.5, 1.5, 1.0), 'j': (5.0, 5.0, 4.0), 'yaw_r': 0.0, 'yaw_a': 1.5},
    'fast':    {'a': (2.2, 2.2, 1.0), 'j': (6.0, 6.0, 3.0), 'yaw_r': 1.2, 'yaw_a': 1.5},
    'evade':   {'a': (6.0, 6.0, 1.0), 'j': (30.0, 30.0, 3.0), 'yaw_r': 1.2, 'yaw_a': 1.5},   # R10: escape dinámico
}


def yaw_accel_limit(P, margin=0.5):
    """
    Aceleración de yaw que el vehículo puede sostener sin comprometer empuje ni
    roll/pitch: km·Σ|Δf| / Jz con Δf = min(f_max − f_hover, f_hover), × margen.
    X3: ≈0.57 rad/s² (autoridad de yaw baja, km pequeño); nano: > 1.5 → se acota.
    """
    df = min(P.f_max - P.hover_f, P.hover_f)
    return float(min(1.5, margin * P.km * 4 * df / P.J[2]))


class ReferenceGenerator:
    """Velocidad y yaw suavizados (jerk/aceleración limitados)."""

    def __init__(self, limits=REF_LIMITS['normal'], yaw0=0.0):
        self.vel = JerkLimited(limits['a'], limits['j'])
        self.yaw = YawShaper(limits['yaw_r'], limits['yaw_a'], yaw0)
        self.v_target = np.zeros(3)
        self.yaw_target = yaw0

    def set_evade(self, on):
        """R10: perfil agresivo solo durante el escape ante obstáculos dinámicos."""
        if getattr(self, '_evade', False) == on or self.sup.mode != 'NORMAL':
            return
        self._evade = on
        L = REF_LIMITS['evade' if on else self.limits]
        self.ref.vel.set_limits(L['a'], L['j'])
        self.ctrl.tilt_max = np.radians(40 if on else 25)

    def set_target(self, v, yaw):
        self.v_target = np.asarray(v, float)
        self.yaw_target = float(yaw)

    def update(self, dt):
        v, a = self.vel.update(self.v_target, dt)
        psi, r = self.yaw.update(self.yaw_target, dt)
        return v, a, psi, r


class FlightStack:
    def __init__(self, P, yaw0=0.0, z_min=0.35, shaping=True, limits='normal'):
        self.P = P
        self.limits = limits
        lim = dict(REF_LIMITS[limits], yaw_a=yaw_accel_limit(P))
        self.ref = ReferenceGenerator(lim, yaw0=yaw0)
        self.ctrl = CascadedController(P)
        self.sup = StabilityMonitor()
        self.z_min = z_min
        self.shaping = shaping
        self.hold_z = None
        self.hold_yaw = yaw0

    def set_evade(self, on):
        """R10: perfil agresivo solo durante el escape ante obstáculos dinámicos."""
        if getattr(self, '_evade', False) == on or self.sup.mode != 'NORMAL':
            return
        self._evade = on
        L = REF_LIMITS['evade' if on else self.limits]
        self.ref.vel.set_limits(L['a'], L['j'])
        self.ctrl.tilt_max = np.radians(40 if on else 25)

    def set_target(self, v, yaw):
        if self.sup.mode == 'NORMAL':
            self.ref.set_target(v, yaw)

    def update(self, st, t, dt):
        mode, entered = self.sup.update(st, self.ctrl.saturated_rp, t, dt)
        if entered:                                  # transferencia sin salto
            _, _, yaw = euler_from_rot(st['R'])
            self.ref.vel.reset(st['v'], self.ref.vel.a)   # conserva a_ref → sin salto de jerk
            self.ref.yaw.reset(yaw)
            self.hold_yaw = yaw
            self.hold_z = max(st['p'][2], self.z_min)
            self.ctrl.vi[:2] = 0
            self.ctrl.freeze_vi = True
            self.ctrl.tilt_max = np.radians(15)
            L = REF_LIMITS['recover']
            self.ref.vel.set_limits(L['a'], L['j'])
        if mode == 'RECOVER':
            vz = np.clip(1.5 * (self.hold_z - st['p'][2]), -0.3, 0.5)
            self.ref.v_target = np.array([0.0, 0.0, vz])
            self.ref.yaw_target = self.hold_yaw
        elif self.ctrl.tilt_max < np.radians(25):    # salida de RECOVER
            self.ctrl.tilt_max = np.radians(25)
            self.ctrl.freeze_vi = False
            self.ref.vel.reset(st['v'], self.ref.vel.a)   # bumpless también a la salida
            L = REF_LIMITS[self.limits]
            self.ref.vel.set_limits(L['a'], L['j'])

        if self.shaping:
            v_ref, a_ff, psi_ref, r_ff = self.ref.update(dt)
        else:                                        # ablación: escalón directo
            v_ref, a_ff, psi_ref, r_ff = self.ref.v_target, np.zeros(3), self.ref.yaw_target, 0.0
        w_cmd, info = self.ctrl.update(st, v_ref, a_ff, psi_ref, r_ff, dt)
        info.update(v_ref=v_ref, a_ref=a_ff, psi_ref=psi_ref, mode=mode)
        return w_cmd, info
