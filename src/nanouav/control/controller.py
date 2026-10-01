"""
_06_control/controller.py — Control en cascada hasta velocidad de motores.

  velocidad (PI + feed-forward de aceleración, límite de inclinación)
    → vector de empuje deseado + yaw → R_des  (Lee et al., SO(3))
    → tasa angular deseada (P sobre error de actitud, prioridad a inclinación)
    → torque (PI de tasa escalado por inercia + compensación giroscópica)
    → mezclador con manejo de saturación (yaw cede primero, luego desplazamiento
      de colectivo tipo airmode, luego escalado de roll/pitch)
    → ω_motor = sqrt(f / kf)

Ganancias derivadas de la constante de tiempo del motor τm (planta dominante):
  rate  K_r ≈ 0.45/τm  (margen de fase ≈ 65°)   att K_a ≈ K_r/4   vel K_v ≈ K_a/3
Así el mismo código sirve para el X3 de Gazebo y para el nano-UAV.
Anti-windup: los integradores se congelan con saturación del mezclador o del
límite de inclinación.
"""
import numpy as np
from .vehicle import G


def vee(M):
    return np.array([M[2, 1], M[0, 2], M[1, 0]])


class CascadedController:
    def __init__(self, P, tilt_max_deg=25.0):
        self.P = P
        tm = P.tau_motor
        kr = 0.45 / tm
        ka = kr / 4.0
        kv = min(ka / 3.0, 2.0)
        self.K = {
            'vel_p': np.array([kv, kv, 2.5]),
            'vel_i': np.array([0.35, 0.35, 0.8]),
            'att_p': np.array([ka, ka, 2.5]),
            'rate_p': np.array([kr, kr, 0.5 * kr]),
            'rate_i': np.array([0.3 * kr, 0.3 * kr, 0.15 * kr]),
            'rate_max': np.array([3.5, 3.5, 1.5]),
        }
        self.tilt_max = np.radians(tilt_max_deg)
        self.vi = np.zeros(3)
        self.ri = np.zeros(3)
        self.saturated = False        # cualquier saturación (congela integradores)
        self.saturated_rp = False     # roll/pitch comprometidos (lo usa el supervisor)
        self.freeze_vi = False        # lo activa el supervisor en RECOVER
        self.tilt_limited = False

    def reset_integrators(self):
        self.vi[:] = 0
        self.ri[:] = 0

    # ── lazo de velocidad → empuje y actitud deseada ──────────────────────
    def velocity_to_attitude(self, v, v_ref, a_ff, psi_ref, dt):
        K = self.K
        e = v_ref - v
        a = a_ff + K['vel_p'] * e + self.vi
        a[2] = np.clip(a[2], -0.6 * G, (self.P.thrust_to_weight * 0.85 - 1) * G)
        a_xy_max = (G + a[2]) * np.tan(self.tilt_max)
        n = np.linalg.norm(a[:2])
        self.tilt_limited = n > a_xy_max
        if self.tilt_limited:
            a[:2] *= a_xy_max / n
        if not (self.saturated or self.tilt_limited or self.freeze_vi):
            self.vi += K['vel_i'] * e * dt
            self.vi = np.clip(self.vi, -2.0, 2.0)
        F = self.P.mass * (a + np.array([0, 0, G]))
        b3 = F / np.linalg.norm(F)
        b1c = np.array([np.cos(psi_ref), np.sin(psi_ref), 0.0])
        b2 = np.cross(b3, b1c)
        b2 /= np.linalg.norm(b2)
        b1 = np.cross(b2, b3)
        R_des = np.column_stack([b1, b2, b3])
        return F, R_des

    # ── lazo de actitud y tasa → torques ──────────────────────────────────
    def attitude_rate(self, R, w, R_des, r_ff, dt):
        K = self.K
        eR = 0.5 * vee(R_des.T @ R - R.T @ R_des)
        w_des = -K['att_p'] * eR
        w_des[2] += r_ff
        w_des = np.clip(w_des, -K['rate_max'], K['rate_max'])
        ew = w_des - w
        if not self.saturated:
            self.ri += K['rate_i'] * ew * dt
            self.ri = np.clip(self.ri, -2.0, 2.0)
        alpha = K['rate_p'] * ew + self.ri
        Jv = self.P.Jv
        return Jv * alpha + np.cross(w, Jv * w), w_des

    # ── mezclador con prioridades ─────────────────────────────────────────
    def mix(self, T, tau):
        P = self.P
        fmax = P.f_max
        base = P.A_inv @ np.array([T, tau[0], tau[1], 0.0])
        fy = P.A_inv @ np.array([0.0, 0.0, 0.0, tau[2]])
        sat = sat_rp = False
        # 1) yaw cede primero
        s = 1.0
        for i in range(4):
            if fy[i] > 1e-12:
                s = min(s, max(0.0, (fmax - base[i]) / fy[i]) if base[i] < fmax else 0.0)
            elif fy[i] < -1e-12:
                s = min(s, max(0.0, -base[i] / fy[i]) if base[i] > 0 else 0.0)
        if s < 1.0:
            sat = True
        f = base + s * fy
        # 2) desplazar colectivo (airmode) para preservar roll/pitch
        lo, hi = f.min(), f.max()
        if lo < 0 or hi > fmax:
            sat = sat_rp = True
            if hi - lo <= fmax:
                f += -lo if lo < 0 else fmax - hi
            else:                                  # 3) escalar roll/pitch
                m = f.mean()
                k = fmax / (hi - lo)
                f = m + k * (f - m)
                f += max(0.0, -f.min()) - max(0.0, f.max() - fmax)
        self.saturated, self.saturated_rp = sat, sat_rp
        f = np.clip(f, 0.0, fmax)
        return np.sqrt(f / P.kf), f

    def update(self, st, v_ref, a_ff, psi_ref, r_ff, dt):
        F, R_des = self.velocity_to_attitude(st['v'], v_ref, a_ff, psi_ref, dt)
        R = st['R']
        T = float(F @ R[:, 2])
        T = max(T, 0.1 * self.P.mass * G)
        tau, w_des = self.attitude_rate(R, st['w'], R_des, r_ff, dt)
        w_cmd, f = self.mix(T, tau)
        return w_cmd, {'T': T, 'tau': tau, 'w_des': w_des, 'f': f, 'R_des': R_des}
