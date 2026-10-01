"""
R4 · Velocidad máxima teórica (Loquercio et al. 2021, Sci. Robot., ec. 9–11, extendiendo a
Falanga et al. 2019) aplicada a nuestros perfiles, y comparación ESQUIVAR vs FRENAR.

  v_max = s / (t_s + t_p + t_rot + sqrt(2 r_obs / a_lat))       (esquivar, ec. 11)
  t_rot = sqrt(2 φ J / T_max)                                   (ec. 9)
  frenar: v·τ + v²/(2a) ≤ s − r_obs                             (ley 'brake' de evasion.py)

Añadidos propios: t_mot (constante de tiempo de motores), a_lat con altura mantenida
(g·tanφ, empuje ≤ c_max), y la versión con los LÍMITES AUTOIMPUESTOS del stack
(referencias 'fast' a = 2.2 m/s², A_BRAKE = 1.0 m/s², T_REACT = 0.6 s).
"""
import math, json, os, sys
import numpy as np
from nanouav.control.vehicle import PROFILES, G
from nanouav import paths

T_S, T_P = 0.10, 0.06          # 10 fps; captura→comando en E3 (LATENCY)
R_POLE = 0.15


def roll_torque(P):
    ys = np.array([abs(y) for _, y in P.rotor_xy])
    return 2 * P.f_max * ys.mean()          # un lado a f_max, el otro a 0


def dodge_vmax(P, s, t_p=T_P, t_s=T_S, a_cap=None, alt_hold=True, t_extra=0.0):
    J = P.Jv[0]; Tm = roll_torque(P); c_max = 4 * P.f_max / P.mass
    r_obs = R_POLE + P.radius
    best = (0, None)
    for phi in np.radians(np.linspace(1, 80, 400)):
        if alt_hold:
            if G / math.cos(phi) > c_max:
                break
            a_lat = G * math.tan(phi)
        else:
            a_lat = c_max * math.sin(phi)
        if a_cap is not None:
            a_lat = min(a_lat, a_cap)
        t_rot = math.sqrt(2 * phi * J / Tm) + P.tau_motor + t_extra
        v = s / (t_s + t_p + t_rot + math.sqrt(2 * r_obs / a_lat))
        if v > best[0]:
            best = (v, math.degrees(phi), t_rot, math.sqrt(2 * r_obs / a_lat))
    return best


def stop_vmax(P, s, a, tau):
    gap = max(s - P.radius - R_POLE, 0)
    return -a * tau + math.sqrt((a * tau) ** 2 + 2 * a * gap)


out = {}
for name, P in PROFILES.items():
    c_max = 4 * P.f_max / P.mass
    r = {'T/W': P.thrust_to_weight, 'a_lat_max_althold': math.sqrt(c_max ** 2 - G ** 2)}
    for s in (1.5, 3.0, 4.0):
        v, phi, trot, tman = dodge_vmax(P, s)
        r[f's={s}'] = {
            'dodge_physical': round(v, 2), 'phi_deg': round(phi, 1), 't_rot_ms': round(1e3 * trot, 1),
            't_maneuver_ms': round(1e3 * tman, 0),
            'dodge_paper_formula': round(dodge_vmax(P, s, alt_hold=False)[0], 2),
            'dodge_stack_limits(a=2.2,+T_REACT)': round(dodge_vmax(P, s, a_cap=2.2, t_extra=0.6 - T_S - T_P)[0], 2),
            'stop_stack(a=1.0,tau=0.6)': round(stop_vmax(P, s, 1.0, 0.6), 2),
            'stop_physical(a=a_lat_max,tau=0.16)': round(stop_vmax(P, s, r['a_lat_max_althold'], T_S + T_P), 2),
        }
        # sensibilidad: ¿qué término de latencia pesa más? (derivada numérica de v_max)
    for s in (3.0,):
        base = dodge_vmax(P, s, a_cap=2.2)[0]
        r['sens_s=3_a2.2'] = {
            'fps 10→20': round(dodge_vmax(P, s, t_s=0.05, a_cap=2.2)[0] - base, 2),
            'latencia 60→20 ms': round(dodge_vmax(P, s, t_p=0.02, a_cap=2.2)[0] - base, 2),
            'alcance 3→4 m': round(dodge_vmax(P, 4.0, a_cap=2.2)[0] - base, 2),
            'a_lat 2.2→4.0': round(dodge_vmax(P, s, a_cap=4.0)[0] - base, 2),
        }
    out[name] = r
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(str(paths.results_dir()), 'r4_speed_limit.json'), 'w'), indent=1)
