"""
_04_policy/mission.py — Capa de misión sobre EvasionPlanner.

  WaypointMission : recorrido A → B → C … (tramos rectos; la evasión desvía y
                    vuelve a la recta del tramo). Cambio de tramo por radio de
                    aceptación o al sobrepasar el waypoint; frena hasta 0 en el último.
  TargetTracker   : filtro α-β (posición + velocidad) sobre medidas ruidosas del
                    objetivo (UWB / GNSS-RTK / detector visual), tolera pérdidas.
  TargetFollower  : sigue un objetivo móvil a distancia d_follow con feed-forward
                    de su velocidad, apuntando la cámara hacia él.

Todas solo fijan el tramo (A, B), la velocidad de crucero y el modo de yaw del
planificador; la evasión, el filtro de seguridad y la suavidad no cambian.
"""
import math
import numpy as np


class WaypointMission:
    def __init__(self, waypoints, v_cruise=1.0, accept=0.35, z=None, yaw_mode='velocity'):
        self.wps = [np.asarray(w, float) for w in waypoints]     # incluye el inicio A
        self.i = 1
        self.v_cruise, self.accept, self.z, self.yaw_mode = v_cruise, accept, z, yaw_mode
        self.done = False
        self.t_reach = []

    @property
    def segment(self):
        return self.wps[self.i - 1], self.wps[self.i]

    def update(self, planner, p, t=None):
        A, B = self.segment
        last = self.i == len(self.wps) - 1
        d = np.linalg.norm(p[:2] - B[:2])
        seg = B[:2] - A[:2]
        passed = np.dot(p[:2] - A[:2], seg) > np.dot(seg, seg)       # sobrepasó B
        if not last and (d < self.accept or passed):
            self.t_reach.append(t)
            self.i += 1
            A, B = self.segment
            last = self.i == len(self.wps) - 1
        elif last and d < planner.ACCEPT + 0.05 and not self.done:
            self.done = True
            self.t_reach.append(t)
        planner.set_segment(A, B, arrive=last, v_cruise=self.v_cruise)
        planner.yaw_mode = self.yaw_mode
        if self.z is not None:
            planner.z_cruise = float(self.z)
        return self.done


class TargetTracker:
    """Filtro α-β 2D. update(meas|None, dt) → (pos, vel)."""

    def __init__(self, alpha=0.45, beta=0.12, t_coast=0.6):
        self.alpha, self.beta = alpha, beta
        self.x = None
        self.v = np.zeros(2)
        self.t_since = 0.0
        self.t_coast = t_coast            # extrapolar como mucho t_coast s tras perder el objetivo
        self.last_meas = None

    def update(self, meas, dt):
        if self.x is None:
            if meas is None:
                return None, None
            self.x = np.asarray(meas, float)[:2].copy()
            return self.x, self.v
        xp = self.x + self.v * dt
        if meas is None:                     # pérdida: predicción acotada (no perseguir un fantasma)
            self.t_since += dt
            if self.t_since <= self.t_coast:
                self.x = xp
                self.v *= 0.98
            else:
                self.v *= 0.5
            return self.x, self.v
        self.last_meas = np.asarray(meas, float)[:2].copy()
        r = np.asarray(meas, float)[:2] - xp
        self.x = xp + self.alpha * r
        self.v = self.v + (self.beta / dt) * r
        self.t_since = 0.0
        return self.x, self.v

    def predict(self, dt):
        return None if self.x is None else self.x + self.v * dt


class VisualTargetSensor:
    """
    Medida del objetivo como la entregaría un detector a bordo (bbox), no un oráculo.
    Modelo tomado de Su et al., EAAI 177:114890 (2026) — FeatherYOLO + IBVS en Crazyflie:
      * solo hay medida si el objetivo está dentro del FOV de la cámara y no ocluido;
      * rumbo  ← X_offset del centro del bbox (ruido de píxel ∝ ancho del bbox);
      * rango  ← tamaño aparente (su A_scale): d = f·W/w_px, con sesgo por episodio
                 del tamaño supuesto del objetivo (persona alta/baja, ropa) → error ∝ d;
      * falsos negativos crecientes con la distancia + pérdidas por iluminación.
    range_src='tof' (propuesta propia): si el rumbo cae dentro del FOV del ToF (40°),
    el rango sale de la zona ToF/CNN-calibrada que cubre el bbox → error ~cm, sin
    depender del tamaño supuesto. Fuera de los 40° se usa el tamaño aparente.
    """

    def __init__(self, rng, range_src='bbox', hfov=1.047, width=96, tof_hfov=0.698,
                 d_max=5.0, p_det=0.97, size_bias_sd=0.10, px_sd=0.5, rel_sd=0.06):
        self.rng, self.range_src = rng, range_src
        self.hfov, self.tof_half = hfov, tof_hfov / 2
        self.f = (width / 2) / math.tan(hfov / 2)
        self.d_max, self.p_det = d_max, p_det
        self.size_bias = float(np.exp(rng.normal(0, size_bias_sd)))   # r_supuesto / r_real
        self.px_sd, self.rel_sd = px_sd, rel_sd
        self.n, self.n_det, self.t_last, self.max_gap = 0, 0, None, 0.0

    def _occluded(self, world, cam, tgt_xy, target_obj, d):
        D = np.array([tgt_xy[0] - cam[0], tgt_xy[1] - cam[1], 0.0])
        D /= max(np.linalg.norm(D), 1e-9)
        for o in world.objects:
            if o is target_obj:
                continue
            t, _ = o.intersect(cam, D[None, :])
            if float(t[0]) < d - 0.05:
                return True
        return False

    def measure(self, t, p, yaw, tgt_xy, tgt_r, world=None, target_obj=None):
        self.n += 1
        if self.t_last is None:
            self.t_last = t
        dv = np.asarray(tgt_xy[:2], float) - p[:2]
        d = float(np.linalg.norm(dv))
        brg = math.atan2(dv[1], dv[0])
        off = (brg - yaw + math.pi) % (2 * math.pi) - math.pi
        half_ang = math.atan2(tgt_r, max(d, 1e-3))
        ok = abs(off) + 0.5 * half_ang < self.hfov / 2 and 0.3 < d < self.d_max
        if ok:
            p_d = self.p_det * float(np.clip(1.0 - (d - 3.0) / 3.0, 0.3, 1.0))
            ok = self.rng.random() < p_d
        if ok and world is not None:
            cam = np.array([p[0], p[1], p[2] if len(p) > 2 else 0.5])
            ok = not self._occluded(world, cam, tgt_xy, target_obj, d)
        if not ok:
            self.max_gap = max(self.max_gap, t - self.t_last)
            return None
        self.n_det += 1
        self.max_gap = max(self.max_gap, t - self.t_last)
        self.t_last = t
        w_px = 2 * self.f * tgt_r / d
        w_meas = max(w_px + self.rng.normal(0, self.rel_sd * w_px + self.px_sd), 1.0)
        x_px = self.f * math.tan(off) + self.rng.normal(0, 0.5 * self.rel_sd * w_px + self.px_sd)
        off_m = math.atan2(x_px, self.f)
        if self.range_src == 'tof' and abs(off_m) < self.tof_half:
            d_m = d + self.rng.normal(0, 0.02 + 0.02 * d)          # zona ToF/CNN calibrada + r supuesto
        else:
            d_m = 2 * self.f * tgt_r * self.size_bias / w_meas     # rango por tamaño aparente
        a = yaw + off_m
        return p[:2] + d_m * np.array([math.cos(a), math.sin(a)])

    @property
    def stats(self):
        return {'det_rate': self.n_det / max(self.n, 1), 'max_gap_s': self.max_gap,
                'size_bias': self.size_bias}


class TargetFollower:
    """
    Seguimiento a distancia: v_des = v_obj·u + k·(d − d_follow), acotada.
    El carril es la recta dron → objetivo (re-anclada cada frame); la cámara
    apunta al objetivo (yaw_mode='target') para no perderlo del campo visual.

    guidance='ibvs' añade la capa de guiado de Su et al. (2026), ablación G0→G6:
    zona muerta con histéresis sobre el error de distancia (entra < dz_in, sale > dz_out),
    filtro paso-bajo α=0.3 del comando y retroceso acotado (v_back_max): su única
    colisión fue retrocediendo hacia lo que la cámara no ve.
    """

    def __init__(self, d_follow=1.3, v_max=1.2, k=0.8, latency=0.15, target_r=0.25,
                 guidance='base', dz_in=0.10, dz_out=0.20, lp_alpha=0.3, v_back_max=0.35):
        self.d_follow, self.v_max, self.k, self.latency = d_follow, v_max, k, latency
        self.target_r = target_r
        self.guidance, self.dz_in, self.dz_out = guidance, dz_in, dz_out
        self.lp_alpha, self.v_back_max = lp_alpha, v_back_max
        self._hold, self._v_f = False, 0.0
        # reacquisición (pérdida > t_lost): hover y barrido de yaw hacia el lado de salida
        self.reacquire, self.t_lost, self.w_sweep = True, 0.6, 1.0
        self._brg, self._brg_rate = None, 0.0

    def _shape(self, v_ff, err):
        if self.guidance != 'ibvs':
            return v_ff + self.k * err
        if self._hold and abs(err) > self.dz_out:
            self._hold = False
        elif not self._hold and abs(err) < self.dz_in:
            self._hold = True
        v = v_ff + (0.0 if self._hold else self.k * err)
        self._v_f = self.lp_alpha * self._v_f + (1 - self.lp_alpha) * v
        return max(self._v_f, -self.v_back_max)

    def update(self, planner, p, tracker):
        tp = tracker.predict(self.latency)
        if tp is None:
            planner.set_segment(p[:2], p[:2] + [1.0, 0.0], arrive=True, v_cruise=0.0)
            return None
        dvec = tp - p[:2]
        d = float(np.linalg.norm(dvec))
        u = dvec / max(d, 1e-6)
        if tracker.t_since == 0.0:                      # rumbo y su tendencia al último avistamiento
            b = math.atan2(dvec[1], dvec[0])
            if self._brg is not None:
                db = (b - self._brg + math.pi) % (2 * math.pi) - math.pi
                self._brg_rate = 0.7 * self._brg_rate + 0.3 * db
            self._brg = b
        elif self.reacquire and tracker.t_since > self.t_lost and self._brg is not None:
            sgn = 1.0 if self._brg_rate >= 0 else -1.0
            a = self._brg + sgn * min(self.w_sweep * (tracker.t_since - self.t_lost) + 0.35, math.pi)
            look = p[:2] + 2.0 * np.array([math.cos(a), math.sin(a)])
            planner.set_segment(p[:2], p[:2] + 0.5 * u, arrive=True, v_cruise=0.0)
            planner.yaw_mode, planner.look_point = 'target', look
            planner.exclude = (tp, self.target_r + 0.35)
            return d
        v_des = float(np.clip(self._shape(tracker.v @ u, d - self.d_follow), -self.v_max, self.v_max))
        planner.set_segment(p[:2], tp, arrive=False, v_cruise=v_des)
        planner.yaw_mode = 'target'
        planner.look_point = tp
        planner.exclude = (tp, self.target_r + 0.35)
        return d
