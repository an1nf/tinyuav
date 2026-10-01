"""
_04_policy/evasion.py — Planificador reactivo de evasión (reemplaza la semielipse).

Problemas de la v11 que corrige:
  * vy saltaba de 0 a B·W al iniciar la semielipse (escalón de velocidad) → ahora
    el planificador solo entrega OBJETIVOS; la suavidad la garantiza el
    ReferenceGenerator (aceleración y jerk acotados).
  * Usaba la lista de obstáculos del SDF (ground truth) para la altura objetivo y
    el obstáculo más cercano → ahora todo sale de la percepción.
  * El suelo visto por la cámara/ToF (inclinados 8.6° abajo) se confundía con
    obstáculos → se retro-proyecta cada píxel a 3D y se descarta suelo y lo que
    está fuera de la banda de altura del dron.

Pipeline por frame:
  profundidad calibrada (96×96) + ToF (8×8) + pose de captura
    → nube de puntos mundo → filtro suelo / banda de altura
    → grilla 2D local con memoria (decaimiento + limpieza por rayos libres)
    → perfil lateral y(x) por programación dinámica (4 m adelante, pendiente
      acotada, histéresis con el plan anterior) → permite slalom entre obstáculos
    → objetivo (v_mundo, yaw); modo {CRUISE, AVOID, RETURN, BRAKE} informativo

Marco del camino: eje x a lo largo de path_heading, y lateral, carril y = lane_y.
"""
import math
import numpy as np
import cv2

from nanouav.config import CAM_HFOV, CAM_PITCH, CAM_OFFSET, TOF_HFOV, TOF_GRID, D_MAX
from nanouav.scene.renderer import camera_rays, body_rot, rot_y


class LocalGrid:
    """
    Grilla 2D mundo con conteo de impactos. El olvido lo hace sobre todo la
    limpieza por espacio libre (rayos que atraviesan la celda, ×0.5): así un
    obstáculo que sale del campo visual (al pasar a su lado) NO se olvida, y un
    fantasma de la CNN se borra en cuanto se vuelve a observar esa zona.
    """

    def __init__(self, res=0.1, xr=(-4.0, 14.0), yr=(-6.0, 6.0), decay=0.995):
        self.res, self.x0, self.y0 = res, xr[0], yr[0]
        self.nx = int((xr[1] - xr[0]) / res)
        self.ny = int((yr[1] - yr[0]) / res)
        self.g = np.zeros((self.nx, self.ny), np.float32)
        self.seen = np.zeros((self.nx, self.ny), bool)      # celda observada alguna vez
        self.decay = decay

    def _idx(self, pts):
        ix = ((pts[:, 0] - self.x0) / self.res).astype(int)
        iy = ((pts[:, 1] - self.y0) / self.res).astype(int)
        ok = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        return ix[ok], iy[ok]

    def update(self, occ, free, decay=True):
        if decay:
            self.g *= self.decay
        if len(free):
            ix, iy = self._idx(free)
            self.g[ix, iy] *= getattr(self, 'free_k', 0.5)
            self.seen[ix, iy] = True
        if len(occ):
            ix, iy = self._idx(occ)
            # como máximo +1 por celda y por fuente en cada frame: un solo frame de la
            # CNN no basta para marcar ocupado (umbral 1.5); el ToF (peso 2) sí
            if len(ix):
                self.seen[ix, iy] = True
                cells = np.unique(np.stack([ix, iy], 1), axis=0)
                self.g[cells[:, 0], cells[:, 1]] += 1.0
            np.minimum(self.g, 6.0, out=self.g)

    def mark_seen(self, c, r):
        ix0, ix1 = int((c[0] - r - self.x0) / self.res), int((c[0] + r - self.x0) / self.res) + 1
        iy0, iy1 = int((c[1] - r - self.y0) / self.res), int((c[1] + r - self.y0) / self.res) + 1
        self.seen[max(ix0, 0):ix1, max(iy0, 0):iy1] = True

    def unknown_along(self, p, u, half_w, L=1.2, step=0.1):
        """Distancia a la primera celda NUNCA observada en el corredor p + s·u (s ≤ L)."""
        perp = np.array([-u[1], u[0]])
        for s_ in np.arange(0.2, L + 1e-6, step):
            pts = p + s_ * u + np.outer(np.linspace(-half_w, half_w, 5), perp)
            ix = ((pts[:, 0] - self.x0) / self.res).astype(int)
            iy = ((pts[:, 1] - self.y0) / self.res).astype(int)
            ok = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
            if (~ok).any() or not self.seen[ix, iy].all():
                return float(s_)
        return np.inf

    def occupied(self, thr=1.5):
        ix, iy = np.nonzero(self.g > thr)
        return np.stack([self.x0 + (ix + 0.5) * self.res, self.y0 + (iy + 0.5) * self.res], 1)


_TOF_RAYS = {}


def _tof_rays():
    """Rayos (cuerpo) por el centro de cada zona ToF y su cos respecto al eje."""
    if 'r' not in _TOF_RAYS:
        half = math.tan(TOF_HFOV / 2)
        c = (np.arange(TOF_GRID) + 0.5) / TOF_GRID * 2 * half - half
        U, V = np.meshgrid(c, c)
        d = np.stack([np.ones_like(U), -U, -V], -1)
        n = np.linalg.norm(d, axis=-1, keepdims=True)
        _TOF_RAYS['r'] = (d / n, 1.0 / n[..., 0])
    return _TOF_RAYS['r']


class EvasionPlanner:
    def __init__(self, radius=0.36, v_cruise=0.5, z_cruise=0.65, path_heading=0.0,
                 lane_y=0.0, margin=0.25, inflate=0.0, v_max=None, a_brake=0.8,
                 grid_bounds=((-10.0, 30.0), (-15.0, 15.0))):
        self.radius = radius
        self.half_w = radius + margin              # semiancho del corredor
        self.v_cruise, self.z_cruise = v_cruise, z_cruise
        self.psi_path, self.lane = path_heading, lane_y
        self.D_SLOW = 1.8                          # empieza a frenar si el plan está bloqueado
        self.D_STOP = radius + 0.35                # velocidad de avance → 0
        self.D_EMERG = radius + 0.10               # holgura mínima → maniobra de escape
        self.A_BRAKE = a_brake                     # desaceleración asumida por el filtro de seguridad
        self.T_REACT = 0.4                         # latencia percepción + retardo referencia/control (s)
        self.GAP_BIAS = 0.10                       # error de la grilla (celda 0.1 m)
        self.Z_BAND = 0.30                         # ± altura relevante
        self.SHADOW, self.res_shadow = 0.3, 0.1        # extensión asumida tras superficies vistas
        self.INFLATE = inflate                     # inflado lateral opcional (m); 0 = resultados publicados
        self.GROUND_Z = 0.15                       # por debajo = suelo (tolera error de profundidad)
        self.Z_MAP_MIN = 0.40                      # no mapear durante el despegue
        self.K_LAT, self.V_LAT = 1.0, 0.45
        self.K_Z, self.V_Z = 1.2, 0.35
        self.YAW_FOLLOW, self.YAW_MAX = 0.5, 0.35
        # planificación lateral (DP)
        self.DX, self.NX = 0.2, 20                 # 4 m hacia adelante
        self.DY, self.NY = 0.1, 23                 # carril ± 2.3 m
        self.SLOPE_MAX = 0.8                       # |dy/dx| máx (≈39°)
        self.SIDESTEP = 1.0                        # desplazamiento lateral puro permitido (m)
        self.W_LANE, self.W_DY, self.W_PREV, self.W_BLOCK = 0.3, 1.0, 0.4, 100.0
        self.plan_x = self.plan_y = None
        self.v_prev = None
        self.bin0_blocked = False
        self.TARGET_ALPHA = 0.5                    # EMA de objetivos a 10 Hz (τ ≈ 0.15 s)
        self.grid = LocalGrid(xr=grid_bounds[0], yr=grid_bounds[1])
        # misión: segmento A→B (marco del camino con origen en A)
        self.v_max = v_max if v_max is not None else v_cruise
        self.origin = np.zeros(2)
        self.goal = None                           # B (mundo, xy) o None = avanzar indefinidamente
        self.arrive = False                        # frenar hasta 0 en B
        self.A_ARRIVE = 0.6                        # desaceleración de llegada (m/s²)
        self.ACCEPT = 0.25                         # radio de llegada (m)
        self.D_SENSE = 3.0                         # alcance fiable de percepción (m) → tope de velocidad
        self.yaw_mode = 'path'                     # 'path' | 'velocity' | 'target'
        self.look_point = None                     # para yaw_mode='target'
        self.yaw_cmd = path_heading
        self.exclude = None                        # (xy, radio): región del objetivo seguido
        self.FOV_OK, self.FOV_LIM, self.V_BLIND = math.radians(20), math.radians(45), 0.0
        self._seed_seen = False
        self._stall, self.V_CREEP = 0.0, 0.15        # avance lento si el desconocido bloquea > 1.5 s
        self.speed_law = 'ramp'                    # 'ramp' (E3) | 'brake' (misiones / alta velocidad)
        self.plan_w = None                         # plan anterior en coordenadas mundo (histéresis)
        # R5 (Loquercio 2021): modo 'consistente' — restricciones de módulos compatibles entre sí
        #   * pendiente del plan ≤ tan(FOV_OK) (antes el DP permitía 45° = FOV_LIM → k_fov = 0)
        #   * seguimiento con horizonte v·T_LOOK (antes solo el 1er bin: viraje tardío)
        #   * el retroceso (backoff) no lo anula el limitador de FOV (se retrocede sobre lo ya visto)
        self.CONSISTENT = False
        self.T_LOOK = 0.6
        self.BACKOFF_MAX = 0.6                     # m de retroceso ciego por bloqueo (modo consistente)
        # R9 (Shang et al., NeurIPS 2025 — auto-calibración contra 'snapshots'): corrección de deriva
        # alineando cada observación con el mapa local ya construido (el mapa es la snapshot)
        self.DRIFT_CORR = False
        self.p_corr = np.zeros(2)
        self.DC_R, self.DC_STEP, self.DC_GAIN = 0.25, 0.05, 0.5
        self.dc_log = []
        # R9b: estimación del sesgo de velocidad alineando el frame actual con un keyframe de hace
        # ~VO_DT s (snapshot reciente). El lazo (e3) resta v_bias_hat a la velocidad estimada.
        self.VO = False
        self.VO_DT, self.VO_R, self.VO_STEP, self.VO_K = 1.0, 0.4, 0.05, 0.5
        self.p_jump = np.zeros(2)
        # R10b (Shang et al. 2025, campos receptivos ERF/IRF de anuros, adaptado a profundidad):
        #   excitación = celda ocupada AHORA que antes se vio LIBRE; inhibición = lo ya mapeado o
        #   nunca visto no dispara (suprime lo estático, incluido lo que entra al FOV al avanzar).
        #   Los grupos dinámicos se siguen (velocidad α-β) y su trayectoria predicha entra al plan.
        self.RF_DYN = False
        self.DYN_HORIZON, self.DYN_R = 1.5, 0.18
        self.V_ESC = 1.2                           # m/s escape lateral ante objeto dinámico
        self._tracks = []
        self.dyn_log = []
        self.v_bias_hat = np.zeros(2)
        self._kf = []
        self.vo_log = []
        self.mode = 'CRUISE'
        self.cp, self.sp = math.cos(path_heading), math.sin(path_heading)
        self.last = {}

    # ── geometría ─────────────────────────────────────────────────────────
    def set_segment(self, A, B, arrive=True, v_cruise=None):
        """Define el tramo A→B: el carril es la recta A→B (marco del camino con origen A)."""
        A, B = np.asarray(A, float)[:2], np.asarray(B, float)[:2]
        d = B - A
        if np.linalg.norm(d) > 1e-3:
            self.psi_path = math.atan2(d[1], d[0])
            self.cp, self.sp = math.cos(self.psi_path), math.sin(self.psi_path)
        self.origin, self.goal, self.arrive, self.lane = A, B, arrive, 0.0
        if v_cruise is not None:
            self.v_cruise = float(v_cruise)

    def v_sense_cap(self):
        """Velocidad máx. tal que se pueda frenar dentro del alcance fiable de percepción."""
        a, tau = self.A_BRAKE, self.T_REACT
        gap = max(self.D_SENSE - self.radius - self.GAP_BIAS - 0.25, 0.0)
        return -a * tau + math.sqrt((a * tau) ** 2 + 2 * a * gap)

    def _to_path(self, xy):
        xy = np.asarray(xy, float) - self.origin
        x, y = xy[..., 0], xy[..., 1]
        return np.stack([self.cp * x + self.sp * y, -self.sp * x + self.cp * y], -1)

    def _from_path(self, pp):
        x, y = pp[..., 0], pp[..., 1]
        return np.stack([self.cp * x - self.sp * y, self.sp * x + self.cp * y], -1) + self.origin

    def _cam(self, pose):
        p, yaw, pitch, roll = pose
        Rb = body_rot(yaw, pitch, roll)
        return np.asarray(p) + Rb @ np.asarray(CAM_OFFSET), Rb @ rot_y(CAM_PITCH)

    @staticmethod
    def flying_pixels(depth, rel=0.2):
        """
        Píxeles en discontinuidades de profundidad: la CNN suaviza los bordes y
        produce profundidades intermedias que, retro-proyectadas, crean 'estelas'
        fantasma detrás de los obstáculos (zonas ocluidas que ya no se limpian).
        """
        k = np.ones((5, 5), np.uint8)
        dmax = cv2.dilate(depth, k)
        dmin = cv2.erode(depth, k)
        return (dmax - dmin) > rel * dmin

    def _backproject(self, depth, rays, cos_ax, O, Rc, step=1, valid=None):
        d = depth[::step, ::step]
        r = rays[::step, ::step] @ Rc.T
        rng_ = d / cos_ax[::step, ::step]
        P = O + r * rng_[..., None]
        far = d >= D_MAX - 0.3
        if valid is not None:
            ok = valid[::step, ::step]
            P, rng_, r, far = P[ok], rng_[ok], r[ok], far[ok]
        # espacio libre: muestras a lo largo del rayo antes del impacto
        F = np.concatenate([(O + r * (f * rng_)[..., None]).reshape(-1, 3)
                            for f in getattr(self, 'FREE_FRACS', (0.1, 0.2, 0.3, 0.5, 0.7, 0.85))])
        return P.reshape(-1, 3), far.reshape(-1), F

    def _filter(self, P, far, z, O=None):
        keep = (~far) & (P[:, 2] > self.GROUND_Z) & (np.abs(P[:, 2] - z) < self.Z_BAND)
        P = P[keep]
        if O is None or not len(P):
            return P
        # sombra de oclusión: la cara trasera no se ve → se asume que el obstáculo
        # se extiende SHADOW m detrás de la superficie observada (se limpia en cuanto
        # rayos libres atraviesan esa zona desde otro ángulo)
        d = P[:, :2] - np.asarray(O)[:2]
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-6)
        ext = [P]
        for k in np.arange(self.res_shadow, self.SHADOW + 1e-6, self.res_shadow):
            Q = P.copy()
            Q[:, :2] += d * k
            ext.append(Q)
        if self.INFLATE > 0:          # inflado lateral (esquinas vistas de canto)
            perp = np.stack([-d[:, 1], d[:, 0]], 1)
            for sgn in (-1, 1):
                Q = P.copy()
                Q[:, :2] += sgn * perp * self.INFLATE
                ext.append(Q)
        return np.concatenate(ext)

    def _free(self, F, z):
        return F[(F[:, 2] > self.GROUND_Z) & (np.abs(F[:, 2] - z) < self.Z_BAND)]

    # ── planificación lateral por programación dinámica ───────────────────
    def _plan(self, occ_p, xd, yd):
        """
        Perfil lateral y(x) libre de obstáculos en [xd, xd+LOOK] (DP sobre una
        grilla x×y). Costos: desvío del carril, cambio lateral, diferencia con el
        plan anterior (histéresis → sin oscilar entre izquierda/derecha).
        Restricción de pendiente |Δy/Δx| ≤ SLOPE_MAX (maniobra suave y factible).
        Retorna (xs, ys, x_bloqueo) — x_bloqueo = inf si el plan es libre.
        """
        xs = xd + np.arange(1, self.NX + 1) * self.DX
        ys = self.lane + np.arange(-self.NY, self.NY + 1) * self.DY
        nx, ny = len(xs), len(ys)
        blocked = np.zeros((nx, ny), bool)
        if len(occ_p):
            near = occ_p[(occ_p[:, 0] > xd - self.half_w) & (occ_p[:, 0] < xs[-1] + self.half_w)]
            if len(near):
                dxm = xs[:, None] - near[None, :, 0]                  # (nx, M)
                for j, y in enumerate(ys):
                    d2 = dxm ** 2 + (y - near[None, :, 1]) ** 2
                    blocked[:, j] = (d2 < self.half_w ** 2).any(axis=1)
        prev = np.full(nx, yd)
        if self.plan_w is not None:
            pp = self._to_path(self.plan_w)
            o = np.argsort(pp[:, 0])
            if pp[o[-1], 0] > xs[0]:
                prev = np.interp(xs, pp[o, 0], pp[o, 1])
        node = (self.W_LANE * np.abs(ys[None, :] - self.lane) +
                self.W_PREV * np.abs(ys[None, :] - prev[:, None]) + self.W_BLOCK * blocked)
        max_step = int(round(self.SLOPE_MAX * self.DX / self.DY))
        if self.CONSISTENT:                              # pendiente ≤ tan(FOV_OK): el FOV no frena el plan
            max_step = max(1, int(math.floor(math.tan(self.FOV_OK) * self.DX / self.DY + 1e-6)))
        J = np.full((nx, ny), np.inf)
        arg = np.zeros((nx, ny), int)
        reach = np.abs(ys - yd) <= self.SLOPE_MAX * self.DX + 1e-6
        cost0 = self.W_DY * np.abs(ys - yd)
        # paso lateral (sidestep): cualquier y alcanzable deslizando de costado por
        # celdas libres contiguas del primer bin — evita bloqueos con v_avance = 0
        j0 = int(np.argmin(np.abs(ys - yd)))
        side = np.zeros(ny, bool)
        for step in (1, -1):
            j = j0
            while 0 <= j < ny and not blocked[0, j] and abs(ys[j] - yd) <= self.SIDESTEP:
                side[j] = True
                j += step
        if self.CONSISTENT and (reach & ~blocked[0]).any():
            side[:] = False          # paso lateral puro solo si no hay salida con pendiente factible
        cost0 = np.where(reach, cost0, 3.0 * cost0)
        J[0] = np.where(reach | side, node[0] + cost0, np.inf)
        idx = np.arange(ny)
        for i in range(1, nx):
            best = np.full(ny, np.inf)
            barg = np.zeros(ny, int)
            for k in range(-max_step, max_step + 1):
                src = idx - k
                ok = (src >= 0) & (src < ny)
                c = np.full(ny, np.inf)
                c[ok] = J[i - 1, src[ok]] + self.W_DY * abs(k) * self.DY
                better = c < best
                best[better], barg[better] = c[better], src[better]
            J[i] = best + node[i]
            arg[i] = barg
        path = np.zeros(nx, int)
        path[-1] = int(np.argmin(J[-1]))
        for i in range(nx - 1, 0, -1):
            path[i - 1] = arg[i, path[i]]
        yp = ys[path]
        hit = blocked[np.arange(nx), path]
        x_block = float(xs[hit.argmax()] - xd) if hit.any() else np.inf
        self.plan_x, self.plan_y = xs, yp
        self.plan_w = self._from_path(np.stack([xs, yp], 1))
        self.bin0_blocked = bool(blocked[0, j0])
        return xs, yp, x_block

    def _front(self, occ_p, xd, yc, look=4.0):
        """Distancia al primer obstáculo en el corredor recto y = yc (freno de emergencia)."""
        if not len(occ_p):
            return np.inf
        dx = occ_p[:, 0] - xd
        m = (np.abs(occ_p[:, 1] - yc) < self.half_w) & (dx > 0) & (dx < look)
        return float(dx[m].min()) if m.any() else np.inf

    def _perceive(self, state, depth_m=None, tof=None, pose=None):
        """Percepción → grilla local (compartido por todos los planificadores).
        Retorna (p corregida (3,), z, celdas ocupadas xy (+ predicción dinámica))."""
        p = np.asarray(state['p'], float).copy()
        if pose is None:
            pose = (p, state['yaw'], 0.0, 0.0)
        if self.DRIFT_CORR:
            p[:2] += self.p_corr
            pose = (np.asarray(pose[0], float) + np.r_[self.p_corr, 0.0], *pose[1:])
        z = p[2]
        O, Rc = self._cam(pose)
        occ, free, surf = [], [], []
        if z < self.Z_MAP_MIN:
            depth_m = tof = None
        if depth_m is not None:
            rays, cos_ax = camera_rays(CAM_HFOV)
            P, far, F = self._backproject(depth_m, rays, cos_ax, O, Rc, step=3,
                                          valid=~self.flying_pixels(depth_m))
            occ.append(self._filter(P, far, z, O))
            free.append(self._free(F, z))
            surf.append(P[(~far) & (P[:, 2] > self.GROUND_Z) & (np.abs(P[:, 2] - z) < self.Z_BAND)])
        if tof is not None:
            rays, cos_ax = _tof_rays()
            P, far, F = self._backproject(tof, rays, cos_ax, O, Rc)
            Pt = self._filter(P, far, z, O)
            occ += [Pt, Pt]                                  # ToF pesa doble (escala absoluta)
            surf += [P[(~far) & (P[:, 2] > self.GROUND_Z) & (np.abs(P[:, 2] - z) < self.Z_BAND)]] * 2
            free.append(self._free(F, z))
        free = np.concatenate(free) if free else np.zeros((0, 3))
        if self.VO and surf:
            self._vo_update(state.get('t'), np.concatenate(surf))
        dyn_c = []
        if self.RF_DYN:
            dyn_c = self._rf_detect(state.get('t'), np.concatenate(surf) if surf else np.zeros((0, 3)), p)
            if dyn_c:
                # no memorizar lo dinámico (sin estelas en la grilla)
                C_ = np.array(dyn_c)
                occ = [o[np.min(np.hypot(o[:, 0, None] - C_[None, :, 0], o[:, 1, None] - C_[None, :, 1]), 1) > 0.35]
                       if len(o) else o for o in occ]
        if self.DRIFT_CORR and occ:
            d = self._drift_align(np.concatenate(occ))
            if d is not None:
                self.p_corr += d
                p[:2] += d
                occ = [o + np.r_[d, 0.0] for o in occ]
                free = free + np.r_[d, 0.0]
        if not occ:
            self.grid.update(np.zeros((0, 3)), free)
        for k, o in enumerate(occ):                      # cada fuente suma por separado
            self.grid.update(o, free if k == 0 else np.zeros((0, 3)), decay=(k == 0))

        occ_xy = self.grid.occupied()
        if self.RF_DYN:
            pred = self._rf_predicted()
            if len(pred):
                occ_xy = np.concatenate([occ_xy.reshape(-1, 2), pred]) if len(occ_xy) else pred
        return p, z, occ_xy

    # ── actualización ─────────────────────────────────────────────────────
    def _rf_detect(self, t, pts, p):
        """Excitación − inhibición sobre la grilla → centroides dinámicos; actualiza pistas."""
        g = self.grid
        if t is None:
            return []
        if len(pts):
            ix = ((pts[:, 0] - g.x0) / g.res).astype(int)
            iy = ((pts[:, 1] - g.y0) / g.res).astype(int)
            ok = (ix >= 0) & (ix < g.nx) & (iy >= 0) & (iy < g.ny)
            pts, ix, iy = pts[ok], ix[ok], iy[ok]
            exc = g.seen[ix, iy] & (g.g[ix, iy] < 0.3)          # visto libre antes → ahora ocupado
            exc &= np.hypot(pts[:, 0] - p[0], pts[:, 1] - p[1]) < 4.0
            d_inh = getattr(self, 'RF_EDGE_INH', 0.0)
            if d_inh > 0:
                # R15: inhibición lateral junto a contornos ESTÁTICOS (como el IRF): el sangrado de bordes de
                # la CNN, que cambia con el punto de vista (sobre todo al rotar), dispara excitaciones pegadas
                # a los obstáculos. 'Estático' = ocupado de forma continua hace > RF_STATIC_AGE s (la estela de
                # un objeto que se mueve no cuenta: cada celda suya está ocupada una fracción de segundo).
                occ_now = g.g > 1.5
                if getattr(self, '_occ_t', None) is None or self._occ_t.shape != g.g.shape:
                    self._occ_t = np.full(g.g.shape, np.inf, np.float32)
                newly = occ_now & ~np.isfinite(self._occ_t)
                self._occ_t[newly] = t
                self._occ_t[~occ_now] = np.inf
                static = occ_now & (t - self._occ_t > getattr(self, 'RF_STATIC_AGE', 1.0))
                if static.any() and exc.any():
                    dist = cv2.distanceTransform((~static).astype(np.uint8), cv2.DIST_L2, 3) * g.res
                    exc &= dist[ix, iy] > d_inh
            D = pts[exc, :2]
        else:
            D = np.zeros((0, 2))
        cents = []
        used = np.zeros(len(D), bool)
        for k in range(len(D)):                                  # agrupamiento voraz r = 0.3 m
            if used[k]:
                continue
            m = (~used) & (np.hypot(D[:, 0] - D[k, 0], D[:, 1] - D[k, 1]) < 0.3)
            used |= m
            if m.sum() >= getattr(self, 'RF_MIN_PTS', 3):
                cents.append(D[m].mean(0))
        # asociación con pistas (vecino más cercano, compuerta 0.6 m) + α-β
        new_tracks = []
        for c in cents:
            best, bd = None, 0.6
            for tr in self._tracks:
                pr = tr['p'] + tr['v'] * (t - tr['t'])
                d = np.hypot(*(c - pr))
                if d < bd:
                    best, bd = tr, d
            if best is None:
                new_tracks.append({'p': c, 'v': np.zeros(2), 't': t, 'n': 1, 'p0': c.copy(), 't0': t})
            else:
                dt = max(t - best['t'], 0.05)
                pr = best['p'] + best['v'] * dt
                r = c - pr
                best['p'] = pr + 0.6 * r
                best['v'] = best['v'] + (0.4 / dt) * r
                best['t'], best['n'] = t, best['n'] + 1
                best['hit'] = True
        self._tracks = [tr for tr in self._tracks if t - tr['t'] < 0.35] + new_tracks
        self.dyn_log.append((t, len(cents), [(tr['p'].copy(), tr['v'].copy(), tr['n']) for tr in self._tracks]))
        return cents

    def _rf_confirmed(self, tr, n_min=3):
        """Pista dinámica confirmada. R15 (DroneMOT: el ruido del detector domina en hover): además de
        n y |v|, opcionalmente exige desplazamiento NETO ≥ RF_NET_DISP desde que nació la pista — el
        parpadeo de la CNN produce pistas con velocidad aparente pero sin desplazamiento coherente."""
        if tr['n'] < n_min or np.hypot(*tr['v']) < 0.3:
            return False
        nd = getattr(self, 'RF_NET_DISP', 0.0)
        if nd > 0 and float(np.hypot(*(tr['p'] - tr.get('p0', tr['p'])))) < nd:
            return False
        if getattr(self, 'RF_EGO_REJ', 0) and getattr(self, '_ego', None) is not None:
            # R15 (DroneMOT, predicción con movimiento dual): una pista cuya velocidad la explica la
            # ROTACIÓN propia (v ≈ ω × r, el 'objeto' gira con la cámara) es un artefacto de la CNN fijo en
            # la imagen, no un objeto que se mueve en el mundo.
            p_d, om = self._ego
            if abs(om) > 0.2:
                r = tr['p'] - p_d[:2]
                v_rot = om * np.array([-r[1], r[0]])
                nv, nr = np.hypot(*tr['v']), np.hypot(*v_rot)
                if nr > 0.1 and float(tr['v'] @ v_rot) / (nv * nr) > 0.7 and 0.4 < nv / nr < 1.8:
                    return False
        return True

    def _rf_escape(self, p, v):
        """
        Respuesta tipo campo de potencial (gradiente lejos de la trayectoria del objeto): si el
        punto de máxima aproximación (CPA) de una pista confirmada cae dentro de DYN_HORIZON con
        distancia < holgura, velocidad de escape perpendicular a la velocidad del objeto.
        """
        best = None
        for tr in self._tracks:
            if not self._rf_confirmed(tr, 3):
                continue
            rp, rv = tr['p'] - p[:2], tr['v'] - v[:2]
            vv = float(rv @ rv)
            if vv < 1e-6:
                continue
            t_cpa = -float(rp @ rv) / vv
            if not (0.0 < t_cpa < self.DYN_HORIZON):
                continue
            miss = float(np.hypot(*(rp + rv * t_cpa)))
            if miss < self.radius + self.DYN_R + 0.25 and (best is None or t_cpa < best[0]):
                best = (t_cpa, tr, miss)
        if best is None:
            return None
        _, tr, _ = best
        u = tr['v'] / np.hypot(*tr['v'])
        e = np.array([-u[1], u[0]])
        side = float(e @ (p[:2] - tr['p']))
        if abs(side) < 0.05:                       # en la línea: el lado con más espacio libre del plan
            side = 1.0 if (self.plan_y is not None and np.mean(self.plan_y) - self.lane >= 0) else -1.0
        e = e * np.sign(side)
        return self.V_ESC * e

    def _rf_predicted(self):
        """Volumen barrido predicho de las pistas dinámicas confirmadas (n ≥ 2, |v| > 0.3 m/s)."""
        out = []
        for tr in self._tracks:
            if not self._rf_confirmed(tr, 2):
                continue
            for tau in np.arange(0.0, self.DYN_HORIZON + 1e-6, 0.1):
                c = tr['p'] + tr['v'] * tau
                for ang in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                    out.append(c + self.DYN_R * np.array([np.cos(ang), np.sin(ang)]))
                out.append(c)
        return np.array(out) if out else np.zeros((0, 2))

    def _vo_update(self, t, pts):
        """Alinea los impactos actuales con el keyframe de hace ~VO_DT s → sesgo de velocidad."""
        if t is None or len(pts) < 20:
            return
        pts = pts[:: max(1, len(pts) // 300), :2]
        self._kf.append((t, pts))
        while self._kf and t - self._kf[0][0] > self.VO_DT + 0.05:
            self._kf.pop(0)
        t0, old = self._kf[0]
        if t - t0 < 0.7 * self.VO_DT or len(old) < 20:
            return
        res = 0.05
        lo = old.min(0) - self.VO_R - 0.2
        shp = (np.ceil((old.max(0) - lo + self.VO_R + 0.2) / res).astype(int) + 1)
        H = np.zeros(shp, np.float32)
        oi = ((old - lo) / res).astype(int)
        np.add.at(H, (oi[:, 0], oi[:, 1]), 1.0)
        H = cv2.GaussianBlur(H, (5, 5), 1.0)
        offs = np.arange(-self.VO_R, self.VO_R + 1e-9, self.VO_STEP)
        S = np.zeros((len(offs), len(offs)))
        for i, dx in enumerate(offs):
            for j, dy in enumerate(offs):
                q = ((pts + [dx, dy] - lo) / res).astype(int)
                ok = (q[:, 0] >= 0) & (q[:, 0] < shp[0]) & (q[:, 1] >= 0) & (q[:, 1] < shp[1])
                S[i, j] = H[q[ok, 0], q[ok, 1]].sum()
        c = len(offs) // 2
        bi, bj = np.unravel_index(np.argmax(S), S.shape)
        if S[bi, bj] <= 0:
            return
        shift = np.array([offs[bi], offs[bj]])
        # condicionamiento por eje (corredores: el eje longitudinal puede ser ambiguo)
        good = np.array([S[bi, bj] - min(S[max(bi - 2, 0), bj], S[min(bi + 2, len(offs) - 1), bj]) > 0.15 * S[bi, bj],
                         S[bi, bj] - min(S[bi, max(bj - 2, 0)], S[bi, min(bj + 2, len(offs) - 1)]) > 0.15 * S[bi, bj]])
        edge = np.array([bi in (0, len(offs) - 1), bj in (0, len(offs) - 1)])
        good &= ~edge
        dt = t - t0
        # puntos actuales = X + δ(t); keyframe = X + δ(t0) → shift = δ(t0) − δ(t) = −∫sesgo
        # Esquema a ~1 Hz sin solape: corrige la deriva de la ventana en posición (p_jump, lo
        # aplica el lazo) y fija la velocidad a la tasa medida; luego reinicia el keyframe.
        b_meas = -shift / dt
        self.p_jump = np.where(good, shift, 0.0)
        self.v_bias_hat = np.where(good, (1 - self.VO_K) * self.v_bias_hat + self.VO_K * (self.v_bias_hat + b_meas),
                                   self.v_bias_hat)
        self._kf = [(t, pts + self.p_jump)]
        self.vo_log.append((t, shift.copy(), good.copy(), self.v_bias_hat.copy()))

    def _drift_align(self, pts):
        """
        Desplazamiento (dx, dy) que mejor alinea los impactos del frame con las celdas ya ocupadas
        del mapa (correlación sobre una rejilla ±DC_R). Devuelve la corrección a aplicar
        (ganancia DC_GAIN) o None si el mapa/observación no la determinan. Cada eje se acepta solo
        si está bien condicionado (en un corredor el eje longitudinal puede ser ambiguo).
        """
        g = self.grid
        if len(pts) < 30 or (g.g > 1.5).sum() < 20:
            return None
        pts = pts[:: max(1, len(pts) // 400), :2]
        offs = np.arange(-self.DC_R, self.DC_R + 1e-9, self.DC_STEP)
        S = np.zeros((len(offs), len(offs)))
        occm = np.minimum(g.g, 3.0) * (g.g > 1.0)
        for i, dx in enumerate(offs):
            for j, dy in enumerate(offs):
                ix = ((pts[:, 0] + dx - g.x0) / g.res).astype(int)
                iy = ((pts[:, 1] + dy - g.y0) / g.res).astype(int)
                ok = (ix >= 0) & (ix < g.nx) & (iy >= 0) & (iy < g.ny)
                S[i, j] = occm[ix[ok], iy[ok]].sum()
        c = len(offs) // 2
        s0 = S[c, c]
        bi, bj = np.unravel_index(np.argmax(S), S.shape)
        if S[bi, bj] < 1.15 * s0 + 5.0:                       # sin mejora clara → no corregir
            return None
        corr = np.array([offs[bi], offs[bj]])
        # condicionamiento por eje: la mejora debe venir de ese eje (no plana a lo largo de él)
        if S[bi, bj] - S[c, bj] < 0.05 * S[bi, bj]:
            corr[0] = 0.0
        if S[bi, bj] - S[bi, c] < 0.05 * S[bi, bj]:
            corr[1] = 0.0
        if not corr.any():
            return None
        self.dc_log.append(corr.copy())
        return self.DC_GAIN * corr

    def update(self, state, depth_m=None, tof=None, pose=None):
        """
        state: {'p': (3,), 'v': (3,), 'yaw': float}
        depth_m: profundidad calibrada 96×96 (m) o None
        tof: 8×8 (m) o None
        pose: (p, yaw, pitch, roll) en el instante de captura (por defecto state)
        Retorna (v_objetivo_mundo (3,), yaw_objetivo, info)
        """
        p, z, occ_xy = self._perceive(state, depth_m, tof, pose)
        t_ = state.get('t')
        if t_ is not None:                                  # ego-movimiento (yaw) para RF_EGO_REJ
            if getattr(self, '_yaw_prev', None) is not None and t_ > self._yaw_prev[1]:
                om = ((state['yaw'] - self._yaw_prev[0] + math.pi) % (2 * math.pi) - math.pi) / (t_ - self._yaw_prev[1])
                self._ego = (p.copy(), float(om))
            self._yaw_prev = (state['yaw'], t_)
        occ_p = self._to_path(occ_xy) if len(occ_xy) else occ_xy
        xd, yd = self._to_path(p[:2])
        occ_plan = occ_p
        if self.exclude is not None and len(occ_p):
            # objetivo seguido: no es un obstáculo a rodear (el filtro de seguridad sí lo ve)
            ce = self._to_path(np.asarray(self.exclude[0])[:2])
            occ_plan = occ_p[np.hypot(occ_p[:, 0] - ce[0], occ_p[:, 1] - ce[1]) > self.exclude[1]]
        xs, yp, x_block = self._plan(occ_plan, xd, yd)
        d_here = self._front(occ_plan, xd, yd)
        if len(occ_p):
            dv = np.array([xd, yd]) - occ_p
            dist = np.hypot(dv[:, 0], dv[:, 1])
            order = np.argsort(dist)[:40]
            kmin = int(order[0])
            c_near, away = float(dist[kmin]), dv[kmin] / max(dist[kmin], 1e-6)
        else:
            c_near, away, order = np.inf, np.zeros(2), []

        # seguimiento del perfil SIN cortar esquinas: referencia en el primer bin
        # + feed-forward de pendiente (v_lat = v_fwd·dy/dx)
        y_goal = float(yp[0])
        slope0 = float((yp[1] - yp[0]) / self.DX)
        y_align = y_goal
        if self.CONSISTENT:
            # objetivo lateral a v·T_LOOK por delante (seguimiento de trayectoria, no del 1er punto)
            x_look = xd + max(self.DX, abs(self.v_cruise) * self.T_LOOK)
            y_goal = float(np.interp(x_look, np.concatenate([[xd], xs]), np.concatenate([[yd], yp])))
            i_l = int(np.clip(np.searchsorted(xs, x_look), 1, len(xs) - 1))
            slope0 = float((yp[i_l] - yp[i_l - 1]) / self.DX)
        slope = np.abs(np.diff(np.concatenate([[yd], yp[:int(1.2 / self.DX)]]))).max() / self.DX
        k_slope = 1.0 - 0.5 * min(slope / self.SLOPE_MAX, 1.0)
        if self.CONSISTENT:     # pendiente ya acotada por el FOV: frenado leve solo por curvatura
            k_slope = 1.0 - 0.25 * min(slope / max(math.tan(self.FOV_OK), 1e-3), 1.0)
        if self.speed_law == 'brake':
            # velocidad desde distancia de frenado real: v·τ + v²/2a ≤ distancia libre del plan
            a, tau = self.A_BRAKE, self.T_REACT
            gap = x_block - self.radius - self.GAP_BIAS - 0.15
            v_blk = (-a * tau + math.sqrt((a * tau) ** 2 + 2 * a * max(gap, 0.0))) if np.isfinite(gap) else np.inf
            v_lim = min(self.v_cruise, v_blk, self.v_sense_cap())
        else:                                            # rampa lineal (resultados E3 publicados)
            v_lim = self.v_cruise * float(np.clip((x_block - self.D_STOP) / (self.D_SLOW - self.D_STOP), 0, 1))
        arrived, d_goal = False, np.inf
        if self.goal is not None and self.arrive:
            gp = self._to_path(self.goal)
            d_goal = float(np.hypot(gp[0] - xd, gp[1] - yd))
            # frenado por distancia a lo largo del tramo: se detiene en la perpendicular
            # por B y corrige lo lateral después (nunca lo sobrepasa aunque vaya desviado)
            v_lim = min(v_lim, math.sqrt(2 * self.A_ARRIVE * max(gp[0] - xd - 0.05, 0.0)))
            arrived = d_goal < self.ACCEPT
        dbg = {'v_cruise': self.v_cruise, 'v_lim': v_lim, 'k_slope': k_slope, 'x_block': x_block}
        v_fwd = v_lim * k_slope if self.v_cruise >= 0 else self.v_cruise   # v_cruise<0: retroceso (seguimiento)
        if arrived:
            v_fwd = 0.0
        dbg['k_align'] = float(np.clip(1.0 - abs(y_align - yd) / 0.4, 0.0, 1.0))
        v_fwd *= dbg['k_align']   # primero alinearse
        if z < self.z_cruise - 0.15:                     # despegue: primero altura
            v_fwd = 0.0
        # límite de velocidad por holgura al obstáculo más cercano
        if v_fwd > 0:
            v_fwd = min(v_fwd, self.v_cruise * float(np.clip((c_near - self.radius) / 0.4, 0.5, 1.0)))
        dbg['v_fwd_pre_barrier'] = v_fwd
        lat_err = y_goal - yd
        v_lat_max = max(self.V_LAT, 0.6 * abs(self.v_cruise))
        v_lat = float(np.clip(self.K_LAT * lat_err + max(v_fwd, 0.0) * slope0, -v_lat_max, v_lat_max))
        # bolsillo: el primer bin frente al dron está bloqueado y no hay paso lateral
        # libre → retroceder despacio hasta recuperar maniobrabilidad (evita deadlock)
        backoff = self.bin0_blocked and x_block < self.D_STOP
        if self.CONSISTENT:
            # R8: el retroceso fuera del FOV se acota a BACKOFF_MAX por bloqueo (antes, con deriva de
            # estimación, el dron retrocedió 5.7 m a ciegas); se rearma cuando el bloqueo desaparece
            if backoff:
                self._backoff_d = getattr(self, '_backoff_d', 0.0) + 0.15 * 0.1
                if self._backoff_d > self.BACKOFF_MAX:
                    backoff = False
                    v_fwd = 0.0
            else:
                self._backoff_d = 0.0
        if backoff:
            v_fwd, v_lat = -0.15, 0.0
        # filtro de seguridad (tipo barrera): la componente de velocidad hacia cada
        # celda ocupada cercana no puede exceder la que permite frenar antes de tocarla
        vp = np.array([v_fwd, v_lat])
        for k in order:
            if dist[k] > 1.2:
                break
            u = -dv[k] / max(dist[k], 1e-6)                    # hacia el obstáculo
            gap = max(dist[k] - self.radius - self.GAP_BIAS, 0.0)
            a, tau = self.A_BRAKE, self.T_REACT
            lim = -a * tau + math.sqrt((a * tau) ** 2 + 2 * a * gap)   # v·τ + v²/2a ≤ gap
            comp = float(vp @ u)
            if comp > lim:
                vp -= (comp - lim) * u
        v_fwd, v_lat = float(vp[0]), float(vp[1])
        dbg['v_after_barrier'] = float(np.hypot(v_fwd, v_lat))
        brake = c_near < self.D_EMERG
        if brake:                                        # escape: alejarse del punto más cercano
            v_fwd, v_lat = 0.3 * away[0], 0.3 * away[1]
        v_z = float(np.clip(self.K_Z * (self.z_cruise - z), -self.V_Z, self.V_Z))
        v_world = np.array([self.cp * v_fwd - self.sp * v_lat, self.sp * v_fwd + self.cp * v_lat, v_z])
        v_des_dir = v_world[:2].copy()                   # dirección deseada (antes de escalar)
        # espacio desconocido: con FOV de 60° el corredor cercano (< half_w/tan30° ≈ 1 m) solo
        # se ve de antes; tras un giro puede no haberse observado → frenar hasta verlo
        if self.speed_law == 'brake' and not brake:
            vh = np.hypot(*v_world[:2])
            if vh > 0.05:
                if not self._seed_seen:                  # entorno del despegue: libre
                    self.grid.mark_seen(p[:2], 1.0)
                    self._seed_seen = True
                # la huella del propio dron es espacio libre por definición (fix R2: antes solo
                # se marcaba al despegar; tras girar, el corredor < ~1 m quedaba "desconocido")
                self.grid.mark_seen(p[:2], self.radius + 0.1)
                s_unk = self.grid.unknown_along(p[:2], v_world[:2] / vh, self.radius + 0.1)
                if np.isfinite(s_unk):
                    a, tau = self.A_BRAKE, self.T_REACT
                    lim = -a * tau + math.sqrt((a * tau) ** 2 + 2 * a * max(s_unk - 0.3, 0.0))
                    creeping = self._stall > 1.5
                    lim = max(lim, self.V_CREEP if creeping else 0.0)   # anti-bloqueo
                    if vh > lim:
                        v_world[:2] *= lim / vh
                    dbg['k_unknown'] = min(lim / vh, 1.0)
                    # fix R2: el contador NO se reinicia mientras se avanza en modo creep
                    # (antes lim=V_CREEP ≥ 0.1 lo reseteaba y el creep duraba un solo frame)
                    self._stall = self._stall + 0.1 if (lim < 0.1 or creeping) and lim < 0.5 * vh else 0.0
                else:
                    self._stall = 0.0
        if self.speed_law == 'brake' and not brake and not (self.CONSISTENT and backoff):
            vh = np.hypot(v_world[0], v_world[1])
            if vh > self.V_BLIND:
                th = abs((math.atan2(v_world[1], v_world[0]) - state['yaw'] + math.pi) % (2 * math.pi) - math.pi)
                k = float(np.clip(1.0 - (th - self.FOV_OK) / (self.FOV_LIM - self.FOV_OK), 0.0, 1.0))
                v_world[:2] *= max(k, min(self.V_BLIND / vh, 1.0))
                dbg['k_fov'] = k
        # filtro de objetivos (ruido frame a frame de la CNN) — sin filtrar si hay que frenar
        slowing = self.v_prev is not None and np.hypot(*v_world[:2]) < np.hypot(*self.v_prev[:2]) - 0.05
        if self.v_prev is not None and not brake and not (slowing and self.speed_law == 'brake'):
            v_world = self.v_prev + self.TARGET_ALPHA * (v_world - self.v_prev)
        if self.RF_DYN:
            v_esc = self._rf_escape(p, np.asarray(state['v'], float))
            if v_esc is not None:
                v_world[:2] = v_esc
                self.mode_override = 'EVADE'
            else:
                self.mode_override = None
        self.v_prev = v_world.copy()
        yaw_off = np.clip(self.YAW_FOLLOW * math.atan2(v_lat, max(v_fwd, 0.3)), -self.YAW_MAX, self.YAW_MAX)

        dev = np.abs(yp - self.lane).max()
        if arrived:
            mode = 'ARRIVED'
        elif brake or backoff:
            mode = 'BRAKE'
        elif dev > 0.1 and abs(yd - self.lane) <= abs(y_goal - self.lane) + 0.02:
            mode = 'AVOID'
        elif abs(yd - self.lane) > 0.1:
            mode = 'RETURN'
        else:
            mode = 'CRUISE'
        if self.RF_DYN and getattr(self, 'mode_override', None):
            mode = self.mode_override
        self.mode = mode
        # orientación: a lo largo del camino, hacia la velocidad, o apuntando al objetivo
        if self.yaw_mode == 'target' and self.look_point is not None:
            dlp = np.asarray(self.look_point)[:2] - p[:2]
            if np.hypot(*dlp) > 0.3:
                self.yaw_cmd = math.atan2(dlp[1], dlp[0])
        elif self.yaw_mode == 'velocity':
            if np.hypot(*v_des_dir) > 0.1:
                self.yaw_cmd = math.atan2(v_des_dir[1], v_des_dir[0])
        else:
            self.yaw_cmd = self.psi_path + float(yaw_off)
        info = {'mode': mode, 'x_block': x_block, 'd_here': d_here, 'c_near': c_near, 'y_goal': y_goal,
                'n_occ': len(occ_xy), 'plan': (xs, yp), 'plan_w': self.plan_w, 'arrived': arrived,
                'd_goal': d_goal}
        dbg['v_out'] = float(np.hypot(*v_world[:2]))
        dbg['backoff'] = bool(backoff)
        self.dbg = dbg
        self.last = info
        return v_world, self.yaw_cmd, info
