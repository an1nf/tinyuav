"""
MHPlanner · planificador local con conjunto de trayectorias (Loquercio et al., Science Robotics 2021,
"Learning High-Speed Flight in the Wild"), adaptado a nuestro mapa de profundidad monocular + ToF.

Qué se toma del paper y por qué:
  1. Referencia global + planificador local receding-horizon. El paper condiciona la política a una
     trayectoria de referencia; aquí la referencia es un campo de costo-a-meta (Dijkstra sobre la grilla
     local, desconocido = transitable con penalización, "optimista"), que existe aunque la recta A→B
     esté bloqueada. El planificador de carril (EvasionPlanner) solo mira ±2.3 m de la recta y se
     bloquea cuando los huecos son más angostos que el dron (X3: r = 0.36 m) → deadlock observado en
     Gazebo en x ≈ 0.4.
  2. Hipótesis múltiples: el experto privilegiado del paper muestrea (Metropolis-Hastings) muchas
     trayectorias libres de colisión y la red predice varias (multimodal) con un costo por cada una.
     Aquí se evalúa un abanico de K arcos (rumbo × curvatura) y se elige el de menor costo; histéresis
     sobre la hipótesis previa para no alternar izquierda/derecha.
  3. Consistencia percepción-acción: el paper vuela con el rumbo alineado a la velocidad (la cámara
     mira hacia donde se va). Aquí yaw = rumbo de la trayectoria elegida y la velocidad se limita
     mientras el desalineo supera FOV_OK: primero girar para ver, luego avanzar. Antes, el limitador
     de FOV anulaba cualquier desplazamiento lateral (k_fov = 0 a 90°) → el dron no podía rodear.
  4. Velocidad limitada por el alcance percibido (v·τ + v²/2a ≤ distancia libre y OBSERVADA a lo
     largo de la trayectoria), como en su análisis de velocidad máxima vs. alcance del sensor.

Interfaz idéntica a EvasionPlanner (update → v_mundo, yaw, info); reutiliza _perceive, la grilla,
set_segment, el filtro de barrera y el freno de emergencia.
"""
import math
import numpy as np
import cv2

from nanouav import settings
from nanouav.planning.evasion import EvasionPlanner, LocalGrid


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class MHGrid(LocalGrid):
    """LocalGrid + distancia mínima desde la que cada celda se vio ocupada (confianza de la evidencia)."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.dmin = np.full((self.nx, self.ny), np.inf, np.float32)
        self.origin = None
        self.R1, self.R2 = 1.5, 2.5
        self.RANGE_W = False            # R13: probado, empeora (paredes lejanas "abiertas" → recorre la pared)

    def update(self, occ, free, decay=True):
        """Como LocalGrid.update, pero la evidencia de la CÁMARA (primera fuente del frame, decay=True) pesa
        según la distancia: la CNN es fiable de cerca y deja fantasmas a 2–4 m. El ToF (llamadas siguientes)
        pesa 1 a cualquier distancia. Además guarda la distancia mínima de observación de cada celda."""
        if decay:
            self.g *= self.decay
        if len(free):
            ix, iy = self._idx(free)
            self.g[ix, iy] *= getattr(self, 'free_k', 0.5)
            self.seen[ix, iy] = True
            gone = self.g[ix, iy] < 0.5
            self.dmin[ix[gone], iy[gone]] = np.inf
        if len(occ):
            ix = ((occ[:, 0] - self.x0) / self.res).astype(int)
            iy = ((occ[:, 1] - self.y0) / self.res).astype(int)
            ok = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
            ix, iy, P = ix[ok], iy[ok], occ[ok]
            if len(ix):
                self.seen[ix, iy] = True
                d = np.hypot(P[:, 0] - self.origin[0], P[:, 1] - self.origin[1]).astype(np.float32) \
                    if self.origin is not None else np.zeros(len(ix), np.float32)
                np.minimum.at(self.dmin, (ix, iy), d)
                key = ix.astype(np.int64) * self.ny + iy
                u, inv = np.unique(key, return_inverse=True)
                dm = np.full(len(u), np.inf, np.float32)
                np.minimum.at(dm, inv, d)
                w = np.ones(len(u), np.float32)
                if decay and self.RANGE_W:                        # cámara/CNN: peso por distancia (opcional)
                    w = np.where(dm < self.R1, 1.0, np.where(dm < self.R2, 0.6, 0.35)).astype(np.float32)
                self.g[u // self.ny, u % self.ny] += w
            np.minimum(self.g, 6.0, out=self.g)


class MHPlanner(EvasionPlanner):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.CONSISTENT = True
        self.speed_law = 'brake'
        self.N_HEAD, self.TURN = 36, (-0.6, -0.3, 0.0, 0.3, 0.6)   # 10° × 5 curvaturas = 180 hipótesis
        self.T_H, self.DT_H = 2.5, 0.125                         # horizonte (s), paso de evaluación
        self.W_CLR, self.C_PREF = 0.4, 0.35                      # penalización por holgura < r_safe + C_PREF
        self.W_HYST, self.W_TURN, self.W_UNK = 1.5, 0.15, 0.6
        self.TRUST_DIST = False         # R13: probado, crea 'huecos móviles' en paredes vistas de lejos
        self.R_TRUST, self.OBS_FAR = 2.0, 4.0                       # m; costo de obstáculo lejano (poco fiable)
        self.OBS_COST = 25.0                                     # por celda dentro de r_safe (soft)
        self.UNK_COST = 0.5                                      # costo extra por celda desconocida (Dijkstra)
        self.CTG_RES = 0.1                                       # m, resolución del campo de costo-a-meta
        self.CTG_PERIOD = 0.3                                    # s entre recálculos del campo de costo
        self.NEAR_GOAL = 1.2
        self.SCAN_RATE = 0.6                                     # rad/s de giro de exploración sin salida
        self._ctg, self._ctg_t, self._ctg_box = None, -1e9, None
        self._head_prev = None
        self._stuck = 0.0
        self.mh_log = []
        # ── pasos estrechos (R13) ──
        # grilla fina 5 cm (con 10 cm la holgura se cuantiza ±5 cm: una puerta de 1.0 m para el X3, Ø 0.72,
        # deja 0.14 m por lado y el error de celda la hacía infactible); el campo de costo-a-meta sigue a 10 cm
        G = self.grid
        res = settings.number('MH_GRID_RES', 0.1)
        self.grid = MHGrid(res=res, xr=(G.x0, G.x0 + G.nx * G.res), yr=(G.y0, G.y0 + G.ny * G.res))
        # con celdas de 5 cm, los rayos rasantes y los errores de profundidad borraban paredes (×0.5 por
        # frame): una pared vista de canto desaparecía del mapa → choque. Borrado más lento y muestras de
        # espacio libre solo hasta el 70 % del rango (lejos de la superficie medida).
        # (R13, probado y descartado: borrado lento free_k 0.8 / memoria 0.999 → los fantasmas de la CNN
        #  persisten y el dron rodea de más; se mantiene el mapeo de R12 y se prohíbe moverse a ciegas)
        self.BLIND_OK = False
        # margen de seguridad dependiente de la velocidad: m(v) = M_MIN + M_V·v (a 0.5 m/s = 0.15 = GAP_BIAS
        # anterior; al pasar despacio por un hueco se admite M_MIN). Velocidad ≤ (holgura − r − M_MIN)/M_V.
        self.M_MIN, self.M_V, self.V_SQUEEZE = 0.05, 0.4, 0.12
        self._ctg_m = None
        self._route, self.ROUTE_K = None, 0.6
        self._insp, self._reject, self.reject_log = None, None, []
        self.INSPECT_D, self.INSPECT_T = 1.1, 1.0             # m antes del cruce dudoso; s mirándolo de frente
        self.D_EMERG = self.radius + 0.02
        self.STUCK_SCAN, self.STUCK_BACK = 4.0, 9.0          # s sin progreso → girar / retroceder por la estela
        self._trail, self._back = [], None
        # anti-estancamiento: si en STAG_T s no mejora ni la distancia a B ni el costo-a-meta, se relaja el
        # margen duro por escalones (M_RELAX) — el ruido de mapa de ±2–3 cm no puede cerrar un paso real —,
        # y si sigue estancado, SCAN (girar para observar) y luego retroceso por la estela.
        self.M_RELAX = (0.05, 0.03)
        self.M_ABS = 0.015                                  # margen mínimo absoluto (paso por la línea de referencia)
        self.STAG_T = 3.0
        self._m_eff, self._relax = self.M_MIN, 0
        self._best_d, self._best_ctg, self._prog_t = np.inf, np.inf, None

    def set_segment(self, A, B, arrive=True, v_cruise=None):
        super().set_segment(A, B, arrive, v_cruise)
        self._ctg, self._head_prev, self._route = None, None, None   # nueva meta → recalcular referencia

    # ── mapa: holgura y costo-a-meta ────────────────────────────────────
    def _goal_xy(self, p):
        if self.goal is not None:
            return np.asarray(self.goal, float)
        xd, _ = self._to_path(p[:2])                              # sin meta: 6 m por delante sobre el carril
        return self._from_path(np.array([[xd + 6.0, self.lane]]))[0]

    def _fields(self, p, goal, t):
        G = self.grid
        occ = G.g > 1.5
        free = (~occ).astype(np.uint8)
        clr = cv2.distanceTransform(free, cv2.DIST_L2, 5) * G.res - 0.5 * G.res   # m a la celda ocupada
        if not occ.any():
            clr[:] = 1e3
        self._clr = clr
        if self._ctg is not None and t - self._ctg_t < self.CTG_PERIOD and self._ctg_m == self._m_eff:
            return
        self._ctg_m = self._m_eff
        # caja de cálculo: dron y meta ± 4 m
        lo = np.minimum(p[:2], goal) - 4.0
        hi = np.maximum(p[:2], goal) + 4.0
        # campo a 10 cm: holgura máxima de cada bloque 2×2 (optimista: no cierra huecos por cuantización)
        f = int(round(self.CTG_RES / G.res))
        nxc, nyc = G.nx // f, G.ny // f
        clr_c = clr[:nxc * f, :nyc * f].reshape(nxc, f, nyc, f).max(axis=(1, 3))
        seen_c = G.seen[:nxc * f, :nyc * f].reshape(nxc, f, nyc, f).any(axis=(1, 3))
        i0 = max(int((lo[0] - G.x0) / self.CTG_RES), 0); i1 = min(int((hi[0] - G.x0) / self.CTG_RES) + 1, nxc)
        j0 = max(int((lo[1] - G.y0) / self.CTG_RES), 0); j1 = min(int((hi[1] - G.y0) / self.CTG_RES) + 1, nyc)
        c = clr_c[i0:i1, j0:j1]
        r_safe = self.radius + self._m_eff
        cost = 1.0 + self.UNK_COST * (~seen_c[i0:i1, j0:j1]) \
            + 4.0 * np.clip((r_safe + self.C_PREF - c) / self.C_PREF, 0, 1)
        # obstáculos del mapa como costo ALTO pero finito: la CNN deja fantasmas a 2–4 m (fuera del alcance
        # del ToF) que, como restricción dura, encierran al dron (campo = inf). La restricción dura de colisión
        # la aplican los arcos (≤ 1.25 m, donde el mapa es fiable); la referencia global es optimista.
        # graduado: dentro de r (choque aun sin margen) = muy caro; entre r y r+m = caro pero atravesable
        # (un paso estrecho sigue siendo preferible a un rodeo de 15 m; la factibilidad la deciden los arcos)
        # confianza de la evidencia (hallazgo R13): celdas vistas ocupadas solo desde > R_TRUST m la CNN estrecha o cierra los pasos (bordes
        # difusos, fondo lejano); tratar esas celdas como muro hace que un pasillo real a 3 m "no exista" y el
        # dron busque un rodeo de 15 m. Lejos → costo moderado (optimista); cerca → el costo alto normal.
        dmin_c = G.dmin[:nxc * f, :nyc * f].reshape(nxc, f, nyc, f).min(axis=(1, 3))[i0:i1, j0:j1]
        w_obs = np.where(dmin_c < self.R_TRUST, self.OBS_COST, self.OBS_FAR) if self.TRUST_DIST else self.OBS_COST
        cost[c < r_safe] += 3.0
        cost = np.where(c < self.radius, cost + w_obs, cost)
        # histéresis de ruta: las celdas de la ruta anterior cuestan menos → el dron no alterna entre
        # rodeos cuando el mapa de la CNN fluctúa (compromiso con el paso elegido mientras no se cierre de cerca)
        if self._route is not None and len(self._route):
            ri = ((self._route[:, 0] - G.x0) / self.CTG_RES).astype(int) - i0
            rj = ((self._route[:, 1] - G.y0) / self.CTG_RES).astype(int) - j0
            okr = (ri >= 0) & (ri < c.shape[0]) & (rj >= 0) & (rj < c.shape[1])
            disc = np.ones(c.shape, np.float32)
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    a_, b_ = np.clip(ri[okr] + di, 0, c.shape[0] - 1), np.clip(rj[okr] + dj, 0, c.shape[1] - 1)
                    disc[a_, b_] = self.ROUTE_K
            cost = np.where(c >= self.radius, cost * disc, cost)
        D = np.full(c.shape, np.inf, np.float32)
        gi = int(np.clip((goal[0] - G.x0) / self.CTG_RES - i0, 0, c.shape[0] - 1))
        gj = int(np.clip((goal[1] - G.y0) / self.CTG_RES - j0, 0, c.shape[1] - 1))
        cost[gi, gj] = min(cost[gi, gj], 1.0)
        D[gi, gj] = 0.0
        if self._reject is not None:
            self._reject *= 0.99                                   # el rechazo se olvida (~20 s): el mapa cambia
            cost = cost + self._reject[i0:i1, j0:j1]
        cost = cost.astype(np.float32) * self.CTG_RES
        steps = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                 (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)]
        for _ in range(4 * max(c.shape)):
            D0 = D.copy()
            for di, dj, w in steps:
                src = D[max(-di, 0):D.shape[0] - max(di, 0), max(-dj, 0):D.shape[1] - max(dj, 0)]
                dst = (slice(max(di, 0), D.shape[0] - max(-di, 0) if di < 0 else D.shape[0]),
                       slice(max(dj, 0), D.shape[1] - max(-dj, 0) if dj < 0 else D.shape[1]))
                np.minimum(D[dst], src + w * cost[dst], out=D[dst])
            if np.array_equal(D, D0):
                break
        self._ctg, self._ctg_t, self._ctg_box = D, t, (i0, j0)
        r = self._streamline(p, 12.0, self.CTG_RES)             # ruta completa (para la histéresis)
        self._route = r

    def _inspect(self, t, p, yaw, v_nom):
        """Percepción activa ante un paso dudoso (R13). La ruta de referencia cruza celdas que el mapa da
        por ocupadas (típico de la CNN: los bordes difusos de una puerta la 'rellenan' vistos de lado o de
        lejos, y de frente a ~1 m se ve abierta). Entonces: ir por la ruta hasta ~1 m antes del cruce,
        mirarlo de frente INSPECT_T s y decidir: si se abrió, seguir; si no, se rechaza ese cruce
        (penalización permanente en el campo) y la referencia busca otra ruta. Nunca queda detenido."""
        R = self._route
        if R is None or len(R) < 3:
            self._insp = None
            return None
        rc = self._lookup(self._clr, R, fill=1e3)
        n3 = min(len(R), int(3.0 / self.CTG_RES))
        blk = np.nonzero(rc[:n3] < self.radius + self.M_ABS)[0]
        if not len(blk):
            self._insp = None
            return None
        ib = int(blk[0])
        look = R[min(ib + 3, len(R) - 1)]
        iv = max(0, ib - int(self.INSPECT_D / self.CTG_RES))
        view = R[iv]
        # ¿mismo cruce que antes? (si el sitio cambia más de 0.5 m, es otro cruce)
        if self._insp is None or np.hypot(*(self._insp['look'] - look)) > 0.5:
            self._insp = {'look': look, 't0': None}
        d_view = float(np.hypot(*(view - p[:2])))
        dl = look - p[:2]
        h_look = math.atan2(dl[1], dl[0])
        if d_view > 0.25 and iv > 0:
            # seguir la ruta (centrada, ya libre hasta iv) hacia el punto de observación
            tgt = R[min(3, iv)]
            u = (tgt - p[:2]) / max(np.hypot(*(tgt - p[:2])), 1e-6)
            c_path = float(rc[:iv + 1].min())
            sp = min(v_nom, max((c_path - self.radius - self.M_ABS) / self.M_V, self.V_SQUEEZE),
                     math.sqrt(2 * self.A_BRAKE * d_view))
            head = math.atan2(u[1], u[0]) if d_view > 1.0 else h_look
            th = abs(_wrap(math.atan2(u[1], u[0]) - yaw))
            if th > self.FOV_LIM and d_view > 1.0:
                sp = 0.0                                          # girar primero
            self._insp['t0'] = None
            return u * sp, head, 'GO_VIEW'
        # en el punto de observación: mirar el cruce de frente y esperar
        if abs(_wrap(h_look - yaw)) > 0.25:
            self._insp['t0'] = None
            return np.zeros(2), h_look, 'INSPECT'
        if self._insp['t0'] is None:
            self._insp['t0'] = t
        if t - self._insp['t0'] < self.INSPECT_T:
            return np.zeros(2), h_look, 'INSPECT'
        # sigue cerrado tras mirarlo de frente → rechazar este cruce y replanificar
        G = self.grid
        ci = int((look[0] - G.x0) / self.CTG_RES); cj = int((look[1] - G.y0) / self.CTG_RES)
        rr = int(0.4 / self.CTG_RES)
        if self._reject is None:
            self._reject = np.zeros((int(round(G.nx * G.res / self.CTG_RES)), int(round(G.ny * G.res / self.CTG_RES))), np.float32)
        self._reject[max(ci - rr, 0):ci + rr + 1, max(cj - rr, 0):cj + rr + 1] += self.OBS_COST
        self.reject_log.append((round(t, 1), np.round(look, 2).tolist()))
        self._ctg, self._route, self._insp = None, None, None
        return np.zeros(2), yaw, 'REJECT'

    def _stagnation(self, t, p, goal):
        d = float(np.hypot(*(goal - p[:2])))
        ctg = float(self._lookup(self._ctg, p[None, :2], self._ctg_box, res=self.CTG_RES)[0]) \
            if self._ctg is not None else np.inf
        if self._prog_t is None or d < self._best_d - 0.3 or (np.isfinite(ctg) and ctg < self._best_ctg - 0.5):
            self._best_d, self._best_ctg = min(self._best_d, d), min(self._best_ctg, ctg) if np.isfinite(ctg) else self._best_ctg
            self._prog_t = t
        stag = t - self._prog_t
        c_here = float(self._lookup(self._clr, p[None, :2], fill=1e3)[0]) if hasattr(self, '_clr') else 1e3
        lvl = min(int(stag // self.STAG_T), len(self.M_RELAX) - 1)
        if lvl > self._relax:
            if self._relax == 0:
                self._relax_d = d
            self._relax = lvl
        elif self._relax and d < self._relax_d - 1.0 and c_here > self.radius + 0.3:
            self._relax = 0                                        # cruzó y avanzó ≥ 1 m: margen normal
        self._m_eff = self.M_RELAX[self._relax]
        self._stag = stag

    def _lookup(self, F, pts, box=(0, 0), fill=np.inf, res=None):
        G = self.grid
        res = res or G.res
        ix = ((pts[..., 0] - G.x0) / res).astype(int) - box[0]
        iy = ((pts[..., 1] - G.y0) / res).astype(int) - box[1]
        ok = (ix >= 0) & (ix < F.shape[0]) & (iy >= 0) & (iy < F.shape[1])
        out = np.full(pts.shape[:-1], fill, np.float64)
        out[ok] = F[ix[ok], iy[ok]]
        return out

    # ── hipótesis ───────────────────────────────────────────────────────
    def _streamline(self, p, L, ds):
        """Hipótesis extra: descenso por el campo de costo-a-meta (la referencia misma, centrada en los
        huecos por la penalización de proximidad). Remuestreada a paso ds, longitud L."""
        if self._ctg is None:
            return None
        D, (i0, j0), r = self._ctg, self._ctg_box, self.CTG_RES
        G = self.grid
        i = int((p[0] - G.x0) / r) - i0; j = int((p[1] - G.y0) / r) - j0
        if not (0 <= i < D.shape[0] and 0 <= j < D.shape[1]):
            return None
        pts = [p[:2].copy()]
        nb = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        for _ in range(int(L / r * 1.5) + 2):
            best, bij = D[i, j], None
            for di, dj in nb:
                a, b = i + di, j + dj
                if 0 <= a < D.shape[0] and 0 <= b < D.shape[1] and D[a, b] < best:
                    best, bij = D[a, b], (a, b)
            if bij is None:
                break
            i, j = bij
            pts.append(np.array([G.x0 + (i + i0 + 0.5) * r, G.y0 + (j + j0 + 0.5) * r]))
        pts = np.array(pts)
        if len(pts) < 3:
            return None
        seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        sq = np.arange(1, int(L / ds) + 1) * ds
        sq = np.minimum(sq, seg[-1])
        xs, ys = np.interp(sq, seg, pts[:, 0]), np.interp(sq, seg, pts[:, 1])
        # suavizado leve (quita el zigzag de 8-vecinos)
        k = np.ones(5) / 5
        xs = np.convolve(np.r_[[xs[0]] * 2, xs, [xs[-1]] * 2], k, 'valid')
        ys = np.convolve(np.r_[[ys[0]] * 2, ys, [ys[-1]] * 2], k, 'valid')
        return np.stack([xs, ys], -1)

    def _rollouts(self, p, v_nom):
        th0 = np.arange(self.N_HEAD) * 2 * math.pi / self.N_HEAD
        H, W = np.meshgrid(th0, np.array(self.TURN), indexing='ij')
        H, W = H.ravel(), W.ravel()
        ts = np.arange(1, int(self.T_H / self.DT_H) + 1) * self.DT_H
        th = H[:, None] + W[:, None] * ts[None, :]
        ds = v_nom * self.DT_H
        xs = p[0] + np.cumsum(ds * np.cos(th), 1)
        ys = p[1] + np.cumsum(ds * np.sin(th), 1)
        return H, W, np.stack([xs, ys], -1), ds

    def update(self, state, depth_m=None, tof=None, pose=None):
        t = float(state.get('t', 0.0) or 0.0)
        self.grid.origin = np.asarray(state['p'], float)[:2]
        p, z, occ_xy = self._perceive(state, depth_m, tof, pose)
        v_now = np.asarray(state['v'], float)[:2]
        yaw = float(state['yaw'])
        goal = self._goal_xy(p)
        self._stagnation(t, p, goal)
        self._fields(p, goal, t)
        G, r_safe = self.grid, self.radius + self.GAP_BIAS
        a, tau = self.A_BRAKE, self.T_REACT
        self.grid.mark_seen(p[:2], self.radius + 0.1)

        d_goal = float(np.hypot(*(goal - p[:2])))
        arrived = self.goal is not None and self.arrive and d_goal < self.ACCEPT
        v_nom = max(abs(self.v_cruise), 0.2)
        H, Wt, P, ds = self._rollouts(p, v_nom)                 # (K,), (K,), (K,N,2)
        sl = self._streamline(p, v_nom * self.T_H, ds)
        has_sl = False
        if sl is not None and len(sl) == P.shape[1]:
            has_sl = True
            d0 = sl[min(3, len(sl) - 1)] - p[:2]
            P = np.concatenate([P, sl[None]], 0)
            H = np.r_[H, math.atan2(d0[1], d0[0])]
            Wt = np.r_[Wt, 0.0]
        K, N = P.shape[:2]
        clr = self._lookup(self._clr, P, fill=1e3)             # holgura en cada punto
        seen = self._lookup(G.seen.astype(np.float32), P, fill=0.0) > 0.5
        s = np.arange(1, N + 1) * ds
        r_hard = self.radius + self._m_eff
        bad = clr < r_hard
        # paso estrecho: la línea de referencia (centrada en el hueco) puede pasar con margen M_ABS, despacio
        if has_sl:
            bad[-1] = clr[-1] < self.radius + self.M_ABS
        first_bad = np.where(bad.any(1), s[bad.argmax(1)], np.inf)
        first_unk = np.where((~seen).any(1), s[(~seen).argmax(1)], np.inf)
        s_free = np.minimum(first_bad, first_unk)              # libre Y observado
        v_ok = -a * tau + np.sqrt((a * tau) ** 2 + 2 * a * np.maximum(s_free - 0.3, 0.0))
        creeping = self._stuck > 1.5
        v_ok = np.where((first_unk <= first_bad) & creeping, np.maximum(v_ok, self.V_CREEP), v_ok)
        # velocidad según holgura en el primer metro: m(v) = M_MIN + M_V·v ≤ holgura − r
        n1 = max(int(1.0 / ds), 1)
        c1 = clr[:, :n1].min(1)
        v_clr = np.clip((c1 - r_hard) / self.M_V, self.V_SQUEEZE, v_nom)
        v_ex = np.minimum(np.minimum(v_ok, v_nom), v_clr)
        # extremo efectivo: lo que se avanza a la velocidad ejecutable en el horizonte
        n_eff = np.clip(np.round(v_ex / v_nom * N).astype(int) - 1, -1, N - 1)
        n_bad = np.where(bad.any(1), bad.argmax(1) - 2, N - 1)     # solo cuenta el tramo libre del arco
        n_eff = np.minimum(n_eff, n_bad)
        end = np.where(n_eff[:, None] >= 0, P[np.arange(K), np.maximum(n_eff, 0)], p[:2][None, :])
        ctg_end = self._lookup(self._ctg, end, self._ctg_box, res=self.CTG_RES)
        ctg_here = float(self._lookup(self._ctg, p[None, :2], self._ctg_box, res=self.CTG_RES)[0])
        feasible = first_bad > 0.6                             # al menos 0.6 m del arco sin colisión
        pen_clr = self.W_CLR * np.clip((r_hard + self.C_PREF - clr[:, :max(int(1.2 / ds), 1)]) / self.C_PREF,
                                       0, 1).mean(1)
        dpsi = np.abs(_wrap(H - yaw))
        J = ctg_end + pen_clr + self.W_TURN * dpsi + self.W_UNK * (first_unk < 1.0)
        if self._head_prev is not None:
            J += self.W_HYST * np.abs(_wrap(H - self._head_prev)) / math.pi
        J = np.where(feasible & np.isfinite(ctg_end), J, np.inf)
        k = int(np.argmin(J))
        ok = np.isfinite(J[k])
        squeeze = ok and has_sl and k == K - 1 and c1[k] < r_hard
        # ¿progresa? (si ninguna hipótesis reduce el costo-a-meta, contar atasco → creep en desconocido)
        progress = ok and (not np.isfinite(ctg_here) or ctg_here - ctg_end[k] > 0.02)
        self._stuck = 0.0 if progress or arrived else self._stuck + 0.1

        mode = 'CRUISE'
        c_near = float(np.min(self._lookup(self._clr, p[None, :2], fill=1e3))) + self.radius * 0  # m
        if arrived:
            v_xy, mode = np.zeros(2), 'ARRIVED'
            head = self._head_prev if self._head_prev is not None else yaw
        elif d_goal < self.NEAR_GOAL and self.goal is not None and self.arrive and \
                (self._lookup(self._clr, p[:2] + np.linspace(0, 1, 12)[:, None] * (goal - p[:2]), fill=1e3) > r_hard).all():
            # tramo final: aproximación directa (holonómica) con frenado de llegada, yaw fijo
            u = (goal - p[:2]) / max(d_goal, 1e-6)
            sp = min(v_nom, math.sqrt(2 * self.A_ARRIVE * max(d_goal - 0.05, 0.0)), 1.2 * d_goal)
            v_xy, mode = u * sp, 'APPROACH'
            head = self._head_prev if self._head_prev is not None else yaw
        elif ok:
            i_l = min(max(int(0.5 / ds), 1), N) - 1                # dirección hacia el punto a 0.5 m
            dvec = P[k, i_l] - p[:2]
            u = dvec / max(np.linalg.norm(dvec), 1e-6)
            head = math.atan2(u[1], u[0])                          # la cámara mira hacia donde se va
            sp = float(v_ex[k])
            if self.goal is not None and self.arrive:
                sp = min(sp, math.sqrt(2 * self.A_ARRIVE * max(d_goal - 0.05, 0.0)))
            # consistencia percepción-acción: avanzar solo lo que la cámara ya ve (girar primero)
            th = abs(_wrap(math.atan2(u[1], u[0]) - yaw))
            k_fov = float(np.clip(1.0 - (th - self.FOV_OK) / (self.FOV_LIM - self.FOV_OK), 0.0, 1.0))
            # hacia espacio YA observado (p.ej. retroceder o desplazarse dentro de un pasillo visto) no hace
            # falta girar la cámara: la consistencia percepción-acción solo exige ver lo desconocido
            known = (self.BLIND_OK and first_unk[k] > sp * tau + sp * sp / (2 * a) + 0.6
                     and c1[k] > self.radius + 0.2 and th < math.radians(100))
            if known and th > self.FOV_OK:
                k_fov = 1.0
                if sl is not None:
                    dsl = sl[min(len(sl) - 1, 8)] - p[:2]
                    head = math.atan2(dsl[1], dsl[0])
                else:
                    head = self._head_prev if self._head_prev is not None else yaw
            sp *= k_fov
            v_xy = u * sp
            lateral = abs(_wrap(head - self.psi_path)) > math.radians(15)
            mode = 'SQUEEZE' if squeeze else ('AVOID' if lateral else 'CRUISE')
            if sp < 0.05:
                mode = 'TURN'
        else:
            # sin hipótesis factible: detenerse y girar para observar (exploración de rumbo)
            v_xy, mode = np.zeros(2), 'SCAN'
            head = yaw + self.SCAN_RATE
        if not arrived and mode != 'APPROACH' and (self._stag > self.STAG_T or self._insp is not None):
            ins = self._inspect(t, p, yaw, v_nom)          # solo tras estancarse: el vuelo normal no cambia
            if ins is not None:
                v_xy, head, mode = ins
                squeeze = False
        self._head_prev = head

        # filtro de barrera (igual que EvasionPlanner) + freno de emergencia
        if len(occ_xy):
            dv = p[:2] - occ_xy
            dist = np.hypot(dv[:, 0], dv[:, 1])
            order = np.argsort(dist)[:40]
            for kk in order:
                if dist[kk] > 1.2:
                    break
                uo = -dv[kk] / max(dist[kk], 1e-6)
                gap = max(dist[kk] - self.radius - (self.M_ABS if squeeze else self._m_eff), 0.0)
                lim = -a * tau + math.sqrt((a * tau) ** 2 + 2 * a * gap)
                comp = float(v_xy @ uo)
                if comp > lim:
                    v_xy = v_xy - (comp - lim) * uo
            c_near = float(dist[order[0]])
            if c_near < (self.radius + 0.005 if squeeze else self.D_EMERG):
                away = dv[order[0]] / max(c_near, 1e-6)
                v_xy, mode = 0.3 * away, 'BRAKE'
        else:
            c_near = np.inf
        if z < self.z_cruise - 0.15:                           # despegue: primero altura
            v_xy = np.zeros(2)
        v_z = float(np.clip(self.K_Z * (self.z_cruise - z), -self.V_Z, self.V_Z))
        v_world = np.array([v_xy[0], v_xy[1], v_z])
        # suavizado de objetivos (sin filtrar si se frena)
        slowing = self.v_prev is not None and np.hypot(*v_world[:2]) < np.hypot(*self.v_prev[:2]) - 0.05
        if self.v_prev is not None and mode != 'BRAKE' and not slowing:
            v_world = self.v_prev + self.TARGET_ALPHA * (v_world - self.v_prev)
        self.v_prev = v_world.copy()
        self.yaw_cmd = head
        self.mode = mode
        plan = (P[k, :, 0], P[k, :, 1]) if ok else (np.array([p[0]]), np.array([p[1]]))
        self.plan_w = np.stack(plan, 1)
        info = {'mode': mode, 'x_block': float(first_bad[k]) if ok else 0.0, 'd_here': float(s_free[k]) if ok else 0.0,
                'c_near': c_near, 'y_goal': 0.0, 'n_occ': len(occ_xy), 'plan': plan, 'plan_w': self.plan_w,
                'arrived': arrived, 'd_goal': d_goal if self.goal is not None else np.inf,
                'ctg': ctg_here, 'n_feasible': int(np.isfinite(J).sum())}
        self.dbg = {'v_out': float(np.hypot(*v_world[:2])), 'stuck': self._stuck, 'head': head}
        self.last = info
        return v_world, self.yaw_cmd, info
