"""
MHPlanner · local planner with a trajectory set (Loquercio et al., Science Robotics 2021,
"Learning High-Speed Flight in the Wild"), adapted to our monocular depth map + ToF.

What is taken from the paper and why:
  1. Global reference + receding-horizon local planner. The paper conditions the policy on a
     reference trajectory; here the reference is a cost-to-go field (Dijkstra over the local grid,
     unknown = traversable with a penalty, "optimistic"), which exists even when the straight line A→B
     is blocked. The lane planner (EvasionPlanner) only looks ±2.3 m around the line and gets
     stuck when the gaps are narrower than the drone (X3: r = 0.36 m) → deadlock observed in
     Gazebo at x ≈ 0.4.
  2. Multiple hypotheses: the paper's privileged expert samples (Metropolis-Hastings) many
     collision-free trajectories and the network predicts several (multimodal) with a cost for each.
     Here a fan of K arcs (heading × curvature) is evaluated and the lowest-cost one is chosen; hysteresis
     on the previous hypothesis avoids alternating left/right.
  3. Perception-action consistency: the paper flies with the heading aligned to the velocity (the camera
     looks where the drone is going). Here yaw = heading of the chosen trajectory and the speed is limited
     while the misalignment exceeds FOV_OK: first turn to see, then advance. Previously, the FOV
     limiter cancelled any lateral motion (k_fov = 0 at 90°) → the drone could not detour.
  4. Speed limited by the perceived range (v·τ + v²/2a ≤ free AND OBSERVED distance along
     the trajectory), as in their analysis of maximum speed vs. sensor range.

Same interface as EvasionPlanner (update → v_world, yaw, info); reuses _perceive, the grid,
set_segment, the barrier filter and the emergency brake.
"""
import math
import numpy as np
import cv2

from nanouav import settings
from nanouav.planning.evasion import EvasionPlanner, LocalGrid


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class MHGrid(LocalGrid):
    """LocalGrid + minimum distance from which each cell was seen occupied (evidence confidence)."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.dmin = np.full((self.nx, self.ny), np.inf, np.float32)
        self.origin = None
        self.R1, self.R2 = 1.5, 2.5
        self.RANGE_W = False            # R13: tested, makes things worse (distant walls "open" → drone follows the wall)

    def update(self, occ, free, decay=True):
        """Like LocalGrid.update, but CAMERA evidence (first source of the frame, decay=True) is weighted
        by distance: the CNN is reliable up close and leaves phantoms at 2–4 m. The ToF (subsequent calls)
        weighs 1 at any distance. It also stores the minimum observation distance of each cell."""
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
                if decay and self.RANGE_W:                        # camera/CNN: distance weighting (optional)
                    w = np.where(dm < self.R1, 1.0, np.where(dm < self.R2, 0.6, 0.35)).astype(np.float32)
                self.g[u // self.ny, u % self.ny] += w
            np.minimum(self.g, 6.0, out=self.g)


class MHPlanner(EvasionPlanner):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.CONSISTENT = True
        self.speed_law = 'brake'
        self.N_HEAD, self.TURN = 36, (-0.6, -0.3, 0.0, 0.3, 0.6)   # 10° × 5 curvatures = 180 hypotheses
        self.T_H, self.DT_H = 2.5, 0.125                         # horizon (s), evaluation step
        self.W_CLR, self.C_PREF = 0.4, 0.35                      # penalty for clearance < r_safe + C_PREF
        self.W_HYST, self.W_TURN, self.W_UNK = 1.5, 0.15, 0.6
        self.TRUST_DIST = False         # R13: tested, creates 'moving gaps' in walls seen from afar
        self.R_TRUST, self.OBS_FAR = 2.0, 4.0                       # m; cost of a distant (unreliable) obstacle
        self.OBS_COST = 25.0                                     # per cell inside r_safe (soft)
        self.UNK_COST = 0.5                                      # extra cost per unknown cell (Dijkstra)
        self.CTG_RES = 0.1                                       # m, resolution of the cost-to-go field
        self.CTG_PERIOD = 0.3                                    # s between cost-field recomputations
        self.NEAR_GOAL = 1.2
        self.SCAN_RATE = 0.6                                     # rad/s exploration turn when there is no way out
        self._ctg, self._ctg_t, self._ctg_box = None, -1e9, None
        self._head_prev = None
        self._stuck = 0.0
        self.mh_log = []
        # ── narrow passages (R13) ──
        # Map cell size from NANOUAV_MH_GRID_RES (default 0.1 m). With 10 cm cells the clearance is quantized
        # to ±5 cm, so a 1.0 m door for the X3 (Ø 0.72 m, 0.14 m per side) can look infeasible; 0.05 m fixes
        # that with good depth but lets grazing rays erase walls with the CNN (R13). The cost-to-go field
        # always uses 10 cm.
        G = self.grid
        res = settings.number('MH_GRID_RES', 0.1)
        self.grid = MHGrid(res=res, xr=(G.x0, G.x0 + G.nx * G.res), yr=(G.y0, G.y0 + G.ny * G.res))
        # R13, tested and discarded: slower erasing (free_k 0.8) and longer memory (0.999) — CNN phantoms
        # persist and the drone detours too much. The R12 mapping is kept and blind motion is forbidden.
        self.BLIND_OK = False
        # speed-dependent safety margin: m(v) = M_MIN + M_V·v (at 0.5 m/s = 0.15 = the previous GAP_BIAS;
        # passing slowly through a gap allows M_MIN). Speed ≤ (clearance − r − M_MIN)/M_V.
        self.M_MIN, self.M_V, self.V_SQUEEZE = 0.05, 0.4, 0.12
        self._ctg_m = None
        self._route, self.ROUTE_K = None, 0.6
        self._insp, self._reject, self.reject_log = None, None, []
        self.INSPECT_D, self.INSPECT_T = 1.1, 1.0             # m before the doubtful crossing; s looking at it head-on
        self.D_EMERG = self.radius + 0.02
        self.STUCK_SCAN, self.STUCK_BACK = 4.0, 9.0          # s without progress → turn / back up along the trail
        self._trail, self._back = [], None
        # anti-stagnation: if within STAG_T s neither the distance to B nor the cost-to-go improves, the
        # hard margin is relaxed in steps (M_RELAX) — map noise of ±2–3 cm must not close a real passage —,
        # and if still stagnated, SCAN (turn to observe) and then back up along the trail.
        self.M_RELAX = (0.05, 0.03)
        self.M_ABS = 0.015                                  # absolute minimum margin (passage along the reference line)
        self.STAG_T = 3.0
        self._m_eff, self._relax = self.M_MIN, 0
        self._best_d, self._best_ctg, self._prog_t = np.inf, np.inf, None

    def set_segment(self, A, B, arrive=True, v_cruise=None):
        super().set_segment(A, B, arrive, v_cruise)
        self._ctg, self._head_prev, self._route = None, None, None   # new goal → recompute reference
        # new goal → stagnation bookkeeping starts over (otherwise the best distance/cost of the previous
        # segment makes the new one look stagnated after STAG_T and margins get relaxed in normal flight)
        self._m_eff, self._relax = self.M_MIN, 0
        self._best_d, self._best_ctg, self._prog_t = np.inf, np.inf, None
        self._insp = None

    # ── map: clearance and cost-to-go ───────────────────────────────────
    def _goal_xy(self, p):
        if self.goal is not None:
            return np.asarray(self.goal, float)
        xd, _ = self._to_path(p[:2])                              # no goal: 6 m ahead along the lane
        return self._from_path(np.array([[xd + 6.0, self.lane]]))[0]

    def _fields(self, p, goal, t):
        G = self.grid
        occ = G.g > 1.5
        free = (~occ).astype(np.uint8)
        clr = cv2.distanceTransform(free, cv2.DIST_L2, 5) * G.res - 0.5 * G.res   # m to the nearest occupied cell
        if not occ.any():
            clr[:] = 1e3
        self._clr = clr
        if self._ctg is not None and t - self._ctg_t < self.CTG_PERIOD and self._ctg_m == self._m_eff:
            return
        self._ctg_m = self._m_eff
        # computation box: drone and goal ± 4 m
        lo = np.minimum(p[:2], goal) - 4.0
        hi = np.maximum(p[:2], goal) + 4.0
        # field at 10 cm: maximum clearance of each 2×2 block (optimistic: does not close gaps through quantization)
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
        # map obstacles as a HIGH but finite cost: the CNN leaves phantoms at 2–4 m (beyond the ToF
        # range) that, as a hard constraint, enclose the drone (field = inf). The hard collision constraint
        # is enforced by the arcs (≤ 1.25 m, where the map is reliable); the global reference is optimistic.
        # graded: inside r (collision even without margin) = very expensive; between r and r+m = expensive but
        # traversable (a narrow passage is still preferable to a 15 m detour; the arcs decide feasibility)
        # evidence confidence (R13 finding): for cells seen occupied only from > R_TRUST m, the CNN narrows or
        # closes passages (blurry edges, distant background); treating those cells as walls makes a real corridor
        # at 3 m "not exist" and the drone looks for a 15 m detour. Far → moderate cost (optimistic); near → the
        # normal high cost.
        dmin_c = G.dmin[:nxc * f, :nyc * f].reshape(nxc, f, nyc, f).min(axis=(1, 3))[i0:i1, j0:j1]
        w_obs = np.where(dmin_c < self.R_TRUST, self.OBS_COST, self.OBS_FAR) if self.TRUST_DIST else self.OBS_COST
        cost[c < r_safe] += 3.0
        cost = np.where(c < self.radius, cost + w_obs, cost)
        # route hysteresis: cells of the previous route cost less → the drone does not alternate between
        # detours when the CNN map fluctuates (commitment to the chosen passage unless it closes up close)
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
            self._reject *= 0.99                                   # rejection is forgotten (~20 s): the map changes
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
        r = self._streamline(p, 12.0, self.CTG_RES)             # full route (for the hysteresis)
        self._route = r

    def _inspect(self, t, p, yaw, v_nom):
        """Active perception at a doubtful passage (R13). The reference route crosses cells that the map
        marks as occupied (typical of the CNN: the blurry edges of a door 'fill it in' when seen from the side
        or from afar, while head-on at ~1 m it looks open). So: follow the route up to ~1 m before the crossing,
        look at it head-on for INSPECT_T s and decide: if it opened, continue; if not, that crossing is rejected
        (permanent penalty in the field) and the reference looks for another route. It never stays stopped."""
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
        # same crossing as before? (if the location moves more than 0.5 m, it is a different crossing)
        if self._insp is None or np.hypot(*(self._insp['look'] - look)) > 0.5:
            self._insp = {'look': look, 't0': None}
        d_view = float(np.hypot(*(view - p[:2])))
        dl = look - p[:2]
        h_look = math.atan2(dl[1], dl[0])
        if d_view > 0.25 and iv > 0:
            # follow the route (centered, already free up to iv) toward the observation point
            tgt = R[min(3, iv)]
            u = (tgt - p[:2]) / max(np.hypot(*(tgt - p[:2])), 1e-6)
            c_path = float(rc[:iv + 1].min())
            sp = min(v_nom, max((c_path - self.radius - self.M_ABS) / self.M_V, self.V_SQUEEZE),
                     math.sqrt(2 * self.A_BRAKE * d_view))
            head = math.atan2(u[1], u[0]) if d_view > 1.0 else h_look
            th = abs(_wrap(math.atan2(u[1], u[0]) - yaw))
            if th > self.FOV_LIM and d_view > 1.0:
                sp = 0.0                                          # turn first
            self._insp['t0'] = None
            return u * sp, head, 'GO_VIEW'
        # at the observation point: face the crossing and wait
        if abs(_wrap(h_look - yaw)) > 0.25:
            self._insp['t0'] = None
            return np.zeros(2), h_look, 'INSPECT'
        if self._insp['t0'] is None:
            self._insp['t0'] = t
        if t - self._insp['t0'] < self.INSPECT_T:
            return np.zeros(2), h_look, 'INSPECT'
        # still closed after looking at it head-on → reject this crossing and replan
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
            self._relax = 0                                        # crossed and advanced ≥ 1 m: normal margin
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

    # ── hypotheses ──────────────────────────────────────────────────────
    def _streamline(self, p, L, ds):
        """Extra hypothesis: descent along the cost-to-go field (the reference itself, centered in the
        gaps by the proximity penalty). Resampled at step ds, length L."""
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
        # light smoothing (removes the 8-neighbor zigzag)
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
        clr = self._lookup(self._clr, P, fill=1e3)             # clearance at each point
        seen = self._lookup(G.seen.astype(np.float32), P, fill=0.0) > 0.5
        s = np.arange(1, N + 1) * ds
        r_hard = self.radius + self._m_eff
        bad = clr < r_hard
        # narrow passage: the reference line (centered in the gap) may pass with margin M_ABS, slowly
        if has_sl:
            bad[-1] = clr[-1] < self.radius + self.M_ABS
        first_bad = np.where(bad.any(1), s[bad.argmax(1)], np.inf)
        first_unk = np.where((~seen).any(1), s[(~seen).argmax(1)], np.inf)
        s_free = np.minimum(first_bad, first_unk)              # free AND observed
        v_ok = -a * tau + np.sqrt((a * tau) ** 2 + 2 * a * np.maximum(s_free - 0.3, 0.0))
        creeping = self._stuck > 1.5
        v_ok = np.where((first_unk <= first_bad) & creeping, np.maximum(v_ok, self.V_CREEP), v_ok)
        # speed from the clearance over the first meter: m(v) = M_MIN + M_V·v ≤ clearance − r
        n1 = max(int(1.0 / ds), 1)
        c1 = clr[:, :n1].min(1)
        v_clr = np.clip((c1 - r_hard) / self.M_V, self.V_SQUEEZE, v_nom)
        v_ex = np.minimum(np.minimum(v_ok, v_nom), v_clr)
        # effective endpoint: how far the drone gets at the executable speed within the horizon
        n_eff = np.clip(np.round(v_ex / v_nom * N).astype(int) - 1, -1, N - 1)
        n_bad = np.where(bad.any(1), bad.argmax(1) - 2, N - 1)     # only the free segment of the arc counts
        n_eff = np.minimum(n_eff, n_bad)
        end = np.where(n_eff[:, None] >= 0, P[np.arange(K), np.maximum(n_eff, 0)], p[:2][None, :])
        ctg_end = self._lookup(self._ctg, end, self._ctg_box, res=self.CTG_RES)
        ctg_here = float(self._lookup(self._ctg, p[None, :2], self._ctg_box, res=self.CTG_RES)[0])
        feasible = first_bad > 0.6                             # at least 0.6 m of the arc collision-free
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
        # making progress? (if no hypothesis reduces the cost-to-go, count as stuck → creep into unknown)
        progress = ok and (not np.isfinite(ctg_here) or ctg_here - ctg_end[k] > 0.02)
        self._stuck = 0.0 if progress or arrived else self._stuck + 0.1

        mode = 'CRUISE'
        c_near = float(np.min(self._lookup(self._clr, p[None, :2], fill=1e3))) + self.radius * 0  # m
        if arrived:
            v_xy, mode = np.zeros(2), 'ARRIVED'
            head = self._head_prev if self._head_prev is not None else yaw
        elif d_goal < self.NEAR_GOAL and self.goal is not None and self.arrive and \
                (self._lookup(self._clr, p[:2] + np.linspace(0, 1, 12)[:, None] * (goal - p[:2]), fill=1e3) > r_hard).all():
            # final segment: direct (holonomic) approach with arrival braking, fixed yaw
            u = (goal - p[:2]) / max(d_goal, 1e-6)
            sp = min(v_nom, math.sqrt(2 * self.A_ARRIVE * max(d_goal - 0.05, 0.0)), 1.2 * d_goal)
            v_xy, mode = u * sp, 'APPROACH'
            head = self._head_prev if self._head_prev is not None else yaw
        elif ok:
            i_l = min(max(int(0.5 / ds), 1), N) - 1                # direction toward the point at 0.5 m
            dvec = P[k, i_l] - p[:2]
            u = dvec / max(np.linalg.norm(dvec), 1e-6)
            head = math.atan2(u[1], u[0])                          # the camera looks where the drone is going
            sp = float(v_ex[k])
            if self.goal is not None and self.arrive:
                sp = min(sp, math.sqrt(2 * self.A_ARRIVE * max(d_goal - 0.05, 0.0)))
            # perception-action consistency: only advance into what the camera already sees (turn first)
            th = abs(_wrap(math.atan2(u[1], u[0]) - yaw))
            k_fov = float(np.clip(1.0 - (th - self.FOV_OK) / (self.FOV_LIM - self.FOV_OK), 0.0, 1.0))
            # toward ALREADY observed space (e.g. backing up or moving inside a corridor already seen) there is
            # no need to turn the camera: perception-action consistency only requires seeing the unknown
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
            # no feasible hypothesis: stop and turn to observe (heading exploration)
            v_xy, mode = np.zeros(2), 'SCAN'
            head = yaw + self.SCAN_RATE
        if not arrived and mode != 'APPROACH' and (self._stag > self.STAG_T or self._insp is not None):
            ins = self._inspect(t, p, yaw, v_nom)          # only after stagnating: normal flight is unchanged
            if ins is not None:
                v_xy, head, mode = ins
                squeeze = False
        self._head_prev = head

        # barrier filter (same as EvasionPlanner) + emergency brake
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
        if z < self.z_cruise - 0.15:                           # takeoff: altitude first
            v_xy = np.zeros(2)
        v_z = float(np.clip(self.K_Z * (self.z_cruise - z), -self.V_Z, self.V_Z))
        v_world = np.array([v_xy[0], v_xy[1], v_z])
        # target smoothing (unfiltered when braking)
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
