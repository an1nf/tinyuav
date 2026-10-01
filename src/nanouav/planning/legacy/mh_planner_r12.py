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

from nanouav.planning.evasion import EvasionPlanner


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class MHPlannerV1(EvasionPlanner):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.CONSISTENT = True
        self.speed_law = 'brake'
        self.N_HEAD, self.TURN = 36, (-0.6, -0.3, 0.0, 0.3, 0.6)   # 10° × 5 curvatures = 180 hypotheses
        self.T_H, self.DT_H = 2.5, 0.125                         # horizon (s), evaluation step
        self.W_CLR, self.C_PREF = 3.0, 0.35                      # penalty for clearance < r_safe + C_PREF
        self.W_HYST, self.W_TURN, self.W_UNK = 1.5, 0.15, 0.6
        self.OBS_COST = 25.0                                     # per cell inside r_safe (soft)
        self.UNK_COST = 0.5                                      # extra cost per unknown cell (Dijkstra)
        self.CTG_PERIOD = 0.3                                    # s between cost-field recomputations
        self.NEAR_GOAL = 1.2
        self.SCAN_RATE = 0.6                                     # rad/s exploration turn when there is no way out
        self._ctg, self._ctg_t, self._ctg_box = None, -1e9, None
        self._head_prev = None
        self._stuck = 0.0
        self.mh_log = []

    def set_segment(self, A, B, arrive=True, v_cruise=None):
        super().set_segment(A, B, arrive, v_cruise)
        self._ctg, self._head_prev = None, None                 # new goal → recompute reference

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
        if self._ctg is not None and t - self._ctg_t < self.CTG_PERIOD:
            return
        # computation box: drone and goal ± 4 m
        lo = np.minimum(p[:2], goal) - 4.0
        hi = np.maximum(p[:2], goal) + 4.0
        i0 = max(int((lo[0] - G.x0) / G.res), 0); i1 = min(int((hi[0] - G.x0) / G.res) + 1, G.nx)
        j0 = max(int((lo[1] - G.y0) / G.res), 0); j1 = min(int((hi[1] - G.y0) / G.res) + 1, G.ny)
        c = clr[i0:i1, j0:j1]
        r_safe = self.radius + self.GAP_BIAS
        cost = 1.0 + self.UNK_COST * (~G.seen[i0:i1, j0:j1]) \
            + 4.0 * np.clip((r_safe + self.C_PREF - c) / self.C_PREF, 0, 1)
        # map obstacles as a HIGH but finite cost: the CNN leaves phantoms at 2–4 m (beyond the ToF
        # range) that, as a hard constraint, enclose the drone (field = inf). The hard collision constraint
        # is enforced by the arcs (≤ 1.25 m, where the map is reliable); the global reference is optimistic.
        cost[c < r_safe] += self.OBS_COST
        D = np.full(c.shape, np.inf, np.float32)
        gi = int(np.clip((goal[0] - G.x0) / G.res - i0, 0, c.shape[0] - 1))
        gj = int(np.clip((goal[1] - G.y0) / G.res - j0, 0, c.shape[1] - 1))
        cost[gi, gj] = min(cost[gi, gj], 1.0)
        D[gi, gj] = 0.0
        cost = cost.astype(np.float32) * G.res
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

    def _lookup(self, F, pts, box=(0, 0), fill=np.inf):
        G = self.grid
        ix = ((pts[..., 0] - G.x0) / G.res).astype(int) - box[0]
        iy = ((pts[..., 1] - G.y0) / G.res).astype(int) - box[1]
        ok = (ix >= 0) & (ix < F.shape[0]) & (iy >= 0) & (iy < F.shape[1])
        out = np.full(pts.shape[:-1], fill, np.float64)
        out[ok] = F[ix[ok], iy[ok]]
        return out

    # ── hypotheses ──────────────────────────────────────────────────────
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
        p, z, occ_xy = self._perceive(state, depth_m, tof, pose)
        v_now = np.asarray(state['v'], float)[:2]
        yaw = float(state['yaw'])
        goal = self._goal_xy(p)
        self._fields(p, goal, t)
        G, r_safe = self.grid, self.radius + self.GAP_BIAS
        a, tau = self.A_BRAKE, self.T_REACT
        self.grid.mark_seen(p[:2], self.radius + 0.1)

        d_goal = float(np.hypot(*(goal - p[:2])))
        arrived = self.goal is not None and self.arrive and d_goal < self.ACCEPT
        v_nom = max(abs(self.v_cruise), 0.2)
        H, Wt, P, ds = self._rollouts(p, v_nom)                 # (K,), (K,), (K,N,2)
        K, N = P.shape[:2]
        clr = self._lookup(self._clr, P, fill=1e3)             # clearance at each point
        seen = self._lookup(G.seen.astype(np.float32), P, fill=0.0) > 0.5
        s = np.arange(1, N + 1) * ds
        bad = clr < r_safe
        first_bad = np.where(bad.any(1), s[bad.argmax(1)], np.inf)
        first_unk = np.where((~seen).any(1), s[(~seen).argmax(1)], np.inf)
        s_free = np.minimum(first_bad, first_unk)              # free AND observed
        v_ok = -a * tau + np.sqrt((a * tau) ** 2 + 2 * a * np.maximum(s_free - 0.3, 0.0))
        creeping = self._stuck > 1.5
        v_ok = np.where((first_unk <= first_bad) & creeping, np.maximum(v_ok, self.V_CREEP), v_ok)
        v_ex = np.minimum(v_ok, v_nom)
        # effective endpoint: how far the drone gets at the executable speed within the horizon
        n_eff = np.clip(np.round(v_ex / v_nom * N).astype(int) - 1, -1, N - 1)
        n_bad = np.where(bad.any(1), bad.argmax(1) - 2, N - 1)     # only the free segment of the arc counts
        n_eff = np.minimum(n_eff, n_bad)
        end = np.where(n_eff[:, None] >= 0, P[np.arange(K), np.maximum(n_eff, 0)], p[:2][None, :])
        ctg_end = self._lookup(self._ctg, end, self._ctg_box)
        ctg_here = float(self._lookup(self._ctg, p[None, :2], self._ctg_box)[0])
        feasible = first_bad > 0.6                             # at least 0.6 m of the arc collision-free
        pen_clr = self.W_CLR * np.clip((r_safe + self.C_PREF - clr[:, :max(int(1.2 / ds), 1)]) / self.C_PREF,
                                       0, 1).mean(1)
        dpsi = np.abs(_wrap(H - yaw))
        J = ctg_end + pen_clr + self.W_TURN * dpsi + self.W_UNK * (first_unk < 1.0)
        if self._head_prev is not None:
            J += self.W_HYST * np.abs(_wrap(H - self._head_prev)) / math.pi
        J = np.where(feasible & np.isfinite(ctg_end), J, np.inf)
        k = int(np.argmin(J))
        ok = np.isfinite(J[k])
        # making progress? (if no hypothesis reduces the cost-to-go, count as stuck → creep into unknown)
        progress = ok and (not np.isfinite(ctg_here) or ctg_here - ctg_end[k] > 0.02)
        self._stuck = 0.0 if progress or arrived else self._stuck + 0.1

        mode = 'CRUISE'
        c_near = float(np.min(self._lookup(self._clr, p[None, :2], fill=1e3))) + self.radius * 0  # m
        if arrived:
            v_xy, mode = np.zeros(2), 'ARRIVED'
            head = self._head_prev if self._head_prev is not None else yaw
        elif d_goal < self.NEAR_GOAL and self.goal is not None and self.arrive and \
                (self._lookup(self._clr, p[:2] + np.linspace(0, 1, 8)[:, None] * (goal - p[:2]), fill=1e3) > r_safe).all():
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
            sp *= k_fov
            v_xy = u * sp
            lateral = abs(_wrap(head - self.psi_path)) > math.radians(15)
            mode = 'AVOID' if lateral else 'CRUISE'
            if sp < 0.05:
                mode = 'TURN'
        else:
            # no feasible hypothesis: stop and turn to observe (heading exploration)
            v_xy, mode = np.zeros(2), 'SCAN'
            head = yaw + self.SCAN_RATE
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
                gap = max(dist[kk] - self.radius - self.GAP_BIAS, 0.0)
                lim = -a * tau + math.sqrt((a * tau) ** 2 + 2 * a * gap)
                comp = float(v_xy @ uo)
                if comp > lim:
                    v_xy = v_xy - (comp - lim) * uo
            c_near = float(dist[order[0]])
            if c_near < self.D_EMERG:
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
