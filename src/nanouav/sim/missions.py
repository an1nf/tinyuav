"""
E4 · Misiones: recorrido por waypoints, seguimiento de objetivo móvil y alta velocidad.

  waypoints : A → B → C en un campo de obstáculos, barrido de velocidad de crucero
  follow    : seguir un objetivo que se mueve entre obstáculos (medidas ruidosas
              10 Hz con pérdidas, filtro α-β), manteniéndolo en cámara
  speed     : Monte Carlo de corredores (E3) a 1.0 / 1.5 / 2.0 m/s
  popup     : obstáculo que aparece de golpe delante del dron a alta velocidad

Todo usa el mismo lazo que E3 (render → CNN int8 → ToF → planificador → control
→ motores → 6-DOF); la capa de misión solo fija tramo, velocidad y modo de yaw.

Uso:
    python e4_missions.py                                  # todo, x3
    python e4_missions.py --profile nano --only follow,popup --n 10
"""
import os, sys, json, math, time, argparse
import numpy as np
from nanouav import paths

ROOT = str(paths.repo_root())

from nanouav.scene.renderer import World, Box, Cylinder, Texture, random_ground, random_lighting
from nanouav.planning.mission import WaypointMission, TargetTracker, TargetFollower, VisualTargetSensor
from nanouav.control.vehicle import PROFILES
import nanouav.sim.closed_loop as E3
from nanouav.sim.closed_loop import run_episode, random_corridor_world, C

ASSETS = str(paths.results_dir())


def fast_kw(v):
    """Parámetros del planificador y de la referencia según velocidad."""
    fast = v > 0.8
    return ({'v_max': v, 'a_brake': 1.0 if fast else 0.8}, 'fast' if fast else 'normal')


def set_brake_law(planner):
    planner.speed_law = 'brake'
    planner.W_LANE = 0.6          # misiones: volver a la recta del tramo pesa más que la histéresis
    planner.T_REACT = 0.6         # frame 0.1 s + latencia 0.06 + rampa de jerk + retardo de seguimiento
    planner.GAP_BIAS = 0.15       # error de mapa (CNN + celda) observado a mayor velocidad


# ── mundos ─────────────────────────────────────────────────────────────────
def _seg_dist(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-9), 0, 1)
    return np.linalg.norm(p - (a + t * ab))


def field_world(rng, wps, n=12, keep=0.9, on_path=3):
    """Obstáculos altos dispersos; algunos sobre las rectas del recorrido."""
    wps = [np.asarray(w, float)[:2] for w in wps]
    lo = np.min(wps, 0) - 1.0
    hi = np.max(wps, 0) + 1.0
    objs, placed = [], []
    for k in range(n + on_path):
        for _ in range(60):
            if k < on_path:                          # sobre un tramo
                i = int(rng.integers(0, len(wps) - 1))
                c = wps[i] + rng.uniform(0.3, 0.7) * (wps[i + 1] - wps[i]) + rng.normal(0, 0.15, 2)
            else:
                c = rng.uniform(lo, hi)
            if (min(np.linalg.norm(c - w) for w in wps) > keep and
                    all(np.linalg.norm(c - q) > 1.3 for q in placed)):
                break
        placed.append(c)
        tex = Texture('uniform', albedo=rng.uniform(0.15, 0.9))
        if rng.random() < 0.5:
            hx, hy = rng.uniform(0.1, 0.3, 2)
            objs.append(Box((c[0], c[1], 0.75), (hx, hy, 0.75), rng.uniform(-1, 1), tex))
        else:
            objs.append(Cylinder(c[0], c[1], rng.uniform(0.08, 0.28), 0.0, rng.uniform(1.0, 2.0), tex))
    return World(objs, random_ground(rng), **random_lighting(rng))


class MovingTarget:
    """Objetivo visible (cilindro) que recorre una curva suave a velocidad v."""

    def __init__(self, center, ax, ay, v, r=0.2, phase=0.0):
        self.c, self.ax, self.ay, self.v, self.r = np.asarray(center, float), ax, ay, v, r
        L = 2 * math.pi * math.sqrt((ax ** 2 + ay ** 2) / 2)      # perímetro aprox. de la elipse
        self.w = 2 * math.pi * v / L
        self.phase = phase
        self.obj = Cylinder(*self.pos(0.0), r, 0.0, 1.2, Texture('stripes', 0.85, 0.2, 0.25))

    def pos(self, t):
        a = self.phase + self.w * t
        return self.c + np.array([self.ax * math.sin(a), self.ay * math.sin(2 * a) / 2])

    def step(self, t):
        self.obj.cx, self.obj.cy = self.pos(t)


# ── escenarios ─────────────────────────────────────────────────────────────
def run_waypoints(P, v, seed, variant, log=False):
    rng = np.random.default_rng(seed)
    wps = [(-1.0, 0.0), (3.0 + rng.uniform(-0.5, 0.5), 2.5 + rng.uniform(-0.5, 0.5)),
           (6.5 + rng.uniform(-0.5, 0.5), -1.5 + rng.uniform(-0.5, 0.5)), (10.0, 0.5)]
    w = field_world(rng, wps)
    kw, lim = fast_kw(v)
    mis = WaypointMission(wps, v_cruise=v, yaw_mode='velocity')

    def on_frame(t, st, pl):
        set_brake_law(pl)
        mis.update(pl, st['p'], t)

    straight = sum(np.linalg.norm(np.subtract(wps[i + 1], wps[i])) for i in range(len(wps) - 1))
    yaw0 = math.atan2(wps[1][1] - wps[0][1], wps[1][0] - wps[0][0])
    s, R, _, _ = run_episode(w, P, seed=seed, start=(wps[0][0], wps[0][1], 0.3), t_max=25 + 2.5 * straight / v,
                             start_yaw=yaw0,
                             variant=variant, on_frame=on_frame, done_fn=lambda t, p: mis.done,
                             planner_kw=kw, limits=lim)
    xy = R[:, [C['x'], C['y']]]
    s['path_ratio'] = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)) / straight)
    s['v_max_real'] = float(np.linalg.norm(R[:, [C['vx'], C['vy']]], axis=1).max())
    s['wp_times'] = [round(x, 1) for x in mis.t_reach if x is not None]
    return s, R, w, wps


def run_follow(P, v_target, seed, variant, T=45.0, noise=0.05, dropout=0.1,
               sensor='oracle', guidance='base'):
    """sensor: 'oracle' (posición + ruido, E4 original) | 'bbox' | 'bbox_tof' (VisualTargetSensor)."""
    rng = np.random.default_rng(seed)
    tgt = MovingTarget((4.0, 0.0), 3.2, 2.4, v_target, phase=rng.uniform(0, 2 * math.pi))
    path = np.array([tgt.pos(t) for t in np.linspace(0, 2 * math.pi / tgt.w, 200)])
    # obstáculos lejos de la trayectoria del objetivo
    objs = []
    for _ in range(200):
        c = rng.uniform([-1.5, -3.5], [9.5, 3.5])
        if (np.min(np.linalg.norm(path - c, axis=1)) > 1.0 and
                all(np.hypot(o.center[0] - c[0], o.center[1] - c[1]) > 1.3 if isinstance(o, Box)
                    else np.hypot(o.cx - c[0], o.cy - c[1]) > 1.3 for o in objs)):
            tex = Texture('uniform', albedo=rng.uniform(0.15, 0.9))
            objs.append(Cylinder(c[0], c[1], rng.uniform(0.1, 0.25), 0.0, 1.6, tex) if rng.random() < 0.5
                        else Box((c[0], c[1], 0.75), (0.2, 0.2, 0.75), rng.uniform(-1, 1), tex))
        if len(objs) >= 10:
            break
    w = World(objs + [tgt.obj], random_ground(rng), **random_lighting(rng))
    p0 = tgt.pos(0.0)
    start = (p0[0] - 2.2, p0[1], 0.3)
    trk = TargetTracker()
    fol = TargetFollower(v_max=max(1.2, 1.6 * v_target), target_r=tgt.r, guidance=guidance)
    kw, lim = fast_kw(fol.v_max)
    last_t = [0.0]
    dist_log = []
    vis = None if sensor == 'oracle' else VisualTargetSensor(
        np.random.default_rng(seed + 10_000), range_src='tof' if sensor == 'bbox_tof' else 'bbox')

    def on_frame(t, st, pl):
        set_brake_law(pl)
        if vis is None:
            meas = None if rng.random() < dropout else tgt.pos(t) + rng.normal(0, noise, 2)
        else:
            meas = vis.measure(t, st['p'], st['yaw'], tgt.pos(t), tgt.r, w, tgt.obj)
        trk.update(meas, max(t - last_t[0], 0.05))
        last_t[0] = t
        fol.update(pl, st['p'], trk)

    def on_step(t, sim):
        tgt.step(t)

    def extra(t):
        return list(tgt.pos(t))

    # colisión con el objetivo = colisión (está en world.objects)
    s, R, _, _ = run_episode(w, P, seed=seed, start=start, t_max=T, variant=variant, on_frame=on_frame,
                             on_step=on_step, done_fn=lambda t, p: t >= T - 0.01, planner_kw=kw,
                             limits=lim, record_extra=extra)
    tx, ty = R[:, len(C)], R[:, len(C) + 1]
    d = np.hypot(tx - R[:, C['x']], ty - R[:, C['y']])
    bearing = np.arctan2(ty - R[:, C['y']], tx - R[:, C['x']])
    off = np.abs((bearing - R[:, C['yaw']] + np.pi) % (2 * np.pi) - np.pi)
    m = R[:, 0] > 5.0                                      # tras la convergencia inicial
    s['follow_err_mean'] = float(np.mean(np.abs(d[m] - fol.d_follow)))
    s['follow_err_p95'] = float(np.percentile(np.abs(d[m] - fol.d_follow), 95))
    s['target_in_fov_pct'] = float(100 * np.mean(off[m] < 0.5 * 1.047))
    s['min_dist_target'] = float(d[m].min())
    s['success'] = bool(not s['collided'] and not s['ground_contact'])
    if vis is not None:
        s.update(vis.stats)
        s['lost_gt5s'] = bool(vis.max_gap > 5.0)            # criterio de fallo de Su et al.
        s['success'] = s['success'] and not s['lost_gt5s']
    s['sensor'], s['guidance'] = sensor, guidance
    return s, R, w


def run_popup(P, v, seed, variant, D=2.5):
    rng = np.random.default_rng(seed)
    w = World([], random_ground(rng), **random_lighting(rng))
    kw, lim = fast_kw(v)
    state = {'done': False, 't': None, 'v': None}

    def on_frame(t, st, pl):
        set_brake_law(pl)
        pl.set_segment((-1.5, 0.0), (20.0, 0.0), arrive=False, v_cruise=v)

    def on_step(t, sim):
        if not state['done'] and sim.p[0] > 2.5:
            half_w = rng.uniform(0.3, 0.6)
            w.objects.append(Box((sim.p[0] + D + 0.2, rng.uniform(-0.2, 0.2), 0.8), (0.2, half_w, 0.8), 0.0,
                                 Texture('uniform', albedo=rng.uniform(0.2, 0.8))))
            state.update(done=True, t=t, v=float(np.linalg.norm(sim.v[:2])))

    s, R, _, _ = run_episode(w, P, seed=seed, start=(-1.5, 0.0, 0.3), t_max=40, variant=variant,
                             on_frame=on_frame, on_step=on_step, done_fn=lambda t, p: p[0] > 9.0,
                             planner_kw=kw, limits=lim)
    s['v_at_popup'] = state['v']
    s['popup_dist'] = D
    return s, R, w


# ── gráficas ───────────────────────────────────────────────────────────────
def _ax(ax, t):
    ax.set_facecolor('#111111'); ax.tick_params(colors='#888', labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor('#333')
    ax.set_title(t, color='white', fontsize=9)


def _draw_world(ax, w, skip=None):
    from matplotlib.patches import Rectangle, Circle
    for o in w.objects:
        if o is skip:
            continue
        if isinstance(o, Box):
            ax.add_patch(Rectangle((o.center[0] - o.half[0], o.center[1] - o.half[1]), 2 * o.half[0],
                                   2 * o.half[1], angle=np.degrees(o.yaw), rotation_point='center',
                                   color='#E24B4A', alpha=0.7))
        elif isinstance(o, Cylinder):
            ax.add_patch(Circle((o.cx, o.cy), o.r, color='#E24B4A', alpha=0.7))


def plot_missions(wp_runs, follow_run, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, len(wp_runs) + 1, figsize=(5.2 * (len(wp_runs) + 1), 4.8), facecolor='#1a1a1a')
    for ax, (v, s, R, w, wps) in zip(axs, wp_runs):
        _ax(ax, f'Waypoints v={v} m/s · {"OK" if s["success"] else "FALLO"} · {s["time_s"]:.0f}s')
        _draw_world(ax, w)
        sp = np.linalg.norm(R[:, [C['vx'], C['vy']]], axis=1)
        sc = ax.scatter(R[::20, C['x']], R[::20, C['y']], c=sp[::20], cmap='viridis', s=4, vmin=0, vmax=max(v, 0.5))
        W = np.array(wps)
        ax.plot(W[:, 0], W[:, 1], 'w--', lw=0.7)
        ax.scatter(W[:, 0], W[:, 1], c='white', marker='x')
        ax.set_aspect('equal')
        plt.colorbar(sc, ax=ax, fraction=0.03).ax.tick_params(colors='#888', labelsize=7)
    if follow_run is not None:
        s, R, w = follow_run
        ax = axs[-1]
        _ax(ax, f'Seguimiento · err medio {s["follow_err_mean"]:.2f} m · en FOV {s["target_in_fov_pct"]:.0f}%')
        _draw_world(ax, w, skip=w.objects[-1])
        ax.plot(R[:, len(C)], R[:, len(C) + 1], color='#EF9F27', lw=1, label='objetivo')
        ax.plot(R[:, C['x']], R[:, C['y']], color='#5DCAA5', lw=1, label='dron')
        ax.legend(fontsize=7, facecolor='#222', labelcolor='white')
        ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor='#1a1a1a')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--profile', default='x3', choices=list(PROFILES))
    ap.add_argument('--variant', default='M')
    ap.add_argument('--n', type=int, default=6, help='repeticiones por condición (MC)')
    ap.add_argument('--speeds', default='0.5,1.0,1.5,2.0')
    ap.add_argument('--only', default='waypoints,follow,speed,popup')
    a = ap.parse_args()
    P = PROFILES[a.profile]
    speeds = [float(x) for x in a.speeds.split(',')]
    todo = a.only.split(',')
    os.makedirs(ASSETS, exist_ok=True)
    out = {'profile': P.name}
    t0 = time.time()
    keys = ['success', 'collided', 'timeout', 'time_s', 'min_clearance_m', 'max_tilt_deg', 'p99_jerk',
            'sat_pct', 'recoveries']

    def short(s, extra=()):
        return {k: (round(s[k], 2) if isinstance(s[k], float) else s[k]) for k in list(keys) + list(extra) if k in s}

    wp_runs, follow_run = [], None
    if 'waypoints' in todo:
        out['waypoints'] = {}
        for v in speeds:
            res = []
            for i in range(a.n):
                s, R, w, wps = run_waypoints(P, v, 500 + i, a.variant)
                res.append(short(s, ('path_ratio', 'v_max_real', 'wp_times')))
                if i == 0:
                    wp_runs.append((v, s, R, w, wps))
                print(f'[wp v={v}] {i}: {res[-1]}  ({time.time() - t0:.0f}s)', flush=True)
            out['waypoints'][str(v)] = res
    if 'follow' in todo:
        out['follow'] = {}
        for vt in (0.4, 0.8):
            res = []
            for i in range(a.n):
                s, R, w = run_follow(P, vt, 700 + i, a.variant)
                res.append(short(s, ('follow_err_mean', 'follow_err_p95', 'target_in_fov_pct', 'min_dist_target')))
                if follow_run is None:
                    follow_run = (s, R, w)
                print(f'[follow v_obj={vt}] {i}: {res[-1]}  ({time.time() - t0:.0f}s)', flush=True)
            out['follow'][str(vt)] = res
    if 'speed' in todo:
        out['speed'] = {}
        for v in speeds:
            kw, lim = fast_kw(v)
            res = []
            for i in range(a.n):
                w = random_corridor_world(np.random.default_rng(1000 + i))
                s, _, _, _ = run_episode(w, P, seed=i, goal_x=6.5, t_max=60, variant=a.variant,
                                         on_frame=lambda t, st, pl, v=v: (set_brake_law(pl),
                                                                          setattr(pl, 'v_cruise', v)),
                                         planner_kw=kw, limits=lim)
                res.append(short(s))
                print(f'[speed v={v}] {i}: {res[-1]}  ({time.time() - t0:.0f}s)', flush=True)
            out['speed'][str(v)] = res
    if 'popup' in todo:
        out['popup'] = {}
        for v in speeds:
            res = []
            for i in range(a.n):
                s, _, _ = run_popup(P, v, 900 + i, a.variant)
                res.append(short(s, ('v_at_popup',)))
                print(f'[popup v={v}] {i}: {res[-1]}  ({time.time() - t0:.0f}s)', flush=True)
            out['popup'][str(v)] = res

    summ = {}
    for sec in ('waypoints', 'speed', 'popup', 'follow'):
        if sec in out:
            summ[sec] = {k: {'ok': sum(r['success'] for r in L), 'n': len(L),
                             'collisions': sum(r['collided'] for r in L),
                             't_med': float(np.median([r['time_s'] for r in L])),
                             'tilt_med': float(np.median([r['max_tilt_deg'] for r in L])),
                             'jerk99_med': float(np.median([r['p99_jerk'] for r in L]))}
                         for k, L in out[sec].items()}
            if sec == 'follow':
                for k, L in out[sec].items():
                    summ[sec][k].update(err_med=float(np.median([r['follow_err_mean'] for r in L])),
                                        fov_med=float(np.median([r['target_in_fov_pct'] for r in L])))
    out['summary'] = summ
    json.dump(out, open(os.path.join(ASSETS, f'e4_results_{P.name}.json'), 'w'), indent=2)
    if wp_runs or follow_run:
        plot_missions(wp_runs, follow_run, os.path.join(ASSETS, f'e4_missions_{P.name}.png'))
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
