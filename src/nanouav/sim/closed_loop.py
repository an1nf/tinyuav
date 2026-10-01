"""
E3 · Simulación en lazo cerrado percepción → control → motores.

  render (cámara 10 Hz, pose real del dron) → CNN int8 → calibración ToF
  → planificador de evasión (con latencia de percepción) → referencia suave
  → control en cascada 500 Hz → mezclador → motores (1er orden) → dinámica 6-DOF

Escenarios:
  gazebo     : réplica del world SDF, vuelo de x=-1.5 a x=4.5
  kick       : igual + golpe de torque (≈3.5 rad/s) y ráfaga lateral en plena evasión
  montecarlo : N mundos aleatorios con obstáculos en el carril
  ablation   : mismos mundos, referencia SIN suavizado (escalones como en v11)

Uso:
    python e3_closed_loop.py                       # todo, perfil x3 (Gazebo)
    python e3_closed_loop.py --profile nano --n 20
"""
import os, sys, json, time, argparse
import math
import numpy as np
from nanouav import paths

ROOT = str(paths.repo_root())

from nanouav.config import pool_to_tof
from nanouav.scene.renderer import (gazebo_world, render_view, World, Box, Cylinder,
                                Texture, random_ground, random_lighting)
from nanouav.sensors.tof import tof_8x8
from nanouav.perception.cnn import predict_depth_norm, is_real_mode
from nanouav.perception.tof_calibrator import ToFCalibrator, fuse_tof, fuse
from nanouav.planning.evasion import EvasionPlanner
from nanouav.control.vehicle import PROFILES
from nanouav.control.quad_sim import QuadSim, euler_from_rot
from nanouav.control.flight_stack import FlightStack

ASSETS = str(paths.results_dir())
MODE_IDX = {'CRUISE': 0, 'AVOID': 1, 'RETURN': 2, 'BRAKE': 3, 'ARRIVED': 0}
DT_CTRL = 0.002
T_CAM = 0.1
INFLATE = 0.0
LATENCY = 0.06          # captura → comando (render + CNN + planificación)


def random_corridor_world(rng):
    """Obstáculos en el carril (x 1–6 m) + laterales; siempre queda un hueco ≥ 1.2 m."""
    objs, placed = [], []
    n = int(rng.integers(2, 5))
    for k in range(n):
        for _ in range(40):
            x = rng.uniform(1.2, 5.5)
            y = rng.uniform(-0.2, 0.2) if k == 0 else rng.uniform(-1.6, 1.6)
            if all(abs(x - a) > 1.3 or abs(y - b) > 1.4 for a, b in placed):
                break
        placed.append((x, y))
        tex = Texture('uniform', albedo=rng.uniform(0.15, 0.9))
        if rng.random() < 0.5:
            hx, hy = rng.uniform(0.1, 0.3, 2)
            objs.append(Box((x, y, 0.75), (hx, hy, 0.75), rng.uniform(-1, 1), tex))
        else:
            objs.append(Cylinder(x, y, rng.uniform(0.08, 0.3), 0.0, rng.uniform(1.0, 2.0), tex))
    if rng.random() < 0.5:           # obstáculo bajo que debe sobrevolarse, no esquivarse
        objs.append(Box((rng.uniform(1, 5), rng.uniform(-0.3, 0.3), 0.1), (0.2, 0.2, 0.1), 0.0,
                        Texture('uniform', albedo=rng.uniform(0.2, 0.8))))
    return World(objs, random_ground(rng), **random_lighting(rng))


def run_episode(world, P, seed=0, start=(-1.5, 0.0, 0.3), goal_x=4.5, t_max=40.0,
                shaping=True, disturbances=(), variant='M', log=False, inflate=None,
                on_frame=None, on_step=None, done_fn=None, planner_kw=None, limits='normal',
                start_yaw=0.0, record_extra=None, est_noise=None, thrust_m=1.0, planner_cls=None):
    """
    Hooks (E4 misiones): on_frame(t, st, planner) antes de planificar cada frame;
    on_step(t, sim) cada paso de control (mundo dinámico); done_fn(t, p) → fin exitoso;
    record_extra(t) → lista de floats añadida a cada fila del registro.
    est_noise (R8, Loquercio 2021 §S1): dict(sb, sw, tau) → el planificador y el controlador
    ven v_est = v + sesgo Gauss-Markov(σ=sb, τ) + blanco(σ=sw, 100 Hz) y p_est = p + ∫(v_est − v);
    la cámara, el ToF y las colisiones usan el estado verdadero. thrust_m: factor de empuje (≤ 1).
    """
    rng = np.random.default_rng(seed)
    sim = QuadSim(P, pos=start, yaw=start_yaw, gyro_noise=0.01, seed=seed)
    fs = FlightStack(P, yaw0=start_yaw, shaping=shaping, limits=limits)
    planner = (planner_cls or EvasionPlanner)(radius=P.radius, v_cruise=0.5, z_cruise=0.65,
                             inflate=INFLATE if inflate is None else inflate, **(planner_kw or {}))
    planner.yaw_cmd = start_yaw
    cal = ToFCalibrator()
    t, t_next_cam = 0.0, 0.0
    pending = []                       # (t_aplicar, v, yaw)
    rec, frames = [], []
    min_clear, collided, reached = np.inf, False, False
    coll_info = None
    fs.set_target(np.array([0, 0, 0.3]), 0.0)
    nrng = np.random.default_rng(seed + 777)
    v_bias, v_white, p_drift, t_white = np.zeros(3), np.zeros(3), np.zeros(3), -1.0
    vo_fix = np.zeros(3)
    while t < t_max:
        st = sim.state()
        st_true = st
        if est_noise:
            a_gm = DT_CTRL / est_noise.get('tau', 1.0)
            v_bias += -a_gm * v_bias + est_noise['sb'] * math.sqrt(2 * a_gm) * nrng.normal(size=3)
            if t - t_white >= 0.01:
                v_white, t_white = est_noise['sw'] * nrng.normal(size=3), t
            dv = v_bias + v_white
            p_drift += dv * DT_CTRL
            st = dict(st)
            st['v'] = st['v'] + dv
            st['p'] = st['p'] + p_drift
        if getattr(planner, 'VO', False):                      # R9b: corrección de sesgo por VO
            b = np.r_[planner.v_bias_hat, 0.0]
            vo_fix -= b * DT_CTRL
            if planner.p_jump.any():                           # corrección de posición a ~1 Hz
                vo_fix += np.r_[planner.p_jump, 0.0]
                planner.p_jump = np.zeros(2)
            st = dict(st)
            st['v'] = st['v'] - b
            st['p'] = st['p'] + vo_fix
        roll, pitch, yaw = euler_from_rot(st['R'])
        if t >= t_next_cam:                                    # percepción
            t_next_cam += T_CAM
            gray, depth_gt = render_view(world, st_true['p'], yaw, pitch, roll, rng=rng)
            tof = tof_8x8(depth_gt, rng=rng)
            dn = predict_depth_norm(gray, variant, depth_gt, tof=tof)
            depth_cal = fuse(cal.update_and_scale(dn, tof), tof)
            if on_frame is not None:
                on_frame(t, {'p': st['p'], 'v': st['v'], 'yaw': yaw}, planner)
            planner._p_true = np.asarray(st_true['p'])            # solo diagnóstico (R9)
            v_t, yaw_t, info = planner.update({'p': st['p'], 'v': st['v'], 'yaw': yaw, 't': t},
                                              depth_cal, tof, (st['p'], yaw, pitch, roll))
            pending.append((t + LATENCY, v_t, yaw_t))
            if getattr(planner, 'RF_DYN', False) and getattr(planner, 'EVADE_AGGR', False):
                fs.set_evade(info['mode'] == 'EVADE')
            if log:
                frames.append({'t': t, 'gray': gray, 'depth': depth_cal, 'gt': depth_gt,
                               'mode': info['mode'], 'scale': cal.scale_ema})
            plan_mode = info['mode']
        while pending and pending[0][0] <= t:
            _, v_t, yaw_t = pending.pop(0)
            fs.set_target(v_t, yaw_t)
        sim.F_ext[:] = 0
        sim.T_ext[:] = 0
        for (t0, dur, F, T) in disturbances:
            if t0 <= t < t0 + dur:
                sim.F_ext += F
                sim.T_ext += T
        if on_step is not None:
            on_step(t, sim)
        w_cmd, ci = fs.update(st, t, DT_CTRL)
        if thrust_m != 1.0:
            w_cmd = np.asarray(w_cmd) * math.sqrt(thrust_m)      # f ∝ ω² → pérdida de empuje
        sim.step(w_cmd, DT_CTRL)
        t += DT_CTRL
        c = world.clearance(sim.p)
        min_clear = min(min_clear, c)
        if c < P.radius and not collided:
            collided = True
            coll_info = {'pos': np.round(sim.p, 2).tolist(), 'v': np.round(sim.v, 2).tolist(),
                         'plan_mode': plan_mode, 'c_near_map': round(planner.last.get('c_near', -1), 2),
                         'yaw': round(float(yaw), 2)}
        rec.append((t, *sim.p, *sim.v, *sim.acc, roll, pitch, yaw, *sim.w,
                    *ci['v_ref'], *ci['a_ref'], ci['psi_ref'],
                    ci['mode'] == 'RECOVER', fs.ctrl.saturated_rp,
                    *(sim.wm / P.w_max), MODE_IDX.get(plan_mode, 0),
                    *(record_extra(t) if record_extra else [])))
        reached = done_fn(t, sim.p) if done_fn else sim.p[0] > goal_x
        if collided or reached:
            break
    R = np.array(rec, float)
    summ = summarize(R, P, collided, min_clear, bool(reached), fs)
    if coll_info:
        summ['collision'] = coll_info
    return summ, R, frames, fs


COLS = ['t', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'ax', 'ay', 'az', 'roll', 'pitch', 'yaw',
        'wx', 'wy', 'wz', 'vrx', 'vry', 'vrz', 'arx', 'ary', 'arz', 'psir', 'recover', 'sat',
        'm0', 'm1', 'm2', 'm3', 'plan']
C = {k: i for i, k in enumerate(COLS)}


def summarize(R, P, collided, min_clear, reached, fs):
    dt = DT_CTRL
    ground = bool(((R[:, 0] > 3.0) & (R[:, C['z']] < 0.05)).any())
    tilt = np.degrees(np.arccos(np.clip(np.cos(R[:, C['roll']]) * np.cos(R[:, C['pitch']]), -1, 1)))
    a_ref = R[:, [C['arx'], C['ary'], C['arz']]]
    j_ref = np.diff(a_ref, axis=0) / dt
    # jerk real: derivada de la aceleración filtrada a 10 Hz (excluye vibración numérica)
    k = int(0.1 / dt)
    acc = R[:, [C['ax'], C['ay'], C['az']]]
    acc_f = np.apply_along_axis(lambda s: np.convolve(s, np.ones(k) / k, 'same'), 0, acc)
    j_act = np.linalg.norm(np.diff(acc_f[k:-k], axis=0) / dt, axis=1)
    wz = R[:, C['wz']]
    wz_f = np.convolve(wz, np.ones(k) / k, 'same')
    ev = fs.sup.events
    return {
        'success': bool(reached and not collided and not ground),
        'collided': bool(collided), 'ground_contact': bool(ground),
        'timeout': bool(not reached and not collided),
        'time_s': float(R[-1, 0]), 'min_clearance_m': float(min_clear),
        'max_tilt_deg': float(tilt.max()),
        'max_ref_jerk': float(np.abs(j_ref).max()) if len(j_ref) else 0.0,
        'p99_jerk': float(np.percentile(j_act, 99)) if len(j_act) else 0.0,
        'max_yaw_rate': float(np.abs(wz_f).max()),
        'max_yaw_acc': float(np.abs(np.diff(wz_f[k:-k]) / dt).max()) if len(wz_f) > 2 * k + 1 else 0.0,
        'sat_pct': float(100 * R[:, C['sat']].mean()),
        'max_motor_pct': float(100 * R[:, C['m0']:C['m3'] + 1].max()),
        'z_dev_max': float(np.abs(R[R[:, 0] > 3, C['z']] - 0.65).max()) if (R[:, 0] > 3).any() else 0.0,
        'recoveries': [(round(a, 2), round(b - a, 2), c) for a, b, c in ev],
        'in_recovery_at_end': fs.sup.mode == 'RECOVER',
    }


# ── Gráficas ───────────────────────────────────────────────────────────────
def plot_episode(R, world, title, path, frames=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle
    DARK, BG = '#111111', '#1a1a1a'

    def sty(ax, t):
        ax.set_facecolor(DARK); ax.tick_params(colors='#888', labelsize=8)
        for s in ax.spines.values(): s.set_edgecolor('#333')
        ax.set_title(t, color='white', fontsize=9)

    fig = plt.figure(figsize=(15, 9), facecolor=BG)
    fig.suptitle(title, color='white')
    gs = fig.add_gridspec(3, 3)
    ax = fig.add_subplot(gs[:, 0]); sty(ax, 'Trayectoria (vista superior)')
    for o in world.objects:
        if isinstance(o, Box):
            ax.add_patch(Rectangle((o.center[0] - o.half[0], o.center[1] - o.half[1]),
                                   2 * o.half[0], 2 * o.half[1], angle=np.degrees(o.yaw),
                                   rotation_point='center', color='#E24B4A', alpha=0.7))
        elif isinstance(o, Cylinder):
            ax.add_patch(Circle((o.cx, o.cy), o.r, color='#E24B4A', alpha=0.7))
    modes = R[:, C['plan']].astype(int)
    cols = np.array(['#5DCAA5', '#EF9F27', '#378ADD', '#E24B4A'])
    ax.scatter(R[::25, C['x']], R[::25, C['y']], c=cols[modes[::25]], s=4)
    rec = R[:, C['recover']] > 0
    if rec.any():
        ax.scatter(R[rec, C['x']][::10], R[rec, C['y']][::10], c='#9B59B6', s=10, label='RECOVER')
        ax.legend(fontsize=7, facecolor='#222', labelcolor='white')
    ax.set_aspect('equal'); ax.set_xlim(-2, 5.5); ax.set_ylim(-2.2, 2.2)
    ax.set_xlabel('x (m)  · verde crucero · naranja evasión · azul retorno · rojo freno', color='#888', fontsize=7)

    t = R[:, 0]
    panels = [
        ('Velocidad vs referencia (m/s)', [('vx', 'vrx', '#5DCAA5'), ('vy', 'vry', '#EF9F27'), ('vz', 'vrz', '#378ADD')]),
        ('Inclinación y yaw (°)', None),
        ('Aceleración de referencia (m/s²)', [('arx', None, '#5DCAA5'), ('ary', None, '#EF9F27'), ('arz', None, '#378ADD')]),
        ('Motores (% ω_max)', None),
        ('Tasas angulares (rad/s)', [('wx', None, '#5DCAA5'), ('wy', None, '#EF9F27'), ('wz', None, '#378ADD')]),
        ('Altura (m)', [('z', None, '#5DCAA5')]),
    ]
    for i, (ttl, series) in enumerate(panels):
        ax = fig.add_subplot(gs[i % 3, 1 + i // 3]); sty(ax, ttl)
        if ttl.startswith('Inclinación'):
            ax.plot(t, np.degrees(R[:, C['roll']]), color='#5DCAA5', lw=1, label='roll')
            ax.plot(t, np.degrees(R[:, C['pitch']]), color='#EF9F27', lw=1, label='pitch')
            ax.plot(t, np.degrees(R[:, C['yaw']]), color='#378ADD', lw=1, label='yaw')
            ax.plot(t, np.degrees(R[:, C['psir']]), color='#378ADD', lw=0.8, ls='--')
        elif ttl.startswith('Motores'):
            for m, c in zip(['m0', 'm1', 'm2', 'm3'], ['#5DCAA5', '#EF9F27', '#378ADD', '#E24B4A']):
                ax.plot(t, 100 * R[:, C[m]], color=c, lw=0.8, label=m)
            ax.axhline(100, color='white', lw=0.6, ls='--')
        else:
            for a, b, c in series:
                ax.plot(t, R[:, C[a]], color=c, lw=1, label=a)
                if b:
                    ax.plot(t, R[:, C[b]], color=c, lw=0.8, ls='--')
        if rec.any():
            ax.fill_between(t, *ax.get_ylim(), where=rec, color='#9B59B6', alpha=0.15, lw=0)
        ax.legend(fontsize=6, facecolor='#222', labelcolor='white', ncol=4)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=BG)
    plt.close(fig)


def plot_mc(res_s, res_a, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    DARK, BG = '#111111', '#1a1a1a'
    keys = [('p99_jerk', 'Jerk real p99 (m/s³)'), ('max_tilt_deg', 'Inclinación máx (°)'),
            ('sat_pct', 'Saturación roll/pitch (%)'), ('min_clearance_m', 'Holgura mínima (m)')]
    fig, axs = plt.subplots(1, 4, figsize=(15, 4), facecolor=BG)
    fig.suptitle(f'Monte Carlo · suavizado vs escalones · éxito {sum(r["success"] for r in res_s)}/{len(res_s)} '
                 f'vs {sum(r["success"] for r in res_a)}/{len(res_a)}', color='white')
    for ax, (k, ttl) in zip(axs, keys):
        ax.set_facecolor(DARK); ax.tick_params(colors='#888', labelsize=8)
        for s in ax.spines.values(): s.set_edgecolor('#333')
        ax.set_title(ttl, color='white', fontsize=9)
        bp = ax.boxplot([[r[k] for r in res_s], [r[k] for r in res_a]], patch_artist=True)
        for b, c in zip(bp['boxes'], ['#5DCAA5', '#E24B4A']):
            b.set_facecolor(c)
        ax.set_xticks([1, 2]); ax.set_xticklabels(['suavizado', 'escalones'], color='#aaa')
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=BG)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--profile', default='x3', choices=list(PROFILES))
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--variant', default='M')
    ap.add_argument('--inflate', type=float, default=0.0, help='inflado lateral del mapa (m), p.ej. 0.1')
    ap.add_argument('--only-mc', action='store_true', help='solo Monte Carlo')
    a = ap.parse_args()
    P = PROFILES[a.profile]
    global INFLATE
    INFLATE = a.inflate
    os.makedirs(ASSETS, exist_ok=True)
    print(f'[E3] perfil {P.name} · T/W {P.thrust_to_weight:.2f} · CNN '
          f'{"REAL int8" if is_real_mode(a.variant) else "FAKE"} ({a.variant})')
    out = {'profile': P.name}

    if not a.only_mc:
        # 1) world Gazebo
        s, R, frames, _ = run_episode(gazebo_world(), P, seed=1, variant=a.variant, log=True)
        out['gazebo'] = s
        print('[gazebo]', json.dumps(s))
        plot_episode(R, gazebo_world(), f'E3 · world Gazebo · {P.name}', os.path.join(ASSETS, f'e3_gazebo_{P.name}' + ('' if a.variant == 'M' else f'_{a.variant}') + '.png'))

        # 2) golpe de torque (Δω nominal 6 rad/s en 60 ms) + ráfaga lateral 0.35 g durante la evasión
        def kick(dw):
            Tk = np.array([1.0, 0.6, 0.0]) * (dw * P.J[0] / 0.06)
            Fg = np.array([0.0, 1.0, 0.0]) * 0.35 * P.mass * 9.81
            return [(4.2, 0.06, np.zeros(3), Tk), (8.5, 0.4, Fg, np.zeros(3))]
        s, R, _, _ = run_episode(gazebo_world(), P, seed=2, disturbances=kick(6.0), variant=a.variant)
        out['kick'] = s
        print('[kick]', json.dumps(s))
        plot_episode(R, gazebo_world(), f'E3 · golpe Δω≈6 rad/s t=4.2 s + ráfaga 0.35 g t=8.5 s · {P.name}',
                     os.path.join(ASSETS, f'e3_kick_{P.name}' + ('' if a.variant == 'M' else f'_{a.variant}') + '.png'))
        # envolvente de recuperación
        env = []
        for dw in (3.0, 6.0, 9.0, 12.0):
            s, R, _, _ = run_episode(gazebo_world(), P, seed=3, disturbances=kick(dw)[:1], variant=a.variant)
            m = (R[:, 0] > 4.1) & (R[:, 0] < 8)
            tilt = np.degrees(np.arccos(np.clip(np.cos(R[m, C['roll']]) * np.cos(R[m, C['pitch']]), -1, 1)))
            rec = [r for r in s['recoveries'] if r[0] < 5]
            env.append({'dw_rad_s': dw, 'tilt_peak_deg': float(tilt.max()),
                        'z_min_m': float(R[m, C['z']].min()), 'recovery_s': rec[0][1] if rec else 0.0,
                        'success': s['success'], 'ground_contact': s['ground_contact']})
            print('  kick', env[-1])
        out['kick_envelope'] = env

    # 3) Monte Carlo y ablación sin suavizado
    res_s, res_a = [], []
    t0 = time.time()
    for i in range(a.n):
        w = random_corridor_world(np.random.default_rng(1000 + i))
        rs, _, _, _ = run_episode(w, P, seed=i, goal_x=6.5, t_max=60, variant=a.variant)
        w = random_corridor_world(np.random.default_rng(1000 + i))
        ra, _, _, _ = run_episode(w, P, seed=i, goal_x=6.5, t_max=60, shaping=False, variant=a.variant)
        res_s.append(rs); res_a.append(ra)
        print(f'  mc {i:02d}: suave {"OK " if rs["success"] else "FAIL"} clear={rs["min_clearance_m"]:.2f} '
              f'jerk99={rs["p99_jerk"]:.1f} tilt={rs["max_tilt_deg"]:.1f} | escalón '
              f'{"OK " if ra["success"] else "FAIL"} jerk99={ra["p99_jerk"]:.1f} tilt={ra["max_tilt_deg"]:.1f} '
              f'({time.time() - t0:.0f}s)', flush=True)
    agg = lambda L, k: float(np.median([r[k] for r in L]))
    out['montecarlo'] = {
        'n': a.n,
        'success_smooth': sum(r['success'] for r in res_s),
        'success_step': sum(r['success'] for r in res_a),
        'median': {k: {'smooth': agg(res_s, k), 'step': agg(res_a, k)}
                   for k in ['p99_jerk', 'max_tilt_deg', 'sat_pct', 'min_clearance_m', 'max_yaw_acc', 'time_s']},
        'collisions_smooth': sum(r['collided'] for r in res_s),
        'collisions_step': sum(r['collided'] for r in res_a),
        'timeouts_smooth': sum(r['timeout'] for r in res_s),
        'timeouts_step': sum(r['timeout'] for r in res_a),
        'failures_smooth': [i for i, r in enumerate(res_s) if not r['success']],
    }
    vtag = '' if a.variant == 'M' else f'_{a.variant}'
    keep = ('success', 'collided', 'timeout', 'time_s', 'min_clearance_m', 'p99_jerk', 'max_tilt_deg')
    out['runs'] = {'suave': [{k: r[k] for k in keep} for r in res_s],
                   'escalon': [{k: r[k] for k in keep} for r in res_a]}
    plot_mc(res_s, res_a, os.path.join(ASSETS, f'e3_montecarlo_{P.name}{vtag}' + (f'_inflate{a.inflate:g}' if a.inflate else '') + '.png'))
    tag = P.name + vtag + (f'_inflate{a.inflate:g}' if a.inflate else '') + ('_mc' if a.only_mc else '')
    json.dump(out, open(os.path.join(ASSETS, f'e3_results_{tag}.json'), 'w'), indent=2)
    print(json.dumps(out['montecarlo'], indent=2))


if __name__ == '__main__':
    main()
