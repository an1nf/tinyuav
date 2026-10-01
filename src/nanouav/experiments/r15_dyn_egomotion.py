"""
R15 · Falsas detecciones dinámicas según el tipo de movimiento propio (inspirado en DroneMOT,
Wang, Wang & Li 2024: el movimiento del dron se descompone en hover / traslación / rotación y cada uno
se compensa distinto; la rotación es la que produce los mayores saltos de píxel).

Pregunta: la capa RF_DYN (R10, excitación = celda vista libre que ahora aparece ocupada) ¿cuántas pistas
dinámicas FALSAS confirma en un mundo ESTÁTICO, y depende del tipo de ego-movimiento?
Y con un balón real cruzando: ¿lo detecta igual en cada régimen?

Lazo abierto: la cámara sigue una trayectoria prescrita (sin colisiones posibles: el carril y = 0 está
libre), se renderiza, CNN M + ToF + calibración como en vuelo, y se llama al planificador solo para
percibir/mapear/seguir. Métricas por minuto: pistas falsas confirmadas (n ≥ 3, |v| > 0.3 m/s, a > 0.6 m
de cualquier objeto dinámico), activaciones de EVADE, y para el balón: recall (fracción de su paso por el
campo visual con pista confirmada a < 0.4 m) y latencia de la primera confirmación.
Uso: python r15_dyn_egomotion.py --n 6 [--gate yaw]
"""
import os, sys, json, math, time, argparse
import numpy as np
from nanouav import paths
ROOT = str(paths.repo_root())
from nanouav.scene.renderer import World, Box, Cylinder, Sphere, Texture, render_view, random_ground, random_lighting
from nanouav.sensors.tof import tof_8x8
from nanouav.perception.cnn import predict_depth_norm
from nanouav.perception.tof_calibrator import ToFCalibrator, fuse
from nanouav.planning.evasion import EvasionPlanner

T_EP, DT = 20.0, 0.1


def static_world(rng):
    objs = []
    for x in np.arange(0.5, 11.0, 1.2):
        for side in (-1, 1):
            if rng.random() < 0.8:
                y = side * rng.uniform(0.9, 2.5)
                t = Texture(str(rng.choice(['uniform', 'noise', 'stripes'])), albedo=rng.uniform(0.2, 0.9),
                            albedo2=rng.uniform(0.1, 0.5), scale=rng.uniform(0.15, 0.5), seed=int(rng.integers(1 << 30)))
                if rng.random() < 0.5:
                    objs.append(Cylinder(x + rng.uniform(-0.3, 0.3), y, rng.uniform(0.08, 0.3), 0.0, rng.uniform(1.0, 2.0), t))
                else:
                    h = rng.uniform(0.1, 0.3, 2)
                    objs.append(Box((x, y, 0.75), (h[0], h[1], 0.75), rng.uniform(-1, 1), t))
    return World(objs, random_ground(rng), **random_lighting(rng))


def trajectory(kind, t, v=0.5):
    x = -1.5 + (v * t if kind in ('T', 'TR') else 0.0)
    yaw = 0.0
    if kind in ('R', 'TR'):
        yaw = 1.0 * math.sin(2 * math.pi * t / 6.0)            # ±57°, |ω| ≤ 1.05 rad/s
    rate = 1.0 * 2 * math.pi / 6.0 * math.cos(2 * math.pi * t / 6.0) if kind in ('R', 'TR') else 0.0
    vel = np.array([v if kind in ('T', 'TR') else 0.0, 0.0, 0.0])
    return np.array([x, 0.0, 0.65]), yaw, rate, vel


def episode(kind, seed, ball, gate=None, cfg=None):
    rng = np.random.default_rng(seed)
    w = static_world(rng)
    b = None
    if ball:
        b = Sphere((50, 50, 0.65), 0.15, Texture('uniform', albedo=0.85))
        w.objects.append(b)
    pl = EvasionPlanner(radius=0.36, v_cruise=0.5)
    pl.speed_law, pl.CONSISTENT, pl.RF_DYN = 'brake', True, True
    pl.set_segment((-1.5, 0.0), (20.0, 0.0), arrive=False)
    for k_, v_ in (cfg or {}).items():
        setattr(pl, k_, v_)
    cal = ToFCalibrator()
    seen_fp, n_evade, conf_ball, vis_ball, first_conf = set(), 0, 0, 0, None
    t_launch = 3.0
    for k in range(int(T_EP / DT)):
        t = k * DT
        p, yaw, rate, vel = trajectory(kind, t)
        bpos = None
        if b is not None and t >= t_launch:                     # cruza y = −2.5 → 2.5 a 1 m/s en x = p + 2.2
            x_b = p[0] + 2.2 if kind in ('H', 'R') else -1.5 + 0.5 * t_launch + 2.2 + 0.5 * 2.5
            bpos = np.array([x_b, -2.5 + 1.0 * (t - t_launch), 0.65])
            if bpos[1] > 2.5:
                bpos = None
        if b is not None:
            b.center = tuple(bpos) if bpos is not None else (50.0, 50.0, 0.65)
        gray, dgt = render_view(w, p, yaw, 0.0, 0.0, rng=rng)
        tof = tof_8x8(dgt, rng=rng)
        dn = predict_depth_norm(gray, 'M', tof=tof)
        depth = fuse(cal.update_and_scale(dn, tof), tof)
        if gate == 'yaw' and abs(rate) > 0.35:
            pl.RF_DYN_SUPPRESS = True                         # (se usa abajo: se descartan detecciones nuevas)
        else:
            pl.RF_DYN_SUPPRESS = False
        n_before = {id(tr) for tr in pl._tracks}
        _, _, info = pl.update({'p': p, 'v': vel, 'yaw': yaw, 't': t}, depth, tof, (p, yaw, 0.0, 0.0))
        if pl.RF_DYN_SUPPRESS:                                 # DMP-like: en rotación rápida no se crean pistas
            pl._tracks = [tr for tr in pl._tracks if id(tr) in n_before]
        if info.get('mode') == 'EVADE':
            n_evade += 1
        # ¿balón en el campo visual (±30°) y a < 4 m?
        in_view = False
        if bpos is not None:
            d = bpos[:2] - p[:2]
            ang = abs((math.atan2(d[1], d[0]) - yaw + math.pi) % (2 * math.pi) - math.pi)
            in_view = ang < math.radians(30) and np.hypot(*d) < 4.0
            vis_ball += in_view
        got = False
        for tr in pl._tracks:
            if not pl._rf_confirmed(tr, 3):
                continue
            if bpos is not None and np.hypot(*(tr['p'] - bpos[:2])) < 0.6:
                got = True
            else:
                seen_fp.add(id(tr))
        if got and in_view:
            conf_ball += 1
            if first_conf is None:
                first_conf = t - t_launch
    return {'fp_per_min': len(seen_fp) / (T_EP / 60), 'evade_s': n_evade * DT,
            'ball_recall': (conf_ball / vis_ball) if vis_ball else None, 'ball_first_s': first_conf}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--kinds', default='H,T,R,TR')
    ap.add_argument('--gate', default='')
    ap.add_argument('--cfg', default='', help='p.ej. "RF_NET_DISP=0.4,RF_MIN_PTS=5"')
    a = ap.parse_args()
    cfg = {k: float(v) for k, v in (x.split('=') for x in a.cfg.split(',') if x)}
    out, t0 = {}, time.time()
    for kind in a.kinds.split(','):
        for ball in (False, True):
            L = [episode(kind, 1500 + i, ball, a.gate or None, cfg) for i in range(a.n)]
            key = f'{kind}_{"ball" if ball else "static"}'
            out[key] = L
            fp = np.mean([r['fp_per_min'] for r in L])
            ev = np.mean([r['evade_s'] for r in L])
            rec = [r['ball_recall'] for r in L if r['ball_recall'] is not None]
            lat = [r['ball_first_s'] for r in L if r['ball_first_s'] is not None]
            print(f'{key:10s} FP/min {fp:5.2f}  EVADE {ev:4.1f} s/20 s  recall_balón '
                  f'{(np.mean(rec) if rec else float("nan")):.2f}  detectado {len(lat)}/{len(L)} '
                  f'lat_med {(np.median(lat) if lat else float("nan")):.2f} s  ({time.time() - t0:.0f}s)', flush=True)
    json.dump(out, open(os.path.join(str(paths.results_dir()), f'r15_dyn_egomotion{"_" + a.gate if a.gate else ""}{"_" + a.cfg.replace("=", "").replace(",", "_") if a.cfg else ""}.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
