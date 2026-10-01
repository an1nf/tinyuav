"""
R8 · Sensibilidad a ruido de estimación y de empuje (Loquercio 2021 §S1, Tabla S1).
Corredores E3, ley 'brake' ± modo consistente, CNN M int8. El mapa local del planificador
está en coordenadas mundo: la deriva de posición estimada lo "corre" respecto al mundo real.
Uso: python r8_state_noise.py --law consistent --v 1.0 --n 10
"""
import os, sys, json, time, math, argparse
import numpy as np
from nanouav import paths
ROOT = str(paths.repo_root())
from nanouav.sim.closed_loop import run_episode, random_corridor_world
from nanouav.sim.missions import fast_kw, set_brake_law
from nanouav.control.vehicle import PROFILES

LEVELS = {'sin ruido': None,
          'bajo (0.05)': {'sb': 0.05, 'sw': 0.05, 'tau': 1.0},
          'medio (0.10)': {'sb': 0.10, 'sw': 0.10, 'tau': 1.0},
          'alto (0.20)': {'sb': 0.20, 'sw': 0.20, 'tau': 1.0},
          'solo sesgo 0.20': {'sb': 0.20, 'sw': 0.0, 'tau': 1.0},
          'solo blanco 0.20': {'sb': 0.0, 'sw': 0.20, 'tau': 1.0}}

ap = argparse.ArgumentParser()
ap.add_argument('--profile', default='nano')
ap.add_argument('--law', default='consistent')
ap.add_argument('--v', type=float, default=1.0)
ap.add_argument('--n', type=int, default=10)
ap.add_argument('--levels', default=','.join(LEVELS))
ap.add_argument('--thrust', action='store_true', help='factor de empuje m ~ U(0.9, 1) por episodio')
ap.add_argument('--drift-corr', action='store_true')
ap.add_argument('--vo', action='store_true')
ap.add_argument('--variant', default='M')
ap.add_argument('--decay', type=float, default=None, help='olvido por frame del mapa local (def. 0.995)')
a = ap.parse_args()
P = PROFILES[a.profile]
kw, lim = fast_kw(a.v)
out, t0 = {}, time.time()


def on_frame(t, st, pl):
    set_brake_law(pl)
    pl.v_cruise = a.v
    pl.CONSISTENT = a.law == 'consistent'
    if pl.CONSISTENT:
        pl.FOV_OK = math.radians(28)
    if a.decay is not None:
        pl.grid.decay = a.decay
    pl.DRIFT_CORR = a.drift_corr
    pl.VO = a.vo


for lev in a.levels.split(','):
    L = []
    for i in range(a.n):
        w = random_corridor_world(np.random.default_rng(1000 + i))
        m = float(np.random.default_rng(50 + i).uniform(0.9, 1.0)) if a.thrust else 1.0
        s, R, _, _ = run_episode(w, P, seed=i, goal_x=6.5, t_max=60, on_frame=on_frame, planner_kw=kw,
                                 limits=lim, est_noise=LEVELS[lev], thrust_m=m, variant=a.variant)
        r = {k: (round(float(s[k]), 3) if isinstance(s[k], (float, np.floating)) else s[k])
             for k in ('success', 'collided', 'timeout', 'time_s', 'min_clearance_m', 'p99_jerk', 'max_tilt_deg')}
        r['thrust_m'] = round(m, 3)
        L.append(r)
        print(f'[{a.law} {lev}] {i}: {r} ({time.time() - t0:.0f}s)', flush=True)
    out[lev] = L
tag = f"{P.name}_{a.law}_{a.variant}_v{a.v}{'_thrust' if a.thrust else ''}_{len(a.levels.split(','))}lv{'_dec'+str(a.decay) if a.decay else ''}{'_dc' if a.drift_corr else ''}{'_vo' if a.vo else ''}"
json.dump(out, open(os.path.join(str(paths.results_dir()), f'r8_noise_{tag}.json'), 'w'), indent=1)
for lev, L in out.items():
    print(f"{lev:14s} ok {sum(r['success'] for r in L)}/{len(L)} col {sum(r['collided'] for r in L)} "
          f"t_med {np.median([r['time_s'] for r in L]):.1f} clear_min {np.min([r['min_clearance_m'] for r in L]):.2f}")
