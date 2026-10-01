"""
R6 · Does the 'consistent' mode (R5) generalize to worlds with several obstacles?
Random corridors from the E3 Monte Carlo (random_corridor_world), 'brake' law, int8 CNN M.
Usage: nanouav run r06 --law consistent --speeds 1.0,1.5 --n 8
"""
import os, sys, json, time, math, argparse
import numpy as np
from nanouav import paths
ROOT = str(paths.repo_root())
from nanouav.sim.closed_loop import run_episode, random_corridor_world
from nanouav.sim.missions import fast_kw, set_brake_law
from nanouav.control.vehicle import PROFILES

ap = argparse.ArgumentParser()
ap.add_argument('--profile', default='nano')
ap.add_argument('--law', default='brake')
ap.add_argument('--speeds', default='1.0,1.5')
ap.add_argument('--n', type=int, default=8)
ap.add_argument('--variant', default='M')
a = ap.parse_args()
P = PROFILES[a.profile]
out, t0 = {}, time.time()
for v in [float(x) for x in a.speeds.split(',')]:
    kw, lim = fast_kw(v)
    L = []

    def on_frame(t, st, pl, v=v):
        set_brake_law(pl)
        pl.v_cruise = v
        pl.CONSISTENT = a.law == 'consistent'
        if pl.CONSISTENT:
            pl.FOV_OK = math.radians(28)

    for i in range(a.n):
        w = random_corridor_world(np.random.default_rng(1000 + i))
        s, _, _, _ = run_episode(w, P, seed=i, goal_x=6.5, t_max=60, on_frame=on_frame, planner_kw=kw, limits=lim,
                                 variant=a.variant)
        L.append({k: (round(float(s[k]), 3) if isinstance(s[k], (float, np.floating)) else s[k])
                  for k in ('success', 'collided', 'timeout', 'time_s', 'min_clearance_m', 'p99_jerk', 'max_tilt_deg')})
        print(f'[{a.law} v={v}] {i}: {L[-1]} ({time.time() - t0:.0f}s)', flush=True)
    out[str(v)] = L
json.dump(out, open(os.path.join(str(paths.results_dir()), f'r6_corridor_{P.name}_{a.law}_{a.variant}_n{a.n}_v{a.speeds}.json'), 'w'), indent=1)
for v, L in out.items():
    print(v, f"ok {sum(r['success'] for r in L)}/{len(L)} col {sum(r['collided'] for r in L)} "
             f"t_med {np.median([r['time_s'] for r in L]):.1f} clear_med {np.median([r['min_clearance_m'] for r in L]):.2f}")
