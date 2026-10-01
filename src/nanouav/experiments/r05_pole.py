"""
R5 · Controlled pole experiment (Falanga 2019 / Loquercio 2021 Fig. 5) in our loop.

A single pole on the straight path, with a cruise-speed sweep. It is run twice:
  NANOUAV_FAKE_CNN=1 nanouav run r05 ...   → privileged depth (GT + light noise)
                     nanouav run r05 ...   → real int8 CNN M
The difference separates PERCEPTION failures from PLANNING/CONTROL failures.

Extra metrics: speed when passing the pole, whether it braked (<0.2 m/s) before dodging,
distance to the pole when the lateral deviation starts (>0.1 m).
"""
import os, sys, json, time, argparse
import numpy as np
from nanouav import paths, settings

ROOT = str(paths.repo_root())
from nanouav.scene.renderer import World, Cylinder, Texture, random_ground, random_lighting
from nanouav.control.vehicle import PROFILES
from nanouav.sim.closed_loop import run_episode, C
from nanouav.sim.missions import fast_kw, set_brake_law

X_POLE = 4.0


def run_pole(P, v, seed, variant='M', law='brake'):
    rng = np.random.default_rng(seed)
    r = rng.uniform(0.1, 0.25)
    y0 = rng.uniform(-0.15, 0.15)
    w = World([Cylinder(X_POLE, y0, r, 0.0, 2.0, Texture('uniform', albedo=rng.uniform(0.2, 0.8)))],
              random_ground(rng), **random_lighting(rng))
    kw, lim = fast_kw(v)

    def on_frame(t, st, pl):
        set_brake_law(pl)
        pl.speed_law = 'brake'
        pl.CONSISTENT = (law == 'consistent')
        if pl.CONSISTENT:
            pl.FOV_OK = __import__('math').radians(28)
        pl.set_segment((-1.5, 0.0), (20.0, 0.0), arrive=False, v_cruise=v)

    s, R, _, _ = run_episode(w, P, seed=seed, start=(-1.5, 0.0, 0.3), t_max=15 + 14 / v, variant=variant,
                             on_frame=on_frame, done_fn=lambda t, p: p[0] > 8.0, planner_kw=kw, limits=lim)
    x, y = R[:, C['x']], R[:, C['y']]
    sp = np.hypot(R[:, C['vx']], R[:, C['vy']])
    near = np.abs(x - X_POLE) < 0.5
    appr = (x > X_POLE - 2.5) & (x < X_POLE) & (R[:, 0] > 3.0)
    dev = np.abs(y) > 0.1
    k = np.argmax(dev & (R[:, 0] > 3.0)) if (dev & (R[:, 0] > 3.0)).any() else None
    s.update(pole_r=r, v_cmd=v,
             v_pass=float(sp[near].mean()) if near.any() else float('nan'),
             v_max=float(sp.max()),
             stopped=bool(appr.any() and sp[appr].min() < 0.2),
             dodge_start_d=float(X_POLE - x[k]) if k is not None else float('nan'),
             max_lat=float(np.abs(y).max()))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--profile', default='nano', choices=list(PROFILES))
    ap.add_argument('--speeds', default='0.5,1.0,1.5,2.0,2.5,3.0')
    ap.add_argument('--n', type=int, default=5)
    ap.add_argument('--law', default='brake')
    ap.add_argument('--tag', default='')
    ap.add_argument('--variant', default='M')
    a = ap.parse_args()
    P = PROFILES[a.profile]
    mode = 'gt' if settings.flag('FAKE_CNN') else 'cnn'
    out, t0 = {}, time.time()
    for v in [float(x) for x in a.speeds.split(',')]:
        L = []
        for i in range(a.n):
            s = run_pole(P, v, 300 + i, variant=a.variant, law=a.law)
            L.append({k: (round(float(s[k]), 3) if isinstance(s[k], (float, np.floating)) else s[k])
                      for k in ('success', 'collided', 'timeout', 'time_s', 'min_clearance_m', 'v_pass',
                                'v_max', 'stopped', 'dodge_start_d', 'max_lat', 'pole_r', 'max_tilt_deg')})
            print(f'[{mode} v={v}] {i}: {L[-1]} ({time.time() - t0:.0f}s)', flush=True)
        out[str(v)] = L
    summ = {v: {'ok': sum(r['success'] for r in L), 'n': len(L), 'col': sum(r['collided'] for r in L),
                'stopped': sum(r['stopped'] for r in L),
                'v_pass_med': float(np.nanmedian([r['v_pass'] for r in L])),
                'dodge_d_med': float(np.nanmedian([r['dodge_start_d'] for r in L])),
                't_med': float(np.median([r['time_s'] for r in L]))} for v, L in out.items()}
    res = {'profile': P.name, 'mode': mode, 'law': a.law, 'runs': out, 'summary': summ}
    json.dump(res, open(os.path.join(str(paths.results_dir()), f'r5_pole_{P.name}_{mode}_{a.law}_{a.variant}{a.tag}.json'), 'w'), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
