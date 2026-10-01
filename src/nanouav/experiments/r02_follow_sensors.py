"""
R2 · Target following (E4 follow) with a realistic visual measurement of the target.

Compares, with the same worlds/seeds:
  oracle/base    : original E4 (position + 5 cm noise, 10 % dropouts, no FOV)
  bbox/base      : onboard detector (FOV, occlusion, range from apparent size)
  bbox/ibvs      : + guidance layer from Su et al. 2026 (hysteresis, LP, bounded backoff)
  bbox_tof/ibvs  : + range from ToF/CNN within the central 40° (our own proposal)

Usage: nanouav run r02 --profile nano --n 4 --vt 0.4,0.8
       NANOUAV_FAKE_CNN=1 nanouav run r02 --n 2        # quick debugging
Output: results/r2_follow_{profile}.json
"""
import os, sys, json, time, argparse
import numpy as np
from nanouav import paths

ROOT = str(paths.repo_root())
from nanouav.sim.missions import run_follow
from nanouav.control.vehicle import PROFILES

CONFIGS = [('oracle', 'base'), ('bbox', 'base'), ('bbox', 'ibvs'), ('bbox_tof', 'ibvs')]
KEYS = ['success', 'collided', 'lost_gt5s', 'follow_err_mean', 'follow_err_p95', 'target_in_fov_pct',
        'min_dist_target', 'det_rate', 'max_gap_s', 'size_bias', 'p99_jerk', 'max_tilt_deg']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--profile', default='nano', choices=list(PROFILES))
    ap.add_argument('--variant', default='M')
    ap.add_argument('--n', type=int, default=4)
    ap.add_argument('--vt', default='0.4,0.8')
    ap.add_argument('--T', type=float, default=45.0)
    ap.add_argument('--configs', default=','.join(f'{a}/{b}' for a, b in CONFIGS))
    a = ap.parse_args()
    P = PROFILES[a.profile]
    cfgs = [tuple(c.split('/')) for c in a.configs.split(',')]
    out, t0 = {}, time.time()
    for vt in [float(x) for x in a.vt.split(',')]:
        for sensor, guid in cfgs:
            key = f'v{vt}/{sensor}/{guid}'
            out[key] = []
            for i in range(a.n):
                s, _, _ = run_follow(P, vt, 700 + i, a.variant, T=a.T, sensor=sensor, guidance=guid)
                r = {k: (round(float(s[k]), 3) if isinstance(s.get(k), (float, np.floating)) else s.get(k))
                     for k in KEYS if k in s}
                out[key].append(r)
                print(f'[{key}] {i}: {r}  ({time.time() - t0:.0f}s)', flush=True)
    summ = {}
    for k, L in out.items():
        summ[k] = {'ok': sum(bool(r['success']) for r in L), 'n': len(L),
                   'collisions': sum(bool(r['collided']) for r in L),
                   'lost5s': sum(bool(r.get('lost_gt5s')) for r in L),
                   'err_med': float(np.median([r['follow_err_mean'] for r in L])),
                   'err_p95_med': float(np.median([r['follow_err_p95'] for r in L])),
                   'fov_med': float(np.median([r['target_in_fov_pct'] for r in L])),
                   'min_d_min': float(np.min([r['min_dist_target'] for r in L]))}
    res = {'profile': P.name, 'variant': a.variant, 'runs': out, 'summary': summ}
    json.dump(res, open(os.path.join(str(paths.results_dir()), f'r2_follow_{P.name}_vt{a.vt}.json'), 'w'), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
