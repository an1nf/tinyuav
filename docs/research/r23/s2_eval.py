"""Stage 2, closed loop: the small students A/B/C in the R18 test forests (straight-line reference only)."""
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import D  # noqa: E402
_M = {}


def job(j):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import tensorflow as tf
    from nanouav.agile import envs as EV
    from nanouav.agile.rollout import Config, Rollout
    from s2_train import Policy
    name, v, seed = j
    if name not in _M:
        _M[name] = Policy(tf.keras.models.load_model(os.path.join(D, f's2_model_{name}.keras'), compile=False))
    env = EV.TEST['forest'](seed)
    R = Rollout(env, Config(v_des=v, seed=seed, omega_ref='goal' if name in ('D', 'F') else 'line'),
                'student', _M[name])
    R.trace = []
    r, _, log = R.run()
    lat = float(np.mean([abs(row[2] - env.start[1]) for row in log]))
    dz = float(np.mean([abs(row[3] - env.start[2]) for row in log]))
    prog = float(max(row[1] for row in log) - env.start[0]) / float(env.goal[0] - env.start[0])
    return {'model': name, 'v': v, 'seed': seed, 'success': r['success'], 'crashed': r['crashed'],
            'time_s': r['time_s'], 'mean_abs_lateral_m': round(lat, 2),
            'mean_abs_dz_m': round(dz, 2), 'progress': round(min(prog, 1.0), 3),
            'all_collide_frac': round(float(np.mean([max(d['clear']) < 0.2 for d in R.trace])), 3) if R.trace else None}


if __name__ == '__main__':
    for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[k] = '1'
    models = sys.argv[1].split(',') if len(sys.argv) > 1 else ['A', 'B', 'C']
    speeds = [float(x) for x in (sys.argv[2].split(',') if len(sys.argv) > 2 else ['3', '5'])]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    jobs = [(m, v, 1000 + s) for v in speeds for s in range(n) for m in models]
    t0 = time.time()
    out = []
    with mp.get_context('spawn').Pool(2) as pool:
        for r in pool.imap_unordered(job, jobs):
            out.append(r)
            print(json.dumps(r), f'{time.time() - t0:.0f} s', flush=True)
    fn = os.path.join(D, 's2_closed_loop.json')
    old = [r for r in json.load(open(fn)) if r['model'] not in models] if os.path.exists(fn) else []
    json.dump(old + out, open(fn, 'w'), indent=1)
    for m in models:
        for v in speeds:
            R = [r for r in out if r['model'] == m and r['v'] == v]
            print(f'{m} v={v:g}: {sum(r["success"] for r in R)}/{len(R)}  crashes {sum(r["crashed"] for r in R)}  '
                  f'all-collide {np.mean([r["all_collide_frac"] for r in R]):.2f}  |lateral| {np.mean([r["mean_abs_lateral_m"] for r in R]):.2f} m')
