"""Stage 2 data: states from gbl and blind flights in training environments, with SGM depth, and three label
sets per state: priv (M-H on the full map, τ_gbl reference: R18's labels), line (full map, straight-line
reference: the reference the student is told), obs (visible obstacles only + straight line)."""
import os
import sys
import time
import multiprocessing as mp

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import D  # noqa: E402
OUT = os.path.join(D, 's2')
BASE = 4242


def episode(job):
    from nanouav.agile import envs as EV
    from nanouav.agile import trajectory as TR
    from nanouav.agile.expert import Expert
    from nanouav.agile.geometry import Field
    from nanouav.agile.rollout import Config, Rollout, local_esdf, straight_ref
    from nanouav.agile.stereo import observe_full
    i, pilot = job
    fn = os.path.join(OUT, f'ep_{i:04d}_{pilot}.npz')
    if os.path.exists(fn):
        return fn
    rng = np.random.default_rng([BASE, i, 7 if pilot == 'gbl' else 9])
    env = EV.training_env(i, base_seed=BASE)
    v = float(rng.uniform(3.0, 8.0))
    cfg = Config(v_des=v, seed=i, label_every=3)
    R = Rollout(env, cfg, pilot, label=True)
    k = [0]
    H, W = cfg.rig.size
    zero = (np.zeros((H, W), np.uint8), np.zeros((H, W), np.float32), np.zeros((H, W), np.float32))

    def obs(p, yaw, pitch, roll):
        render = k[0] % cfg.label_every == 0
        k[0] += 1
        return observe_full(env.world, cfg.rig, p, yaw, pitch, roll, R.rng)[:3] if render else zero
    R.observer = obs
    R.defer = []
    R.run()
    line = straight_ref(env, v)
    fng = Field(env.world, ground=False)
    out = {key: [] for key in ('depth', 'v', 'R', 'omega', 'omega_gbl', 'z', 'esdf', 'priv', 'line', 'p', 'yaw', 'ok')}
    for j, it in enumerate(R.defer):
        p, ref, vel, yaw = it['_p'], it['_ref'], it['_v'], it['_yaw']
        s0, _ = line.closest(p)
        ref_line = line.ahead(s0)
        labs = []
        g = ref[-1] - p                                    # τ_gbl point 1 s ahead of the closest gbl state
        om_g = TR.rotate_to_heading(g / max(np.linalg.norm(g), 1e-6), yaw).astype(np.float32)
        for world, rf in ((env.world, ref), (env.world, ref_line)):
            L = Expert(world, seed=i * 1000 + j).label(p, rf, v0=vel)['trajs']
            labs.append(TR.to_heading_frame(L, p, yaw).astype(np.float32) if len(L) == 3 else np.zeros((3, 10, 3), np.float32))
        okk = all(np.abs(x).sum() > 0 for x in labs)
        for key, val in (('depth', it['depth']), ('v', it['v']), ('R', it['R']), ('omega', it['omega']),
                         ('omega_gbl', om_g), ('z', it['z']),
                         ('esdf', local_esdf(fng, p, yaw, p[2], cfg)), ('priv', labs[0]), ('line', labs[1]),
                         ('p', p.astype(np.float32)), ('yaw', np.float32(yaw)), ('ok', okk)):
            out[key].append(val)
    np.savez_compressed(fn, env=i, pilot=pilot, v_des=v, **{key: np.array(val) for key, val in out.items()})
    return fn


if __name__ == '__main__':
    for kk in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[kk] = '1'
    os.makedirs(OUT, exist_ok=True)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    jobs = [(i, p) for i in range(n) for p in ('gbl', 'blind')]
    t0 = time.time()
    with mp.get_context('spawn').Pool(2) as pool:
        for m, fn in enumerate(pool.imap_unordered(episode, jobs)):
            if m % 10 == 0:
                print(f'{m + 1}/{len(jobs)} episodes · {time.time() - t0:.0f} s', flush=True)
    print('done', time.time() - t0)
