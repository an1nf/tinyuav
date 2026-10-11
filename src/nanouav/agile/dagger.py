"""DAgger for the student policy (Loquercio et al. 2021, Methods C).

The student is rolled out in the training environments and every visited state is labeled by the privileged
expert. The student only flies while the drone is within ξ of τ_gbl; otherwise τ_gbl is tracked (the paper
uses an MPC). ξ starts at 0 (pure imitation along τ_gbl), the student is retrained on all the data every
30 environments and ξ ← min(ξ + 0.25, 6). 850 environments give ~90 000 samples in the paper; for the
narrow gap the policy is fine-tuned on 100 extra environments (~10 000 samples).

Collection runs in parallel processes on the CPU (`workers`); training runs in the main process (GPU if
available). Everything is resumable: environments whose shard exists are skipped, and a round whose model
exists is not retrained.

Not given in the paper and chosen here: the speed of each training rollout (uniform in [v_min, v_max]) and
the epochs per retraining.
"""
import json
import math
import multiprocessing as mp
import os
import time

import numpy as np

from nanouav.agile import envs as EV
from nanouav.agile.rollout import Config, Rollout


def save_shard(path, samples, stats):
    keys = ('depth', 'v', 'R', 'omega', 'trajs', 'costs', 'esdf', 'z')
    arr = {k: np.stack([s[k] for s in samples]) if samples else np.zeros((0,)) for k in keys}
    tmp = path + '.tmp.npz'
    np.savez_compressed(tmp, stats=json.dumps(stats), **arr)
    os.replace(tmp, path)


def _worker_init():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''          # collection on CPU; the GPU is for training
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')


_POLICY = {}


def _policy(path):
    if path is None:
        return None
    if path not in _POLICY:
        from nanouav.agile.policy import load_policy
        _POLICY.clear()
        _POLICY[path] = load_policy(path)
    return _POLICY[path]


def collect_one(job):
    """job = dict(i, out, model, xi, v_min, v_max, cam_hz, kind, base_seed) → stats."""
    i, out = job['i'], job['out']
    path = os.path.join(out, f'env_{i:04d}.npz')
    if os.path.exists(path):
        return json.loads(str(np.load(path)['stats']))
    rng = np.random.default_rng([job.get('base_seed', 0), i, 0x5D])
    kind = job.get('kind')
    v = float(rng.uniform(job['v_min'], job['v_max']))
    cfg = Config(v_des=v, xi=job['xi'], cam_hz=job['cam_hz'], seed=i, label_every=job.get('label_every', 1),
                 omega_ref=job.get('omega_ref', 'line'), label_ref=job.get('label_ref', 'gbl'))
    t0 = time.time()
    try:
        mission = None
        if kind == 'follow':                          # R19: follow a walking person (nanouav.follow)
            from nanouav.follow.mission import FollowMission
            from nanouav.follow.people import people_env
            env = people_env(int(np.random.default_rng([job.get('base_seed', 0), i, 0x19]).integers(1 << 30)))
            mission = FollowMission(env, seed=i)
        elif kind == 'balloon':                       # R20: follow a balloon through the forest
            from nanouav.follow.balloon import balloon_env
            from nanouav.follow.balloon_mission import BalloonMission
            env = balloon_env(int(np.random.default_rng([job.get('base_seed', 0), i, 0x20]).integers(1 << 30)),
                              color=job.get('color', 'red'))
            mission = BalloonMission(env, seed=i)
        else:
            env = (EV.gap_finetune_env if kind == 'gap' else EV.training_env)(i, job.get('base_seed', 0))
        r, samples, _ = Rollout(env, cfg, 'dagger', _policy(job['model']), label=True, mission=mission).run()
    except RuntimeError as e:                         # no global path (blocked environment)
        r, samples, env = {'error': str(e), 'env': f'{kind}_{i}'}, [], None
    r.update(i=i, n_samples=len(samples), wall_s=round(time.time() - t0, 1), xi=job['xi'],
             meta=env.meta if env is not None else {})
    save_shard(path, samples, r)
    return r


def run(out, n_envs=850, per_round=30, workers=4, epochs=3, batch=8, v_min=3.0, v_max=10.0, cam_hz=15.0,
        xi_step=0.25, xi_max=6.0, net_size=(120, 160), weights=None, kind='train', init_model=None,
        base_seed=0, label_every=1, xi_start=None, train_kw=None, omega_ref='line', label_ref='gbl', log=print):
    from nanouav.agile.policy import train, write_meta
    data = os.path.join(out, 'data_' + kind)
    os.makedirs(data, exist_ok=True)
    state_f = os.path.join(out, f'dagger_{kind}.json')
    state = json.load(open(state_f)) if os.path.exists(state_f) else {'rounds': []}
    model, xi = init_model, 0.0
    if kind in ('gap', 'follow', 'balloon'):
        xi = xi_max                                   # fine-tuning: the trained student flies
    if xi_start is not None:                          # e.g. a new skill (balloon: changing height)
        xi = xi_start
    n_rounds = math.ceil(n_envs / per_round)
    ctx = mp.get_context('spawn')
    for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):   # one core per worker
        os.environ.setdefault(k, '1')
    for r in range(n_rounds):
        mpath = os.path.join(out, f'student_{kind}_r{r:02d}.keras')
        ids = list(range(r * per_round, min((r + 1) * per_round, n_envs)))
        jobs = [dict(i=i, out=data, model=model, xi=xi, v_min=v_min, v_max=v_max, cam_hz=cam_hz,
                     kind=kind, base_seed=base_seed, label_every=label_every, omega_ref=omega_ref,
                     label_ref=label_ref) for i in ids]
        t0 = time.time()
        if workers > 1:
            with ctx.Pool(workers, initializer=_worker_init) as pool:
                stats = pool.map(collect_one, jobs, chunksize=1)
        else:
            stats = [collect_one(j) for j in jobs]
        ok = [s for s in stats if 'success' in s]
        summ = {'round': r, 'xi': xi, 'envs': len(ids), 'samples': int(sum(s['n_samples'] for s in stats)),
                'success': int(sum(s['success'] for s in ok)), 'crashed': int(sum(s['crashed'] for s in ok)),
                'student_frac': round(float(np.mean([s['student_frac'] for s in ok])) if ok else 0.0, 3),
                'collect_s': round(time.time() - t0, 1)}
        log(f'[dagger {kind}] round {r + 1}/{n_rounds} · ξ = {xi:.2f} · {summ["samples"]} samples · '
            f'success {summ["success"]}/{len(ok)} · student {summ["student_frac"]:.0%} of frames · {summ["collect_s"]} s')
        if not os.path.exists(mpath):
            _, hist = train(data if kind == 'train' else [os.path.join(out, 'data_train'), data], mpath,
                            epochs=epochs, batch=batch, init_from=model, net_size=net_size, weights=weights, log=log,
                            **(train_kw or {}))
            summ['val'] = hist[-1]['val']
            write_meta(mpath, omega_ref=omega_ref, label_ref=label_ref)
        model = mpath
        xi = min(xi + xi_step, xi_max)
        state['rounds'] = [x for x in state['rounds'] if x['round'] != r] + [summ]
        json.dump(state, open(state_f, 'w'), indent=1)
    final = os.path.join(out, f'student_{kind}.keras')
    if model and os.path.exists(model):
        import shutil
        shutil.copy(model, final)
        if os.path.exists(model + '.json'):
            shutil.copy(model + '.json', final + '.json')
    return final
