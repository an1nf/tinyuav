"""DAgger inside Flightmare (Loquercio et al. 2021, Methods C, in the simulator of the paper).

Same algorithm as nanouav.agile.dagger: the student flies while it is within ξ of τ_gbl, otherwise τ_gbl
is tracked; every visited state is labelled by the privileged M-H expert; retrain on all the data every
`per_round` environments; ξ ← min(ξ + step, ξ_max). What changes:

  · the images come from the Unity player (one instance → the rollouts run one after the other in the main
    process) at VGA, like the paper, and the depth is SGM on that stereo pair;
  · the environments are `arena.arena_env` (the scanned static scene + spawned trunks, exact geometry);
  · the expert labels — the slow part, ~0.4 s each — are computed by a pool of CPU workers while the next
    rollout flies (`Rollout.defer`).

Resumable like the other DAgger: environments whose shard exists are skipped, rounds whose model exists
are not retrained.
"""
import json
import math
import multiprocessing as mp
import os
import time

import numpy as np

_ARENA = None
TILT_V, TILT_DEG = 7.0, 30.0           # the paper: depth sensor tilted 30° up for flights at ≥ 7 m/s


def tilted(rig, v):
    """The rig for a flight at v (m/s): tilted up TILT_DEG at v ≥ TILT_V (the network infers it from v)."""
    import dataclasses
    return dataclasses.replace(rig, tilt_up_deg=TILT_DEG if v >= TILT_V - 1e-6 else 0.0)


def _init(arena_path):
    global _ARENA
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    from nanouav.flightmare import arena as A
    _ARENA = A.load(arena_path)


def _label(job):
    """One deferred state → the expert label fields of a sample (or None if the expert found nothing)."""
    from nanouav.agile import trajectory as TR
    from nanouav.agile.expert import Expert
    from nanouav.agile.geometry import Field
    from nanouav.agile.rollout import Config, local_esdf
    from nanouav.scene.renderer import Box
    boxes = [Box(tuple(c), tuple(h)) for c, h in job['boxes']]
    world = _ARENA.with_boxes(boxes)
    p, ref, v, yaw = job['p'], job['ref'], job['v'], job['yaw']
    L = Expert(world, seed=job['seed'], z_min=world.floor + 0.4).label(p, ref, v0=v)
    if not len(L['trajs']):
        return None
    return {'trajs': TR.to_heading_frame(L['trajs'], p, yaw).astype(np.float32),
            'costs': L['costs'].astype(np.float32),
            'esdf': local_esdf(Field(world, ground=False), p, yaw, p[2], Config())}


def _finish(path, r, items, labels):
    from nanouav.agile.dagger import save_shard
    samples = []
    for it, lab in zip(items, labels):
        if lab is None:
            continue
        s = {k: it[k] for k in ('depth', 'v', 'R', 'omega', 'z')}
        s.update(lab)
        samples.append(s)
    r['n_samples'] = len(samples)
    save_shard(path, samples, r)
    return r


def run(client, arena_path, out, rig, n_envs=300, per_round=30, workers=6, epochs=3, batch=16, v_min=3.0,
        v_max=10.0, cam_hz=15.0, init_model=None, xi_start=1.0, xi_step=0.5, xi_max=6.0, base_seed=0,
        train_kw=None, tag='fm', omega_ref='line', label_ref='gbl', log=print):
    from nanouav.agile import dagger as D
    from nanouav.agile.policy import train, write_meta
    from nanouav.agile.rollout import Config, Rollout
    from nanouav.flightmare import arena as A
    from nanouav.flightmare.run import FlightmareObserver
    arena = A.load(arena_path)
    data = os.path.join(out, f'data_{tag}')
    os.makedirs(data, exist_ok=True)
    state_f = os.path.join(out, f'dagger_{tag}.json')
    state = json.load(open(state_f)) if os.path.exists(state_f) else {'rounds': []}
    model, xi = init_model, xi_start
    ctx = mp.get_context('spawn')
    pool = ctx.Pool(workers, initializer=_init, initargs=(arena_path,))
    n_rounds = math.ceil(n_envs / per_round)
    try:
        for r in range(n_rounds):
            mpath = os.path.join(out, f'student_{tag}_r{r:02d}.keras')
            ids = list(range(r * per_round, min((r + 1) * per_round, n_envs)))
            pending, stats, t0 = [], [], time.time()
            for i in ids:
                path = os.path.join(data, f'env_{i:04d}.npz')
                if os.path.exists(path):
                    stats.append(json.loads(str(np.load(path)['stats'])))
                    continue
                rng = np.random.default_rng([base_seed, i, 0x5E])
                v = float(rng.uniform(v_min, v_max))
                try:
                    env = A.arena_env(arena, i, base_seed, 'train')
                except RuntimeError as e:
                    stats.append({'error': str(e), 'i': i, 'n_samples': 0})
                    continue
                client.place(env.world.objects)
                rig_v = tilted(rig, v)
                client.set_tilt(rig_v.tilt_up_deg)
                cfg = Config(v_des=v, xi=xi, cam_hz=cam_hz, seed=i, rig=rig_v, omega_ref=omega_ref,
                             label_ref=label_ref)
                R = Rollout(env, cfg, 'dagger', D._policy(model), label=True)
                R.observer = FlightmareObserver(client, rig_v)
                R.defer = []
                te = time.time()
                res, _, _ = R.run()
                res.update(i=i, v=round(v, 2), xi=xi, wall_s=round(time.time() - te, 1), meta=env.meta)
                bx = [(list(b.center), list(b.half)) for b in env.world.objects]
                jobs = [{'boxes': bx, 'p': it['_p'], 'ref': it['_ref'], 'v': it['_v'], 'yaw': it['_yaw'],
                         'seed': i * 1000 + k} for k, it in enumerate(R.defer)]
                pending.append((path, res, R.defer, pool.map_async(_label, jobs, chunksize=4)))
                done = [x for x in pending if x[3].ready()]
                for x in done:
                    stats.append(_finish(x[0], x[1], x[2], x[3].get()))
                    pending.remove(x)
                log(f"[fm dagger] env {i} · v {v:.1f} · {'ok' if res['success'] else 'CRASH' if res['crashed'] else 'fail'}"
                    f" · student {res.get('student_frac', 0):.0%} · {len(R.defer)} frames · {res['wall_s']} s")
            for x in pending:
                stats.append(_finish(x[0], x[1], x[2], x[3].get()))
            ok = [s for s in stats if 'success' in s]
            summ = {'round': r, 'xi': xi, 'envs': len(ids), 'samples': int(sum(s.get('n_samples', 0) for s in stats)),
                    'success': int(sum(s['success'] for s in ok)), 'crashed': int(sum(s['crashed'] for s in ok)),
                    'student_frac': round(float(np.mean([s.get('student_frac', 0) for s in ok])) if ok else 0.0, 3),
                    'collect_s': round(time.time() - t0, 1)}
            log(f'[fm dagger] round {r + 1}/{n_rounds} · ξ = {xi:.2f} · {summ["samples"]} samples · success '
                f'{summ["success"]}/{len(ok)} · student {summ["student_frac"]:.0%} of frames · {summ["collect_s"]} s')
            if not os.path.exists(mpath):
                _, hist = train(data, mpath, epochs=epochs, batch=batch, init_from=model, log=log, **(train_kw or {}))
                summ['val'] = hist[-1]['val']
                write_meta(mpath, omega_ref=omega_ref, label_ref=label_ref)
            model = mpath
            xi = min(xi + xi_step, xi_max)
            state['rounds'] = [x for x in state['rounds'] if x['round'] != r] + [summ]
            json.dump(state, open(state_f, 'w'), indent=1)
    finally:
        pool.terminate()
    final = os.path.join(out, f'student_{tag}.keras')
    if model and os.path.exists(model):
        import shutil
        shutil.copy(model, final)
        if os.path.exists(model + '.json'):
            shutil.copy(model + '.json', final + '.json')
    return final
