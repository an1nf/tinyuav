"""R18 · Recreation of Loquercio et al. 2021 (*Learning high-speed flight in the wild*).

  nanouav run r18 sgm                                  # SGM depth quality vs ground truth
  nanouav run r18 pilots --world forest --v 3,5,7,10   # expert / τ_gbl / blind baselines (no network)
  nanouav run r18 dagger --envs 850 --workers 7        # data collection + training (resumable)
  nanouav run r18 dagger --kind gap --envs 100 --init results/r18/student_train.keras   # narrow-gap fine-tuning
  nanouav run r18 eval --model results/r18/student_train.keras --world forest --v 3,5,7,10
  nanouav run r18 eval --model … --world gap --v 3,5,7

Results go to $NANOUAV_RESULTS/r18 (default results/r18). See docs/research/R18_agile_autonomy.md.
"""
import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

from nanouav import paths


def _out(a):
    d = a.out or os.path.join(str(paths.results_dir()), 'r18')
    os.makedirs(d, exist_ok=True)
    return d


def opts_of(a):
    """Rollout options from the command line → (dict, file-name suffix of the non-default ones)."""
    o = {'proj': getattr(a, 'proj', 'scaled'), 'tscale': a.tscale, 'vcap': a.vcap, 'select': a.select}
    suf = ''.join([f"_{o['proj']}" if o['proj'] != 'scaled' else '', f"_{o['tscale']}" if o['tscale'] != 'chord' else '',
                   f"_cap{o['vcap']:g}" if o['vcap'] else '', f"_{o['select']}" if o['select'] != 'snap' else ''])
    return o, suf


def apply_opts(R, o):
    R.PROJ = o.get('proj', 'scaled')
    R.TSCALE = o.get('tscale', 'chord')
    R.V_CAP = o.get('vcap') or None
    R.SELECT = o.get('select', 'snap')
    return R


def _eval_job(job):
    from nanouav.agile import dagger, envs as EV
    from nanouav.agile.rollout import Config, Rollout
    dagger._worker_init()
    env = EV.TEST[job['world']](job['seed'])
    cfg = Config(v_des=job['v'], cam_hz=job['cam_hz'], seed=job['seed'])
    pol = dagger._policy(job.get('model')) if job['pilot'] == 'student' else None
    try:
        R = Rollout(env, cfg, job['pilot'], pol)
        apply_opts(R, job)
        r, _, log = R.run()
    except RuntimeError as e:
        r, log = {'success': False, 'crashed': False, 'timeout': False, 'error': str(e)}, []
    r.update(seed=job['seed'], world=job['world'])
    if job.get('traj_dir'):
        np.save(os.path.join(job['traj_dir'], f"{job['world']}_{job['pilot']}{job.get('tag', '')}_v{job['v']:g}_s{job['seed']}.npy"),
                np.array([row[:7] for row in log], float))
    return r


def _sweep(a, pilots, model=None):
    out = _out(a)
    speeds = [float(x) for x in a.v.split(',')]
    seeds = list(range(a.seed0, a.seed0 + a.n))
    tdir = os.path.join(out, 'traj')
    os.makedirs(tdir, exist_ok=True)
    o, suf = opts_of(a)
    tag = (os.path.splitext(os.path.basename(model))[0] if model else 'pilots') + suf
    if a.tag:
        tag += f'_{a.tag}'
    jobs = [dict(world=a.world, pilot=p, v=v, seed=s, cam_hz=a.cam_hz, model=model, traj_dir=tdir,
                 tag=f'_{tag}' if model else '', **o) for p in pilots for v in speeds for s in seeds]
    t0 = time.time()
    for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):   # one core per worker
        os.environ.setdefault(k, '1')
    ctx = mp.get_context('spawn')
    with ctx.Pool(a.workers) as pool:
        res = pool.map(_eval_job, jobs, chunksize=1)
    table = {}
    for j, r in zip(jobs, res):
        table.setdefault(f"{j['pilot']}_v{j['v']:g}", []).append(r)
    fn = os.path.join(out, f'eval_{a.world}_{tag}.json')
    old = json.load(open(fn)) if os.path.exists(fn) else {}
    old.update(table)
    json.dump(old, open(fn, 'w'), indent=1)
    print(f'{"pilot":10s} {"v":>5s} {"success":>8s} {"crash":>6s} {"t_med":>6s} {"v_mean":>6s}')
    for p in pilots:
        for v in speeds:
            L = table[f'{p}_v{v:g}']
            ok = [r for r in L if r.get('success')]
            tm = np.median([r['time_s'] for r in ok]) if ok else float('nan')
            vm = np.median([r['mean_speed'] for r in ok]) if ok else float('nan')
            print(f'{p:10s} {v:5.1f} {len(ok):3d}/{len(L):<4d} {sum(r.get("crashed", 0) for r in L):6d} {tm:6.1f} {vm:6.1f}')
    print(f'→ {fn}  ({time.time() - t0:.0f} s)')


def cmd_sgm(a):
    from nanouav.agile import envs as EV
    from nanouav.agile.stereo import StereoRig, depth_metrics, observe
    rig = StereoRig(width=a.width, height=a.height)
    rows = []
    for s in range(a.n):
        for name in ('forest', 'gap'):
            e = EV.TEST[name](s)
            rng = np.random.default_rng(s)
            for x in (0.0, 4.0):
                p = e.start + [x, 0, 0]
                d, g = observe(e.world, rig, p, 0.0, rng=rng)
                m = depth_metrics(d, g)
                m.update(world=name, seed=s, x=x)
                rows.append(m)
    for k in ('valid', 'valid_gt', 'rel_med', 'rel_p90'):
        print(f'{k:9s} {np.nanmedian([r[k] for r in rows]):.3f}')
    json.dump(rows, open(os.path.join(_out(a), f'sgm_{a.width}x{a.height}.json'), 'w'), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('sgm', 'pilots', 'dagger', 'eval', 'train', 'diagnose', 'video', 'export'):
        p = sub.add_parser(name)
        p.add_argument('--out', default='')
        p.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
        p.add_argument('--cam-hz', type=float, default=15.0)
        if name == 'sgm':
            p.add_argument('--n', type=int, default=5)
            p.add_argument('--width', type=int, default=320)
            p.add_argument('--height', type=int, default=240)
        if name in ('pilots', 'eval', 'diagnose'):
            p.add_argument('--world', default='forest', choices=['forest', 'gap'])
            p.add_argument('--v', default='3,5,7,10')
            p.add_argument('--n', type=int, default=10)
            p.add_argument('--seed0', type=int, default=1000)
            p.add_argument('--pilots', default='expert,gbl,blind' if name == 'pilots' else 'student')
        if name in ('eval', 'diagnose'):
            p.add_argument('--model', required=True)
        if name == 'export':
            p.add_argument('--model', required=True, help='.keras student → .tflite next to it (Gazebo, hardware)')
        if name == 'video':
            p.add_argument('--from-gz', default='', help='a Gazebo flight (results/gz_runs/agile/*.json) instead of our simulator')
            p.add_argument('--model', default='', help='student model (not needed for --pilot expert|gbl)')
            p.add_argument('--world', default='forest', choices=['forest', 'gap'])
            p.add_argument('--v', type=float, default=3.0)
            p.add_argument('--seed', type=int, default=1000)
            p.add_argument('--pilot', default='student', choices=['student', 'expert', 'gbl', 'blind'])
            p.add_argument('--no-chase', dest='chase', action='store_false', help='skip the chase camera (faster)')
            p.add_argument('--proj', default='scaled', choices=['scaled', 'raw'])
            p.add_argument('--tag', default='')
        if name in ('pilots', 'eval', 'diagnose', 'video'):
            if name != 'video':
                p.add_argument('--proj', default='scaled', choices=['scaled', 'raw'],
                               help='projection of the predicted points onto the flown polynomial (raw: ≤ 0.16.1)')
                p.add_argument('--tag', default='')
            p.add_argument('--tscale', default='chord', choices=['chord', 'arc'],
                           help='time scale from the straight distance (chord, ≤ 0.17.1) or the path length (arc)')
            p.add_argument('--vcap', type=float, default=0.0, help='peak speed of a plan ≤ vcap · v (0: no cap)')
            p.add_argument('--select', default='snap', choices=['snap', 'consistent'],
                           help='choice among the hypotheses (consistent: keep the side unless the cost says so)')
        if name in ('dagger', 'train'):
            p.add_argument('--kind', default='train', choices=['train', 'gap'])
            p.add_argument('--envs', type=int, default=850)
            p.add_argument('--per-round', type=int, default=30)
            p.add_argument('--epochs', type=int, default=3)
            p.add_argument('--batch', type=int, default=8)
            p.add_argument('--v-min', type=float, default=3.0)
            p.add_argument('--v-max', type=float, default=10.0)
            p.add_argument('--init', default='')
            p.add_argument('--imagenet', action='store_true', help='pre-trained MobileNet-V3 (downloads weights)')
            p.add_argument('--net-size', default='120x160', help='H x W of the depth input (paper: 480x640)')
            p.add_argument('--lr', type=float, default=1e-3)
            p.add_argument('--lr-end', type=float, default=None, help='cosine decay of the learning rate down to this')
            p.add_argument('--best', action='store_true', help='keep the epoch with the lowest validation R-WTA')
            p.add_argument('--val-by-env', action='store_true', help='validation on whole environments')
        if name == 'train':
            p.add_argument('--data', default='', help='comma-separated shard directories (default data_<kind>)')
            p.add_argument('--name', default='', help='output model name (default student_<kind>.keras)')
    a = ap.parse_args()
    if a.cmd == 'sgm':
        return cmd_sgm(a)
    if a.cmd == 'pilots':
        return _sweep(a, a.pilots.split(','))
    if a.cmd == 'eval':
        return _sweep(a, a.pilots.split(','), model=os.path.abspath(a.model))
    if a.cmd == 'export':
        from nanouav.agile.policy import TFLitePolicy, export_tflite, load_policy
        out = export_tflite(a.model)
        rng = np.random.default_rng(0)                     # check: same output as the Keras model
        d, v = rng.uniform(0, 10, (120, 160)).astype(np.float32), rng.normal(0, 3, 3)
        A, B = load_policy(a.model)(d, v, np.eye(3).ravel(), [1, 0, 0]), TFLitePolicy(out)(d, v, np.eye(3).ravel(), [1, 0, 0])
        err = max(float(np.abs(A[0] - B[0]).max()), float(np.abs(A[1] - B[1]).max()))
        print(f'→ {out}  (max difference with the Keras model: {err:.2e})')
        return
    if a.cmd == 'video' and a.from_gz:
        from nanouav.agile import video
        return video.from_gz(a.from_gz, _out(a), chase=a.chase)
    if a.cmd == 'video':
        from nanouav.agile import video
        o, suf = opts_of(a)
        return video.run(a.model, _out(a), world=a.world, v=a.v, seed=a.seed, pilot=a.pilot, cam_hz=a.cam_hz,
                         opts=o, chase=a.chase, suffix=suf + (f'_{a.tag}' if a.tag else ''))
    if a.cmd == 'diagnose':
        from nanouav.agile import diagnose
        o, suf = opts_of(a)
        return diagnose.run(a.model, _out(a), world=a.world, speeds=[float(x) for x in a.v.split(',')], n=a.n,
                            seed0=a.seed0, workers=a.workers, cam_hz=a.cam_hz, opts=o,
                            tag=a.tag or f"{os.path.splitext(os.path.basename(a.model))[0]}_{o['proj']}{suf.replace('_raw', '')}")
    net = tuple(int(x) for x in a.net_size.lower().split('x'))
    if a.cmd == 'train':
        from nanouav.agile.policy import train
        out = _out(a)
        data = [os.path.join(out, d) if not os.path.isabs(d) else d for d in a.data.split(',')] if a.data \
            else os.path.join(out, 'data_' + a.kind)
        name = a.name or f'student_{a.kind}.keras'
        train(data, os.path.join(out, name if name.endswith('.keras') else name + '.keras'), epochs=a.epochs,
              batch=a.batch, init_from=a.init or None, net_size=net, weights='imagenet' if a.imagenet else None,
              lr=a.lr, lr_end=a.lr_end, keep_best=a.best, val_by_env=a.val_by_env)
        return
    from nanouav.agile import dagger
    m = dagger.run(_out(a), n_envs=a.envs, per_round=a.per_round, workers=a.workers, epochs=a.epochs, batch=a.batch,
                   v_min=a.v_min, v_max=a.v_max, cam_hz=a.cam_hz, net_size=net,
                   weights='imagenet' if a.imagenet else None, kind=a.kind, init_model=a.init or None,
                   train_kw=dict(lr=a.lr, lr_end=a.lr_end, keep_best=a.best, val_by_env=a.val_by_env))
    print('student →', m)


if __name__ == '__main__':
    main()
