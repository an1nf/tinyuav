"""Navigation in the simulated forests: baselines, DAgger student, evaluation and video.

  tinyuav nav pilots --v 3,5,7,10                      # expert, global reference, blind
  tinyuav nav dagger --envs 850                         # student: data collection and training
  tinyuav nav eval --pilots diffphys --model M.keras --v 3,5,7,10 --n 30 --edge
  tinyuav nav video --pilot diffphys --model M.keras --v 10 --edge --view sgm
  tinyuav nav compare A.json B.json                     # paired by seed (exact McNemar)
"""
import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

from tinyuav import paths

OPTS = {'proj': 'scaled', 'tscale': 'chord', 'vcap': 0.0, 'select': 'snap', 'lookahead': 0.0, 'keep': False,
        'fan': False, 'pool': 'stride', 'latency': 0.05, 'tilt': False, 'edge': False, 'slow': 0.0, 'carve': False,
        'vga': False, 'w_clear': 0.0, 'dp_gt': False, 'dp_q': 0.0, 'dp_persist': False, 'yaw_fwd': False,
        'dp_amax': 0.0}


def _out(a):
    d = a.out or os.path.join(str(paths.results_dir()), 'nav')
    os.makedirs(d, exist_ok=True)
    return d


def opts_of(a):
    """Rollout options from the arguments and the file-name suffix of the non-default ones."""
    o = {k: getattr(a, k, v) for k, v in OPTS.items()}
    suf = ''
    for k, v in o.items():
        if v == OPTS[k]:
            continue
        suf += f'_{k}' if isinstance(v, bool) else f'_{k}{v:g}' if isinstance(v, float) else f'_{v}'
    return o, suf


def apply_opts(R, o):
    R.PROJ, R.TSCALE, R.SELECT = o.get('proj', 'scaled'), o.get('tscale', 'chord'), o.get('select', 'snap')
    R.V_CAP = o.get('vcap') or None
    R.LOOKAHEAD = float(o.get('lookahead') or 0.0)
    R.KEEP, R.FAN, R.EDGE = bool(o.get('keep')), bool(o.get('fan')), bool(o.get('edge'))
    R.POOL = o.get('pool') or 'stride'
    R.SLOW = float(o.get('slow') or 0.0)
    R.CARVE = bool(o.get('carve'))
    R.W_CLEAR = float(o.get('w_clear') or 0.0)
    R.DP_GT, R.DP_Q = bool(o.get('dp_gt')), float(o.get('dp_q') or 0.0)
    R.DP_PERSIST, R.YAW_FWD = bool(o.get('dp_persist')), bool(o.get('yaw_fwd'))
    R.DP_AMAX = float(o.get('dp_amax') or 0.0)
    return R


def add_flight_opts(p, select='snap', keep=False):
    p.add_argument('--proj', default='scaled', choices=['scaled', 'raw'], help='student: projection of its points')
    p.add_argument('--tscale', default='chord', choices=['chord', 'arc'], help='student: time scale of a plan')
    p.add_argument('--vcap', type=float, default=0.0, help='student: peak plan speed <= vcap * v (0: off)')
    p.add_argument('--select', default=select,
                   choices=['snap', 'consistent', 'oracle', 'sgm', 'oracle_cost', 'sgm_cost'],
                   help='student: choice among hypotheses (sgm: least snap among those free in its own SGM)')
    p.add_argument('--lookahead', type=float, default=0.0, help='sgm/oracle: also check this many s past the plan')
    p.add_argument('--keep', action=argparse.BooleanOptionalAction, default=keep,
                   help='sgm/oracle/reactive: keep the current plan when no new one is free')
    p.add_argument('--fan', action='store_true', help='sgm/oracle: fall back to fan candidates')
    p.add_argument('--pool', default='stride', choices=['stride', 'min'], help='SGM sampling for the sgm choice')
    p.add_argument('--latency', type=float, default=0.05, help='s from capture to command')
    p.add_argument('--edge', action='store_true', help='drop SGM pixels on depth discontinuities')
    p.add_argument('--slow', type=float, default=0.0, help='reactive: retry at this fraction of v')
    p.add_argument('--w-clear', type=float, default=0.0, help='reactive: weight of the soft clearance cost (m^2)')
    p.add_argument('--carve', action='store_true', help='free-space evidence in the perceived obstacles')
    p.add_argument('--dp-gt', action='store_true', help='diffphys: true depth instead of SGM')
    p.add_argument('--dp-q', type=float, default=0.0, help='diffphys: block quantile of the depth input')
    p.add_argument('--dp-amax', type=float, default=0.0, help='diffphys: clip the horizontal acceleration (m/s^2)')
    p.add_argument('--dp-persist', action='store_true', help='diffphys: a near block must persist two frames')
    p.add_argument('--yaw-fwd', action='store_true', help='yaw follows the velocity only toward the goal')
    p.add_argument('--tilt', action='store_true', help='camera 30 deg up at >= 7 m/s')


def _eval_job(job):
    from tinyuav.nav import dagger
    from tinyuav.nav.rollout import Config, Rollout, straight_ref
    from tinyuav.nav.student import omega_of
    from tinyuav.sim import envs as EV
    dagger._worker_init()
    env = EV.TEST[job['world']](job['seed'])
    om = omega_of(job['model']) if job['pilot'] == 'student' and job.get('model') else 'line'
    kw = {}
    if job.get('tilt') or job.get('vga'):
        import dataclasses
        from tinyuav.sim.stereo import StereoRig
        rig = StereoRig(width=640, height=480, num_disp=96, texture_win=11) if job.get('vga') else StereoRig()
        kw['rig'] = dataclasses.replace(rig, tilt_up_deg=30.0 if (job.get('tilt') and job['v'] >= 7.0 - 1e-6) else 0.0)
    cfg = Config(v_des=job['v'], cam_hz=job['cam_hz'], seed=job['seed'], omega_ref=om, latency=job.get('latency') or 0.05,
                 **kw)
    pol = dagger._policy(job.get('model')) if job['pilot'] == 'student' else \
        (job.get('model') if job['pilot'] == 'diffphys' else None)
    try:
        R = Rollout(env, cfg, job['pilot'], pol)
        apply_opts(R, job)
        r, _, log = R.run()
    except RuntimeError as e:
        r, log = {'success': False, 'crashed': False, 'timeout': False, 'error': str(e)}, []
    r.update(seed=job['seed'], world=job['world'])
    if log:
        line = straight_ref(env, job['v'])
        r['lateral_m'] = round(float(np.mean([line.closest(np.array(row[1:4]))[1] for row in log])), 3)
    return r


def _sweep(a, pilots, model=None):
    out = _out(a)
    speeds = [float(x) for x in a.v.split(',')]
    seeds = list(range(a.seed0, a.seed0 + a.n))
    o, suf = opts_of(a)
    tag = (os.path.splitext(os.path.basename(model))[0] if model else 'pilots') + suf + (f'_{a.tag}' if a.tag else '')
    jobs = [dict(world=a.world, pilot=p, v=v, seed=s, cam_hz=a.cam_hz, model=model, **o)
            for p in pilots for v in speeds for s in seeds]
    t0 = time.time()
    for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(k, '1')
    res = []
    with mp.get_context('spawn').Pool(a.workers) as pool:
        for j, r in zip(jobs, pool.imap(_eval_job, jobs, chunksize=1)):
            res.append(r)
            st = 'ok' if r.get('success') else 'CRASH' if r.get('crashed') else 'error' if r.get('error') else 'fail'
            print(f"[{len(res)}/{len(jobs)} {time.time() - t0:.0f} s] {j['pilot']} v={j['v']:g} seed {j['seed']}: {st} "
                  f"{r.get('time_s', float('nan')):.1f} s" + (f"  {r['error']}" if r.get('error') else ''), flush=True)
    table = {}
    for j, r in zip(jobs, res):
        table.setdefault(f"{j['pilot']}_v{j['v']:g}", []).append(r)
    fn = os.path.join(out, f'eval_{a.world}_{tag}.json')
    old = json.load(open(fn)) if os.path.exists(fn) else {}
    old.update(table)
    json.dump(old, open(fn, 'w'), indent=1)
    print(f'{"pilot":10s} {"v":>5s} {"success":>8s} {"crash":>6s} {"t_med":>6s} {"v_mean":>6s} {"lat_m":>6s} {"dec_ms":>7s}')
    for p in pilots:
        for v in speeds:
            L = table[f'{p}_v{v:g}']
            ok = [r for r in L if r.get('success')]
            tm = np.median([r['time_s'] for r in ok]) if ok else float('nan')
            vm = np.median([r['mean_speed'] for r in ok]) if ok else float('nan')
            lat = np.mean([r['lateral_m'] for r in L if 'lateral_m' in r]) if any('lateral_m' in r for r in L) else float('nan')
            dm = [r['decide_ms'] for r in L if r.get('decide_ms') is not None]
            print(f'{p:10s} {v:5.1f} {len(ok):3d}/{len(L):<4d} {sum(r.get("crashed", 0) for r in L):6d} {tm:6.1f} {vm:6.1f} '
                  f'{lat:6.2f} {np.median(dm) if dm else float("nan"):7.1f}')
    print(f'{fn}  ({time.time() - t0:.0f} s)')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('pilots', 'eval', 'video', 'dagger', 'train'):
        p = sub.add_parser(name)
        p.add_argument('--out', default='')
        p.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
        p.add_argument('--cam-hz', type=float, default=15.0)
        if name in ('pilots', 'eval', 'video'):
            p.add_argument('--world', default='forest', choices=['forest', 'gap'])
            p.add_argument('--tag', default='')
            p.add_argument('--vga', action='store_true', help='640 x 480 stereo pair')
            add_flight_opts(p)
        if name in ('pilots', 'eval'):
            p.add_argument('--v', default='3,5,7,10')
            p.add_argument('--n', type=int, default=10)
            p.add_argument('--seed0', type=int, default=1000)
            p.add_argument('--pilots', default='expert,gbl,blind' if name == 'pilots' else 'student')
        if name == 'eval':
            p.add_argument('--model', required=True)
        if name == 'video':
            p.add_argument('--model', default='')
            p.add_argument('--v', type=float, default=3.0)
            p.add_argument('--seed', type=int, default=1000)
            p.add_argument('--pilot', default='student', choices=['student', 'expert', 'gbl', 'blind', 'reactive', 'diffphys'])
            p.add_argument('--view', default='perception', choices=['perception', 'sgm'])
        if name in ('dagger', 'train'):
            p.add_argument('--kind', default='train', choices=['train', 'gap'])
            p.add_argument('--envs', type=int, default=850)
            p.add_argument('--per-round', type=int, default=30)
            p.add_argument('--epochs', type=int, default=3)
            p.add_argument('--batch', type=int, default=8)
            p.add_argument('--v-min', type=float, default=3.0)
            p.add_argument('--v-max', type=float, default=10.0)
            p.add_argument('--init', default='')
            p.add_argument('--imagenet', action='store_true', help='ImageNet MobileNet-V3 weights')
            p.add_argument('--net-size', default='120x160', help='H x W of the depth input')
            p.add_argument('--lr', type=float, default=1e-3)
            p.add_argument('--lr-end', type=float, default=None, help='cosine decay to this rate')
            p.add_argument('--best', action='store_true', help='keep the epoch with the lowest validation loss')
            p.add_argument('--val-by-env', action='store_true', help='validate on whole environments')
            p.add_argument('--omega', default='line', choices=['line', 'gbl', 'goal'], help='direction input of the student')
            p.add_argument('--label-ref', default='gbl', choices=['gbl', 'ctg'], help='reference of the expert labels')
            p.add_argument('--head', default='gap', choices=['gap', 'spatial'], help='new models: feature head')
        if name == 'train':
            p.add_argument('--data', default='', help='comma-separated shard directories (default data_<kind>)')
            p.add_argument('--name', default='', help='model name (default student_<kind>.keras)')
    p = sub.add_parser('compare', help='eval JSONs on the same seeds, paired against the first (McNemar)')
    p.add_argument('files', nargs='+', help='eval_*.json; a.json+b.json joins runs')
    a = ap.parse_args(argv)
    if a.cmd == 'compare':
        from tinyuav.nav.stats import compare
        return compare(a.files)
    if a.cmd == 'pilots':
        return _sweep(a, a.pilots.split(','))
    if a.cmd == 'eval':
        return _sweep(a, a.pilots.split(','), model=os.path.abspath(a.model))
    if a.cmd == 'video':
        from tinyuav.nav import video
        o, suf = opts_of(a)
        return video.run(a.model, _out(a), world=a.world, v=a.v, seed=a.seed, pilot=a.pilot, cam_hz=a.cam_hz,
                         opts=o, suffix=suf + (f'_{a.tag}' if a.tag else ''), view=a.view)
    net = tuple(int(x) for x in a.net_size.lower().split('x'))
    if a.cmd == 'train':
        from tinyuav.nav.student import train
        out = _out(a)
        data = [d if os.path.isabs(d) else os.path.join(out, d) for d in a.data.split(',')] if a.data \
            else os.path.join(out, 'data_' + a.kind)
        name = a.name or f'student_{a.kind}.keras'
        train(data, os.path.join(out, name if name.endswith('.keras') else name + '.keras'), epochs=a.epochs,
              batch=a.batch, init_from=a.init or None, net_size=net, weights='imagenet' if a.imagenet else None,
              lr=a.lr, lr_end=a.lr_end, keep_best=a.best, val_by_env=a.val_by_env, head=a.head)
        return
    from tinyuav.nav import dagger
    m = dagger.run(_out(a), n_envs=a.envs, per_round=a.per_round, workers=a.workers, epochs=a.epochs, batch=a.batch,
                   v_min=a.v_min, v_max=a.v_max, cam_hz=a.cam_hz, net_size=net,
                   weights='imagenet' if a.imagenet else None, kind=a.kind, init_model=a.init or None,
                   train_kw=dict(lr=a.lr, lr_end=a.lr_end, keep_best=a.best, val_by_env=a.val_by_env, head=a.head),
                   omega_ref=a.omega, label_ref=a.label_ref)
    print('student:', m)


if __name__ == '__main__':
    main()
