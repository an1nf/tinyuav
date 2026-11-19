"""Follow a balloon through a forest.

  tinyuav follow train --init NAV.keras --v 1,7 --name follow.keras     # follow policy, differentiable sim
  tinyuav follow eval --model follow.keras --density 0.0625             # same simulator, no perception
  tinyuav follow sim --pilots gbl,direct,diffphys --model follow.keras --density 0.0625 --speed 7 --edge --fast-target
  tinyuav follow video --model follow.keras --speed 7 --fast-target --view sgm
"""
import argparse
import json
import multiprocessing as mp
import os
import time

import numpy as np

from tinyuav import paths


def _out(a):
    d = a.out or os.path.join(str(paths.results_dir()), 'follow')
    os.makedirs(d, exist_ok=True)
    return d


def _env(job):
    from tinyuav.follow.balloon import balloon_env
    return balloon_env(job['seed'], density=job.get('density'), n_distractors=job.get('distractors'),
                       speed=job.get('speed'), color=job.get('color', 'red'), p_same=job.get('p_same', 0.25),
                       z_lo=job.get('z_lo', 1.2), z_hi=job.get('z_hi', 2.6))


def fly(job, trace=False):
    """One balloon flight in our simulator: (result, rollout, log)."""
    from tinyuav.follow.mission import BalloonMission
    from tinyuav.nav import dagger
    from tinyuav.nav.rollout import Config, Rollout
    env = _env(job)
    M = BalloonMission(env, standoff=job['standoff'], seed=job['seed'], opts=job.get('opts'))
    pol = dagger._policy(job['model']) if job['pilot'] == 'student' else \
        (job['model'] if job['pilot'] == 'diffphys' else None)
    R = Rollout(env, Config(v_des=2.0, cam_hz=job['cam_hz'], seed=job['seed']), job['pilot'], pol, mission=M)
    R.EDGE = bool(job.get('edge'))
    if trace:
        R.trace, R.trace_depth, R.record_cam = [], True, True
        R.dp_trace = [] if job['pilot'] == 'diffphys' else None
    r, _, log = R.run()
    M.label_trace(R.trace)
    return r, R, log


def _sim_job(job):
    from tinyuav.nav import dagger
    dagger._worker_init()
    try:
        r = fly(job)[0]
    except RuntimeError as e:
        r = {'success': False, 'crashed': False, 'failed': False, 'error': str(e)}
    r.update(seed=job['seed'])
    return r


def add_lost_opts(p):
    p.add_argument('--fast-target', action='store_true', help='--q-acc 5 --yaw-ext 0.5 --gate-v 1 --relock-free --turn-pred')
    p.add_argument('--q-acc', type=float, default=None, help='filter process noise (default 2 m/s^2)')
    p.add_argument('--yaw-ext', type=float, default=None, help='while lost, look at the estimate extrapolated <= s')
    p.add_argument('--gate-v', type=float, default=None, help='while lost, re-lock gate grows at |v_target| + this')
    p.add_argument('--relock-free', action='store_true', help='first measurement after a re-lock skips the gate')
    p.add_argument('--turn-pred', action='store_true', help='while lost, extrapolate the last turn rate')


def lost_opts(a):
    """Mission options from --fast-target and its parts."""
    from tinyuav.follow.mission import FAST_TARGET
    o = dict(FAST_TARGET) if getattr(a, 'fast_target', False) else {}
    for k in ('q_acc', 'yaw_ext', 'gate_v'):
        if getattr(a, k, None) is not None:
            o[k] = getattr(a, k)
    for k in ('relock_free', 'turn_pred'):
        if getattr(a, k, False):
            o[k] = True
    return o


def print_table(table, pilots):
    print(f'{"pilot":8s} {"success":>8s} {"crash":>5s} {"lost":>4s} {"locked":>6s} {"true":>5s} {"d_err":>5s} {"relock":>6s}')
    for p in pilots:
        L = table[p]
        f = lambda k: np.nanmedian([r.get(k) if r.get(k) is not None else np.nan for r in L])
        print(f'{p:8s} {sum(r.get("success", 0) for r in L):3d}/{len(L):<4d} {sum(r.get("crashed", 0) for r in L):5d} '
              f'{sum(r.get("failed", 0) for r in L):4d} {f("locked_frac"):6.2f} {f("true_frac"):5.2f} '
              f'{f("dist_err_med"):5.2f} {f("relocks"):6.0f}')


def _job(a, pilot, seed):
    return dict(pilot=pilot, seed=seed, cam_hz=a.cam_hz, standoff=a.standoff, color=a.color, density=a.density,
                speed=a.speed, model=os.path.abspath(a.model) if a.model else None, edge=a.edge, opts=lost_opts(a),
                distractors=getattr(a, 'distractors', None), p_same=getattr(a, 'p_same', 0.25),
                z_lo=a.z[0] if hasattr(a, 'z') else 1.2, z_hi=a.z[1] if hasattr(a, 'z') else 2.6)


def cmd_sim(a):
    pilots = a.pilots.split(',')
    jobs = [_job(a, p, s) for p in pilots for s in range(a.seed0, a.seed0 + a.n)]
    for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(k, '1')
    t0 = time.time()
    with mp.get_context('spawn').Pool(a.workers) as pool:
        res = pool.map(_sim_job, jobs, chunksize=1)
    table = {}
    for j, r in zip(jobs, res):
        table.setdefault(j['pilot'], []).append(r)
    fn = os.path.join(_out(a), f'balloon_{a.tag or "sim"}.json')
    json.dump(table, open(fn, 'w'), indent=1)
    print_table(table, pilots)
    print(f'{fn}  ({time.time() - t0:.0f} s)')


def cmd_video(a):
    from tinyuav.nav import dagger
    from tinyuav.nav.video import render_episode
    dagger._worker_init()
    job = _job(a, a.pilot, a.seed)
    r, R, log = fly(job, trace=True)
    env = R.env
    name = (os.path.splitext(os.path.basename(a.model))[0] if a.model else a.pilot) + (f'_{a.tag}' if a.tag else '')
    d = os.path.join(_out(a), 'video')
    os.makedirs(d, exist_ok=True)
    v = env.meta['speed']
    fn = os.path.join(d, f'balloon_{name}_d{a.density:g}_v{v:g}_s{a.seed}_{a.view}.mp4')
    res = 'EXITO' if r['success'] else ('CHOQUE' if r['crashed'] else 'GLOBO PERDIDO')
    render_episode(log, R.trace, R.cfg.rig, fn, fps=a.cam_hz, result=f"{res} ({r['time_s']:.1f} s)", view=a.view,
                   decisions=R.dp_trace)
    if R.dp_trace is not None:
        with open(fn[:-4] + '.json', 'w') as fh:
            json.dump({'result': r, 'decisions': R.dp_trace}, fh, default=float)
    print(f"{res} · {r['time_s']} s · d_err {r.get('dist_err_med')} m · {fn}")


def cmd_train(a):
    from tinyuav.follow import diffphys as FD
    FD.W_VIS = a.w_vis
    FD.BURN = tuple(float(x) for x in a.burn.split(','))
    print(FD.train(_out(a), a.init, iters=a.iters, batch=a.batch, lr=a.lr, v_range=tuple(float(x) for x in a.v.split(',')),
                   seed=a.seed, name=a.name, log=lambda s: print(s, flush=True)))


def cmd_eval(a):
    from tinyuav.follow import diffphys as FD
    from tinyuav.nav import diffphys as DP
    FD.evaluate(a.model, B=a.n, v_range=tuple(float(x) for x in a.v.split(',')), T=int(a.seconds / DP.DT),
                density=a.density)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('train', help='fine-tune a navigation policy to follow a moving target')
    p.add_argument('--init', required=True, help='navigation model to start from')
    p.add_argument('--name', default='diffphys_follow.keras')
    p.add_argument('--iters', type=int, default=1000)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--w-vis', type=float, default=1.0, help='weight of the line-of-sight barrier')
    p.add_argument('--v', default='1,4', help='target speed range (m/s)')
    p.add_argument('--burn', default='0,30', help='s of flight without gradient before each episode')
    p.add_argument('--seed', type=int, default=0)
    p = sub.add_parser('eval', help='crashes, visibility and distance error in the differentiable simulator')
    p.add_argument('--model', required=True)
    p.add_argument('--n', type=int, default=128)
    p.add_argument('--v', default='1,4')
    p.add_argument('--seconds', type=float, default=60.0)
    p.add_argument('--density', type=float, default=None, help='trees per m^2 (default 1/49-1/16 at random)')
    for name in ('sim', 'video'):
        p = sub.add_parser(name, help='balloon mission with perception' if name == 'sim' else 'video of one flight')
        p.add_argument('--model', default='')
        p.add_argument('--standoff', type=float, default=4.0)
        p.add_argument('--color', default='red', help='red | orange | yellow | green | blue | magenta')
        p.add_argument('--cam-hz', type=float, default=10.0)
        p.add_argument('--tag', default='')
        add_lost_opts(p)
        if name == 'sim':
            p.add_argument('--pilots', default='gbl,direct')
            p.add_argument('--n', type=int, default=10)
            p.add_argument('--seed0', type=int, default=3000)
            p.add_argument('--density', type=float, default=None)
            p.add_argument('--distractors', type=int, default=None)
            p.add_argument('--speed', type=float, default=None)
            p.add_argument('--p-same', type=float, default=0.25, help='chance that a distractor has the target color')
            p.add_argument('--z', type=lambda x: tuple(float(v) for v in x.split(',')), default=(1.2, 2.6),
                           help='balloon height range (m)')
            p.add_argument('--edge', action='store_true', help='diffphys: drop SGM pixels on depth edges')
            p.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
        else:
            p.add_argument('--pilot', default='diffphys', help='diffphys | student | direct | gbl')
            p.add_argument('--seed', type=int, default=3000)
            p.add_argument('--speed', type=float, default=7.0)
            p.add_argument('--density', type=float, default=1 / 16)
            p.add_argument('--no-edge', dest='edge', action='store_false', help='diffphys without the edge filter')
            p.add_argument('--view', default='perception', choices=['perception', 'sgm'])
    for q in sub.choices.values():
        q.add_argument('--results', dest='out', default='', help='results directory (default results/follow)')
    a = ap.parse_args(argv)
    return {'train': cmd_train, 'eval': cmd_eval, 'sim': cmd_sim, 'video': cmd_video}[a.cmd](a)


if __name__ == '__main__':
    main()
