"""Stage 1: how noisy / how privileged are the M-H expert labels the R18 student is trained on?

For states visited along τ_gbl in training environments:
  L_priv(s)  = Expert(full world, seed s).label(p, τ_gbl ahead, v)          (the training labels), s = 0..S-1
  L_obs      = Expert(visible trees only).label(p, straight line ahead, v)  (only what the student can know)
Metrics: R-WTA (policy.rwta formula, ε = 0.05, M = K = 3) between label sets.
"""
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import D as OUT  # noqa: E402


def rwta_np(te, tn, eps=0.05):
    d = ((te[:, None] - tn[None]) ** 2).sum(axis=(-1, -2))          # (K, M)
    M = tn.shape[0]
    one = np.eye(M)[d.argmin(1)]
    a = (1 - eps) * one + eps / (M - 1) * (1 - one)
    return float((a * d).sum())


def visible_world(world, p, yaw, hfov=math.radians(90), rng_max=10.0):
    """Trees the camera can see: inside the field of view, within the SGM range, not fully hidden by nearer
    trunks (2D angular occlusion; trunks are tall)."""
    from nanouav.scene.renderer import Box, Cylinder, Sphere, World
    cand = []
    for o in world.objects:
        if isinstance(o, Cylinder):
            cx, cy, r = o.cx, o.cy, o.r
        elif isinstance(o, Box):
            cx, cy, r = o.center[0], o.center[1], math.hypot(o.half[0], o.half[1])
        elif isinstance(o, Sphere):
            cx, cy, r = o.center[0], o.center[1], o.r
        else:
            continue
        dx, dy = cx - p[0], cy - p[1]
        d = math.hypot(dx, dy)
        if d - r > rng_max or d < 1e-3:
            continue
        th = math.atan2(dy, dx) - yaw
        th = (th + math.pi) % (2 * math.pi) - math.pi
        a = math.asin(min(r / d, 1.0))
        if abs(th) - a > hfov / 2:
            continue
        cand.append((d, th - a, th + a, o))
    cand.sort(key=lambda c: c[0])
    covered, keep = [], []
    for d, lo, hi, o in cand:
        segs = [(lo, hi)]
        for clo, chi in covered:                   # subtract covered intervals
            new = []
            for s0, s1 in segs:
                if chi <= s0 or clo >= s1:
                    new.append((s0, s1))
                else:
                    if clo > s0:
                        new.append((s0, clo))
                    if chi < s1:
                        new.append((chi, s1))
            segs = new
        if sum(s1 - s0 for s0, s1 in segs) > 1e-3:
            keep.append(o)
        covered.append((lo, hi))
    return World(keep, world.ground)


def collect(job):
    """States along τ_gbl in one training environment (no rendering: a dummy observer)."""
    from nanouav.agile import envs as EV
    from nanouav.agile.rollout import Config, Rollout
    i, v = job
    env = EV.training_env(i, base_seed=777)
    cfg = Config(v_des=v, seed=i, label_every=6)
    R = Rollout(env, cfg, 'gbl', label=True)
    H, W = cfg.rig.size
    R.observer = lambda p, yaw, pitch, roll: (np.zeros((H, W), np.uint8), np.zeros((H, W), np.float32),
                                              np.zeros((H, W), np.float32))
    R.defer = []
    r, _, _ = R.run()
    return [{'env': i, 'v': v, 'p': d['_p'], 'ref': d['_ref'], 'vel': d['_v'], 'yaw': d['_yaw']} for d in R.defer]


def label_state(st):
    from nanouav.agile import envs as EV
    from nanouav.agile.expert import Expert
    from nanouav.agile.rollout import straight_ref
    env = EV.training_env(st['env'], base_seed=777)
    p, ref, vel, yaw = st['p'], st['ref'], st['vel'], st['yaw']
    out = {'priv': [], 'n_free': []}
    for s in range(4):
        L = Expert(env.world, seed=1000 + s).label(p, ref, v0=vel)
        out['priv'].append(L['trajs'])
        out['n_free'].append(L['n_free'])
    line = straight_ref(env, st['v'])
    s0, _ = line.closest(p)
    ref_line = line.ahead(s0)
    wv = visible_world(env.world, p, yaw)
    out['obs'] = Expert(wv, seed=1000).label(p, ref_line, v0=vel)['trajs']
    out['obs_gblref'] = Expert(wv, seed=1000).label(p, ref, v0=vel)['trajs']
    out['priv_lineref'] = Expert(env.world, seed=1000).label(p, ref_line, v0=vel)['trajs']
    out['n_vis'] = len(wv.objects)
    out['n_all'] = len(env.world.objects)
    return out


if __name__ == '__main__':
    for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[k] = '1'
    n_env = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    t0 = time.time()
    jobs = [(i, v) for i in range(n_env) for v in (3.0, 7.0)]
    with mp.get_context('spawn').Pool(2) as pool:
        states = sum(pool.map(collect, jobs), [])
    print(f'{len(states)} states in {time.time() - t0:.0f} s', flush=True)
    with mp.get_context('spawn').Pool(2) as pool:
        labs = pool.map(label_state, states, chunksize=2)
    print(f'labelled in {time.time() - t0:.0f} s', flush=True)
    np.save(os.path.join(OUT, 's1_states.npy'), np.array(states, dtype=object), allow_pickle=True)
    np.save(os.path.join(OUT, 's1_labels.npy'), np.array(labs, dtype=object), allow_pickle=True)
