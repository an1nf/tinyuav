"""Why does the student crash? (`nanouav run r18 diagnose`)

Re-runs evaluation episodes with a trace of every student decision (the M hypotheses, their predicted
costs, the chosen one and the *true* clearance of each executed polynomial, from the privileged map) and
classifies each crash by the decisions of its last 0.5 s:

* `todas_chocan` — every hypothesis collides: the network did not see / did not propose a way out
  (perception, imitation);
* `eligio_mal` — a free hypothesis existed but the selection (predicted cost, then snap) took one that
  collides: the cost head;
* `plan_justo` — the chosen plan was free but passed within 15 cm of the limit: too tight for the
  tracking error;
* `seguimiento` — the chosen plan had margin, the drone still hit: tracking, latency, re-planning
  (control).

Also counts how often the chosen hypothesis switches side (left/right of the course) in the last 2 s
(indecision between two modes), and whether the obstacle that was hit was inside the camera's field of
view (±45°) in the last 0.5 s (`fuera_vista`: a tree beside the drone while passing it; the policy has
no memory). One top-down PNG per crash in results/r18/diag_<model>_<proj>/.
"""
import json
import multiprocessing as mp
import os
import time

import numpy as np

R_DRONE = 0.2


def _job(job):
    from nanouav.agile import dagger, envs as EV
    from nanouav.agile.rollout import Config, Rollout
    dagger._worker_init()
    env = EV.TEST[job['world']](job['seed'])
    R = Rollout(env, Config(v_des=job['v'], cam_hz=job['cam_hz'], seed=job['seed']), 'student',
                dagger._policy(job['model']))
    R.PROJ = job['proj']
    R.trace = []
    r, _, log = R.run()
    r.update(seed=job['seed'], world=job['world'], v=job['v'])
    out = {'result': r, 'diag': None}
    if r['crashed']:
        out['diag'] = classify(env, R.trace, log)
        if job.get('png_dir'):
            plot(env, R.trace, log, out['diag'], os.path.join(
                job['png_dir'], f"{job['world']}_v{job['v']:g}_s{job['seed']}.png"))
    out['beta_med'] = float(np.median([d['beta'] for d in R.trace])) if R.trace else None
    return out


def _nearest(world, p):
    best, o_best = np.inf, None
    for o in world.objects:
        d = o.sdf(p)
        if d < best:
            best, o_best = d, o
    if p[2] - 0.0 < best:
        return float(p[2]), 'suelo', None, None
    c = np.array([o_best.cx, o_best.cy]) if hasattr(o_best, 'cx') else np.asarray(getattr(o_best, 'center', p), float)[:2]
    return float(best), type(o_best).__name__, float(getattr(o_best, 'r', 0.0) or 0.0), c


def _bearing(d, c):
    """Angle (deg) of the point c (x, y) from the camera axis at decision d (needs the yaw in the trace)."""
    if c is None or 'att' not in d:
        return None
    ang = np.degrees(np.arctan2(c[1] - d['p'][1], c[0] - d['p'][0]) - d['att'][0])
    return float((ang + 180) % 360 - 180)


def classify(env, trace, log):
    tc = log[-1][0]
    p_c = np.array(log[-1][1:4])
    last = [d for d in trace if tc - 0.5 <= d['t'] <= tc]
    cls = 'seguimiento'
    if any(d['clear'][d['k']] < R_DRONE + 0.15 for d in last):
        cls = 'plan_justo'
    for d in last:
        if d['clear'][d['k']] < R_DRONE:
            cls = 'eligio_mal' if max(d['clear']) >= R_DRONE + 0.1 else 'todas_chocan'
            if cls == 'todas_chocan':
                break
    recent = [d for d in trace if tc - 2.0 <= d['t'] <= tc]
    sides = [np.sign(d['lat'][d['k']]) for d in recent if abs(d['lat'][d['k']]) > 0.3]
    flips = int(np.sum(np.diff(sides) != 0)) if len(sides) > 1 else 0
    dist, kind, radius, c = _nearest(env.world, p_c)
    speed = float(np.linalg.norm(log[-1][4:7]))
    # was the obstacle it hit inside the camera's view (±45°) in the last 0.5 s? A tree beside the drone
    # is outside the image while passing it, and the policy has no memory of it
    bear = [_bearing(d, c) for d in last]
    bear = [b for b in bear if b is not None]
    seen = float(np.mean([abs(b) < 45.0 for b in bear])) if bear else None
    return {'class': cls, 'flips_2s': flips, 't': tc, 'p': np.round(p_c, 2).tolist(), 'speed': round(speed, 2),
            'obstacle': kind, 'obstacle_r': radius, 'n_decisions_0.5s': len(last),
            'bearing_deg': round(bear[-1], 1) if bear else None, 'in_view_0.5s': seen,
            'sel_clear_0.5s': [d['clear'][d['k']] for d in last], 'best_clear_0.5s': [max(d['clear']) for d in last]}


def plot(env, trace, log, diag, fn):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    p_c = np.array(diag['p'])
    fig, ax = plt.subplots(figsize=(6, 6))
    for o in env.world.objects:
        if hasattr(o, 'cx') and np.hypot(o.cx - p_c[0], o.cy - p_c[1]) < 9:
            ax.add_patch(Circle((o.cx, o.cy), o.r, color='0.4'))
            ax.add_patch(Circle((o.cx, o.cy), o.r + R_DRONE, fill=False, ls=':', color='0.6'))
    P = np.array([row[1:4] for row in log if row[0] >= diag['t'] - 4])
    ax.plot(P[:, 0], P[:, 1], 'k-', lw=1.5, label='vuelo (4 s)')
    for j, d in enumerate([d for d in trace if d['t'] >= diag['t'] - 0.6]):
        for k, Wk in enumerate(d['W']):
            Wk = np.array(Wk)
            sel = k == d['k']
            free = d['clear'][k] >= R_DRONE
            ax.plot(Wk[:, 0], Wk[:, 1], '-' if sel else ':', lw=2 if sel else 1,
                    color=('tab:green' if free else 'tab:red'), alpha=0.4 + 0.6 * sel)
    ax.plot(*p_c[:2], 'rx', ms=12, mew=2)
    ax.set_xlim(p_c[0] - 7, p_c[0] + 7)
    ax.set_ylim(p_c[1] - 7, p_c[1] + 7)
    ax.set_aspect('equal')
    ax.set_title(f"{os.path.basename(fn)[:-4]} · {diag['class']} · {diag['speed']:.1f} m/s · cambios de lado "
                 f"{diag['flips_2s']} · obstáculo a {diag.get('bearing_deg')}° del eje",
                 fontsize=9)
    ax.text(0.01, 0.01, 'hipótesis: verde libre / rojo choca, gruesa = elegida (últimos 0.6 s)',
            transform=ax.transAxes, fontsize=7)
    fig.savefig(fn, dpi=110, bbox_inches='tight')
    plt.close(fig)


def run(model, out, world='forest', speeds=(3.0,), n=10, seed0=1000, workers=4, cam_hz=15.0, proj='scaled',
        tag='', png=True):
    tag = tag or f"{os.path.splitext(os.path.basename(model))[0]}_{proj}"
    d = os.path.join(out, f'diag_{tag}')
    os.makedirs(d, exist_ok=True)
    jobs = [dict(world=world, v=float(v), seed=s, cam_hz=cam_hz, model=os.path.abspath(model), proj=proj,
                 png_dir=d if png else None) for v in speeds for s in range(seed0, seed0 + n)]
    for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(k, '1')
    t0 = time.time()
    with mp.get_context('spawn').Pool(workers) as pool:
        res = pool.map(_job, jobs, chunksize=1)
    json.dump(res, open(os.path.join(d, 'diag.json'), 'w'), indent=1)
    print(f'{"v":>5s} {"éxito":>6s} {"choques":>7s} {"todas_chocan":>12s} {"eligio_mal":>10s} {"plan_justo":>10s} '
          f'{"seguimiento":>11s} {"fuera_vista":>11s} {"cambios":>7s} {"β med":>6s}')
    for v in speeds:
        L = [x for x in res if x['result']['v'] == v]
        D = [x['diag'] for x in L if x['diag']]
        c = {k: sum(x['class'] == k for x in D) for k in ('todas_chocan', 'eligio_mal', 'plan_justo', 'seguimiento')}
        fl = np.median([x['flips_2s'] for x in D]) if D else float('nan')
        unseen = sum(1 for x in D if x.get('in_view_0.5s') is not None and x['in_view_0.5s'] < 0.5)
        beta = np.median([x['beta_med'] for x in L if x['beta_med'] is not None])
        print(f'{v:5.1f} {sum(x["result"]["success"] for x in L):3d}/{len(L):<2d} {len(D):7d} {c["todas_chocan"]:12d} '
              f'{c["eligio_mal"]:10d} {c["plan_justo"]:10d} {c["seguimiento"]:11d} {unseen:11d} {fl:7.1f} {beta:6.2f}')
    print(f'→ {d}  ({time.time() - t0:.0f} s): diag.json and one PNG per crash')
    return res
