"""
G1 · Replica of the Gazebo world (gazebo/worlds/nano_uav_world.sdf) in our own simulator.
Reproduces the deadlock observed in Gazebo (X3 stuck at x≈0.4 in front of the 3-box 'wall')
and compares planners: 'lane' (consistent EvasionPlanner) vs 'mh' (MHPlanner, Loquercio 2021).
Usage: nanouav run r12 --planner lane,mh --goals 4.5:0,4:1.5 --n 5 [--profile x3]
"""
import os, sys, json, time, math, argparse
import numpy as np
from nanouav import paths, settings
ROOT = str(paths.repo_root())
from nanouav.scene.renderer import World, Box, Cylinder, Plane, Texture, random_lighting
from nanouav.control.vehicle import PROFILES
from nanouav.sim.closed_loop import run_episode, C
from nanouav.sim.missions import fast_kw, set_brake_law
from nanouav.sim.closed_loop import random_corridor_world


def gazebo_world(rng=None, jitter=0.0):
    j = (lambda: rng.uniform(-jitter, jitter)) if (rng is not None and jitter) else (lambda: 0.0)
    obs = [Box((1.8 + j(), 0.0 + j(), 0.3), (0.2, 0.2, 0.3), tex=Texture('uniform', albedo=0.55)),   # orange
           Box((1.2 + j(), -1.0 + j(), 0.25), (0.175, 0.175, 0.25), tex=Texture('uniform', albedo=0.45)),  # green
           Box((1.0 + j(), 1.1 + j(), 0.4), (0.15, 0.15, 0.4), tex=Texture('uniform', albedo=0.35)),    # blue
           Cylinder(3.0 + j(), 0.4 + j(), 0.18, 0.0, 0.7, Texture('uniform', albedo=0.7))]              # yellow
    ground = Plane(Texture('checker', albedo=0.75, albedo2=0.3, scale=1.0, extent=3.5))
    lk = random_lighting(rng) if rng is not None else {}
    return World(obs, ground, **lk)


def scenario_world(spec, seed):
    """'scn:bosque' (seeded with seed) or 'json:gazebo/worlds/gen/bosque_s3.json' → (World, scenario, balls)."""
    from nanouav.sim import scenarios as SC
    from nanouav.scene.renderer import Sphere
    if spec.startswith('json:'):
        s = json.load(open(spec[5:]))
    else:
        name, *kw = spec[4:].split(';')
        s = SC.make(name, seed, **{k: float(v) for k, v in (x.split('=') for x in kw)})
    rng = np.random.default_rng(seed)
    w = SC.to_world(s, random_lighting(rng))
    balls = []
    for d in s['dynamic']:
        b = Sphere(tuple(d['p0']), d['r'], Texture('uniform', albedo=d.get('tex', {}).get('albedo', 0.8)))
        w.objects.append(b)
        balls.append((b, d))
    return w, s, balls


def parse_cfg(text):
    """'INSPECT_TOF=1,M_ABS=0.02' → {'INSPECT_TOF': 1.0, 'M_ABS': 0.02} (planner attributes)."""
    return {k.strip(): float(v) for k, v in (x.split('=') for x in text.split(',') if x)}


def run(P, planner, goal, seed, variant='M', v=0.5, jitter=0.0, world='gz', cfg=None):
    rng = np.random.default_rng(seed)
    scn, balls = None, []
    if world.startswith(('scn:', 'json:')):
        w, scn, balls = scenario_world(world, seed)
        goal = np.asarray(scn['B'], float)
    else:
        w = gazebo_world(rng, jitter) if world == 'gz' else random_corridor_world(rng)
    corridor = world == 'corridor'
    kw, lim = fast_kw(v)
    A = np.array(scn['A'] if scn else [-1.5, 0.0])
    cls = None
    if planner == 'mh':
        from nanouav.planning.mh_planner import MHPlanner
        cls = MHPlanner
    elif planner == 'mh1':                     # R12 version (before narrow passages), for comparison
        from nanouav.planning.legacy.mh_planner_r12 import MHPlannerV1
        cls = MHPlannerV1
    first = {'done': False}

    def on_frame(t, st, pl):
        if first['done']:
            return
        first['done'] = True
        set_brake_law(pl)
        pl.CONSISTENT, pl.FOV_OK = True, math.radians(28)
        pl.set_segment(A, goal, arrive=not corridor, v_cruise=v)
        for k_, v_ in (cfg or {}).items():
            if not hasattr(pl, k_):
                raise AttributeError(f'planner has no attribute {k_!r} (--cfg)')
            setattr(pl, k_, type(getattr(pl, k_))(v_) if isinstance(getattr(pl, k_), (bool, int, float)) else v_)

    def done(t, p):
        if corridor:
            return p[0] > goal[0] - 0.5
        return np.hypot(p[0] - goal[0], p[1] - goal[1]) < 0.3 and abs(math.hypot(*[0, 0])) < 1

    def on_step(t, sim):                       # balls: back-and-forth (triangle wave) along their axis
        for b, d in balls:
            per = 2 * d['amp'] / d['v']
            ph = (t % per) / per
            sgn = 2 * ph if ph < 0.5 else 2 * (1 - ph)
            b.center = (d['p0'][0] + d['axis'][0] * d['amp'] * sgn, d['p0'][1] + d['axis'][1] * d['amp'] * sgn, d['p0'][2])

    s, R, _, _ = run_episode(w, P, seed=seed, start=(A[0], A[1], 0.3), t_max=scn['t_max'] if scn else 60,
                             variant=variant, on_frame=on_frame, done_fn=done, planner_kw=kw, limits=lim,
                             planner_cls=cls, start_yaw=scn['yaw0'] if scn else 0.0,
                             on_step=on_step if balls else None)
    xy = R[:, [C['x'], C['y']]]
    s['d_goal_end'] = float(np.hypot(*(xy[-1] - goal)))
    s['path_len'] = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
    s['traj'] = xy[::50].round(2).tolist()
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--profile', default='x3')
    ap.add_argument('--planner', default='lane,mh')
    ap.add_argument('--goals', default='4.5:0,4:1.5')
    ap.add_argument('--n', type=int, default=3)
    ap.add_argument('--jitter', type=float, default=0.0)
    ap.add_argument('--variant', default='M')
    ap.add_argument('--v', type=float, default=0.5)
    ap.add_argument('--cfg', default='', help='planner attributes, e.g. "INSPECT_TOF=1" (R17)')
    ap.add_argument('--world', default='gz', help='gz | corridor | scn:bosque[;gap=1.0] | json:path.json (goal = scenario B)')
    a = ap.parse_args()
    P = PROFILES[a.profile]
    goals = [np.array([float(u) for u in g.split(':')]) for g in a.goals.split(',')]
    out, t0 = {}, time.time()
    mode = 'gt' if settings.flag('FAKE_CNN') else 'cnn'
    for pl in a.planner.split(','):
        for g in goals:
            key = f'{pl}_{g[0]:g}_{g[1]:g}'
            L = []
            for i in range(a.n):
                s = run(P, pl, g, 900 + i, a.variant, a.v, a.jitter, a.world, parse_cfg(a.cfg))
                L.append({k: s[k] for k in ('success', 'collided', 'timeout', 'time_s', 'min_clearance_m',
                                            'd_goal_end', 'path_len', 'traj')})
                r = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in L[-1].items() if k != 'traj'}
                print(f'[{mode} {key}] {i}: {r} ({time.time() - t0:.0f}s)', flush=True)
            out[key] = L
            print(f'== {key}: ok {sum(r["success"] for r in L)}/{len(L)} col {sum(r["collided"] for r in L)}', flush=True)
    os.makedirs(str(paths.results_dir()), exist_ok=True)
    ftag = '_fuse2' if settings.flag('FUSE2') else ''        # R14 fusion: keep both results side by side
    ftag += ('_' + a.cfg.replace('=', '').replace(',', '_')) if a.cfg else ''
    json.dump(out, open(os.path.join(str(paths.results_dir()), f'g1_{a.world.replace(":", "-").replace("/", "_").replace(";", "_")}_{a.planner.replace(",", "+")}_{P.name}_{mode}_{a.variant}_j{a.jitter:g}{ftag}.json'), 'w'))


if __name__ == '__main__':
    main()
