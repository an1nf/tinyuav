"""
R10 · Obstáculos dinámicos (Shang et al., NeurIPS 2025: evasión de objetos rápidos con ERF/IRF).

Nuestro planificador supone mundo estático (grilla con memoria). Escenarios, dron a v = 1 m/s
por la recta y = 0, esfera r = 0.12 m (balón) a la altura de vuelo:
  cross : cruza de lado a lado; se lanza para pasar por (x = 4, y = 0) cuando llegaría el dron
  head  : viene de frente (−x) con desfase lateral 0.1 m; velocidad de cierre = v + v_obs
Métricas: colisión con la esfera, éxito (llegar a x = 8), holgura mínima a la esfera.
Uso: python r10_dynamic.py --scen cross --vobs 0.5,1,2,3 --n 5 [--rf]
"""
import os, sys, json, time, math, argparse
import numpy as np
from nanouav import paths
ROOT = str(paths.repo_root())
from nanouav.scene.renderer import World, Sphere, Texture, random_ground, random_lighting
from nanouav.control.vehicle import PROFILES
from nanouav.sim.closed_loop import run_episode, C
from nanouav.sim.missions import fast_kw, set_brake_law

X_MEET, V_DRONE = 4.0, 1.0
AGGR = False
VARIANT = 'M'


def run_dyn(P, scen, v_obs, seed, rf=False):
    rng = np.random.default_rng(seed)
    ball = Sphere((50.0, 50.0, 0.65), 0.12, Texture('uniform', albedo=rng.uniform(0.3, 0.9)))
    w = World([ball], random_ground(rng), **random_lighting(rng))
    kw, lim = fast_kw(V_DRONE)
    st = {'launched': False, 't0': None, 'p0': None, 'vel': None, 'dmin': np.inf}
    side = 1 if rng.random() < 0.5 else -1

    def on_frame(t, s, pl):
        set_brake_law(pl)
        pl.CONSISTENT = True
        pl.FOV_OK = math.radians(28)
        pl.RF_DYN = rf
        pl.EVADE_AGGR = AGGR
        pl.set_segment((-1.5, 0.0), (20.0, 0.0), arrive=False, v_cruise=V_DRONE)

    def on_step(t, sim):
        if not st['launched'] and sim.p[0] > 1.0:
            t_meet = (X_MEET - sim.p[0]) / V_DRONE
            if scen == 'cross':
                p0 = np.array([X_MEET, -side * v_obs * t_meet, 0.65])
                vel = np.array([0.0, side * v_obs, 0.0])
            else:
                p0 = np.array([X_MEET + v_obs * t_meet, 0.1 * side, 0.65])
                vel = np.array([-v_obs, 0.0, 0.0])
            st.update(launched=True, t0=t, p0=p0, vel=vel)
        if st['launched']:
            c = st['p0'] + st['vel'] * (t - st['t0'])
            ball.center = tuple(c)
            st['dmin'] = min(st['dmin'], float(np.linalg.norm(sim.p - c) - ball.r))

    s, R, _, _ = run_episode(w, P, seed=seed, start=(-1.5, 0.0, 0.3), t_max=30, on_frame=on_frame,
                             on_step=on_step, done_fn=lambda t, p: p[0] > 8.0, planner_kw=kw, limits=lim,
                             variant=VARIANT)
    s['d_ball_min'] = st['dmin']
    s['hit_ball'] = bool(st['dmin'] < P.radius)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--profile', default='nano')
    ap.add_argument('--scen', default='cross')
    ap.add_argument('--vobs', default='0.5,1,2,3')
    ap.add_argument('--n', type=int, default=5)
    ap.add_argument('--rf', action='store_true', help='capa dinámica inspirada en ERF/IRF (R10b)')
    ap.add_argument('--aggr', action='store_true', help='perfil de referencia agresivo durante EVADE (R10c)')
    ap.add_argument('--variant', default='M')
    a = ap.parse_args()
    global AGGR, VARIANT
    AGGR, VARIANT = a.aggr, a.variant
    P = PROFILES[a.profile]
    out, t0 = {}, time.time()
    for v in [float(x) for x in a.vobs.split(',')]:
        L = []
        for i in range(a.n):
            s = run_dyn(P, a.scen, v, 600 + i, a.rf)
            L.append({k: (round(float(s[k]), 3) if isinstance(s[k], (float, np.floating)) else s[k])
                      for k in ('success', 'collided', 'hit_ball', 'timeout', 'time_s', 'd_ball_min', 'min_clearance_m')})
            print(f'[{a.scen} vobs={v} rf={a.rf} aggr={a.aggr}] {i}: {L[-1]} ({time.time() - t0:.0f}s)', flush=True)
        out[str(v)] = L
    json.dump(out, open(os.path.join(str(paths.results_dir()), f'r10_dyn_{P.name}_{a.scen}{"_rf" if a.rf else ""}{"_aggr" if a.aggr else ""}_{a.variant}.json'), 'w'), indent=1)
    for v, L in out.items():
        print(f"v_obs={v}: ok {sum(r['success'] for r in L)}/{len(L)} golpes {sum(r['hit_ball'] for r in L)} "
              f"timeouts {sum(r['timeout'] for r in L)} d_min_med {np.median([r['d_ball_min'] for r in L]):.2f}")


if __name__ == '__main__':
    main()
