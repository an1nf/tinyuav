"""Stage 2, variants with the replanned (cost-to-go) reference. Same states, depth, split and network as A/B/C.
  D  labels ctg (reference replanned from p to the goal) + ω to the goal      (test: ω to the goal)
  E  labels ctg                                          + ω to the line      (test: ω to the line)
  F  labels priv (τ_gbl, R18)                            + ω to the goal      (test: ω to the goal)
  G  labels ctg with the line penalty (s2g)              + ω to the line      (test: ω to the line)
  A2 = A on the same subset (all states have every label set, so A2 = A)
Usage: python s2_train_d.py D,E,F,G,A2"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s2_train import train, prep, state, Policy  # noqa: E402

from common import D  # noqa: E402


def load():
    keys = ('depth', 'v', 'R', 'omega', 'omega_gbl', 'z', 'esdf', 'priv', 'line', 'p', 'yaw', 'ok')
    G = os.path.isdir(os.path.join(D, 's2g')) and len(glob.glob(os.path.join(D, 's2g', '*.npz'))) >= 120
    out = {k: [] for k in keys + ('ctg', 'omega_goal', 'ok_d', 'env', 'pilot') + (('ctgl', 'ok_g') if G else ())}
    for fn in sorted(glob.glob(os.path.join(D, 's2', 'ep_*.npz'))):
        z = np.load(fn)
        if len(z['ok']) == 0:
            continue
        zd = np.load(os.path.join(D, 's2d', os.path.basename(fn)))
        for k in keys:
            out[k].append(z[k])
        out['ctg'].append(zd['ctg'])
        out['omega_goal'].append(zd['omega_goal'])
        out['ok_d'].append(zd['ok'])
        if G:
            zg = np.load(os.path.join(D, 's2g', os.path.basename(fn)))
            out['ctgl'].append(zg['ctg'])
            out['ok_g'].append(zg['ok'])
        out['env'].append(np.full(len(z['ok']), int(z['env'])))
        out['pilot'].append(np.full(len(z['ok']), str(z['pilot'])))
    out = {k: np.concatenate(v) for k, v in out.items()}
    m = out['ok'].astype(bool) & out['ok_d'].astype(bool) & (out['ok_g'].astype(bool) if G else True)
    return {k: v[m] for k, v in out.items()}


def open_loop(D_, model, va, omega_key, envs_cache):
    from nanouav.agile import envs as EV
    from nanouav.agile import trajectory as TR
    from nanouav.agile.geometry import Field
    from nanouav.agile.rollout import choose, fit_plan
    pol = Policy(model)
    X = prep(D_['depth'])
    out = []
    tt = np.linspace(0, 1, 21)
    for i in va:
        e = int(D_['env'][i])
        if e not in envs_cache:
            envs_cache[e] = Field(EV.training_env(e, base_seed=4242).world, ground=True)
        F = envs_cache[e]
        tn, c = pol.f(X[i:i + 1], state(D_, omega_key, [i]))
        tn, c = tn.numpy()[0], c.numpy()[0]
        p, yaw, vh = D_['p'][i].astype(float), float(D_['yaw'][i]), D_['v'][i].astype(float)
        c_, s_ = np.cos(yaw), np.sin(yaw)
        v = np.array([c_ * vh[0] - s_ * vh[1], s_ * vh[0] + c_ * vh[1], vh[2]])
        W = TR.from_heading_frame(tn, p, yaw)
        fits = [fit_plan(W[k], p, v, np.zeros(3), max(np.linalg.norm(v), 3.0)) for k in range(len(W))]
        k = choose(fits, c, W)
        cl = [float(F(TR.poly_eval(f[0], tt)).min()) for f in fits]
        out.append((cl[k], max(cl)))
    out = np.array(out)
    return out


if __name__ == '__main__':
    D_ = load()
    envs = np.unique(D_['env'])
    rng = np.random.default_rng(1)
    val_envs = rng.choice(envs, max(4, len(envs) // 5), replace=False)
    va = np.nonzero(np.isin(D_['env'], val_envs))[0]
    tr = np.nonzero(~np.isin(D_['env'], val_envs))[0]
    print(f'{len(D_["ok"])} samples (ok in both label sets), train {len(tr)}, val {len(va)}', flush=True)
    res, cache = {}, {}
    which = sys.argv[1].split(',') if len(sys.argv) > 1 else ['D', 'E', 'F']
    spec = {'D': ('ctg', 'omega_goal', 'omega_goal'), 'E': ('ctg', 'omega', 'omega'),
            'F': ('priv', 'omega_goal', 'omega_goal'), 'A2': ('priv', 'omega', 'omega'),
            'G': ('ctgl', 'omega', 'omega')}
    for name in which:
        lk, ok_train, ok_test = spec[name]
        model, hist = train(D_, lk, ok_train, tr, va)
        model.save(os.path.join(D, f's2_model_{name}.keras'))
        ol = open_loop(D_, model, va, ok_test, cache)
        res[name] = {'labels': lk, 'omega_train': ok_train, 'omega_test': ok_test, 'rwta_train': hist[-1][0],
                     'rwta_val': hist[-1][1], 'rwta_val_min': min(h[1] for h in hist),
                     'open_loop_chosen_free': float(np.mean(ol[:, 0] >= 0.2)),
                     'open_loop_any_free': float(np.mean(ol[:, 1] >= 0.2))}
        np.save(os.path.join(D, f's2_openloop_{name}.npy'), ol)
        print(name, json.dumps(res[name]), flush=True)
    prev = json.load(open(os.path.join(D, 's2_results_d.json'))) if os.path.exists(os.path.join(D, 's2_results_d.json')) else {}
    prev.update(res)
    json.dump(prev, open(os.path.join(D, 's2_results_d.json'), 'w'), indent=1)
