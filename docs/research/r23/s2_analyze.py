"""Per-state open loop for all variants under their test ω, plus label safety (priv, line, ctg)."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s2_train_d import load, open_loop   # noqa: E402
from s2_train import Policy, prep, state  # noqa: E402,F401
from common import D  # noqa: E402
if __name__ == '__main__':
    import tensorflow as tf
    from nanouav.agile import envs as EV
    from nanouav.agile import trajectory as TR
    from nanouav.agile.rollout import fit_plan, straight_ref
    D_ = load()
    envs = np.unique(D_['env'])
    val_envs = np.random.default_rng(1).choice(envs, max(4, len(envs) // 5), replace=False)
    va = np.nonzero(np.isin(D_['env'], val_envs))[0]
    test_omega = {'A': 'omega', 'B': 'omega', 'C': 'omega', 'D': 'omega_goal', 'E': 'omega', 'F': 'omega_goal', 'G': 'omega'}
    res, cache = {}, {}
    for k, om in test_omega.items():
        m = tf.keras.models.load_model(os.path.join(D, f's2_model_{k}.keras'), compile=False)
        res[k] = open_loop(D_, m, va, om, cache)[:, 0]
    tt = np.linspace(0, 1, 21)
    lab = {lk: [] for lk in ('priv', 'line', 'ctg', 'ctgl')}
    dl = []
    for i in va:
        e = int(D_['env'][i])
        F = cache[e]
        p, yaw, vh = D_['p'][i].astype(float), float(D_['yaw'][i]), D_['v'][i].astype(float)
        c_, s_ = np.cos(yaw), np.sin(yaw)
        v = np.array([c_ * vh[0] - s_ * vh[1], s_ * vh[0] + c_ * vh[1], vh[2]])
        for lk in lab:
            W = TR.from_heading_frame(D_[lk][i], p, yaw)
            f = fit_plan(W[0], p, v, np.zeros(3), max(np.linalg.norm(v), 3.0))
            lab[lk].append(float(F(TR.poly_eval(f[0], tt)).min()))
        if ('L', e) not in cache:
            cache[('L', e)] = straight_ref(EV.training_env(e, base_seed=4242), 5.0)
        dl.append(float(cache[('L', e)].closest(p)[1]))
    dl = np.array(dl)
    print(f'{"distance to line":18s} {"n":>4s} | labels: {"priv":>5s} {"line":>5s} {"ctg":>5s} | chosen free: ' + ' '.join(f'{k:>5s}' for k in res))
    for lo, hi in ((0, .5), (.5, 1.5), (1.5, 3), (3, 99), (0, 99)):
        s = (dl >= lo) & (dl < hi)
        print(f'{lo:g}-{hi:g} m{"":12s} {s.sum():4d} |         ' + ' '.join(f'{np.mean(np.array(lab[lk])[s] >= 0.2):5.3f}' for lk in lab)
              + ' |              ' + ' '.join(f'{np.mean(res[k][s] >= 0.2):5.3f}' for k in res))
    json.dump({'d_line': dl.tolist(), **{k: v.tolist() for k, v in res.items()}, **{f'lab_{k}': v for k, v in lab.items()}},
              open(os.path.join(D, 's2_perstate_all.json'), 'w'))
