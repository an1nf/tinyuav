"""Angle between the student's ω (line, 1 s ahead) and where each label set's best trajectory is at 1 s."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s2_train_d import load  # noqa: E402
D_ = load()
from nanouav.agile import envs as EV  # noqa: E402
from nanouav.agile.rollout import straight_ref  # noqa: E402
lines = {}
dl = []
for e, p in zip(D_['env'], D_['p']):
    if e not in lines:
        lines[e] = straight_ref(EV.training_env(int(e), base_seed=4242), 5.0)
    dl.append(lines[e].closest(p.astype(float))[1])
dl = np.array(dl)


def ang(a, b):
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.degrees(np.arccos(np.clip((a * b).sum(1), -1, 1)))


om = D_['omega']
keys = [k for k in ('priv', 'line', 'ctg', 'ctgl') if k in D_]
A = {k: ang(D_[k][:, 0, -1, :], om) for k in keys}
print(f'{"distance to line":18s} {"n":>5s}  ' + '  '.join(f'{k:>12s}' for k in keys) + '   (median / p90 angle to ω_line, deg)')
for lo, hi in ((0, .5), (.5, 1.5), (1.5, 3), (3, 99), (0, 99)):
    s = (dl >= lo) & (dl < hi)
    print(f'{lo:g}-{hi:g} m{"":12s} {s.sum():5d}  ' + '  '.join(f'{np.median(A[k][s]):5.1f} /{np.percentile(A[k][s], 90):5.1f}' for k in keys))
