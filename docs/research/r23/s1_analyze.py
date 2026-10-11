import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_labels import rwta_np  # noqa: E402

from common import D  # noqa: E402
S = np.load(os.path.join(D, 's1_states.npy'), allow_pickle=True)
L = np.load(os.path.join(D, 's1_labels.npy'), allow_pickle=True)


def ok(x):
    return x is not None and len(x) == 3


rows = []
for st, lb in zip(S, L):
    P = lb['priv']
    if not all(ok(x) for x in P) or not ok(lb['obs']) or not ok(lb['priv_lineref']) or not ok(lb['obs_gblref']):
        continue
    # lateral offset of the gbl reference from the straight line, 1 s ahead (heading frame)
    p, ref, yaw = st['p'], st['ref'], st['yaw']
    rows.append({
        'v': st['v'],
        'seed': np.mean([rwta_np(P[0], P[j]) for j in (1, 2, 3)]),
        'ref': rwta_np(P[0], lb['priv_lineref']),          # same full map, line reference instead of τ_gbl
        'map': rwta_np(P[0], lb['obs_gblref']),            # same τ_gbl reference, visible obstacles only
        'both': rwta_np(P[0], lb['obs']),                  # what the student can know: visible + line
        'gbl_off': float(np.abs(ref[-1, 1] - p[1])),       # |y| of τ_ref 1 s ahead from the drone (rough)
        'n_vis': lb['n_vis'], 'n_all': lb['n_all']})

print(f'{len(rows)} states with complete label sets (of {len(S)})')
for v in (3.0, 7.0):
    R = [r for r in rows if r['v'] == v]
    f = lambda k: (np.median([r[k] for r in R]), np.mean([r[k] for r in R]))
    print(f'v={v:g} n={len(R)}')
    for k, name in (('seed', 'other M-H seed (sampling noise)'), ('ref', 'line ref instead of τ_gbl (full map)'),
                    ('map', 'visible obstacles only (τ_gbl ref)'), ('both', 'visible + line ref (student-consistent)')):
        md, mn = f(k)
        print(f'   R-WTA(priv, {name:42s}) median {md:7.2f}  mean {mn:7.2f}   ~RMS/point {np.sqrt(mn / 30):.2f} m')
print('for scale: R18 student R-WTA — train ~9–30, validation by environment ~44 (spatial head, best epoch)')
