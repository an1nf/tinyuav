"""Stage 2, labels with the reference replanned from the current position (`expert.CostToGo`).
  python s2_label.py d   → s2d/: cost-to-go to the goal point (variants D, E), and ω to the goal
  python s2_label.py g   → s2g/: the same with the line penalty W_LINE (variant G)
Saves <dir>/<same name as s2/>.npz with 'ctg' labels, 'omega_goal', 'ok'."""
import glob
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import D  # noqa: E402
VARIANT = {'d': ('s2d', 0.0), 'g': ('s2g', None)}   # None: expert.W_LINE


def work(fn):
    from nanouav.agile import envs as EV
    from nanouav.agile import trajectory as TR
    from nanouav.agile.expert import Expert
    out = os.path.join(D, VARIANT[os.environ.get('R23_LABELS', 'g')][0], os.path.basename(fn))
    if os.path.exists(out):
        return out
    z = np.load(fn)
    e, v_des = int(z['env']), float(z['v_des'])
    env = EV.training_env(e, base_seed=4242)
    from nanouav.agile.expert import W_LINE, CostToGo
    sub, w = VARIANT[os.environ.get('R23_LABELS', 'g')]
    ctg = CostToGo(env, w_line=W_LINE if w is None else w)
    labs, oms, oks = [], [], []
    for j in range(len(z['ok'])):
        p, yaw, vh = z['p'][j].astype(float), float(z['yaw'][j]), z['v'][j].astype(float)
        c_, s_ = np.cos(yaw), np.sin(yaw)
        v = np.array([c_ * vh[0] - s_ * vh[1], s_ * vh[0] + c_ * vh[1], vh[2]])
        g = np.asarray(env.goal, float) - p
        oms.append(TR.rotate_to_heading(g / max(np.linalg.norm(g), 1e-6), yaw).astype(np.float32))
        gr = ctg.ref_from(p, v_des)
        L = Expert(env.world, seed=e * 1000 + j + 17).label(p, gr.ahead(0.0), v0=v)['trajs'] if gr is not None else []
        ok = len(L) == 3
        labs.append(TR.to_heading_frame(L, p, yaw).astype(np.float32) if ok else np.zeros((3, 10, 3), np.float32))
        oks.append(ok)
    np.savez_compressed(out, ctg=np.array(labs), omega_goal=np.array(oms), ok=np.array(oks))
    return out


if __name__ == '__main__':
    for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[k] = '1'
    kind = sys.argv[1] if len(sys.argv) > 1 else 'g'
    os.environ['R23_LABELS'] = kind                     # inherited by the spawned workers
    os.makedirs(os.path.join(D, VARIANT[kind][0]), exist_ok=True)
    files = sorted(glob.glob(os.path.join(D, 's2', 'ep_*.npz')))
    t0 = time.time()
    with mp.get_context('spawn').Pool(2) as pool:
        for m, _ in enumerate(pool.imap_unordered(work, files)):
            if m % 10 == 0:
                print(f'{m + 1}/{len(files)} · {time.time() - t0:.0f} s', flush=True)
    print('done', time.time() - t0, flush=True)
