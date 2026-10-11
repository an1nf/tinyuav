import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from nanouav.agile import envs as EV
from nanouav.agile.rollout import Config, Rollout
from s2_train import Policy
name, v, seed = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
from common import D  # noqa: E402
pol = Policy(tf.keras.models.load_model(os.path.join(D, f's2_model_{name}.keras'), compile=False))
env = EV.TEST['forest'](seed)
R = Rollout(env, Config(v_des=v, seed=seed, omega_ref='goal' if name in ('D', 'F') else 'line'), 'student', pol)
R.trace = []
r, _, log = R.run()
print(json.dumps({k: r[k] for k in ('success', 'crashed', 'time_s')}), 'start', env.start, 'goal', env.goal)
L = np.array([row[:7] for row in log])
for t in np.arange(0, L[-1, 0] + 1e-9, 1.0):
    i = int(np.argmin(abs(L[:, 0] - t)))
    tr = min(R.trace, key=lambda d: abs(d['t'] - t)) if R.trace else None
    print(f't={L[i,0]:5.1f}  p=({L[i,1]:6.2f},{L[i,2]:6.2f},{L[i,3]:5.2f})  |v|={np.linalg.norm(L[i,4:7]):4.2f}',
          (f"clear={tr['clear']} k={tr['k']} lat={tr['lat']} beta={tr['beta']}" if tr else ''))
print('end p', L[-1, 1:4])
