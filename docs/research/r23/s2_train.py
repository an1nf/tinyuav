"""Stage 2: does a consistent direction input make the expert's labels learnable, and the student safer?

Variants (same states, same depth):
  A  labels priv (τ_gbl reference)  + ω from the straight line   ← R18 as implemented
  B  labels priv (τ_gbl reference)  + ω from τ_gbl                ← consistent (ω points where the labels go)
  C  labels line (straight-line ref) + ω from the straight line   ← consistent alternative
At test time only the straight line exists: every variant gets ω from the line.
A small CNN on 60×80 depth (CPU), the R18 losses (R-WTA + collision-cost head)."""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import D  # noqa: E402
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')


def load():
    keys = ('depth', 'v', 'R', 'omega', 'omega_gbl', 'z', 'esdf', 'priv', 'line', 'p', 'yaw', 'ok')
    out = {k: [] for k in keys}
    out['env'], out['pilot'] = [], []
    for fn in sorted(glob.glob(os.path.join(D, 's2', 'ep_*.npz'))):
        z = np.load(fn)
        if len(z['ok']) == 0:
            continue
        for k in keys:
            out[k].append(z[k])
        out['env'].append(np.full(len(z['ok']), int(z['env'])))
        out['pilot'].append(np.full(len(z['ok']), str(z['pilot'])))
    out = {k: np.concatenate(v) for k, v in out.items()}
    m = out['ok'].astype(bool)
    return {k: v[m] for k, v in out.items()}


def build(m=3):
    import tensorflow as tf
    L = tf.keras.layers
    d = L.Input((60, 80, 1))
    s = L.Input((15,))
    x = d
    for c, k in ((16, 5), (32, 3), (64, 3), (64, 3)):
        x = L.LeakyReLU(0.2)(L.Conv2D(c, k, strides=2, padding='same')(x))
    x = L.Dense(128)(L.Flatten()(x))
    x = L.LeakyReLU(0.2)(x)
    y = L.LeakyReLU(0.2)(L.Dense(64)(s))
    y = L.LeakyReLU(0.2)(L.Dense(64)(y))
    h = L.Concatenate()([x, y])
    h = L.LeakyReLU(0.2)(L.Dense(256)(h))
    o = L.Dense(m * 31)(h)
    o = L.Reshape((m, 31))(o)
    traj = L.Reshape((m, 10, 3))(o[..., :30] * 5.0)
    cost = L.Activation('softplus')(o[..., 30])
    return tf.keras.Model([d, s], [traj, cost])


def prep(depth_mm):
    d = depth_mm[:, ::2, ::2].astype(np.float32) / 1000.0
    return (np.clip(d / 10.0, 0, 1) * 2 - 1)[..., None]


def state(D_, omega_key, idx):
    return np.concatenate([D_['v'][idx] / 10.0, D_['R'][idx], D_[omega_key][idx]], 1).astype(np.float32)


def train(D_, label_key, omega_key, tr, va, epochs=40, batch=32, seed=0, log=print):
    import tensorflow as tf
    from nanouav.agile.policy import rwta, collision_target
    tf.keras.utils.set_random_seed(seed)
    model = build()
    steps = epochs * (len(tr) // batch)
    opt = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.CosineDecay(1e-3, steps, alpha=0.01))
    X = prep(D_['depth'])

    def batch_of(ii):
        return (X[ii], state(D_, omega_key, ii), D_[label_key][ii], D_['esdf'][ii].astype(np.float32),
                D_['z'][ii].astype(np.float32))

    def losses(dep, st, te, esdf, z, training):
        tn, c = model([dep, st], training=training)
        l_w = rwta(te, tn)
        ct = tf.stop_gradient(collision_target(tn, esdf, z))
        l_c = tf.reduce_mean(tf.reduce_sum(tf.square(c - ct), -1))
        return 10 * l_w + 0.1 * l_c, l_w, l_c

    @tf.function
    def step(dep, st, te, esdf, z):
        with tf.GradientTape() as tape:
            loss, l_w, l_c = losses(dep, st, te, esdf, z, True)
        opt.apply_gradients(zip(tape.gradient(loss, model.trainable_variables), model.trainable_variables))
        return l_w

    rng = np.random.default_rng(seed)
    hist = []
    for ep in range(epochs):
        idx = rng.permutation(tr)
        lw = [float(step(*batch_of(idx[i:i + batch]))) for i in range(0, len(idx) - batch + 1, batch)]
        vw = [float(losses(*batch_of(va[i:i + 256]), False)[1]) for i in range(0, len(va), 256)]
        hist.append((float(np.mean(lw)), float(np.mean(vw))))
        if ep % 10 == 9 or ep == epochs - 1:
            log(f'  [{label_key}/{omega_key}] epoch {ep + 1}: R-WTA train {hist[-1][0]:.2f} val {hist[-1][1]:.2f}')
    return model, hist


class Policy:
    """Rollout policy: (depth 240×320 SGM m, v_h, R, ω_h) → hypotheses (heading frame), costs."""

    def __init__(self, model):
        import tensorflow as tf
        self.model = model
        self.f = tf.function(lambda d, s: model([d, s], training=False))

    def __call__(self, depth, v_h, R, omega_h):
        from nanouav.agile.rollout import downsample_depth
        d = downsample_depth(depth, (60, 80)) if depth.shape != (60, 80) else depth
        x = (np.clip(d.astype(np.float32) / 10.0, 0, 1) * 2 - 1)[None, ..., None]
        s = np.concatenate([np.asarray(v_h) / 10.0, np.asarray(R).ravel(), np.asarray(omega_h)])[None].astype(np.float32)
        tn, c = self.f(x, s)
        return tn.numpy()[0].astype(float), c.numpy()[0].astype(float)


def open_loop(D_, model, va, envs_cache):
    """Under test conditions (ω from the straight line): is the chosen hypothesis' executed plan free?"""
    from nanouav.agile import envs as EV
    from nanouav.agile import trajectory as TR
    from nanouav.agile.geometry import Field
    from nanouav.agile.rollout import choose, fit_plan
    pol = Policy(model)
    X = prep(D_['depth'])
    chosen_free, any_free = [], []
    for i in va:
        e = int(D_['env'][i])
        if e not in envs_cache:
            env = EV.training_env(e, base_seed=4242)
            envs_cache[e] = Field(env.world, ground=True)
        F = envs_cache[e]
        tn, c = pol.f(X[i:i + 1], state(D_, 'omega', [i]))
        tn, c = tn.numpy()[0], c.numpy()[0]
        p, yaw = D_['p'][i].astype(float), float(D_['yaw'][i])
        vh = D_['v'][i].astype(float)
        c_, s_ = np.cos(yaw), np.sin(yaw)
        v = np.array([c_ * vh[0] - s_ * vh[1], s_ * vh[0] + c_ * vh[1], vh[2]])
        W = TR.from_heading_frame(tn, p, yaw)
        speed = max(np.linalg.norm(v), 3.0)
        fits = [fit_plan(W[k], p, v, np.zeros(3), speed) for k in range(len(W))]
        k = choose(fits, c, W)
        tt = np.linspace(0, 1, 21)
        clear = [float(F(TR.poly_eval(f[0], tt)).min()) for f in fits]
        chosen_free.append(clear[k] >= 0.2)
        any_free.append(max(clear) >= 0.2)
    return float(np.mean(chosen_free)), float(np.mean(any_free))


if __name__ == '__main__':
    D_ = load()
    envs = np.unique(D_['env'])
    rng = np.random.default_rng(1)
    val_envs = rng.choice(envs, max(4, len(envs) // 5), replace=False)
    va = np.nonzero(np.isin(D_['env'], val_envs))[0]
    tr = np.nonzero(~np.isin(D_['env'], val_envs))[0]
    print(f'{len(D_["ok"])} samples, {len(envs)} envs; train {len(tr)}, val {len(va)} ({len(val_envs)} envs)')
    res, cache = {}, {}
    for name, lk, ok in (('A', 'priv', 'omega'), ('B', 'priv', 'omega_gbl'), ('C', 'line', 'omega')):
        model, hist = train(D_, lk, ok, tr, va)
        model.save(os.path.join(D, f's2_model_{name}.keras'))
        cf, af = open_loop(D_, model, va, cache)
        res[name] = {'labels': lk, 'omega_train': ok, 'rwta_train': hist[-1][0], 'rwta_val': hist[-1][1],
                     'open_loop_chosen_free': cf, 'open_loop_any_free': af}
        print(name, json.dumps(res[name]), flush=True)
    json.dump(res, open(os.path.join(D, 's2_results.json'), 'w'), indent=1)
