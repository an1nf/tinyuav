"""Student policy of Loquercio et al. 2021 (Methods B) in TensorFlow/Keras.

Inputs: SGM depth image, platform velocity, attitude (rotation matrix, 9 values) and the desired flight
direction (unit vector toward the reference point 1 s ahead). Velocity, direction and output
trajectories are in the heading frame (x forward, y left, z up, rotated by the yaw); the attitude matrix
keeps roll and pitch.

Architecture (paper): pre-trained MobileNet-V3 on the depth image → 1D convolution → M feature vectors
of 32; velocity + attitude + direction → MLP [64, 32, 32, 32] with LeakyReLU → 1D convolution → one
32-vector per mode; concatenation, then per mode an MLP [64, 128, 128] with LeakyReLU → 10 positions
(1 s horizon) and a collision cost. M = 3.

Loss (Eq. 5–7): λ₁ · R-WTA(T_e, T_n) + λ₂ · Σ_k (c_k − C_collision(τ_n^k))², λ₁ = 10, λ₂ = 0.1, ε = 0.05,
Adam lr 1e-3, minibatch 8. The ground-truth collision cost of the *predicted* trajectories needs the
geometry during training; here every sample stores a local 2D distance map at flight altitude
(heading frame, 0.25 m) plus the altitude, and C_collision is evaluated on it (an approximation:
obstacles are treated as vertical within ±0.4 m of the flight altitude; the ground is exact).
"""
import glob
import json
import os

import numpy as np

M = 3
N_PTS = 10
D_SCALE = 10.0          # depth normalization (m)
V_SCALE = 10.0          # velocity normalization (m/s)
T_SCALE = 5.0           # output trajectory scale (m)
RQ = 0.2
DT = 0.1


def build_model(net_size=(120, 160), weights=None, m=M, head='gap'):
    """head: how the backbone's feature map (4 × 5 × 576 at 120 × 160) reaches the trajectory heads.
    'gap' (≤ 0.18.0): global average pooling — keeps *what* is in the image but discards *where* (left or
    right of the course: exactly what the choice of side around a tree needs). 'spatial': a 1 × 1
    convolution to 32 channels per cell, flattened, so the position of every obstacle is kept."""
    import tensorflow as tf
    L = tf.keras.layers
    H, W = net_size
    d_in = L.Input((H, W, 1), name='depth')
    s_in = L.Input((15,), name='state')
    x = L.Concatenate()([d_in, d_in, d_in])
    bb = tf.keras.applications.MobileNetV3Small(input_shape=(H, W, 3), include_top=False, weights=weights,
                                                include_preprocessing=False)
    if head == 'spatial':
        f = L.LeakyReLU(0.2)(L.Conv2D(32, 1)(bb(x)))
        f = L.Reshape((1, -1))(L.Flatten()(f))
    else:
        f = L.GlobalAveragePooling2D()(bb(x))
        f = L.Reshape((1, -1))(f)
    f = L.Conv1D(m * 32, 1)(f)
    f = L.Reshape((m, 32))(f)
    s = s_in
    for n in (64, 32, 32, 32):
        s = L.LeakyReLU(0.2)(L.Dense(n)(s))
    s = L.Reshape((1, 32))(s)
    s = L.Reshape((m, 32))(L.Conv1D(m * 32, 1)(s))
    h = L.Concatenate(-1)([f, s])
    for n in (64, 128, 128):
        h = L.LeakyReLU(0.2)(L.Dense(n)(h))
    out = L.Dense(N_PTS * 3 + 1)(h)
    traj = L.Reshape((m, N_PTS, 3), name='traj')(out[..., :N_PTS * 3] * T_SCALE)
    cost = L.Activation('softplus', name='cost')(out[..., N_PTS * 3])
    return tf.keras.Model([d_in, s_in], [traj, cost], name='agile_student')


def prep_depth(depth_m):
    """m (0 = invalid) → network input in [−1, 1] (invalid = −1)."""
    return (np.clip(np.asarray(depth_m, np.float32) / D_SCALE, 0, 1) * 2 - 1)[..., None]


def prep_state(v_h, R, omega_h):
    return np.concatenate([np.asarray(v_h, np.float32) / V_SCALE, np.asarray(R, np.float32).ravel(),
                           np.asarray(omega_h, np.float32)])


# ── losses ────────────────────────────────────────────────────────────────
def rwta(te, tn, eps=0.05):
    """te (B,K,10,3) expert labels, tn (B,M,10,3) predictions → mean over the batch of Eq. 5."""
    import tensorflow as tf
    d = tf.reduce_sum(tf.square(te[:, :, None] - tn[:, None]), axis=[-1, -2])        # (B, K, M)
    m = tf.shape(tn)[1]
    one = tf.one_hot(tf.argmin(d, axis=-1), m)
    alpha = (1 - eps) * one + eps / tf.cast(m - 1, tf.float32) * (1 - one)
    return tf.reduce_mean(tf.reduce_sum(tf.stop_gradient(alpha) * d, axis=[1, 2]))


def collision_target(tn, esdf, z, box=(-2.0, 14.0, -8.0, 8.0), res=0.25):
    """Ground-truth collision cost (Eq. 3, integrated with dt = 0.1) of predicted trajectories, from the
    stored local distance map. tn (B,M,10,3) heading frame, esdf (B,ny,nx), z (B,)."""
    import tensorflow as tf
    x0, _, y0, _ = box
    ny, nx = tf.shape(esdf)[1], tf.shape(esdf)[2]
    ix = tf.clip_by_value(tf.cast(tf.floor((tn[..., 0] - x0) / res), tf.int32), 0, nx - 1)
    iy = tf.clip_by_value(tf.cast(tf.floor((tn[..., 1] - y0) / res), tf.int32), 0, ny - 1)
    B = tf.shape(tn)[0]
    b = tf.broadcast_to(tf.range(B)[:, None, None], tf.shape(ix))
    d2 = tf.gather_nd(tf.cast(esdf, tf.float32), tf.stack([b, iy, ix], -1))
    d = tf.minimum(d2, z[:, None, None] + tn[..., 2])
    d = tf.maximum(d, 0.0)
    c = tf.where(d > 2 * RQ, 0.0, -d * d / (RQ * RQ) + 4.0)
    return tf.reduce_sum(c, -1) * DT


# ── data ──────────────────────────────────────────────────────────────────
def load_shards(dirs, max_samples=None):
    files = []
    for d in ([dirs] if isinstance(dirs, str) else dirs):
        files += sorted(glob.glob(os.path.join(d, '**', 'env_*.npz'), recursive=True))
    keys = ('depth', 'v', 'R', 'omega', 'trajs', 'costs', 'esdf', 'z')
    parts = {k: [] for k in keys + ('_env',)}
    n = 0
    for fi, f in enumerate(files):
        z = np.load(f)
        if len(z['v']) == 0:
            continue
        for k in keys:
            parts[k].append(z[k])
        parts['_env'].append(np.full(len(z['v']), fi, np.int32))   # which environment (for a split by env)
        n += len(z['v'])
        if max_samples and n >= max_samples:
            break
    if n == 0:
        raise FileNotFoundError(f'no samples in {dirs}')
    return {k: np.concatenate(v) for k, v in parts.items()}, len(files)


def train(data_dirs, out_path, epochs=3, batch=8, lr=1e-3, init_from=None, net_size=(120, 160),
          weights=None, val_frac=0.05, lam1=10.0, lam2=0.1, seed=0, lr_end=None, keep_best=False,
          val_by_env=False, head='gap', log=print):
    """lr_end: cosine decay of the learning rate from lr to lr_end over all the steps (None: constant).
    keep_best: save the weights of the epoch with the lowest validation R-WTA instead of the last ones.
    val_by_env: validation = whole environments (val_frac of the shards), not random samples of every
    environment (consecutive frames of one flight are nearly equal, so a sample split is optimistic)."""
    import tensorflow as tf
    D, n_files = load_shards(data_dirs)
    n = len(D['v'])
    rng = np.random.default_rng(seed)
    if val_by_env and len(np.unique(D['_env'])) >= 4:
        envs = rng.permutation(np.unique(D['_env']))
        v_envs = envs[:max(1, int(round(len(envs) * val_frac)))]
        is_val = np.isin(D['_env'], v_envs)
        va, tr = np.nonzero(is_val)[0], rng.permutation(np.nonzero(~is_val)[0])
    else:
        idx = rng.permutation(n)
        n_val = max(int(n * val_frac), min(n, 8))
        va, tr = idx[:n_val], idx[n_val:] if n > n_val else idx
    if init_from and os.path.exists(init_from):
        model = tf.keras.models.load_model(init_from, compile=False, safe_mode=False)
    else:
        model = build_model(net_size, weights, head=head)
    steps = max(epochs * (len(tr) // batch), 1)
    sched = tf.keras.optimizers.schedules.CosineDecay(lr, steps, alpha=lr_end / lr) if lr_end else lr
    opt = tf.keras.optimizers.Adam(sched)

    def batch_of(ii):
        dep = (np.clip(D['depth'][ii].astype(np.float32) / 1000.0 / D_SCALE, 0, 1) * 2 - 1)[..., None]
        st = np.concatenate([D['v'][ii] / V_SCALE, D['R'][ii], D['omega'][ii]], 1).astype(np.float32)
        return dep, st, D['trajs'][ii], D['esdf'][ii].astype(np.float32), D['z'][ii].astype(np.float32)

    def _losses(dep, st, te, esdf, z, training):
        tn, c = model([dep, st], training=training)
        l_wta = rwta(te, tn)
        ct = tf.stop_gradient(collision_target(tn, esdf, z))
        l_c = tf.reduce_mean(tf.reduce_sum(tf.square(c - ct), -1))
        return lam1 * l_wta + lam2 * l_c, l_wta, l_c

    @tf.function(jit_compile=False)
    def train_step(dep, st, te, esdf, z):
        with tf.GradientTape() as tape:
            loss, l_wta, l_c = _losses(dep, st, te, esdf, z, True)
        opt.apply_gradients(zip(tape.gradient(loss, model.trainable_variables), model.trainable_variables))
        return loss, l_wta, l_c

    @tf.function(jit_compile=False)
    def val_step(dep, st, te, esdf, z):
        return _losses(dep, st, te, esdf, z, False)

    hist, best = [], (np.inf, None, -1)
    for ep in range(epochs):
        rng.shuffle(tr)
        L = []
        for i in range(0, len(tr) - batch + 1, batch):
            L.append([float(x) for x in train_step(*batch_of(tr[i:i + batch]))])
        V = [[float(x) for x in val_step(*batch_of(va[i:i + batch]))]
             for i in range(0, len(va), batch)]
        h = {'epoch': ep, 'train': np.mean(L, 0).round(4).tolist() if L else None, 'val': np.mean(V, 0).round(4).tolist()}
        hist.append(h)
        log(f'[train] epoch {ep + 1}/{epochs} · {n} samples · loss/R-WTA/cost train {h["train"]} val {h["val"]}')
        if keep_best and h['val'][1] < best[0]:
            best = (h['val'][1], model.get_weights(), ep)
    if keep_best and best[1] is not None:
        model.set_weights(best[1])
        log(f'[train] keeping epoch {best[2] + 1} (val R-WTA {best[0]:.3f})')
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    model.save(out_path)
    json.dump({'samples': n, 'files': n_files, 'history': hist, 'kept_epoch': best[2] if keep_best else epochs - 1,
               'lr': lr, 'lr_end': lr_end, 'val_by_env': bool(val_by_env), 'n_val': int(len(va))},
              open(out_path + '.json', 'w'), indent=1)
    return model, hist


class StudentPolicy:
    """Callable used by the rollout: (depth m, v_h, R, omega_h) → (trajs (M,10,3) heading frame, costs (M,))."""

    def __init__(self, path):
        import tensorflow as tf
        self.model = tf.keras.models.load_model(path, compile=False, safe_mode=False)
        self._f = tf.function(lambda d, s: self.model([d, s], training=False), jit_compile=False)

    def __call__(self, depth, v_h, R, omega_h):
        d = prep_depth(depth)[None]
        s = prep_state(v_h, R, omega_h)[None]
        tn, c = self._f(d, s)
        return tn.numpy()[0].astype(float), c.numpy()[0].astype(float)


def export_tflite(model_path, out_path=None):
    """Keras student → float32 TFLite (for the Gazebo container, which has LiteRT but not TensorFlow, and
    for the hardware). → path of the .tflite."""
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
    # from_keras_model freezes the weights; a concrete function with the model as trackable object keeps
    # them as resource variables, which LiteRT reads as NaN
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    out_path = out_path or os.path.splitext(model_path)[0] + '.tflite'
    open(out_path, 'wb').write(conv.convert())
    return out_path


class TFLitePolicy:
    """Same call as `StudentPolicy`, on a .tflite (ai-edge-litert, or TensorFlow's interpreter)."""

    def __init__(self, path):
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:                       # pragma: no cover - depends on the installation
            from tensorflow.lite import Interpreter
        self.it = Interpreter(model_path=path, num_threads=2)
        self.it.allocate_tensors()
        ins = self.it.get_input_details()
        self.i_depth = next(d['index'] for d in ins if len(d['shape']) == 4)
        self.i_state = next(d['index'] for d in ins if len(d['shape']) == 2)
        outs = self.it.get_output_details()
        self.o_traj = next(d['index'] for d in outs if len(d['shape']) == 4)
        self.o_cost = next(d['index'] for d in outs if len(d['shape']) == 2)

    def __call__(self, depth, v_h, R, omega_h):
        self.it.set_tensor(self.i_depth, prep_depth(depth)[None].astype(np.float32))
        self.it.set_tensor(self.i_state, prep_state(v_h, R, omega_h)[None].astype(np.float32))
        self.it.invoke()
        return (self.it.get_tensor(self.o_traj)[0].astype(float), self.it.get_tensor(self.o_cost)[0].astype(float))


def write_meta(model_path, **kw):
    """Sidecar `<model>.json`: how the model was trained (R23: what its direction input ω points to)."""
    with open(str(model_path) + '.json', 'w') as f:
        json.dump(kw, f)


def omega_of(model_path):
    """The ω convention a model was trained with ('line' for models without a sidecar: R18–R22)."""
    try:
        with open(str(model_path) + '.json') as f:
            return json.load(f).get('omega_ref', 'line')
    except (OSError, ValueError, TypeError):
        return 'line'


def load_policy(path):
    """.keras → StudentPolicy (TensorFlow); .tflite → TFLitePolicy."""
    return TFLitePolicy(path) if str(path).endswith('.tflite') else StudentPolicy(path)
