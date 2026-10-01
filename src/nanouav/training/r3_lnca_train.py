"""
R3 · Comparative training on a reduced (CPU) budget: CNN-M vs LNCA (± early ToF).

All models use EXACTLY the same data, loss (get_depth_loss), augmentation,
epochs and LR. Does not touch models/model_*.tflite or metrics.json (separate output).

Variants:
  M          : current depthwise U-Net (26 k)
  M+cal      : M + per-image global scale from the 8×8 ToF (like ToFCalibrator, without EMA)
  LNCA24     : conv encoder → 12×12×24 → 4 NCA steps → PixelShuffle
  LNCA24+ToF : + 8×8 ToF injected into the 12×12 latent grid (early fusion)

Usage:  nanouav run r03 --n-train 4000 --epochs 15
Output: results/r3_lnca_train.json, models/r3/*.tflite
"""
import os, sys, json, time, argparse
from nanouav import paths
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np

ROOT = str(paths.repo_root())
from nanouav.config import (m_to_norm, norm_to_m, pool_to_tof, ZONE_BOXES, TOF_GRID, D_MIN, D_MAX)
from nanouav.training.train_cnn import _gen, _gazebo_view, thin_scene
from nanouav.perception.model_zoo import build_variant, get_depth_loss, depth_metrics, tflite_arena_kb
from nanouav.perception.lnca_zoo import build_lnca, count_macs

OUT = os.path.join(str(paths.models_dir()), 'r3')
CACHE = os.path.join(str(paths.data_dir()), 'synth_r3.npz')

# ── 8×8 ToF → 12×12 latent grid (fixed table; on the MCU it is a 144-entry gather)
_edges = [ZONE_BOXES[0][k][2] for k in range(TOF_GRID)] + [ZONE_BOXES[0][TOF_GRID - 1][3]]
LAT_IDX = np.full(12, -1, int)
for k in range(12):
    c = 8 * k + 4                                   # latent cell center, in pixels
    for z in range(TOF_GRID):
        if _edges[z] <= c < _edges[z + 1]:
            LAT_IDX[k] = z


def tof_sim(depth_m, rng):
    """8×8 ToF (low percentile per zone) + approx. VL53L5CX noise + out-of-range marked invalid."""
    t = pool_to_tof(depth_m)
    t = t * (1 + rng.normal(0, 0.02, t.shape)) + rng.normal(0, 0.01, t.shape)
    valid = (t > D_MIN) & (t < D_MAX - 0.3)
    return np.clip(t, D_MIN, D_MAX), valid


def tof_to_lat(t, valid):
    g = np.zeros((12, 12, 2), np.float32)
    for i in range(12):
        for j in range(12):
            zi, zj = LAT_IDX[i], LAT_IDX[j]
            if zi >= 0 and zj >= 0 and valid[zi, zj]:
                g[i, j, 0] = np.log(t[zi, zj] / D_MIN) / np.log(D_MAX / D_MIN)   # normalized log-depth
                g[i, j, 1] = 1.0
    return g


def build_data(n_tr, n_va, n_te, n_gz, thin=0.0):
    """thin > 0: thin poles/wires in that fraction of the train/val scenes (own cache); test and
    Gazebo views stay the original ones."""
    cache = CACHE if not thin else CACHE.replace('.npz', f'_thin{thin:g}.npz')
    if os.path.exists(cache):
        z = np.load(cache)
        if len(z['Xtr']) >= n_tr:
            return {k: z[k] for k in z.files}
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    print(f'[data] generating {n_tr}+{n_va}+{n_te}+{n_gz} scenes (thin={thin:g}) → {cache}', flush=True)
    d = {}
    d['Xtr'], d['Ytr'] = _gen(n_tr, 0, thin_scene(thin))
    d['Xva'], d['Yva'] = _gen(n_va, 10_000_000, thin_scene(thin))
    d['Xte'], d['Yte'] = _gen(n_te, 20_000_000)
    d['Xgz'], d['Ygz'] = _gen(n_gz, 30_000_000, _gazebo_view)
    rng = np.random.default_rng(7)
    for s in ('tr', 'va', 'te', 'gz'):
        T, V, L = [], [], []
        for y in d['Y' + s]:
            t, v = tof_sim(norm_to_m(y / 65535.0), rng)
            T.append(t); V.append(v); L.append(tof_to_lat(t, v))
        d['T' + s], d['V' + s], d['L' + s] = np.array(T, np.float32), np.array(V), np.array(L)
    np.savez_compressed(cache, **d)
    return d


def make_ds(X, Y, L, batch, train, with_tof):
    import tensorflow as tf

    def prep(x, y, l):
        return tf.cast(x, tf.float32)[..., None] / 255.0, tf.cast(y, tf.float32)[..., None] / 65535.0, l

    def aug(x, y, l):
        flip = tf.random.uniform(()) < 0.5
        x = tf.cond(flip, lambda: tf.image.flip_left_right(x), lambda: x)
        y = tf.cond(flip, lambda: tf.image.flip_left_right(y), lambda: y)
        l = tf.cond(flip, lambda: tf.image.flip_left_right(l), lambda: l)
        m = tf.reduce_mean(x)
        x = (x - m) * tf.random.uniform((), 0.75, 1.3) + m
        x = x * tf.random.uniform((), 0.8, 1.2) + tf.random.uniform((), -0.06, 0.06)
        x = tf.clip_by_value(x, 1e-4, 1.0) ** tf.random.uniform((), 0.8, 1.25)
        x = x + tf.random.normal(tf.shape(x), 0.0, tf.random.uniform((), 0.0, 0.02))
        # ToF dropout (invalid zones / missing sensor): the network must not rely on it blindly
        keep = tf.cast(tf.random.uniform(()) > 0.15, tf.float32)
        return tf.clip_by_value(x, 0.0, 1.0), y, l * keep

    def pack(x, y, l):
        return ({'input': x, 'tof12': l} if with_tof else x), y

    ds = tf.data.Dataset.from_tensor_slices((X, Y, L))
    if train:
        ds = ds.shuffle(len(X), reshuffle_each_iteration=True)
    ds = ds.map(prep, num_parallel_calls=tf.data.AUTOTUNE)
    if train:
        ds = ds.map(aug, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.map(pack).batch(batch).prefetch(tf.data.AUTOTUNE)


def export_int8(model, X, L, path, with_tof):
    import tensorflow as tf

    def rep():
        for i in np.linspace(0, len(X) - 1, 300).astype(int):
            x = X[i:i + 1, ..., None].astype(np.float32) / 255.0
            yield ({'input': x, 'tof12': L[i:i + 1]} if with_tof else [x])

    c = tf.lite.TFLiteConverter.from_keras_model(model)
    c.optimizations = [tf.lite.Optimize.DEFAULT]
    c.representative_dataset = rep
    c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    c.inference_input_type = c.inference_output_type = tf.int8
    open(path, 'wb').write(c.convert())


def predict_int8(path, X, L):
    import tensorflow as tf
    it = tf.lite.Interpreter(model_path=path, num_threads=2)
    it.allocate_tensors()
    ins = {d['name'].split(':')[0].replace('serving_default_', ''): d for d in it.get_input_details()}
    out = it.get_output_details()[0]
    so, zo = out['quantization']
    res = np.empty((len(X), 96, 96, 1), np.float32)
    for i in range(len(X)):
        for name, d in ins.items():
            s, z = d['quantization']
            v = (X[i:i + 1, ..., None] / 255.0) if 'tof' not in name else L[i:i + 1]
            it.set_tensor(d['index'], np.clip(np.round(v / s + z), -128, 127).astype(np.int8))
        it.invoke()
        res[i] = (it.get_tensor(out['index']).astype(np.float32) - zo) * so
    return res, it


def tof_scale(pred_n, T, V):
    """Per-image global scale: median(ToF / CNN) over valid zones (core of ToFCalibrator)."""
    out = pred_n.copy()
    for i in range(len(pred_n)):
        dm = norm_to_m(pred_n[i, ..., 0])
        c8 = pool_to_tof(dm)
        ok = V[i] & (c8 > D_MIN + 0.05) & (c8 < D_MAX - 0.3)
        if ok.sum() >= 8:
            s = float(np.clip(np.median(T[i][ok] / c8[ok]), 0.4, 2.5))
            out[i, ..., 0] = m_to_norm(dm * s)
    return out


def _warm_start(m, path):
    """Copies weights from a trained model layer by layer (same order of weighted layers and same shapes)."""
    import keras
    from nanouav.perception.model_zoo import get_depth_loss
    # Without deserializing the config (avoids incompatibilities between Keras versions): the
    # architecture is rebuilt from code and only the weights are loaded from the .keras file.
    variant = os.path.basename(path).replace('model_', '').split('.')[0]
    try:
        src = build_variant(variant)
        src.load_weights(path)
    except Exception as e:
        print(f'[warm-start] load_weights failed ({type(e).__name__}); trying load_model', flush=True)
        src = keras.models.load_model(path, compile=False, custom_objects={'depth_loss': get_depth_loss()})
    sw = [l for l in src.layers if l.get_weights()]
    dw = [l for l in m.layers if l.get_weights() and l.name != 'tof_inject']
    n = 0
    for ls, ld in zip(sw, dw):
        if [w.shape for w in ls.get_weights()] == [w.shape for w in ld.get_weights()]:
            ld.set_weights(ls.get_weights()); n += 1
    for l in m.layers:
        if l.name == 'tof_inject':
            l.set_weights([np.zeros_like(w) for w in l.get_weights()])
    print(f'[warm-start] {n}/{len(dw)} layers copied from {os.path.basename(path)}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-train', type=int, default=4000)
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--models', default='M,LNCA24,LNCA24_ToF')
    ap.add_argument('--init-from', default=None,
                    help='trained .keras (e.g. models/model_M.keras): copies weights by name/shape; '
                         'the tof_inject layer starts at zero → the model starts out identical to M')
    ap.add_argument('--install', action='store_true',
                    help='copies the resulting .tflite to models/model_<name>.tflite (used by the closed loop)')
    ap.add_argument('--thin', type=float, default=0.0,
                    help='fraction of train/val scenes with thin poles and wires (R14); own data cache')
    ap.add_argument('--suffix', default='',
                    help='output name suffix (models/model_<name><suffix>.tflite, results key), e.g. _thin')
    ap.add_argument('--out-json', default=str(paths.results_dir() / 'r3_lnca_train.json'))
    a = ap.parse_args()
    import tensorflow as tf, keras
    tf.random.set_seed(0)
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    d = build_data(a.n_train, 500, 600, 300, thin=a.thin)
    print(f'[data] ready ({time.time() - t0:.0f}s)', flush=True)
    for k in list(d):
        if k.endswith('tr'):
            d[k] = d[k][:a.n_train]

    builders = {
        'M': (lambda b=None: build_variant('M', batch_size=b), False),
        'M_ToF': (lambda b=None: build_variant('M', batch_size=b, tof=True), True),
        'LNCA24': (lambda b=None: build_lnca(24, 4, batch_size=b), False),
        'LNCA24_ToF': (lambda b=None: build_lnca(24, 4, tof=True, batch_size=b), True),
        'LNCA32ds_ToF': (lambda b=None: build_lnca(32, 4, enc='ds', tof=True, batch_size=b), True),
    }
    resp = os.path.join(ROOT, a.out_json)
    results = json.load(open(resp)) if os.path.exists(resp) else {}
    results['_setup'] = vars(a)
    Yte = d['Yte'][..., None] / 65535.0
    Ygz = d['Ygz'][..., None] / 65535.0
    for name in a.models.split(','):
        fn, with_tof = builders[name]
        m = fn()
        if a.init_from:
            _warm_start(m, a.init_from)
        steps = int(np.ceil(len(d['Xtr']) / a.batch)) * a.epochs
        m.compile(optimizer=keras.optimizers.Adam(keras.optimizers.schedules.CosineDecay(a.lr, steps, alpha=0.02)),
                  loss=get_depth_loss(),
                  jit_compile=False)   # no XLA: its autotuner cannot find fused cuDNN kernels on Pascal GPUs
        ts = time.time()
        h = m.fit(make_ds(d['Xtr'], d['Ytr'], d['Ltr'], a.batch, True, with_tof),
                  validation_data=make_ds(d['Xva'], d['Yva'], d['Lva'], 64, False, with_tof),
                  epochs=a.epochs, verbose=2)
        t_train = time.time() - ts
        m1 = fn(1)
        m1.set_weights(m.get_weights())
        tag = name + a.suffix
        path = os.path.join(OUT, f'{tag}.tflite')
        export_int8(m1, d['Xtr'], d['Ltr'], path, with_tof)
        pq, it = predict_int8(path, d['Xte'], d['Lte'])
        pg, _ = predict_int8(path, d['Xgz'], d['Lgz'])
        r = {'params_trainable': int(sum(np.prod(w.shape) for w in m.trainable_weights)),
             'MMAC': count_macs(m1) / 1e6, 'arena_kb': tflite_arena_kb(it),
             'tflite_kb': os.path.getsize(path) / 1024, 'train_s': t_train,
             'val_loss': [float(x) for x in h.history['val_loss']],
             'int8_test': depth_metrics(Yte, pq), 'int8_gazebo': depth_metrics(Ygz, pg)}
        r['train_data'] = {'thin': a.thin, 'n_train': a.n_train, 'init_from': a.init_from}
        results[tag] = r
        if not with_tof:              # same network + global ToF calibration (current pipeline)
            results[tag + '+cal'] = {'int8_test': depth_metrics(Yte, tof_scale(pq, d['Tte'], d['Vte'])),
                                      'int8_gazebo': depth_metrics(Ygz, tof_scale(pg, d['Tgz'], d['Vgz']))}
        json.dump(results, open(resp, 'w'), indent=1)
        if a.install:
            import shutil
            shutil.copy(path, os.path.join(str(paths.models_dir()), f'model_{tag}.tflite'))
            m1.save(os.path.join(str(paths.models_dir()), f'model_{tag}.keras'))
            print(f'[install] models/model_{tag}.tflite', flush=True)
        q, g = r['int8_test'], r['int8_gazebo']
        print(f'[{tag}] {t_train:.0f}s  test AbsRel {q["abs_rel"]:.3f} δ1 {q["delta1"]:.3f} '
              f'minErr {q["min_dist_mae"]:.3f} rec<1m {q["obs1m_recall"]:.2f} | gz AbsRel {g["abs_rel"]:.3f} '
              f'δ1 {g["delta1"]:.3f} rec<1m {g["obs1m_recall"]:.2f}', flush=True)
    print(f'[done] {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
