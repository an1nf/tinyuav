"""
train_cnn.py — Depth CNN training v2 (replaces train.py, train_e0d.py,
e2a_train_variants.py and generate_model.py as the entry point).

Improvements:
  * Synthetic dataset v2 aligned with Gazebo (same camera, z-depth, shadows,
    textures, perturbed poses) + optional TartanAir (--tartanair).
  * On-the-fly augmentation (flip, brightness, contrast, gamma, noise).
  * L1 loss weighted toward near range + gradient + SSIM.
  * Correct int8 quantization and evaluation of the int8 model (not just float).
  * Avoidance metrics: minimum-distance error in the ToF region, recall/precision
    of "obstacle < 1 m".
  * Activation arena measured on the TFLite graph (SRAM limit 180 KB).

Usage:
    nanouav run train                         # 4 variants
    nanouav run train --variants M --epochs 40
    nanouav run train --tartanair             # mix with data/tartanair
"""
import os, sys, json, time, argparse
import numpy as np
from nanouav import paths

ROOT = str(paths.repo_root())

from nanouav.config import m_to_norm
from nanouav.scene.renderer import render_scene, render_view, gazebo_world
from nanouav.perception.model_zoo import (VARIANT_CONFIGS, SRAM_ARENA_LIMIT_KB, build_variant,
                               get_depth_loss, depth_metrics, export_tflite,
                               make_interpreter, tflite_predict, tflite_arena_kb)

CNN_DIR = str(paths.models_dir())
DATA_DIR = str(paths.data_dir())
ASSETS = str(paths.results_dir())
CACHE = os.path.join(DATA_DIR, 'synth_v2.npz')


# ── Dataset ────────────────────────────────────────────────────────────────
def _gen(n, seed0, fn=render_scene):
    X = np.empty((n, 96, 96), np.uint8)
    Y = np.empty((n, 96, 96), np.uint16)
    t0 = time.time()
    for i in range(n):
        g, d = fn(seed0 + i)
        X[i] = g
        Y[i] = np.round(m_to_norm(d) * 65535).astype(np.uint16)
        if (i + 1) % 2000 == 0:
            print(f'    {i + 1}/{n}  ({time.time() - t0:.0f}s)', flush=True)
    return X, Y


def _gazebo_view(seed):
    """Views of the SDF world from plausible flight poses (in-domain test)."""
    rng = np.random.default_rng(seed)
    w = gazebo_world()
    for _ in range(50):
        p = np.array([rng.uniform(-2.0, 2.6), rng.uniform(-1.3, 1.3), rng.uniform(0.4, 1.0)])
        if w.clearance(p) > 0.35:
            break
    return render_view(w, p, rng.uniform(-0.6, 0.6), rng.normal(0, 0.06),
                       rng.normal(0, 0.05), rng=rng)


def load_or_build(n_train, n_val, n_test, n_gz):
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        if len(z['Xtr']) >= n_train:
            print(f'[data] cache {CACHE}')
            return {k: z[k] for k in z.files}
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f'[data] generating {n_train}+{n_val}+{n_test} scenes + {n_gz} Gazebo...')
    d = {}
    d['Xtr'], d['Ytr'] = _gen(n_train, 0)
    d['Xva'], d['Yva'] = _gen(n_val, 10_000_000)
    d['Xte'], d['Yte'] = _gen(n_test, 20_000_000)
    d['Xgz'], d['Ygz'] = _gen(n_gz, 30_000_000, _gazebo_view)
    np.savez_compressed(CACHE, **d)
    return d


def load_tartanair(max_samples=6000):
    from nanouav.training.tartanair import build_dataset, DATA_DIR as TA_DIR
    if not os.path.isdir(TA_DIR):
        print('[data] TartanAir not found, skipping')
        return None, None
    X, Y = build_dataset(max_samples)            # float [0,1], norm [0,1]
    return (np.round(X[..., 0] * 255).astype(np.uint8),
            np.round(Y[..., 0] * 65535).astype(np.uint16))


def to_float(X, Y):
    return (X.astype(np.float32)[..., None] / 255.0,
            Y.astype(np.float32)[..., None] / 65535.0)


def make_ds(X, Y, batch, train):
    import tensorflow as tf

    def prep(x, y):
        x = tf.cast(x, tf.float32)[..., None] / 255.0
        y = tf.cast(y, tf.float32)[..., None] / 65535.0
        return x, y

    def aug(x, y):
        flip = tf.random.uniform(()) < 0.5
        x = tf.cond(flip, lambda: tf.image.flip_left_right(x), lambda: x)
        y = tf.cond(flip, lambda: tf.image.flip_left_right(y), lambda: y)
        m = tf.reduce_mean(x)
        x = (x - m) * tf.random.uniform((), 0.75, 1.3) + m
        x = x * tf.random.uniform((), 0.8, 1.2) + tf.random.uniform((), -0.06, 0.06)
        x = tf.clip_by_value(x, 1e-4, 1.0) ** tf.random.uniform((), 0.8, 1.25)
        x = x + tf.random.normal(tf.shape(x), 0.0, tf.random.uniform((), 0.0, 0.02))
        return tf.clip_by_value(x, 0.0, 1.0), y

    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    if train:
        ds = ds.shuffle(min(len(X), 8192), reshuffle_each_iteration=True)
    ds = ds.map(prep, num_parallel_calls=tf.data.AUTOTUNE)
    if train:
        ds = ds.map(aug, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)


# ── Training a single variant ──────────────────────────────────────────────
def train_variant(v, data, epochs, batch, lr):
    import tensorflow as tf
    import keras

    loss = get_depth_loss()
    model = build_variant(v)
    print(f'\n[{v}] parameters: {model.count_params():,}', flush=True)
    steps = int(np.ceil(len(data['Xtr']) / batch)) * epochs
    sched = keras.optimizers.schedules.CosineDecay(lr, steps, alpha=0.02,
                                                   warmup_target=None)
    model.compile(optimizer=keras.optimizers.Adam(sched), loss=loss, metrics=['mae'],
                  jit_compile=False)   # no XLA: its autotuner cannot find fused cuDNN kernels on Pascal GPUs
    t0 = time.time()
    hist = model.fit(make_ds(data['Xtr'], data['Ytr'], batch, True),
                     validation_data=make_ds(data['Xva'], data['Yva'], 64, False),
                     epochs=epochs, verbose=2,
                     callbacks=[keras.callbacks.EarlyStopping(
                         monitor='val_loss', patience=10, restore_best_weights=True)])
    t_train = time.time() - t0

    # Export model with fixed batch = 1 (same graph, same weights)
    m1 = build_variant(v, batch_size=1)
    m1.set_weights(model.get_weights())
    m1.save(os.path.join(CNN_DIR, f'model_{v}.keras'))
    Xrep, _ = to_float(data['Xtr'][:2000], data['Ytr'][:2000])
    tfl = os.path.join(CNN_DIR, f'model_{v}.tflite')
    nbytes = export_tflite(m1, Xrep, tfl)
    return m1, hist.history, t_train, nbytes


def evaluate(v, model, data):
    Xte, Yte = to_float(data['Xte'], data['Yte'])
    Xgz, Ygz = to_float(data['Xgz'], data['Ygz'])
    pf = model.predict(Xte, verbose=0, batch_size=64)
    it = make_interpreter(os.path.join(CNN_DIR, f'model_{v}.tflite'))
    pq = tflite_predict(it, Xte)
    pq_gz = tflite_predict(it, Xgz)
    q = it.get_input_details()[0]['quantization']
    lat = []
    for i in range(60):
        t0 = time.perf_counter()
        tflite_predict(it, Xte[i:i + 1])
        lat.append((time.perf_counter() - t0) * 1000)
    return {
        'params': int(model.count_params()),
        'tflite_kb': os.path.getsize(os.path.join(CNN_DIR, f'model_{v}.tflite')) / 1024,
        'arena_kb': tflite_arena_kb(it),
        'input_quant': [float(q[0]), int(q[1])],
        'cpu_latency_ms_p50': float(np.median(lat[10:])),
        'float_test': depth_metrics(Yte, pf),
        'int8_test': depth_metrics(Yte, pq),
        'int8_gazebo': depth_metrics(Ygz, pq_gz),
    }, (Xgz, Ygz, pq_gz)


# ── Plots ──────────────────────────────────────────────────────────────────
def plots(results, histories, preview):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    DARK, BG = '#111111', '#1a1a1a'
    C = {'XS': '#5DCAA5', 'S': '#378ADD', 'M': '#EF9F27', 'L': '#E24B4A'}

    def sty(ax, t):
        ax.set_facecolor(DARK); ax.tick_params(colors='#888', labelsize=8)
        for s in ax.spines.values(): s.set_edgecolor('#333')
        ax.set_title(t, color='white', fontsize=9)

    vs = list(results)
    fig, axs = plt.subplots(2, 3, figsize=(15, 8), facecolor=BG)
    fig.suptitle('CNN v2 · int8 metrics (synthetic test) and resources', color='white')
    items = [('abs_rel', 'AbsRel ↓'), ('rmse', 'RMSE (m) ↓'), ('min_dist_mae', 'Min-distance error ToF (m) ↓'),
             ('obs1m_recall', 'Obstacle recall <1 m ↑')]
    for ax, (k, t) in zip(axs.flat[:4], items):
        sty(ax, t)
        fl = [results[v]['float_test'][k] for v in vs]
        q8 = [results[v]['int8_test'][k] for v in vs]
        gz = [results[v]['int8_gazebo'][k] for v in vs]
        x = np.arange(len(vs))
        ax.bar(x - 0.27, fl, 0.27, color='#666', label='float')
        ax.bar(x, q8, 0.27, color=[C[v] for v in vs], label='int8')
        ax.bar(x + 0.27, gz, 0.27, color=[C[v] for v in vs], alpha=0.45, label='int8 Gazebo world')
        ax.set_xticks(x); ax.set_xticklabels(vs, color='#aaa')
        ax.legend(fontsize=7, facecolor='#222', labelcolor='white')
    ax = axs.flat[4]; sty(ax, 'Activation arena (KB) vs SRAM limit')
    ax.bar(vs, [results[v]['arena_kb'] for v in vs], color=[C[v] for v in vs])
    ax.axhline(SRAM_ARENA_LIMIT_KB, color='white', ls='--', lw=0.8)
    ax = axs.flat[5]; sty(ax, 'Validation loss')
    for v in vs:
        ax.plot(results[v].get('val_loss_hist', []), color=C[v], label=v)
    ax.legend(fontsize=7, facecolor='#222', labelcolor='white')
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, 'cnn_v2_comparison.png'), dpi=120, facecolor=BG)
    plt.close(fig)

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('d', [(0.9, 0.15, 0.1), (0.95, 0.9, 0.1),
                                                   (0.1, 0.78, 0.3), (0.08, 0.35, 0.8)])
    v, (X, Y, P) = preview
    idx = np.linspace(0, len(X) - 1, 5).astype(int)
    fig, axs = plt.subplots(len(idx), 3, figsize=(7, 2.3 * len(idx)), facecolor=BG)
    fig.suptitle(f'Variant {v} int8 · Gazebo world (render)', color='white')
    for r, i in enumerate(idx):
        for c, (img, cm) in enumerate([(X[i, ..., 0], 'gray'), (Y[i, ..., 0], cmap), (P[i, ..., 0], cmap)]):
            axs[r, c].imshow(img, cmap=cm, vmin=0, vmax=1); axs[r, c].axis('off')
            if r == 0:
                axs[r, c].set_title(['camera', 'GT', 'CNN int8'][c], color='white', fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, 'cnn_v2_preview.png'), dpi=110, facecolor=BG)
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variants', default='XS,S,M,L')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--n-train', type=int, default=12000)
    ap.add_argument('--tartanair', action='store_true')
    a = ap.parse_args()

    import tensorflow as tf
    tf.random.set_seed(0)
    os.makedirs(ASSETS, exist_ok=True)
    data = load_or_build(a.n_train, 1500, 1000, 400)
    data['Xtr'], data['Ytr'] = data['Xtr'][:a.n_train], data['Ytr'][:a.n_train]
    if a.tartanair:
        Xt, Yt = load_tartanair()
        if Xt is not None:
            data['Xtr'] = np.concatenate([data['Xtr'], Xt])
            data['Ytr'] = np.concatenate([data['Ytr'], Yt])
    print(f'[data] train={len(data["Xtr"])} val={len(data["Xva"])} '
          f'test={len(data["Xte"])} gazebo={len(data["Xgz"])}')

    mpath = os.path.join(CNN_DIR, 'metrics.json')
    results = json.load(open(mpath)) if os.path.exists(mpath) else {}
    histories, preview = {}, None
    for v in a.variants.split(','):
        model, hist, t_train, _ = train_variant(v, data, a.epochs, a.batch, a.lr)
        res, prev = evaluate(v, model, data)
        res['train_s'] = t_train
        res['epochs'] = len(hist['loss'])
        res['val_loss_hist'] = [float(x) for x in hist['val_loss']]
        results[v] = res
        histories[v] = hist
        if v == 'M' or preview is None:
            preview = (v, prev)
        json.dump(results, open(mpath, 'w'), indent=2)
        q = res['int8_test']
        print(f'[{v}] int8: AbsRel={q["abs_rel"]:.3f} RMSE={q["rmse"]:.3f}m '
              f'δ1={q["delta1"]:.3f} minErr={q["min_dist_mae"]:.3f}m '
              f'recall<1m={q["obs1m_recall"]:.2f} | arena={res["arena_kb"]:.0f}KB '
              f'tflite={res["tflite_kb"]:.0f}KB', flush=True)

    # default model.tflite = M (used by cnn.py when no variant is requested)
    import shutil
    if os.path.exists(os.path.join(CNN_DIR, 'model_M.tflite')):
        shutil.copy(os.path.join(CNN_DIR, 'model_M.tflite'), os.path.join(CNN_DIR, 'model.tflite'))
    plots({k: results[k] for k in ['XS', 'S', 'M', 'L'] if k in results}, histories, preview)
    print('\n[done] metrics in models/metrics.json · plots in results/cnn_v2_*.png')


if __name__ == '__main__':
    main()
