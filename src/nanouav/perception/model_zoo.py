"""
nanouav/perception/model_zoo.py — Architectures, loss, metrics and int8 export.

Fixes:
  * Input quantization: the representative_dataset uses EXACTLY the same
    distribution as training ([0,1]). Previously img*255-128 was used and the
    input scale ended up at ~0.99 → the int8 image had only 1–2 levels.
  * Truly distinct variants (previously S and M shared filter counts).

Architecture (TFLite Micro / ESP32-S3 compatible):
  stem Conv3×3 s2 → 4 depthwise-separable blocks (48→24→12→6) → bottleneck
  → decoder: 1×1 at low resolution → bilinear upsample ×2 → Add(skip) → DW
  → 48×48×4 → upsample → 96×96×4 → Conv3×3 → Conv1×1 sigmoid.
  Channels are reduced BEFORE upsampling to bound the activation arena.
"""
import os
import numpy as np

from nanouav.config import D_MIN, D_MAX, D_RANGE, pool_to_tof, norm_to_m

VARIANT_CONFIGS = {
    #        f0  f1  f2  f3   bottleneck blocks
    'XS': {'f': (6, 12, 24, 32), 'nb': 1},
    'S':  {'f': (8, 16, 32, 48), 'nb': 1},
    'M':  {'f': (8, 24, 48, 64), 'nb': 2},
    'L':  {'f': (12, 32, 64, 96), 'nb': 3},
}
SRAM_ARENA_LIMIT_KB = 180


# ── Model ──────────────────────────────────────────────────────────────────
def build_variant(variant: str, batch_size=None, tof=False):
    """tof=True (R3): 8×8 ToF resampled to 12×12 [log d, valid] and added at level e3 (12×12)."""
    import keras
    from keras import layers

    f0, f1, f2, f3 = VARIANT_CONFIGS[variant]['f']
    nb = VARIANT_CONFIGS[variant]['nb']

    def conv_bn(x, f, k=1, s=1):
        x = layers.Conv2D(f, k, strides=s, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        return layers.ReLU(6.0)(x)

    def dw_block(x, f, s=1):
        x = layers.DepthwiseConv2D(3, strides=s, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU(6.0)(x)
        return conv_bn(x, f, 1)

    def up_merge(x, skip, f):
        x = conv_bn(x, f, 1)                                   # reduce channels first
        x = layers.UpSampling2D(2, interpolation='bilinear')(x)
        x = layers.Add()([x, skip])
        return dw_block(x, f)

    inp = keras.Input(shape=(96, 96, 1), batch_size=batch_size, name='input')
    e1 = conv_bn(inp, f0, 3, 2)          # 48
    e1 = dw_block(e1, f0)
    e2 = dw_block(e1, f1, 2)             # 24
    e3 = dw_block(e2, f2, 2)             # 12
    ins = inp
    if tof:
        tin = keras.Input(shape=(12, 12, 2), batch_size=batch_size, name='tof12')
        e3 = layers.Add()([e3, layers.Conv2D(f2, 1, name='tof_inject')(tin)])
        ins = [inp, tin]
    b = dw_block(e3, f3, 2)              # 6
    for _ in range(nb):
        b = dw_block(b, f3)
    d = up_merge(b, e3, f2)              # 12
    d = up_merge(d, e2, f1)              # 24
    d = up_merge(d, e1, f0)              # 48
    h = conv_bn(d, 4, 1)
    h = layers.UpSampling2D(2, interpolation='bilinear')(h)   # 96×96×4
    h = conv_bn(h, 4, 3)
    out = layers.Conv2D(1, 1, activation='sigmoid', name='output')(h)
    return keras.Model(ins, out, name=f'depth_{variant}' + ('_tof' if tof else ''))


def get_depth_loss():
    import tensorflow as tf
    import keras

    @keras.saving.register_keras_serializable(package='nano_uav')
    def depth_loss(y_true, y_pred):
        """
        Near-weighted L1 + multi-scale gradient (1×, ½×, ¼×).
        The weighting (1 + 3·(1-y)) prioritizes the <1.5 m range, which is where
        avoidance is decided; the multi-scale gradient keeps edges sharp
        (replaces SSIM: same practical effect, half the CPU cost).
        """
        w = 1.0 + 3.0 * (1.0 - y_true)
        loss = tf.reduce_sum(w * tf.abs(y_true - y_pred)) / tf.reduce_sum(w)
        yt, yp = y_true, y_pred
        for s in range(3):
            dy_t, dx_t = tf.image.image_gradients(yt)
            dy_p, dx_p = tf.image.image_gradients(yp)
            loss += 0.5 / (s + 1) * tf.reduce_mean(tf.abs(dy_t - dy_p) + tf.abs(dx_t - dx_p))
            yt = tf.nn.avg_pool2d(yt, 2, 2, 'VALID')
            yp = tf.nn.avg_pool2d(yp, 2, 2, 'VALID')
        return loss

    return depth_loss


# ── Metrics ────────────────────────────────────────────────────────────────
def depth_metrics(y_true_norm, y_pred_norm):
    """
    Metrics in meters. Normalized inputs (N,96,96[,1]).
    Includes avoidance-oriented metrics over the ToF region.
    """
    yt = norm_to_m(np.squeeze(y_true_norm, -1) if y_true_norm.ndim == 4 else y_true_norm)
    yp = norm_to_m(np.squeeze(y_pred_norm, -1) if y_pred_norm.ndim == 4 else y_pred_norm)
    yp = np.clip(yp, D_MIN, D_MAX)
    abs_rel = float(np.mean(np.abs(yt - yp) / yt))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    ratio = np.maximum(yt / yp, yp / yt)
    d1 = float(np.mean(ratio < 1.25))
    near = yt < 1.5
    mae_near = float(np.mean(np.abs(yt - yp)[near])) if near.any() else float('nan')

    mins_t, mins_p = [], []
    for a, b in zip(yt, yp):
        mins_t.append(pool_to_tof(a).min())
        mins_p.append(pool_to_tof(b).min())
    mins_t, mins_p = np.array(mins_t), np.array(mins_p)
    obs_t, obs_p = mins_t < 1.0, mins_p < 1.0
    tp = float(np.sum(obs_t & obs_p))
    rec = tp / max(obs_t.sum(), 1)
    prec = tp / max(obs_p.sum(), 1)
    return {'abs_rel': abs_rel, 'rmse': rmse, 'delta1': d1, 'mae_near': mae_near,
            'min_dist_mae': float(np.mean(np.abs(mins_t - mins_p))),
            'obs1m_recall': rec, 'obs1m_precision': prec}


# ── int8 export ────────────────────────────────────────────────────────────
def export_tflite(model, X_rep, path, n_rep=300):
    """
    Full-int8 post-training quantization.
    X_rep: float32 (N,96,96,1) in [0,1] — the SAME scale as in training.
    """
    import tensorflow as tf

    def rep():
        idx = np.linspace(0, len(X_rep) - 1, min(n_rep, len(X_rep))).astype(int)
        for i in idx:
            yield [X_rep[i:i + 1].astype(np.float32)]

    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    data = conv.convert()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(data)
    return len(data)


def make_interpreter(path, threads=1):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        from tflite_runtime.interpreter import Interpreter
    it = Interpreter(model_path=path, num_threads=threads)
    it.allocate_tensors()
    return it


def tflite_predict(it, X01):
    """X01 float (N,96,96,1) in [0,1] → normalized float prediction (N,96,96,1)."""
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    si, zi = inp['quantization']
    so, zo = out['quantization']
    res = np.empty(X01.shape, np.float32)
    for i in range(len(X01)):
        q = np.clip(np.round(X01[i:i + 1] / si + zi), -128, 127).astype(np.int8)
        it.set_tensor(inp['index'], q)
        it.invoke()
        res[i] = (it.get_tensor(out['index']).astype(np.float32) - zo) * so
    return res


def tflite_arena_kb(it):
    """
    Lower bound of the (int8) activation arena via tensor-lifetime analysis
    over the TFLite graph in execution order. Does not include kernel scratch
    buffers or TFLM overhead (add ~10–20 %).
    """
    ops = it._get_ops_details()
    tens = {t['index']: t for t in it.get_tensor_details()}
    produced = {}
    last_use = {}
    for k, op in enumerate(ops):
        for t in op['outputs']:
            produced[t] = k
        for t in op['inputs']:
            if t >= 0:
                last_use[t] = k
    inputs = [d['index'] for d in it.get_input_details()]
    for t in inputs:
        produced[t] = -1
    size = {}
    for t, k0 in produced.items():
        shp = tens[t]['shape']
        size[t] = int(np.prod(shp)) * np.dtype(tens[t]['dtype']).itemsize
    peak = 0
    for k in range(len(ops)):
        live = sum(s for t, s in size.items()
                   if produced[t] <= k and last_use.get(t, len(ops)) >= k)
        peak = max(peak, live)
    return peak / 1024.0
