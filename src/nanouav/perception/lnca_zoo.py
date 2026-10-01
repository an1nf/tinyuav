"""
nanouav/perception/lnca_zoo.py — Depth Latent Neural Cellular Automata (LNCA), adapted to 96×96.

Based on Nunn & Nikoubin, SN Computer Science 7:755 (2026):
  encoder (3 conv s2, 8× downsampling) → T NCA steps with shared weights over the
  latent grid → PixelShuffle decoder (conv3×3 → depth_to_space ×8).

Our own adaptations (not in the paper):
  * 96×96 gray input → 12×12 latent grid (cell = 8 px ≈ 5°), almost 1:1 with
    the 8×8 ToF zones (40° / 8 = 5°/zone).
  * 'tof=True': early fusion. The 64 ToF zones (log-depth + validity mask),
    resampled to the 12×12 grid OUTSIDE the network (fixed table on the MCU),
    are added to the latent state through a 1×1 conv. The NCA propagates the metric
    scale from the center (40°) to the periphery (60°) with its lateral 3×3 convs.
  * 'aux_det=True': target-center head (12×12 heatmap, anchor-free style of
    FeatherYOLO — Su et al., EAAI 177:114890) sharing the encoder and the NCA.
  * The state can be initialized from the previous frame's state (recurrent mode,
    'state_in=True'): x0 = z + x_prev, which allows T=1 per frame.

All ops are TFLite-Micro: CONV_2D, DEPTHWISE_CONV_2D, ADD, CONCATENATION,
TRANSPOSE_CONV (one-hot PixelShuffle), LOGISTIC, RELU.
"""
import numpy as np

LNCA_CONFIGS = {
    #          C   hidden  T   encoder
    'N24':  {'C': 24, 'T': 4, 'enc': 'conv'},
    'N32':  {'C': 32, 'T': 4, 'enc': 'conv'},      # ≈ paper (40.8 k with RGB)
    'N32ds': {'C': 32, 'T': 4, 'enc': 'ds'},       # depthwise-separable encoder
}


def build_lnca(C=32, T=4, enc='conv', tof=False, aux_det=False, state_in=False,
               batch_size=None, name='lnca', shuffle='tconv'):
    import keras
    from keras import layers

    inp = keras.Input((96, 96, 1), batch_size=batch_size, name='input')
    ins = [inp]

    def cbr(x, f, k, s, act=True):
        x = layers.Conv2D(f, k, strides=s, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        return layers.ReLU(6.0)(x) if act else x

    def ds(x, f, s, act=True):
        x = layers.DepthwiseConv2D(3, strides=s, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU(6.0)(x)
        return cbr(x, f, 1, 1, act)

    # ── encoder 96 → 48 → 24 → 12 ──────────────────────────────────────────
    x = cbr(inp, C, 3, 2)
    if enc == 'conv':
        x = cbr(x, C, 3, 2)
        z = cbr(x, C, 3, 2, act=False)                 # latent without activation (paper)
    else:
        x = ds(x, C, 2)
        z = ds(x, C, 2, act=False)

    if tof:
        tin = keras.Input((12, 12, 2), batch_size=batch_size, name='tof12')   # [log d, valid]
        ins.append(tin)
        z = layers.Add()([z, layers.Conv2D(C, 1, name='tof_inject')(tin)])
    if state_in:
        sin = keras.Input((12, 12, C), batch_size=batch_size, name='state_in')
        ins.append(sin)
        z = layers.Add()([z, sin])

    # ── NCA: layers created ONCE, called T times (shared weights) ──────────
    perc = layers.DepthwiseConv2D(3, padding='same', name='nca_perceive')
    upd_dw = layers.DepthwiseConv2D(3, padding='same', name='nca_upd_dw')
    upd_pw = layers.Conv2D(C, 1, name='nca_upd_pw', kernel_initializer='zeros')
    s = z
    for _ in range(T):
        p = layers.Concatenate()([s, perc(s)])          # p = [x ; DW3×3(x)]
        d = upd_pw(layers.ReLU()(upd_dw(p)))
        s = layers.Add()([s, d])

    # ── PixelShuffle decoder ───────────────────────────────────────────────
    h = layers.Conv2D(64, 3, padding='same', name='dec_conv')(s)
    if shuffle == 'd2s':
        h = layers.Lambda(lambda t: __import__('tensorflow').nn.depth_to_space(t, 8), name='pixel_shuffle')(h)
    else:
        # Exact PixelShuffle as an 8×8/s8 TRANSPOSE_CONV with fixed one-hot weights
        # (DEPTH_TO_SPACE int8 v2 is not registered in the standard TFLite runtime)
        ps = layers.Conv2DTranspose(1, 8, strides=8, use_bias=False, trainable=False, name='pixel_shuffle')
        h = ps(h)
        k = np.zeros((8, 8, 1, 64), np.float32)
        for i in range(8):
            for j in range(8):
                k[i, j, 0, i * 8 + j] = 1.0
        ps.set_weights([k])
    out = layers.Activation('sigmoid', name='output')(h)
    outs = [out]
    if aux_det:
        hm = layers.Conv2D(16, 3, padding='same', activation='relu')(s)
        outs.append(layers.Conv2D(3, 1, name='det')(hm))   # [obj logit, dx, dy] per 12×12 cell
    if state_in:
        outs.append(layers.Identity(name='state_out')(s))
    return keras.Model(ins, outs if len(outs) > 1 else out, name=name)


def count_macs(model):
    """MACs of Conv2D / DepthwiseConv2D (shared layers are counted once per call)."""
    from keras import layers
    macs = 0
    for layer in model.layers:
        if not isinstance(layer, (layers.Conv2D, layers.DepthwiseConv2D)):
            continue
        k = int(np.prod(layer.kernel_size))
        for node in layer._inbound_nodes:
            out = node.output_tensors[0].shape
            cin = node.input_tensors[0].shape[-1]
            hw = out[1] * out[2]
            if isinstance(layer, layers.DepthwiseConv2D):
                macs += hw * cin * k
            else:
                macs += hw * cin * k * out[-1]
    return int(macs)
