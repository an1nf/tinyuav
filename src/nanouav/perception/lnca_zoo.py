"""
_03_cnn/lnca_zoo.py — Latent Neural Cellular Automata (LNCA) de profundidad, adaptada a 96×96.

Basada en Nunn & Nikoubin, SN Computer Science 7:755 (2026):
  encoder (3 conv s2, 8× abajo) → T pasos NCA con pesos compartidos sobre la
  rejilla latente → decoder PixelShuffle (conv3×3 → depth_to_space ×8).

Adaptaciones propias (no están en el paper):
  * Entrada gris 96×96 → rejilla latente 12×12 (celda = 8 px ≈ 5°), casi 1:1 con
    las zonas del ToF 8×8 (40° / 8 = 5°/zona).
  * 'tof=True': fusión temprana. Las 64 zonas ToF (log-profundidad + máscara de
    validez), re-muestreadas a la rejilla 12×12 FUERA de la red (tabla fija en el
    MCU), se suman al estado latente con una conv 1×1. La NCA propaga la escala
    métrica del centro (40°) hacia la periferia (60°) con sus convs 3×3 laterales.
  * 'aux_det=True': cabeza de centro de objetivo (heatmap 12×12, estilo anchor-free
    de FeatherYOLO — Su et al., EAAI 177:114890) que comparte encoder y NCA.
  * El estado puede inicializarse con el del frame anterior (modo recurrente,
    'state_in=True'): x0 = z + x_prev, lo que permite T=1 por frame.

Todas las ops son TFLite-Micro: CONV_2D, DEPTHWISE_CONV_2D, ADD, CONCATENATION,
TRANSPOSE_CONV (PixelShuffle one-hot), LOGISTIC, RELU.
"""
import numpy as np

LNCA_CONFIGS = {
    #          C   hidden  T   encoder
    'N24':  {'C': 24, 'T': 4, 'enc': 'conv'},
    'N32':  {'C': 32, 'T': 4, 'enc': 'conv'},      # ≈ paper (40.8 k con RGB)
    'N32ds': {'C': 32, 'T': 4, 'enc': 'ds'},       # encoder depthwise-separable
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
        z = cbr(x, C, 3, 2, act=False)                 # latente sin activación (paper)
    else:
        x = ds(x, C, 2)
        z = ds(x, C, 2, act=False)

    if tof:
        tin = keras.Input((12, 12, 2), batch_size=batch_size, name='tof12')   # [log d, válido]
        ins.append(tin)
        z = layers.Add()([z, layers.Conv2D(C, 1, name='tof_inject')(tin)])
    if state_in:
        sin = keras.Input((12, 12, C), batch_size=batch_size, name='state_in')
        ins.append(sin)
        z = layers.Add()([z, sin])

    # ── NCA: capas creadas UNA vez, llamadas T veces (pesos compartidos) ──
    perc = layers.DepthwiseConv2D(3, padding='same', name='nca_perceive')
    upd_dw = layers.DepthwiseConv2D(3, padding='same', name='nca_upd_dw')
    upd_pw = layers.Conv2D(C, 1, name='nca_upd_pw', kernel_initializer='zeros')
    s = z
    for _ in range(T):
        p = layers.Concatenate()([s, perc(s)])          # p = [x ; DW3×3(x)]
        d = upd_pw(layers.ReLU()(upd_dw(p)))
        s = layers.Add()([s, d])

    # ── decoder PixelShuffle ───────────────────────────────────────────────
    h = layers.Conv2D(64, 3, padding='same', name='dec_conv')(s)
    if shuffle == 'd2s':
        h = layers.Lambda(lambda t: __import__('tensorflow').nn.depth_to_space(t, 8), name='pixel_shuffle')(h)
    else:
        # PixelShuffle exacto como TRANSPOSE_CONV 8×8/s8 con pesos one-hot fijos
        # (DEPTH_TO_SPACE int8 v2 no está registrado en el runtime TFLite estándar)
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
        outs.append(layers.Conv2D(3, 1, name='det')(hm))   # [obj logit, dx, dy] por celda 12×12
    if state_in:
        outs.append(layers.Identity(name='state_out')(s))
    return keras.Model(ins, outs if len(outs) > 1 else out, name=name)


def count_macs(model):
    """MACs de Conv2D / DepthwiseConv2D (las capas compartidas se cuentan por cada llamada)."""
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
