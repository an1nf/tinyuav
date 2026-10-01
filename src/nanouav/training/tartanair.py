"""
Depth CNN training with the TartanAir dataset.

Run:
    conda activate nano_uav
    pip install tensorflow-cpu -q   # if not already installed
    nanouav run train-tartanair

Produces:
    models/model.tflite       — replaces the random-weights model
    results/train_loss.png    — loss curve
    results/train_preview.png — visual sample of predictions
"""
import os, sys, json, zipfile, time
import numpy as np
import matplotlib
from nanouav import paths
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT      = str(paths.repo_root())
ZIP_PATH  = os.path.join(str(paths.data_dir()), 'tartanair', 'moge-dataset-tartanair.zip')
DATA_DIR  = os.path.join(str(paths.data_dir()), 'tartanair', 'moge_dataset_tartanair')
OUT_TFL   = os.path.join(str(paths.models_dir()), 'model.tflite')
ASSETS = str(paths.results_dir())
os.makedirs(ASSETS, exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────
IMG_H, IMG_W = 96, 96
DEPTH_MIN    = 0.2    # meters
DEPTH_MAX    = 4.0    # meters (anything outside is clipped)
DEPTH_MM_MAX = 20000  # millimeters — uint16 values above this are discarded
BATCH_SIZE   = 16
EPOCHS       = 60
LR           = 3e-3
VAL_SPLIT    = 0.15
MAX_SAMPLES  = 6000   # use all; lower it for a quick test


# ── 1. Extract if needed ──────────────────────────────────────────────────
def maybe_extract():
    if os.path.isdir(DATA_DIR) and len(os.listdir(DATA_DIR)) > 10:
        print(f'[data] Directory already exists ({len(os.listdir(DATA_DIR))} entries)')
        return
    if not os.path.exists(ZIP_PATH):
        sys.exit(f'No se encuentra {ZIP_PATH}\nDescarga primero:\n'
                 f'  kaggle datasets download -d peters777/moge-dataset-tartanair '
                 f'--path data/tartanair')
    print(f'[data] Extracting {ZIP_PATH} ...')
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(os.path.dirname(DATA_DIR))
    print(f'[data] Extracted to {DATA_DIR}')


# ── 2. Load samples ───────────────────────────────────────────────────────
def _fov_crop_box(size, src_hfov=np.pi / 2, dst_hfov=1.047):
    w, h = size
    f = np.tan(dst_hfov / 2) / np.tan(src_hfov / 2)
    cw, ch = int(w * f), int(h * f)
    return ((w - cw) // 2, (h - ch) // 2, (w - cw) // 2 + cw, (h - ch) // 2 + ch)


def load_sample(sample_dir: str):
    """
    Returns (gray float32 96x96, depth float32 96x96), or None on failure.
    gray:  [0, 1]
    depth: [0, 1]  where 1 = DEPTH_MAX meters
    """
    from PIL import Image
    try:
        img_path = os.path.join(sample_dir, 'image.jpg')
        dep_path = os.path.join(sample_dir, 'depth.png')

        # Image: RGB → gray → 96x96 → [0,1]
        # TartanAir: HFOV 90° → central crop to 60° (drone camera, nanouav.config.CAM_HFOV)
        img = Image.open(img_path).convert('L')
        box = _fov_crop_box(img.size)
        img = img.crop(box).resize((IMG_W, IMG_H), Image.BILINEAR)
        gray = np.array(img, dtype=np.float32) / 255.0

        # Depth: uint16 mm → meters → clip → normalize to [0,1]
        dep_raw = np.array(Image.open(dep_path), dtype=np.float32)
        dep_m   = dep_raw / 1000.0                          # mm → m
        dep_m   = np.clip(dep_m, DEPTH_MIN, DEPTH_MAX)

        # Resize to 96x96
        dep_img = Image.fromarray(dep_m).crop(_fov_crop_box(dep_m.shape[::-1])).resize((IMG_W, IMG_H), Image.BILINEAR)
        dep_96  = np.array(dep_img, dtype=np.float32)

        # Normalize to [0, 1]
        dep_norm = (dep_96 - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN)

        return gray, dep_norm

    except Exception as e:
        return None


def build_dataset(max_samples: int = MAX_SAMPLES):
    samples = sorted([
        os.path.join(DATA_DIR, d)
        for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    ])[:max_samples]

    print(f'[data] Loading {len(samples)} samples...')
    t0 = time.time()

    X, Y = [], []
    fails = 0
    for i, s in enumerate(samples):
        result = load_sample(s)
        if result is None:
            fails += 1
            continue
        gray, depth = result
        X.append(gray)
        Y.append(depth)
        if (i + 1) % 500 == 0:
            print(f'  {i+1}/{len(samples)}  ({time.time()-t0:.1f}s)')

    X = np.array(X, dtype=np.float32)[..., np.newaxis]  # (N, 96, 96, 1)
    Y = np.array(Y, dtype=np.float32)[..., np.newaxis]  # (N, 96, 96, 1)
    print(f'[data] {len(X)} samples loaded, {fails} failed  ({time.time()-t0:.1f}s)')
    print(f'       X: {X.shape}  Y: {Y.shape}')
    return X, Y


# ── 3. Model ──────────────────────────────────────────────────────────────
def build_model():
    import keras
    from keras import layers

    inp = keras.Input(shape=(IMG_H, IMG_W, 1), batch_size=1, name='input')

    def dw_block(x, filters, stride=1):
        x = layers.DepthwiseConv2D(3, strides=stride, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU(6.)(x)
        x = layers.Conv2D(filters, 1, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        return layers.ReLU(6.)(x)

    # Encoder
    e1 = dw_block(inp, 8,  stride=2)   # 48x48x8
    e2 = dw_block(e1,  16, stride=2)   # 24x24x16
    e3 = dw_block(e2,  32, stride=2)   # 12x12x32

    # Bottleneck
    b  = dw_block(e3,  32, stride=1)   # 12x12x32

    # Decoder with skip connections
    d1 = layers.Conv2DTranspose(16, 3, strides=2, padding='same')(b)   # 24x24x16
    d1 = layers.Add()([d1, e2])
    d1 = dw_block(d1, 16)

    d2 = layers.Conv2DTranspose(8,  3, strides=2, padding='same')(d1)  # 48x48x8
    d2 = layers.Add()([d2, e1])
    d2 = dw_block(d2, 8)

    d3 = layers.Conv2DTranspose(4,  3, strides=2, padding='same')(d2)  # 96x96x4
    out = layers.Conv2D(1, 1, activation='sigmoid', name='output')(d3)

    model = keras.Model(inp, out)
    return model


# ── 4. Loss: SSIM + L1 ───────────────────────────────────────────────────
def depth_loss(y_true, y_pred):
    """
    Combines L1 (absolute error) with an image-gradient term.
    More robust than MSE for depth: penalizes blurry edges.
    """
    import tensorflow as tf
    l1 = tf.reduce_mean(tf.abs(y_true - y_pred))

    # Gradient: enforces sharp depth edges
    dy_true, dx_true = tf.image.image_gradients(y_true)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)
    grad = tf.reduce_mean(tf.abs(dy_true - dy_pred) + tf.abs(dx_true - dx_pred))

    return l1 + 0.5 * grad


# ── 5. Evaluation metrics ─────────────────────────────────────────────────
def eval_metrics(y_true, y_pred):
    """
    y_true, y_pred: normalized [0,1] → converted to meters for interpretation.
    """
    yt = y_true * (DEPTH_MAX - DEPTH_MIN) + DEPTH_MIN
    yp = y_pred * (DEPTH_MAX - DEPTH_MIN) + DEPTH_MIN
    abs_rel = np.mean(np.abs(yt - yp) / yt)
    rmse    = np.sqrt(np.mean((yt - yp) ** 2))
    return abs_rel, rmse


# ── 6. Quantize and export TFLite ────────────────────────────────────────
def export_tflite(model, X_sample):
    import tensorflow as tf

    def rep_data():
        # Convert from [0,1] float to [-128,127] int8, simulating the real input
        for i in range(min(200, len(X_sample))):
            img = X_sample[i:i+1]           # (1, 96, 96, 1) float32 [0,1]
            img_q = (img * 255.0 - 128.0).astype(np.float32)
            yield [img_q]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_data
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.int8
    converter.inference_output_type = tf.int8

    tflite_bytes = converter.convert()
    with open(OUT_TFL, 'wb') as f:
        f.write(tflite_bytes)
    print(f'[export] {OUT_TFL}  ({len(tflite_bytes)//1024} KB)')


# ── 7. Visualization ──────────────────────────────────────────────────────
def plot_preview(model, X_val, Y_val, epoch_losses, val_losses):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    _cmap = LinearSegmentedColormap.from_list(
        'depth',
        [(0.22,0.61,0.87),(0.11,0.62,0.46),(0.94,0.62,0.15),(0.88,0.29,0.29)][::-1],
        N=256)

    DARK, BG = '#111', '#1a1a1a'

    # Loss curve
    fig, ax = plt.subplots(figsize=(8, 3), facecolor=BG)
    ax.set_facecolor(DARK)
    ax.plot(epoch_losses, color='#5DCAA5', lw=1.5, label='train')
    ax.plot(val_losses,   color='#EF9F27', lw=1.5, label='val')
    ax.set_title('Training loss', color='white', fontsize=10)
    ax.set_xlabel('epoch', color='#888'); ax.set_ylabel('loss', color='#888')
    ax.tick_params(colors='#888')
    for sp in ax.spines.values(): sp.set_edgecolor('#333')
    ax.legend(fontsize=8, facecolor='#222', edgecolor='#333', labelcolor='white')
    fig.tight_layout()
    plt.savefig(os.path.join(ASSETS, 'train_loss.png'), dpi=120,
                bbox_inches='tight', facecolor=BG)
    plt.close(fig)

    # Previews: image | depth GT | depth pred
    N_PREVIEW = 4
    idxs = np.random.choice(len(X_val), N_PREVIEW, replace=False)
    preds = model.predict(X_val[idxs], verbose=0)
    # Re-index locally: preds has shape (N_PREVIEW, 96, 96, 1)
    idxs_local = np.arange(N_PREVIEW)

    fig, axes = plt.subplots(N_PREVIEW, 3, figsize=(9, N_PREVIEW*2.5), facecolor=BG)
    fig.suptitle('Predictions vs ground truth', color='white', fontsize=11)
    cols = ['Camera 96×96', 'GT depth', 'CNN prediction']

    for row, (glob_idx, loc_idx) in enumerate(zip(idxs, idxs_local)):
        for col, ax in enumerate(axes[row]):
            ax.set_facecolor(DARK); ax.axis('off')
            if row == 0:
                ax.set_title(cols[col], color='white', fontsize=8)
            if col == 0:
                ax.imshow(X_val[glob_idx, :, :, 0], cmap='gray', vmin=0, vmax=1)
            elif col == 1:
                ax.imshow(Y_val[glob_idx, :, :, 0], cmap=_cmap, vmin=0, vmax=1)
            else:
                ax.imshow(preds[loc_idx, :, :, 0], cmap=_cmap, vmin=0, vmax=1)
                ab, rm = eval_metrics(Y_val[glob_idx:glob_idx+1], preds[loc_idx:loc_idx+1])
                ax.set_xlabel(f'AbsRel={ab:.3f}  RMSE={rm:.2f}m',
                              color='#888', fontsize=7)

    fig.tight_layout()
    plt.savefig(os.path.join(ASSETS, 'train_preview.png'), dpi=120,
                bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'[plot] results/train_loss.png  results/train_preview.png')


# ── Main ──────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # v2: training/export live in train_cnn.py (fixed int8 quantization).
    # This file only provides the TartanAir loader.
    import subprocess
    subprocess.run([sys.executable, '-m', 'nanouav.training.train_cnn', '--tartanair'] + sys.argv[1:], check=True)
