"""
R11 · Estimación de velocidad aprendida con frames apilados como canales (TinyNav, arXiv 2603.11071:
20 frames de profundidad como canales de una CNN 2D en ESP32, sin capas recurrentes).

Pregunta: ¿puede una red tiny sobre K frames de CÁMARA y/o ToF 8×8 (+ giróscopo) estimar la
velocidad horizontal del cuerpo con error ≤ 0.05 m/s (requisito derivado en R8/R9)?
Hipótesis: la mitad inferior de la imagen ve el suelo (R7) → flujo óptico sobre el suelo, con
escala dada por el ToF = un "flow deck" frontal.

Uso:
  python r11_vel_net.py gen --n 3000 --seed 0 --out data/r11_a.npz     (paralelizable por semilla)
  python r11_vel_net.py train --data data/r11_a.npz,data/r11_b.npz
"""
import os, sys, json, time, math, argparse
from nanouav import paths
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import numpy as np
ROOT = str(paths.repo_root())
from nanouav.scene.renderer import render_view, sample_world, body_rot
from nanouav.sensors.tof import tof_8x8
from nanouav.config import D_MIN, D_MAX

K, DT = 4, 0.1


def gen(n, seed, full=False):
    import cv2
    from nanouav.sim.closed_loop import random_corridor_world
    rng = np.random.default_rng(seed)
    G = np.zeros((n, 48, 48, K), np.uint8)
    T = np.zeros((n, 8, 8, K), np.float32)
    Y = np.zeros((n, 2), np.float32)            # velocidad cuerpo (vx, vy) m/s
    W = np.zeros((n, 2), np.float32)            # giróscopo yaw-rate (con ruido), altura z (baro/ToF)
    G96 = np.zeros((n if full else 0, 96, 96, K), np.uint8)
    A = np.zeros((n, 2), np.float32)            # pitch, roll (IMU)
    i, t0 = 0, time.time()
    while i < n:
        world = random_corridor_world(rng) if rng.random() < 0.5 else sample_world(rng)
        z = rng.uniform(0.4, 1.0)
        yaw = rng.normal(0, 0.25)
        vb = np.array([rng.uniform(-0.3, 2.0), rng.uniform(-1.0, 1.0)])
        r = rng.uniform(-0.8, 0.8)
        pitch, roll = rng.normal(0.03, 0.05), rng.normal(0, 0.05)
        p = np.array([rng.uniform(-1.0, 0.5), rng.uniform(-0.5, 0.5), z])
        poses = []
        for k in range(K):
            poses.append((p.copy(), yaw))
            c, s_ = math.cos(yaw), math.sin(yaw)
            p[:2] += np.array([c * vb[0] - s_ * vb[1], s_ * vb[0] + c * vb[1]]) * DT
            yaw += r * DT
        if min(world.clearance(q) for q, _ in poses) < 0.25:
            continue
        for k, (q, yw) in enumerate(poses):
            g, d = render_view(world, q, yw, pitch, roll, rng=rng)
            G[i, :, :, k] = cv2.resize(g, (48, 48), interpolation=cv2.INTER_AREA)
            if full:
                G96[i, :, :, k] = g
            T[i, :, :, k] = tof_8x8(d, rng=rng)
        Y[i] = vb
        W[i] = [r + rng.normal(0, 0.02), z + rng.normal(0, 0.02)]
        A[i] = [pitch, roll]
        i += 1
        if i % 500 == 0:
            print(f'  {i}/{n} ({time.time() - t0:.0f}s)', flush=True)
    return (G, T, Y, W, G96, A) if full else (G, T, Y, W)


def build(kind):
    import keras
    from keras import layers
    ins, feats = [], []
    gyro = keras.Input((2,), name='imu')
    ins.append(gyro)
    if kind in ('gray', 'both'):
        g = keras.Input((48, 48, K), name='gray')
        ins.insert(0, g)
        x = layers.Conv2D(16, 3, strides=2, padding='same', activation='relu')(g)     # 24
        x = layers.Conv2D(24, 3, strides=2, padding='same', activation='relu')(x)     # 12
        x = layers.Conv2D(32, 3, strides=2, padding='same', activation='relu')(x)     # 6
        x = layers.Conv2D(16, 1, activation='relu')(x)
        feats.append(layers.Flatten()(x))
    if kind in ('tof', 'both'):
        t = keras.Input((8, 8, K), name='tof')
        ins.insert(len(ins) - 1, t)
        y = layers.Conv2D(16, 3, padding='same', activation='relu')(t)
        y = layers.Conv2D(16, 3, strides=2, padding='same', activation='relu')(y)     # 4
        feats.append(layers.Flatten()(y))
    h = layers.Concatenate()(feats + [gyro])
    h = layers.Dense(64, activation='relu')(h)
    h = layers.Dense(32, activation='relu')(h)
    out = layers.Dense(2, name='v')(h)
    return keras.Model(ins, out, name=f'vel_{kind}')


def prep(G, T, W):
    g = G.astype(np.float32) / 255.0
    g = g - g.mean(axis=(1, 2), keepdims=True)                         # invarianza a brillo
    t = np.log(np.clip(T, D_MIN, D_MAX) / D_MIN) / np.log(D_MAX / D_MIN)
    return g, t, W.astype(np.float32)


def train(files, epochs):
    import keras, tensorflow as tf
    tf.random.set_seed(0)
    Z = [np.load(p) for p in files]
    G, T, Y, W = (np.concatenate([z[k] for z in Z]) for k in ('G', 'T', 'Y', 'W'))
    n = len(Y)
    nte = n // 6
    g, t, w = prep(G, T, W)
    idx = np.random.default_rng(1).permutation(n)
    te, tr = idx[:nte], idx[nte:]
    res = {'n_train': int(len(tr)), 'n_test': int(nte),
           'baseline_std': np.std(Y[te], 0).round(3).tolist()}
    for kind in ('tof', 'gray', 'both'):
        m = build(kind)
        feed = lambda ix: ([g[ix]] if kind != 'tof' else []) + ([t[ix]] if kind != 'gray' else []) + [w[ix]]
        m.compile(optimizer=keras.optimizers.Adam(keras.optimizers.schedules.CosineDecay(2e-3, epochs * (len(tr) // 64 + 1))),
                  loss='huber')
        t0 = time.time()
        m.fit(feed(tr), Y[tr], batch_size=64, epochs=epochs, verbose=0, validation_split=0.1)
        pr = m.predict(feed(te), verbose=0)
        err = pr - Y[te]
        rmse = np.sqrt((err ** 2).mean(0))
        # subconjunto "estructura cercana": algún ToF < 2.5 m en el último frame
        near = (T[te, :, :, -1] < 2.5).reshape(nte, -1).any(1)
        res[kind] = {'params': int(m.count_params()), 'rmse_vx': float(rmse[0]), 'rmse_vy': float(rmse[1]),
                     'mae_vx': float(np.abs(err[:, 0]).mean()), 'mae_vy': float(np.abs(err[:, 1]).mean()),
                     'rmse_near': np.sqrt((err[near] ** 2).mean(0)).round(3).tolist(),
                     'rmse_far': np.sqrt((err[~near] ** 2).mean(0)).round(3).tolist() if (~near).any() else None,
                     'p90_abs': np.percentile(np.abs(err), 90, axis=0).round(3).tolist(),
                     'train_s': time.time() - t0}
        print(kind, json.dumps(res[kind]), flush=True)
        m.save(os.path.join(str(paths.models_dir()), f'r11_vel_{kind}.keras'))
    json.dump(res, open(os.path.join(str(paths.results_dir()), 'r11_vel_net.json'), 'w'), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd')
    ap.add_argument('--n', type=int, default=3000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='data/r11_a.npz')
    ap.add_argument('--data', default='data/r11_a.npz')
    ap.add_argument('--epochs', type=int, default=40)
    a = ap.parse_args()
    if a.cmd == 'genfull':
        os.makedirs(str(paths.data_dir()), exist_ok=True)
        G, T, Y, W, G96, A = gen(a.n, a.seed, full=True)
        np.savez_compressed(os.path.join(ROOT, a.out), G=G, T=T, Y=Y, W=W, G96=G96, A=A)
    elif a.cmd == 'gen':
        os.makedirs(str(paths.data_dir()), exist_ok=True)
        G, T, Y, W = gen(a.n, a.seed)
        np.savez_compressed(os.path.join(ROOT, a.out), G=G, T=T, Y=Y, W=W)
    else:
        train([os.path.join(ROOT, p) for p in a.data.split(',')], a.epochs)
