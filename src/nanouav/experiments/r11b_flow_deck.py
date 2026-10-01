"""
R11b · "Flow deck frontal" geométrico: la mitad inferior de la cámara ve el suelo (R7).
Flujo óptico denso (Farnebäck) sobre píxeles de suelo → cada píxel se proyecta al plano z = 0
con altura (baro/ToF) y actitud (IMU) conocidas → se busca la velocidad (vx, vy) del cuerpo que
mejor explica el flujo, compensando el giro con el giróscopo. Coste robusto recortado (70 %)
para rechazar píxeles que caen sobre obstáculos. Sin aprendizaje.

Uso: python r11b_flow_deck.py data/r11_full_a.npz,data/r11_full_b.npz
"""
import os, sys, json, math
import numpy as np
import cv2
from nanouav import paths
ROOT = str(paths.repo_root())
from nanouav.scene.renderer import camera_rays, body_rot, rot_y
from nanouav.config import CAM_PITCH, CAM_OFFSET, focal_px, IMG_W, IMG_H

DT = 0.1
F = focal_px()
RAYS, _ = camera_rays()


def cam_pose(t_xy, yaw, z, pitch, roll):
    Rb = body_rot(yaw, pitch, roll)
    Rc = Rb @ rot_y(CAM_PITCH)
    O = np.array([t_xy[0], t_xy[1], z]) + Rb @ np.asarray(CAM_OFFSET)
    return O, Rc


def project(X, O, Rc):
    d = (X - O) @ Rc                                  # = Rc^T (X − O) por filas → marco cuerpo-cámara
    ok = d[:, 0] > 1e-3
    u = IMG_W / 2 + F * (-d[:, 1] / np.maximum(d[:, 0], 1e-3)) - 0.5
    v = IMG_H / 2 + F * (-d[:, 2] / np.maximum(d[:, 0], 1e-3)) - 0.5
    return np.stack([u, v], 1), ok


def estimate(frames, r, z, pitch, roll, trim=0.7):
    """frames: (96,96,K) uint8. Devuelve (vx, vy) y el nº de píxeles útiles."""
    O0, Rc0 = cam_pose((0, 0), 0.0, z, pitch, roll)
    D = RAYS.reshape(-1, 3) @ Rc0.T
    tg = np.where(D[:, 2] < -0.05, -O0[2] / np.minimum(D[:, 2], -1e-6), np.inf)
    ground = np.isfinite(tg) & (tg < 3.5)
    X = O0 + D * np.where(ground, tg, 0)[:, None]
    vv, uu = np.mgrid[0:IMG_H, 0:IMG_W]
    pix = np.stack([uu.ravel(), vv.ravel()], 1).astype(np.float64)
    pairs = []
    for k in range(frames.shape[2] - 1):
        a, b = frames[:, :, k], frames[:, :, k + 1]
        flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 9, 3, 5, 1.1, 0).reshape(-1, 2)
        gx = cv2.Sobel(a, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(a, cv2.CV_32F, 0, 1)
        tex = (np.hypot(gx, gy).ravel() > 20)
        m = ground & tex
        if m.sum() < 30:
            continue
        idx = np.nonzero(m)[0]
        if len(idx) > 1500:
            idx = idx[:: len(idx) // 1500]
        pairs.append((X[idx], pix[idx] + flow[idx]))
    if not pairs:
        return None, 0

    def cost(vx, vy):
        c = 0.0
        for Xs, p1 in pairs:
            O1, Rc1 = cam_pose((vx * DT, vy * DT), r * DT, z, pitch, roll)
            q, ok = project(Xs, O1, Rc1)
            e = np.hypot(*(q - p1).T)
            e = np.where(ok, np.minimum(e, 5.0), 5.0)
            e.sort()
            c += e[: max(int(trim * len(e)), 1)].mean()
        return c

    best = (np.inf, 0, 0)
    for vx in np.arange(-0.5, 2.31, 0.1):
        for vy in np.arange(-1.2, 1.21, 0.1):
            c = cost(vx, vy)
            if c < best[0]:
                best = (c, vx, vy)
    _, bx, by = best
    for vx in np.arange(bx - 0.1, bx + 0.101, 0.02):
        for vy in np.arange(by - 0.1, by + 0.101, 0.02):
            c = cost(vx, vy)
            if c < best[0]:
                best = (c, vx, vy)
    return np.array(best[1:]), int(sum(len(p[0]) for p in pairs))


def main(files):
    Z = [np.load(os.path.join(ROOT, p)) for p in files.split(',')]
    G96, Y, W, A, T = (np.concatenate([z[k] for z in Z]) for k in ('G96', 'Y', 'W', 'A', 'T'))
    est, npx = [], []
    for i in range(len(Y)):
        e, n = estimate(G96[i], W[i, 0], W[i, 1], A[i, 0], A[i, 1])
        est.append(e if e is not None else np.full(2, np.nan))
        npx.append(n)
        if (i + 1) % 100 == 0:
            print(f'  {i + 1}/{len(Y)}', flush=True)
    est, npx = np.array(est), np.array(npx)
    ok = np.isfinite(est[:, 0])
    err = est[ok] - Y[ok]
    ae = np.abs(err)
    res = {'n': int(len(Y)), 'con_estimacion': int(ok.sum()),
           'rmse': np.sqrt((err ** 2).mean(0)).round(3).tolist(),
           'mediana_abs': np.median(ae, 0).round(3).tolist(),
           'p75_abs': np.percentile(ae, 75, 0).round(3).tolist(),
           'p90_abs': np.percentile(ae, 90, 0).round(3).tolist(),
           'frac_ambos_ejes_<=0.05': float(np.mean((ae < 0.05).all(1))),
           'frac_ambos_ejes_<=0.10': float(np.mean((ae < 0.10).all(1))),
           'baseline_std': np.std(Y, 0).round(3).tolist()}
    # por textura del suelo observada (nº de píxeles útiles)
    for lo, hi in ((0, 300), (300, 1500), (1500, 10 ** 6)):
        m = (npx[ok] >= lo) & (npx[ok] < hi)
        if m.any():
            res[f'mediana_abs_px[{lo},{hi})'] = np.median(ae[m], 0).round(3).tolist() + [int(m.sum())]
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(str(paths.results_dir()), 'r11b_flow_deck.json'), 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1])
