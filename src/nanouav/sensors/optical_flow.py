"""
_02_optical_flow/optical_flow.py — Lucas-Kanade + divergencia REAL.

Corrige: antes 'divergencia' = mean(|vx + vy|), que responde a cualquier
traslación/rotación de cámara. Ahora se ajusta por mínimos cuadrados un modelo
de flujo afín  v = A·(p − c) + t  y la divergencia es tr(A) = ∂vx/∂x + ∂vy/∂y.
La traslación (t) y la rotación (parte antisimétrica de A) no la afectan.

Para un plano frontal acercándose: tr(A) ≈ 2 / TTC  [1/frame]
  → TTC_s = 2 / (div · fps).   div > 0 = expansión (acercamiento).
"""
import cv2
import numpy as np

FPS_DEFAULT = 10.0            # cámara de Gazebo
TTC_TRIGGER_S = 3.0           # disparar CNN si el tiempo a colisión < 3 s
DIV_THRESHOLD = 2.0 / (TTC_TRIGGER_S * FPS_DEFAULT)     # ≈ 0.067 /frame

_FP_PARAMS = dict(maxCorners=40, qualityLevel=0.05, minDistance=6, blockSize=5)
_LK_PARAMS = dict(winSize=(9, 9), maxLevel=2,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.02))


def affine_flow_fit(p0, p1, center=(48.0, 48.0)):
    """Ajuste afín robusto (1 iteración de rechazo). Retorna (A 2×2, t 2, residuo)."""
    c = np.asarray(center, np.float32)
    X = p0 - c
    V = p1 - p0
    M = np.column_stack([X, np.ones(len(X))])
    sol, *_ = np.linalg.lstsq(M, V, rcond=None)
    res = np.linalg.norm(M @ sol - V, axis=1)
    keep = res < max(3 * np.median(res), 0.3)
    if keep.sum() >= 4 and keep.sum() < len(X):
        sol, *_ = np.linalg.lstsq(M[keep], V[keep], rcond=None)
        res = np.linalg.norm(M[keep] @ sol - V[keep], axis=1)
    return sol[:2].T, sol[2], float(np.mean(res))


def optical_flow(prev: np.ndarray, curr: np.ndarray):
    """
    prev, curr: uint8 (96,96). Retorna (flow_pts, divergence).
    divergence = tr(A) [1/frame]; 0.0 si hay < 6 puntos rastreados.
    """
    pts0 = cv2.goodFeaturesToTrack(prev, mask=None, **_FP_PARAMS)
    if pts0 is None:
        return [], 0.0
    pts1, st, _ = cv2.calcOpticalFlowPyrLK(prev, curr, pts0, None, **_LK_PARAMS)
    g0, g1 = pts0[st == 1], pts1[st == 1]
    flow_pts = [(a[0], a[1], b[0], b[1]) for a, b in zip(g0, g1)]
    if len(g0) < 6:
        return flow_pts, 0.0
    A, _, _ = affine_flow_fit(g0, g1)
    return flow_pts, float(np.trace(A))


def ttc_seconds(divergence: float, fps: float = FPS_DEFAULT) -> float:
    return 2.0 / (divergence * fps) if divergence > 1e-6 else np.inf


def should_trigger_cnn(divergence: float, threshold: float = DIV_THRESHOLD) -> bool:
    return divergence > threshold


def draw_flow(gray: np.ndarray, flow_pts: list) -> np.ndarray:
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for x0, y0, x1, y1 in flow_pts:
        cv2.arrowedLine(vis, (int(x0), int(y0)), (int(x1), int(y1)),
                        (0, 220, 130), 1, tipLength=0.4)
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
