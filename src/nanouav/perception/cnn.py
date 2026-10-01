"""
_03_cnn/cnn.py — Interfaz de inferencia de profundidad.

  predict_depth_norm(gray, variant, depth_gt=None) → float32 (96,96) NORMALIZADA [0,1]
  predict_depth(gray, depth_gt, variant)           → float32 (96,96) en METROS (sin calibrar)

Modo real : _03_cnn/model_{variant}.tflite (fallback model.tflite) con ai-edge-litert.
Modo fake : ground truth + ruido gaussiano (solo para probar el pipeline).

Corrige:
  * Doble escalado: antes se devolvían metros y el calibrador volvía a multiplicar
    por D_MAX. Ahora la conversión vive solo en uav_config.norm_to_m/m_to_norm.
  * Offset de desnormalización: entrenamiento usa (d-0.2)/3.8; antes se invertía con ×4.0.
  * Se ignoraba 'variant' en modo real: cada variante carga su propio .tflite.
"""
import os
import numpy as np

from nanouav import paths, settings

from nanouav.config import norm_to_m, m_to_norm

# Estimaciones de literatura para el ESP32-S3 (se usan en benchmarks/energía).
# Los parámetros y la arena REALES de los modelos entrenados están en metrics.json.
VARIANTS = {
    'XS': {'ms': 55,  'fps': 18, 'jitter_us': 28,  'params_k': 15,  'arena_kb': 40},
    'S':  {'ms': 77,  'fps': 13, 'jitter_us': 42,  'params_k': 50,  'arena_kb': 80},
    'M':  {'ms': 110, 'fps': 9,  'jitter_us': 75,  'params_k': 150, 'arena_kb': 130},
    'L':  {'ms': 200, 'fps': 5,  'jitter_us': 140, 'params_k': 400, 'arena_kb': 175},
}
_SIGMA_M = {'XS': 0.18, 'S': 0.12, 'M': 0.08, 'L': 0.05, 'M_ToF': 0.05}
SRAM_LIMIT_KB = 180

_interps = {}
_real_mode = None


def _model_path(variant):
    d = str(paths.models_dir())
    p = os.path.join(d, f'model_{variant}.tflite')
    if not os.path.exists(p) and variant == 'M_ToF':
        raise FileNotFoundError('Falta _03_cnn/model_M_ToF.tflite: ejecutar r3_lnca_train.py --models M_ToF '
                                '--init-from _03_cnn/model_M.keras --install (ver RUN_LOCAL.md)')
    return p if os.path.exists(p) else os.path.join(d, 'model.tflite')


def _get_interp(variant='M'):
    global _real_mode
    if variant in _interps:
        return _interps[variant]
    path = _model_path(variant)
    it = None
    if os.path.exists(path) and not settings.flag('FAKE_CNN'):
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            try:
                from tflite_runtime.interpreter import Interpreter
            except ImportError:
                Interpreter = None
        if Interpreter is not None:
            it = Interpreter(model_path=path, num_threads=1)
            it.allocate_tensors()
            print(f'[cnn] Modo REAL · {os.path.basename(path)} ({os.path.getsize(path)//1024} KB)')
    _interps[variant] = it
    _real_mode = it is not None
    return it


def _load_interpreter(variant='M'):
    """Compatibilidad con scripts anteriores."""
    _get_interp(variant)


def _tof12(tof):
    """ToF 8×8 medido (m) → rejilla latente 12×12 [log d, válido] (misma tabla que R3)."""
    from nanouav.config import ZONE_BOXES, TOF_GRID, D_MIN, D_MAX
    edges = [ZONE_BOXES[0][k][2] for k in range(TOF_GRID)] + [ZONE_BOXES[0][TOF_GRID - 1][3]]
    idx = np.full(12, -1, int)
    for k in range(12):
        c = 8 * k + 4
        for z in range(TOF_GRID):
            if edges[z] <= c < edges[z + 1]:
                idx[k] = z
    g = np.zeros((12, 12, 2), np.float32)
    if tof is None:
        return g
    valid = (tof > D_MIN) & (tof < D_MAX - 0.3)
    for i in range(12):
        for j in range(12):
            zi, zj = idx[i], idx[j]
            if zi >= 0 and zj >= 0 and valid[zi, zj]:
                g[i, j, 0] = np.log(tof[zi, zj] / D_MIN) / np.log(D_MAX / D_MIN)
                g[i, j, 1] = 1.0
    return g


def _infer_real(it, gray, tof=None):
    ins, out = it.get_input_details(), it.get_output_details()[0]
    so, zo = out['quantization']
    for inp in ins:
        si, zi = inp['quantization']
        if len(inp['shape']) == 4 and inp['shape'][-1] == 2:          # entrada 'tof12' (M_ToF)
            x = _tof12(tof)[None]
        else:
            x = (gray.astype(np.float32) / 255.0)[None, :, :, None]
        it.set_tensor(inp['index'], np.clip(np.round(x / si + zi), -128, 127).astype(np.int8))
    it.invoke()
    y = (it.get_tensor(out['index'])[0, :, :, 0].astype(np.float32) - zo) * so
    return np.clip(y, 0.0, 1.0)


def predict_depth_norm(gray, variant='M', depth_gt=None, tof=None):
    """tof: medida 8×8 (m); solo la usan variantes con fusión temprana (p. ej. 'M_ToF')."""
    it = _get_interp(variant)
    if it is not None:
        return _infer_real(it, gray, tof)
    if depth_gt is None:
        raise RuntimeError('Modo fake requiere depth_gt (no hay modelo .tflite)')
    noisy = depth_gt + np.random.randn(*depth_gt.shape).astype(np.float32) * _SIGMA_M[variant]
    return m_to_norm(noisy)


def predict_depth(gray, depth_gt, variant='M'):
    """Metros sin calibrar (escala aprendida por la CNN)."""
    return norm_to_m(predict_depth_norm(gray, variant, depth_gt))


def is_real_mode(variant='M') -> bool:
    return _get_interp(variant) is not None


def jitter_sample(variant: str, divergence: float = 0.0) -> int:
    v = VARIANTS[variant]
    base = v['jitter_us']
    noise = int(np.random.randn() * base * 0.08)
    return max(5, base + int(abs(divergence) * base * 0.5) + noise)


def fits_in_sram(variant: str) -> bool:
    return VARIANTS[variant]['arena_kb'] <= SRAM_LIMIT_KB
