"""
nanouav/perception/cnn.py — Depth inference interface.

  predict_depth_norm(gray, variant, depth_gt=None) → float32 (96,96) NORMALIZED [0,1]
  predict_depth(gray, depth_gt, variant)           → float32 (96,96) in METERS (uncalibrated)

Real mode : models/model_{variant}.tflite (fallback model.tflite) with ai-edge-litert.
Fake mode : ground truth + Gaussian noise (only for testing the pipeline).

Fixes:
  * Double scaling: previously meters were returned and the calibrator multiplied
    by D_MAX again. Now the conversion lives only in nanouav.config.norm_to_m/m_to_norm.
  * Denormalization offset: training uses (d-0.2)/3.8; previously it was inverted with ×4.0.
  * 'variant' was ignored in real mode: each variant now loads its own .tflite.
"""
import os
import numpy as np

from nanouav import paths, settings

from nanouav.config import norm_to_m, m_to_norm

# Literature estimates for the ESP32-S3 (used in benchmarks/energy).
# The REAL parameter counts and arena sizes of the trained models are in models/metrics.json.
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
    if os.path.exists(p):
        return p
    if variant == 'M_ToF':
        raise FileNotFoundError('Missing models/model_M_ToF.tflite: run `nanouav run r03 --models M_ToF '
                                '--init-from models/model_M.keras --install` (see docs/artefactos.md)')
    alias = os.path.join(d, 'model.tflite')            # legacy: model.tflite = M
    if variant != 'M' and os.path.exists(alias):
        # models are installed but not this one: never fall back to M silently (results would be
        # labelled with a variant that did not run)
        raise FileNotFoundError(f'Missing models/model_{variant}.tflite (installed: '
                                f'{", ".join(sorted(f[6:-7] for f in os.listdir(d) if f.startswith("model_") and f.endswith(".tflite")))})')
    return alias


class TFLiteRuntimeMissing(ImportError):
    """A trained .tflite model exists but no TFLite runtime can be imported.

    Subclass of ImportError on purpose: callers that catch RuntimeError to fall back to ToF-only
    flight (e.g. the Gazebo node) must NOT silently swallow this — it means the experiment would
    otherwise run with fake depth while being labelled as 'cnn'.
    """


def _get_interp(variant='M'):
    global _real_mode
    if variant in _interps:
        return _interps[variant]
    it = None
    path = None if settings.flag('FAKE_CNN') else _model_path(variant)   # fake mode never needs a file
    if path is not None and os.path.exists(path):
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            try:
                from tflite_runtime.interpreter import Interpreter
            except ImportError:
                raise TFLiteRuntimeMissing(
                    f'{path} exists but neither ai_edge_litert nor tflite_runtime can be imported. '
                    'Install it (pip install ai-edge-litert) or run with --fake-cnn / NANOUAV_FAKE_CNN=1 '
                    'to use ground-truth depth on purpose.') from None
        if Interpreter is not None:
            it = Interpreter(model_path=path, num_threads=1)
            it.allocate_tensors()
            print(f'[cnn] REAL mode · {os.path.basename(path)} ({os.path.getsize(path)//1024} KB)')
    _interps[variant] = it
    _real_mode = it is not None
    return it


def _load_interpreter(variant='M'):
    """Backward compatibility with older scripts."""
    _get_interp(variant)


def _tof12(tof):
    """Measured 8×8 ToF (m) → 12×12 latent grid [log d, valid] (same table as R3)."""
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
        if len(inp['shape']) == 4 and inp['shape'][-1] == 2:          # 'tof12' input (M_ToF)
            x = _tof12(tof)[None]
        else:
            x = (gray.astype(np.float32) / 255.0)[None, :, :, None]
        it.set_tensor(inp['index'], np.clip(np.round(x / si + zi), -128, 127).astype(np.int8))
    it.invoke()
    y = (it.get_tensor(out['index'])[0, :, :, 0].astype(np.float32) - zo) * so
    return np.clip(y, 0.0, 1.0)


def predict_depth_norm(gray, variant='M', depth_gt=None, tof=None):
    """tof: 8×8 measurement (m); only used by early-fusion variants (e.g. 'M_ToF')."""
    it = _get_interp(variant)
    if it is not None:
        return _infer_real(it, gray, tof)
    if depth_gt is None:
        raise RuntimeError('Fake mode requires depth_gt (no .tflite model available)')
    noisy = depth_gt + np.random.randn(*depth_gt.shape).astype(np.float32) * _SIGMA_M.get(variant.split('_thin')[0], 0.08)
    return m_to_norm(noisy)


def predict_depth(gray, depth_gt, variant='M'):
    """Uncalibrated meters (scale learned by the CNN)."""
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
