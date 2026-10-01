from dataclasses import dataclass
import numpy as np

from nanouav.sensors.optical_flow import DIV_THRESHOLD


@dataclass
class Decision:
    action: str          # 'libre' | 'precaucion' | 'evadir'
    confidence: float    # 0.0 – 1.0
    triggered_by: str    # 'none' | 'tof' | 'flow' | 'cnn'
    min_dist: float      # metros


def _classify(min_dist: float, triggered_by: str) -> Decision:
    if min_dist < 0.5:
        return Decision('evadir', 1.0 - min_dist / 0.5, triggered_by, min_dist)
    if min_dist < 1.0:
        return Decision('precaucion', 1.0 - (min_dist - 0.5) / 0.5, triggered_by, min_dist)
    return Decision('libre', 0.0, triggered_by, min_dist)


def policy_p1(depth_pred: np.ndarray) -> Decision:
    """P1: CNN siempre activa."""
    return _classify(float(depth_pred.min()), 'cnn')


def policy_p2(tof: np.ndarray, depth_pred: np.ndarray,
              tof_threshold: float = 1.5) -> Decision:
    """P2: ToF decide si ejecutar CNN."""
    tof_min = float(tof.min())
    if tof_min > tof_threshold:
        return _classify(tof_min, 'tof')
    return _classify(float(depth_pred.min()), 'cnn')


def policy_p3(divergence: float, tof: np.ndarray, depth_pred: np.ndarray,
              div_threshold: float = DIV_THRESHOLD, tof_threshold: float = 1.5) -> Decision:
    """P3: flujo óptico dispara la CNN; ToF como respaldo rápido."""
    tof_min = float(tof.min())
    if tof_min < 0.4:                          # ToF de emergencia
        return _classify(tof_min, 'tof')
    if divergence > div_threshold:             # flujo activa CNN
        return _classify(float(depth_pred.min()), 'cnn')
    if tof_min < tof_threshold:                # ToF como guarda
        return _classify(tof_min, 'tof')
    return _classify(4.0, 'none')              # sin activación → libre


ACTION_COLOR = {'libre': '\033[32m', 'precaucion': '\033[33m', 'evadir': '\033[31m'}
RESET = '\033[0m'


def fmt_decision(d: Decision) -> str:
    c = ACTION_COLOR[d.action]
    return (f"{c}[{d.action.upper():10s}]{RESET} "
            f"dist={d.min_dist:.2f}m  conf={d.confidence:.2f}  via={d.triggered_by}")
