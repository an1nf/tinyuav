"""
nanouav/benchmark/energy_model.py  —  E1-B

Energy simulator based on the real physics of the nano-UAV.

Hardware parameters:
  - Battery: 550 mAh LiPo 1S @ 3.7V = 7326 J
  - Base flight power: ~1.5W (brushed motors in hover + idle MCU)
  - Active CNN power: ~0.214W extra (65mA @ 3.3V, measured with ESP-DL)
  - Actual measured CNN latency: 0.638ms on CPU × factor 8–15 = ESP32-S3

Sources:
  - Palossi et al. IoT-J 2019: Crazyflie AI-deck power consumption
  - Zango & Lan arXiv:2601.13252: nano-UAV energy budget
  - Espressif ESP32-S3 datasheet: active consumption at 240MHz ~100mA @ 3.3V
  - ESP-DL benchmarks: +50-80mA during int8 inference with PIE
"""
import numpy as np
from dataclasses import dataclass, field

# ── Hardware constants ──────────────────────────────────────────────────
BAT_MAH      = 550.0          # mAh
BAT_V        = 3.7            # V
BAT_J        = BAT_MAH * 1e-3 * BAT_V * 3600   # 7326 J
EFFICIENCY   = 0.85           # efficiency of the flight regulator

P_MOTORS_W   = 1.17           # W motors in hover (= 1.5 - MCU)
P_MCU_IDLE_W = 0.33           # W ESP32-S3 @ 240MHz without CNN
P_TOF_W      = 0.050          # W VL53L5CX @ 15Hz
P_IMU_W      = 0.005          # W ICM-42688 always on
P_FLOW_W     = 0.008          # W PMW3901 (hw optical flow, optional)

# CNN: extra consumption during active inference
P_CNN_ACTIVE_W = 0.065 * 3.3  # 65mA @ 3.3V = 0.2145W

# CNN latency measured in E0-C (your CPU)
LAT_CNN_CPU_S  = 0.638e-3     # s
# CPU → ESP32-S3 scale factor
SCALE_LO, SCALE_HI, SCALE_MID = 8.0, 15.0, 11.5

# Mechanical reaction time (from decision to effective braking)
T_REACT_S = 0.050             # 50ms (braking current + inertia)

# CNN policy: fraction of frames with the CNN active (from the E0-C measurement)
CNN_DUTY = {'P1': 1.00, 'P2': 0.80, 'P3': 0.35}

# Model variants
VARIANTS = {
    'XS': {'fps': 18, 'arena_kb': 40},
    'S':  {'fps': 13, 'arena_kb': 80},
    'M':  {'fps': 9,  'arena_kb': 130},
    'L':  {'fps': 5,  'arena_kb': 175},
}


@dataclass
class EnergyBudget:
    """Complete energy budget for one configuration."""
    variant:       str
    policy:        str
    scale_factor:  float

    # Powers (W)
    p_motors:      float = field(init=False)
    p_mcu:         float = field(init=False)
    p_sensors:     float = field(init=False)
    p_cnn_avg:     float = field(init=False)
    p_total:       float = field(init=False)

    # Endurance
    endurance_s:   float = field(init=False)
    endurance_min: float = field(init=False)

    # Maximum safe speed (m/s)
    v_max_evasion:    float = field(init=False)   # 0.5m threshold
    v_max_precaution: float = field(init=False)   # 1.0m threshold

    def __post_init__(self):
        v    = VARIANTS[self.variant]
        fps  = v['fps']

        # CNN latency on the ESP32-S3
        lat_esp_s = LAT_CNN_CPU_S * self.scale_factor

        # Duty cycle: fraction of the time the CNN is running
        duty_cnn  = lat_esp_s * fps   # [0, 1]
        cnn_frac  = CNN_DUTY[self.policy]
        p_cnn_eff = P_CNN_ACTIVE_W * duty_cnn * cnn_frac

        self.p_motors   = P_MOTORS_W
        self.p_mcu      = P_MCU_IDLE_W
        self.p_sensors  = P_TOF_W + P_IMU_W
        self.p_cnn_avg  = p_cnn_eff
        self.p_total    = (self.p_motors + self.p_mcu +
                           self.p_sensors + self.p_cnn_avg) / EFFICIENCY

        self.endurance_s   = BAT_J / self.p_total
        self.endurance_min = self.endurance_s / 60

        # Maximum safe speed:
        # d_threshold = v * (1/fps + T_react + lat_esp)  → v = d / total_time
        t_pipeline = 1.0/fps + T_REACT_S + lat_esp_s
        self.v_max_evasion    = 0.5 / t_pipeline
        self.v_max_precaution = 1.0 / t_pipeline


def compute_all(scale: float = SCALE_MID) -> dict:
    """Computes the EnergyBudget for all variants and policies."""
    results = {}
    for v in VARIANTS:
        results[v] = {}
        for p in CNN_DUTY:
            results[v][p] = EnergyBudget(variant=v, policy=p, scale_factor=scale)
    return results


def baseline() -> float:
    """Endurance without CNN (motors + MCU + sensors only)."""
    p = (P_MOTORS_W + P_MCU_IDLE_W + P_TOF_W + P_IMU_W) / EFFICIENCY
    return BAT_J / p / 60


def endurance_vs_speed(variant: str, policy: str,
                       speed_range: np.ndarray,
                       scale: float = SCALE_MID) -> tuple:
    """
    For a range of flight speeds, computes:
    - Whether it is safe to fly at that speed (d_safe < 0.5m avoidance threshold)
    - The endurance (constant, it does not depend on speed in this model)
    Returns (safe_mask bool, endurance_min float)
    """
    v   = VARIANTS[variant]
    lat = LAT_CNN_CPU_S * scale
    t_p = 1.0 / v['fps'] + T_REACT_S + lat
    d_safe = speed_range * t_p
    safe   = d_safe < 0.5
    budget = EnergyBudget(variant=variant, policy=policy, scale_factor=scale)
    return safe, budget.endurance_min


def sensitivity_analysis(variant: str = 'M', policy: str = 'P3') -> dict:
    """
    Sensitivity analysis: how the endurance changes when varying
    the CPU→ESP32-S3 scale factor (model uncertainty).
    """
    scales  = np.linspace(SCALE_LO, SCALE_HI, 50)
    results = []
    for s in scales:
        b = EnergyBudget(variant=variant, policy=policy, scale_factor=s)
        results.append({
            'scale':     s,
            'lat_ms':    LAT_CNN_CPU_S * s * 1000,
            'endurance': b.endurance_min,
            'v_max':     b.v_max_evasion,
            'p_total':   b.p_total,
        })
    return {'scales': scales, 'results': results}
