"""
_05_benchmark/energy_model.py  —  E1-B

Simulador energético basado en física real del nano-UAV.

Parámetros de hardware:
  - Batería: 550 mAh LiPo 1S @ 3.7V = 7326 J
  - Potencia base vuelo: ~1.5W (motores brushed hover + MCU idle)
  - Potencia CNN activa: ~0.214W adicionales (65mA @ 3.3V, medido ESP-DL)
  - Latencia CNN real medida: 0.638ms en CPU × factor 8–15 = ESP32-S3

Fuentes:
  - Palossi et al. IoT-J 2019: consumo AI-deck Crazyflie
  - Zango & Lan arXiv:2601.13252: presupuesto energético nano-UAVs
  - Espressif ESP32-S3 datasheet: consumo activo 240MHz ~100mA @ 3.3V
  - ESP-DL benchmarks: +50-80mA durante inferencia int8 con PIE
"""
import numpy as np
from dataclasses import dataclass, field

# ── Constantes de hardware ────────────────────────────────────────────────
BAT_MAH      = 550.0          # mAh
BAT_V        = 3.7            # V
BAT_J        = BAT_MAH * 1e-3 * BAT_V * 3600   # 7326 J
EFFICIENCY   = 0.85           # eficiencia del regulador de vuelo

P_MOTORS_W   = 1.17           # W motores en hover (= 1.5 - MCU)
P_MCU_IDLE_W = 0.33           # W ESP32-S3 @ 240MHz sin CNN
P_TOF_W      = 0.050          # W VL53L5CX @ 15Hz
P_IMU_W      = 0.005          # W ICM-42688 siempre activo
P_FLOW_W     = 0.008          # W PMW3901 (flujo óptico hw, opcional)

# CNN: consumo adicional durante inferencia activa
P_CNN_ACTIVE_W = 0.065 * 3.3  # 65mA @ 3.3V = 0.2145W

# Latencia CNN medida en E0-C (tu CPU)
LAT_CNN_CPU_S  = 0.638e-3     # s
# Factor de escala CPU → ESP32-S3
SCALE_LO, SCALE_HI, SCALE_MID = 8.0, 15.0, 11.5

# Tiempo de reacción mecánica (desde decisión hasta frenada efectiva)
T_REACT_S = 0.050             # 50ms (corriente de frenada + inercia)

# Política CNN: fracción de frames con CNN activa (de E0-C medición)
CNN_DUTY = {'P1': 1.00, 'P2': 0.80, 'P3': 0.35}

# Variantes de modelo
VARIANTS = {
    'XS': {'fps': 18, 'arena_kb': 40},
    'S':  {'fps': 13, 'arena_kb': 80},
    'M':  {'fps': 9,  'arena_kb': 130},
    'L':  {'fps': 5,  'arena_kb': 175},
}


@dataclass
class EnergyBudget:
    """Presupuesto energético completo para una configuración."""
    variant:       str
    policy:        str
    scale_factor:  float

    # Potencias (W)
    p_motors:      float = field(init=False)
    p_mcu:         float = field(init=False)
    p_sensors:     float = field(init=False)
    p_cnn_avg:     float = field(init=False)
    p_total:       float = field(init=False)

    # Autonomía
    endurance_s:   float = field(init=False)
    endurance_min: float = field(init=False)

    # Velocidad máxima segura (m/s)
    v_max_evasion:    float = field(init=False)   # umbral 0.5m
    v_max_precaution: float = field(init=False)   # umbral 1.0m

    def __post_init__(self):
        v    = VARIANTS[self.variant]
        fps  = v['fps']

        # Latencia CNN en ESP32-S3
        lat_esp_s = LAT_CNN_CPU_S * self.scale_factor

        # Duty cycle: fracción del tiempo que la CNN está ejecutando
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

        # Velocidad máxima segura:
        # d_threshold = v * (1/fps + T_react + lat_esp)  → v = d / tiempo_total
        t_pipeline = 1.0/fps + T_REACT_S + lat_esp_s
        self.v_max_evasion    = 0.5 / t_pipeline
        self.v_max_precaution = 1.0 / t_pipeline


def compute_all(scale: float = SCALE_MID) -> dict:
    """Computa EnergyBudget para todas las variantes y políticas."""
    results = {}
    for v in VARIANTS:
        results[v] = {}
        for p in CNN_DUTY:
            results[v][p] = EnergyBudget(variant=v, policy=p, scale_factor=scale)
    return results


def baseline() -> float:
    """Autonomía sin CNN (solo motores + MCU + sensores)."""
    p = (P_MOTORS_W + P_MCU_IDLE_W + P_TOF_W + P_IMU_W) / EFFICIENCY
    return BAT_J / p / 60


def endurance_vs_speed(variant: str, policy: str,
                       speed_range: np.ndarray,
                       scale: float = SCALE_MID) -> tuple:
    """
    Para un rango de velocidades de vuelo, calcula:
    - Si es seguro volar a esa velocidad (d_safe < umbral evasión 0.5m)
    - La autonomía (constante, no depende de velocidad en este modelo)
    Retorna (safe_mask bool, endurance_min float)
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
    Análisis de sensibilidad: cómo cambia la autonomía al variar
    el factor de escala CPU→ESP32-S3 (incertidumbre del modelo).
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
