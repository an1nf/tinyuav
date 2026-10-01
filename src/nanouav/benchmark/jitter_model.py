"""
_05_benchmark/jitter_model.py  —  E1-A

Modelo de jitter del PID basado en física del ESP32-S3 dual-core.

Reemplaza jitter_sample() de cnn.py (modelo lineal empírico)
con una simulación de Monte Carlo de los fenómenos reales:

  1. Cambio de contexto FreeRTOS (determinista + variación térmica)
  2. Contención del bus AHB compartido entre núcleos
  3. Cache thrashing: la CNN evicta líneas de caché del núcleo 0
  4. Interrupción DMA de cámara DVP
  5. Interrupción WiFi/BT del radio compartido (spike raro)

Parámetros de literatura:
  - T_CTX = 3.2 µs  (Fenwick et al., FreeRTOS ESP32 latency measurements)
  - BUS_FREQ = 120 MHz AHB (ESP32-S3 TRM, sección 3.3)
  - CACHE_LINE = 64 bytes, L1 = 32 KB por núcleo (ESP32-S3 TRM)
  - MISS_CYCLES = 20 ciclos por cache miss (Espressif perf guide)
  - WiFi IRQ: tasa ~0.8/s, duración ~120 µs (medido en esp-idf examples)

Todas las distribuciones son falsificables con E3-A (medición con GPIO + analizador lógico).
"""
import numpy as np
from dataclasses import dataclass

# ── Constantes del sistema (ESP32-S3) ─────────────────────────────────────
F_CPU_MHZ     = 240.0
F_AHB_MHZ     = 120.0            # bus compartido entre núcleos
CACHE_SIZE_KB = 32.0             # L1 por núcleo
CACHE_LINE_B  = 64               # bytes por línea de caché
MISS_CYCLES   = 20               # ciclos penalización por cache miss
MISS_US       = MISS_CYCLES / F_CPU_MHZ          # 0.0833 µs
BURST_CYCLES  = 16               # ciclos AHB por línea transferida
BURST_US      = BURST_CYCLES / F_AHB_MHZ         # 0.1333 µs

# Cambio de contexto FreeRTOS
T_CTX_US      = 3.2              # µs (determinista)
T_CTX_SIGMA   = 0.8              # µs (variación térmica del oscilador)

# DMA de cámara DVP
DMA_FRAME_KB  = 96 * 96 / 1024  # 9 KB por frame
DMA_BW_US_KB  = 0.8             # µs por KB transferido por DMA
DMA_FRAME_US  = DMA_FRAME_KB * DMA_BW_US_KB  # ~7.2 µs por frame

# Interrupciones WiFi/BT (radio compartido ESP32-S3)
WIFI_IRQ_RATE  = 0.00005         # por µs (= ~50 Hz, modo telemetría activa)
WIFI_IRQ_US    = 120.0           # duración media de la IRQ WiFi

# Configuraciones de prioridad FreeRTOS
# Con prioridad correcta (PID = max), el jitter es el modelo de abajo.
# Sin prioridad correcta, añadir ~170 µs de penalización.
PRIORITY_PENALTY_US = 170.0      # si PID no tiene prioridad máxima

# PID: número de instrucciones en la sección crítica
PID_CRITICAL_INSTR = 200


@dataclass
class JitterParams:
    """Parámetros de configuración del sistema para el modelo."""
    arena_kb:      float   # tamaño de tensor arena CNN (KB)
    inf_ms:        float   # latencia de inferencia (ms)
    fps:           float   # FPS de la CNN
    pid_priority:  str     # 'max' | 'normal' (afecta ~170 µs si no es max)
    wifi_active:   bool    # si el radio WiFi está activo


# Parámetros por variante (de literatura ESP-DL)
VARIANT_PARAMS = {
    'XS': JitterParams(arena_kb=40,  inf_ms=55,  fps=18, pid_priority='max', wifi_active=True),
    'S':  JitterParams(arena_kb=80,  inf_ms=77,  fps=13, pid_priority='max', wifi_active=True),
    'M':  JitterParams(arena_kb=130, inf_ms=110, fps=9,  pid_priority='max', wifi_active=True),
    'L':  JitterParams(arena_kb=175, inf_ms=200, fps=5,  pid_priority='max', wifi_active=True),
}


def _ahb_jitter(params: JitterParams, rng: np.random.Generator) -> float:
    """
    Contención del bus AHB.
    La CNN genera ráfagas de BURST_US µs cada vez que transfiere una línea de caché.
    Fracción del tiempo que el bus está ocupado por la CNN durante inferencia:
        busy_frac = (n_lines * BURST_US) / inf_µs
    """
    n_lines       = params.arena_kb * 1024 / CACHE_LINE_B
    inf_us        = params.inf_ms * 1000
    busy_frac     = min(0.95, (n_lines * BURST_US) / inf_us)
    bursts_per_ms = n_lines / params.inf_ms   # ráfagas por ms de PID

    # Número de colisiones en un ciclo PID de 1 ms
    n_coll = rng.poisson(bursts_per_ms * busy_frac)
    return float(n_coll * BURST_US * rng.exponential(1.0))


def _cache_jitter(params: JitterParams, rng: np.random.Generator) -> float:
    """
    Cache thrashing: la CNN evicta líneas de caché del núcleo 0.
    Fracción de caché eviccionada: min(1, arena / cache_size).
    Prob de que el PID sufra miss en instrucción crítica: evict * 0.55.
    """
    evict_frac = min(1.0, params.arena_kb / CACHE_SIZE_KB)
    p_miss     = evict_frac * 0.55
    # Solo ocurre justo después de que termina la CNN
    if rng.random() > (params.fps / 1000):
        return 0.0
    n_misses = rng.binomial(PID_CRITICAL_INSTR, p_miss * 0.15)
    return float(n_misses * MISS_US)


def _dma_jitter(params: JitterParams, rng: np.random.Generator) -> float:
    """
    Interrupción DMA de cámara.
    La IRQ llega con tasa fps (= params.fps por segundo = fps/1e6 por µs).
    Duración de la IRQ: DMA_FRAME_US con variación exponencial.
    """
    if rng.random() < (params.fps * 1e-3):  # prob en ciclo de 1 ms
        return float(rng.exponential(DMA_FRAME_US * 0.4))
    return 0.0


def _wifi_jitter(params: JitterParams, rng: np.random.Generator) -> float:
    """Spike por interrupción de radio WiFi/BT."""
    if not params.wifi_active:
        return 0.0
    if rng.random() < WIFI_IRQ_RATE * 1000:  # prob en ciclo de 1ms = 50/1000 = 5%
        return float(rng.exponential(WIFI_IRQ_US))
    return 0.0


def sample_jitter(variant: str, n: int = 1,
                  seed: int | None = None) -> np.ndarray:
    """
    Muestrea n valores de jitter del PID para la variante dada.
    Retorna array float32 (n,) en µs.
    """
    params = VARIANT_PARAMS[variant]
    rng    = np.random.default_rng(seed)
    samples = np.zeros(n, dtype=np.float32)

    for i in range(n):
        ctx   = T_CTX_US + rng.normal(0, T_CTX_SIGMA)
        ahb   = _ahb_jitter(params, rng)
        cache = _cache_jitter(params, rng)
        dma   = _dma_jitter(params, rng)
        wifi  = _wifi_jitter(params, rng)
        prio  = PRIORITY_PENALTY_US if params.pid_priority != 'max' else 0.0
        samples[i] = max(0.0, ctx + ahb + cache + dma + wifi + prio)

    return samples


def jitter_stats(variant: str, n: int = 10_000, seed: int = 0) -> dict:
    """Estadísticas completas de la distribución de jitter."""
    s = sample_jitter(variant, n, seed)
    return {
        'variant':   variant,
        'n':         n,
        'mean':      float(np.mean(s)),
        'std':       float(np.std(s)),
        'p50':       float(np.percentile(s, 50)),
        'p95':       float(np.percentile(s, 95)),
        'p99':       float(np.percentile(s, 99)),
        'p999':      float(np.percentile(s, 99.9)),
        'max':       float(s.max()),
        'pct_over_200': float((s > 200).mean() * 100),
        'safe_for_flight': bool(np.percentile(s, 99) < 200),
    }


def component_breakdown(variant: str, n: int = 5000, seed: int = 0) -> dict:
    """
    Descompone el jitter por fuente para entender qué domina.
    Útil para decidir qué optimizar primero.
    """
    params = VARIANT_PARAMS[variant]
    rng    = np.random.default_rng(seed)

    ctx_arr   = np.zeros(n)
    ahb_arr   = np.zeros(n)
    cache_arr = np.zeros(n)
    dma_arr   = np.zeros(n)
    wifi_arr  = np.zeros(n)

    for i in range(n):
        ctx_arr[i]   = T_CTX_US + rng.normal(0, T_CTX_SIGMA)
        ahb_arr[i]   = _ahb_jitter(params, rng)
        cache_arr[i] = _cache_jitter(params, rng)
        dma_arr[i]   = _dma_jitter(params, rng)
        wifi_arr[i]  = _wifi_jitter(params, rng)

    total = ctx_arr + ahb_arr + cache_arr + dma_arr + wifi_arr
    return {
        'ctx':   {'mean': ctx_arr.mean(),   'p99': np.percentile(ctx_arr, 99)},
        'ahb':   {'mean': ahb_arr.mean(),   'p99': np.percentile(ahb_arr, 99)},
        'cache': {'mean': cache_arr.mean(), 'p99': np.percentile(cache_arr, 99)},
        'dma':   {'mean': dma_arr.mean(),   'p99': np.percentile(dma_arr, 99)},
        'wifi':  {'mean': wifi_arr.mean(),  'p99': np.percentile(wifi_arr, 99)},
        'total': {'mean': total.mean(),     'p99': np.percentile(total, 99)},
    }
