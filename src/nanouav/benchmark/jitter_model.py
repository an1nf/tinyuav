"""
nanouav/benchmark/jitter_model.py  —  E1-A

PID jitter model based on the physics of the dual-core ESP32-S3.

Replaces jitter_sample() in nanouav/perception/cnn.py (empirical linear model)
with a Monte Carlo simulation of the real phenomena:

  1. FreeRTOS context switch (deterministic + thermal variation)
  2. Contention on the AHB bus shared between cores
  3. Cache thrashing: the CNN evicts cache lines of core 0
  4. DVP camera DMA interrupt
  5. WiFi/BT interrupt from the shared radio (rare spike)

Parameters from the literature:
  - T_CTX = 3.2 µs  (Fenwick et al., FreeRTOS ESP32 latency measurements)
  - BUS_FREQ = 120 MHz AHB (ESP32-S3 TRM, section 3.3)
  - CACHE_LINE = 64 bytes, L1 = 32 KB per core (ESP32-S3 TRM)
  - MISS_CYCLES = 20 cycles per cache miss (Espressif perf guide)
  - WiFi IRQ: rate ~0.8/s, duration ~120 µs (measured in esp-idf examples)

All distributions are falsifiable with E3-A (measurement with GPIO + logic analyzer).
"""
import numpy as np
from dataclasses import dataclass

# ── System constants (ESP32-S3) ─────────────────────────────────────────
F_CPU_MHZ     = 240.0
F_AHB_MHZ     = 120.0            # bus shared between cores
CACHE_SIZE_KB = 32.0             # L1 per core
CACHE_LINE_B  = 64               # bytes per cache line
MISS_CYCLES   = 20               # penalty cycles per cache miss
MISS_US       = MISS_CYCLES / F_CPU_MHZ          # 0.0833 µs
BURST_CYCLES  = 16               # AHB cycles per transferred line
BURST_US      = BURST_CYCLES / F_AHB_MHZ         # 0.1333 µs

# FreeRTOS context switch
T_CTX_US      = 3.2              # µs (deterministic)
T_CTX_SIGMA   = 0.8              # µs (thermal variation of the oscillator)

# DVP camera DMA
DMA_FRAME_KB  = 96 * 96 / 1024  # 9 KB per frame
DMA_BW_US_KB  = 0.8             # µs per KB transferred by DMA
DMA_FRAME_US  = DMA_FRAME_KB * DMA_BW_US_KB  # ~7.2 µs per frame

# WiFi/BT interrupts (shared ESP32-S3 radio)
WIFI_IRQ_RATE  = 0.00005         # per µs (= ~50 Hz, active telemetry mode)
WIFI_IRQ_US    = 120.0           # mean duration of the WiFi IRQ

# FreeRTOS priority configurations
# With the correct priority (PID = max), the jitter is the model below.
# Without the correct priority, add a ~170 µs penalty.
PRIORITY_PENALTY_US = 170.0      # if the PID does not have maximum priority

# PID: number of instructions in the critical section
PID_CRITICAL_INSTR = 200


@dataclass
class JitterParams:
    """System configuration parameters for the model."""
    arena_kb:      float   # CNN tensor arena size (KB)
    inf_ms:        float   # inference latency (ms)
    fps:           float   # CNN FPS
    pid_priority:  str     # 'max' | 'normal' (adds ~170 µs if not max)
    wifi_active:   bool    # whether the WiFi radio is active


# Per-variant parameters (from the ESP-DL literature)
VARIANT_PARAMS = {
    'XS': JitterParams(arena_kb=40,  inf_ms=55,  fps=18, pid_priority='max', wifi_active=True),
    'S':  JitterParams(arena_kb=80,  inf_ms=77,  fps=13, pid_priority='max', wifi_active=True),
    'M':  JitterParams(arena_kb=130, inf_ms=110, fps=9,  pid_priority='max', wifi_active=True),
    'L':  JitterParams(arena_kb=175, inf_ms=200, fps=5,  pid_priority='max', wifi_active=True),
}


def _ahb_jitter(params: JitterParams, rng: np.random.Generator) -> float:
    """
    AHB bus contention.
    The CNN generates bursts of BURST_US µs each time it transfers a cache line.
    Fraction of the time the bus is occupied by the CNN during inference:
        busy_frac = (n_lines * BURST_US) / inf_µs
    """
    n_lines       = params.arena_kb * 1024 / CACHE_LINE_B
    inf_us        = params.inf_ms * 1000
    busy_frac     = min(0.95, (n_lines * BURST_US) / inf_us)
    bursts_per_ms = n_lines / params.inf_ms   # bursts per ms of PID

    # Number of collisions in a 1 ms PID cycle
    n_coll = rng.poisson(bursts_per_ms * busy_frac)
    return float(n_coll * BURST_US * rng.exponential(1.0))


def _cache_jitter(params: JitterParams, rng: np.random.Generator) -> float:
    """
    Cache thrashing: the CNN evicts cache lines of core 0.
    Fraction of the cache evicted: min(1, arena / cache_size).
    Prob. that the PID suffers a miss on a critical instruction: evict * 0.55.
    """
    evict_frac = min(1.0, params.arena_kb / CACHE_SIZE_KB)
    p_miss     = evict_frac * 0.55
    # Only happens right after the CNN finishes
    if rng.random() > (params.fps / 1000):
        return 0.0
    n_misses = rng.binomial(PID_CRITICAL_INSTR, p_miss * 0.15)
    return float(n_misses * MISS_US)


def _dma_jitter(params: JitterParams, rng: np.random.Generator) -> float:
    """
    Camera DMA interrupt.
    The IRQ arrives at rate fps (= params.fps per second = fps/1e6 per µs).
    IRQ duration: DMA_FRAME_US with exponential variation.
    """
    if rng.random() < (params.fps * 1e-3):  # prob. in a 1 ms cycle
        return float(rng.exponential(DMA_FRAME_US * 0.4))
    return 0.0


def _wifi_jitter(params: JitterParams, rng: np.random.Generator) -> float:
    """Spike caused by a WiFi/BT radio interrupt."""
    if not params.wifi_active:
        return 0.0
    if rng.random() < WIFI_IRQ_RATE * 1000:  # prob. in a 1ms cycle = 50/1000 = 5%
        return float(rng.exponential(WIFI_IRQ_US))
    return 0.0


def sample_jitter(variant: str, n: int = 1,
                  seed: int | None = None) -> np.ndarray:
    """
    Samples n PID jitter values for the given variant.
    Returns a float32 array (n,) in µs.
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
    """Complete statistics of the jitter distribution."""
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
    Breaks the jitter down by source to understand which one dominates.
    Useful for deciding what to optimize first.
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
