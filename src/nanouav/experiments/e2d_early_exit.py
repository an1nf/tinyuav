"""
E2-D · Experimento: early-exit condicional en inferencia CNN.

Ejecutar:
    conda activate nano_uav
    python e2d_early_exit.py

Produce:
    assets/e2d_proxy.png      — early-exit proxy (sin Keras)
    assets/e2d_latency.png    — reducción de latencia acumulada
"""
import os, sys, time
import numpy as np
import matplotlib
from nanouav import paths
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = str(paths.repo_root())

from nanouav.scene.scene import synthetic_scene
from nanouav.scene.renderer import render_scene
from nanouav.sensors.tof import tof_8x8
from nanouav.sensors.optical_flow import optical_flow, should_trigger_cnn
from nanouav.perception.cnn import predict_depth_norm, _load_interpreter, is_real_mode
from nanouav.perception.early_exit import EarlyExitProxy, DANGER_THRESHOLD
from nanouav.perception.tof_calibrator import ToFCalibrator

ASSETS = str(paths.results_dir())
DARK, BG = '#111111', '#1a1a1a'
CNN_DIR = str(paths.models_dir())

N_FRAMES   = 120
VARIANT    = 'M'
CNN_LAT_MS = 0.638   # medido en E0-C

os.makedirs(ASSETS, exist_ok=True)
_load_interpreter()
print(f'[E2-D] CNN {"REAL" if is_real_mode() else "FAKE"} · {N_FRAMES} frames\n')


def _style(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(DARK)
    ax.tick_params(colors='#888', labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor('#333')
    if title:  ax.set_title(title,  color='white', fontsize=9)
    if xlabel: ax.set_xlabel(xlabel, color='#888',  fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, color='#888',  fontsize=8)


# ── Experimento A: Early-exit proxy (sin Keras) ───────────────────────────
proxy = EarlyExitProxy(tof_threshold=0.8)
cal   = ToFCalibrator()

frames     = np.arange(N_FRAMES)
decisions  = []
lats_full  = []   # latencia si siempre CNN completa
lats_proxy = []   # latencia con early-exit proxy
divs       = []
tof_mins   = []
min_dists  = []
prev_gray  = None

for i in range(N_FRAMES):
    t = i * 0.05
    if i % 3 == 0:
        gray, depth_gt = render_scene(seed=i * 11)
    else:
        gray, depth_gt = synthetic_scene(t)

    tof   = tof_8x8(depth_gt)
    _, div = optical_flow(prev_gray, gray) if prev_gray is not None else ([], 0.0)
    prev_gray = gray.copy()

    tof_min = float(tof.min())
    decision = proxy.decide(div, tof_min)

    # Latencia
    lat_full  = CNN_LAT_MS
    lat_proxy = proxy.estimated_latency_ms(decision, CNN_LAT_MS)

    # Profundidad según decisión
    if decision == 'idle':
        depth_m = np.full((96, 96), 4.0, dtype=np.float32)
    elif decision == 'tof_only':
        # Upscale del ToF 8×8 → 96×96 como proxy de profundidad
        import cv2
        depth_m = cv2.resize(tof, (96, 96), interpolation=cv2.INTER_LINEAR)
    else:
        depth_norm = predict_depth_norm(gray, VARIANT, depth_gt)
        depth_m    = cal.update_and_scale(depth_norm, tof)

    decisions.append(decision)
    lats_full.append(lat_full)
    lats_proxy.append(lat_proxy)
    divs.append(div)
    tof_mins.append(tof_min)
    min_dists.append(float(depth_m.min()))

decisions  = np.array(decisions)
lats_full  = np.array(lats_full)
lats_proxy = np.array(lats_proxy)

n_idle     = (decisions == 'idle').sum()
n_tof      = (decisions == 'tof_only').sum()
n_cnn      = (decisions == 'full_cnn').sum()
total_save = (lats_full - lats_proxy).sum()
pct_save   = total_save / lats_full.sum() * 100

print(f'Distribución de decisiones ({N_FRAMES} frames):')
print(f'  idle     (sin CNN):  {n_idle:3d}  ({n_idle/N_FRAMES*100:.0f}%)')
print(f'  tof_only (solo ToF): {n_tof:3d}  ({n_tof/N_FRAMES*100:.0f}%)')
print(f'  full_cnn (CNN comp): {n_cnn:3d}  ({n_cnn/N_FRAMES*100:.0f}%)')
print(f'\nLatencia acumulada:')
print(f'  Sin early-exit: {lats_full.sum():.1f} ms')
print(f'  Con early-exit: {lats_proxy.sum():.1f} ms')
print(f'  Ahorro:         {total_save:.1f} ms  ({pct_save:.0f}%)')


# ── Figura 1: Proxy early-exit ────────────────────────────────────────────
DCOLS = {'idle': '#444', 'tof_only': '#9B59B6', 'full_cnn': '#EF9F27'}

fig, axes = plt.subplots(2, 2, figsize=(13, 8), facecolor=BG)
fig.suptitle('E2-D · Early-exit proxy: flujo óptico + ToF como gate CNN',
             color='white', fontsize=11)

ax = axes[0, 0]
_style(ax, 'Divergencia y decisión de ejecución', 'frame', 'divergencia')
ax.plot(frames, divs, color='#378ADD', lw=1.3, alpha=0.8, label='divergencia')
ax.axhline(proxy.div_threshold, color='#EF9F27', lw=0.8, ls='--',
           label=f'umbral div={proxy.div_threshold}')
for j, dec in enumerate(decisions):
    ax.axvspan(j-0.5, j+0.5, alpha=0.25, color=DCOLS[dec], lw=0)
ax.legend(fontsize=8, facecolor='#222', edgecolor='#333', labelcolor='white')

# Leyenda manual de decisiones
from matplotlib.patches import Patch
leg = [Patch(color=DCOLS[d], alpha=0.6, label=d) for d in DCOLS]
ax.legend(handles=leg, fontsize=7,
          facecolor='#222', edgecolor='#333', labelcolor='white', loc='upper right')

ax = axes[0, 1]
_style(ax, 'Distribución de decisiones', '', 'frames')
counts = [n_idle, n_tof, n_cnn]
labels = ['idle\n(sin CNN)', 'tof_only\n(ToF proxy)', 'full_cnn\n(CNN completa)']
colors = [DCOLS['idle'], DCOLS['tof_only'], DCOLS['full_cnn']]
bars   = ax.bar(labels, counts, color=colors, alpha=0.85)
for bar, cnt in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, cnt + 0.5,
            f'{cnt}\n({cnt/N_FRAMES*100:.0f}%)',
            ha='center', color='white', fontsize=8)

ax = axes[1, 0]
_style(ax, 'Latencia acumulada: full vs early-exit', 'frame', 'ms acumulados')
ax.plot(frames, np.cumsum(lats_full),  color='#E24B4A', lw=1.5, label='Sin early-exit')
ax.plot(frames, np.cumsum(lats_proxy), color='#5DCAA5', lw=1.5, label='Con early-exit')
ax.fill_between(frames,
                np.cumsum(lats_proxy), np.cumsum(lats_full),
                alpha=0.12, color='#5DCAA5', label='tiempo ahorrado')
ax.text(0.98, 0.05, f'ahorro: {pct_save:.0f}%',
        transform=ax.transAxes, color='#5DCAA5', fontsize=10,
        ha='right', fontweight='bold')
ax.legend(fontsize=8, facecolor='#222', edgecolor='#333', labelcolor='white')

ax = axes[1, 1]
_style(ax, 'Distancia mínima detectada', 'frame', 'metros')
ax.plot(frames, min_dists,  color='#5DCAA5', lw=1.5, label='early-exit')
ax.plot(frames, tof_mins,   color='#9B59B6', lw=1.0, alpha=0.7, label='ToF mínimo')
ax.axhline(0.5, color='#E24B4A', lw=0.7, ls='--', label='evasión 0.5m')
ax.axhline(1.0, color='#EF9F27', lw=0.7, ls='--', label='precaución 1.0m')
ax.legend(fontsize=8, facecolor='#222', edgecolor='#333', labelcolor='white')

fig.tight_layout()
out = os.path.join(ASSETS, 'e2d_proxy.png')
plt.savefig(out, dpi=130, bbox_inches='tight', facecolor=BG)
plt.close(fig)
print(f'\n[plot] {out}')


# ── Figura 2: Análisis de umbrales ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=BG)
fig.suptitle('E2-D · Sensibilidad a umbrales del early-exit proxy',
             color='white', fontsize=11)

div_thresholds = np.linspace(0.0, 0.25, 20)
tof_thresholds = np.linspace(0.3, 2.0, 20)

# Para cada umbral de divergencia: % CNN activa y ahorro
pct_cnn  = []
pct_save_arr = []
for dt in div_thresholds:
    p = EarlyExitProxy(div_threshold=dt, tof_threshold=0.8)
    decs  = [p.decide(d, t) for d, t in zip(divs, tof_mins)]
    n_cnn_i = sum(1 for d in decs if d == 'full_cnn')
    lats  = np.array([p.estimated_latency_ms(d, CNN_LAT_MS) for d in decs])
    pct_cnn.append(n_cnn_i / N_FRAMES * 100)
    pct_save_arr.append((lats_full - lats).sum() / lats_full.sum() * 100)

ax = axes[0]
_style(ax, 'Umbral de divergencia: % CNN activa vs ahorro',
       'umbral divergencia', '%')
ax.plot(div_thresholds, pct_cnn,      color='#E24B4A', lw=1.5, label='% CNN activa')
ax.plot(div_thresholds, pct_save_arr, color='#5DCAA5', lw=1.5, label='% ahorro latencia')
ax.axvline(proxy.div_threshold, color='#EF9F27', lw=0.8, ls='--',
           label=f'actual {proxy.div_threshold}')
ax.legend(fontsize=8, facecolor='#222', edgecolor='#333', labelcolor='white')
ax.set_ylim(0, 105)

# Para cada umbral ToF: distribución de decisiones
tof_cnn_pct  = []
tof_tof_pct  = []
for tt in tof_thresholds:
    p = EarlyExitProxy(div_threshold=proxy.div_threshold, tof_threshold=tt)
    decs = [p.decide(d, t) for d, t in zip(divs, tof_mins)]
    tof_cnn_pct.append(sum(1 for d in decs if d == 'full_cnn') / N_FRAMES * 100)
    tof_tof_pct.append(sum(1 for d in decs if d == 'tof_only') / N_FRAMES * 100)

ax = axes[1]
_style(ax, 'Umbral ToF: trade-off CNN vs ToF-only',
       'umbral ToF (m)', '% frames')
ax.plot(tof_thresholds, tof_cnn_pct, color='#EF9F27', lw=1.5, label='full_cnn')
ax.plot(tof_thresholds, tof_tof_pct, color='#9B59B6', lw=1.5, label='tof_only')
ax.axvline(proxy.tof_threshold, color='#5DCAA5', lw=0.8, ls='--',
           label=f'actual {proxy.tof_threshold}m')
ax.legend(fontsize=8, facecolor='#222', edgecolor='#333', labelcolor='white')

fig.tight_layout()
out2 = os.path.join(ASSETS, 'e2d_latency.png')
plt.savefig(out2, dpi=130, bbox_inches='tight', facecolor=BG)
plt.close(fig)
print(f'[plot] {out2}')
print('\n[E2-D] Listo.')
