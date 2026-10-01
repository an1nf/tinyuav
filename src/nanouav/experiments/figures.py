"""
Nano-UAV · Nivel 0 — Ejecución completa del pipeline
Genera todas las visualizaciones en assets/
"""
import os
import sys
import numpy as np
import matplotlib
from nanouav import paths
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cv2

ROOT = str(paths.repo_root())

from nanouav.scene.scene import synthetic_scene
from nanouav.sensors.tof import tof_8x8
from nanouav.sensors.optical_flow import optical_flow, draw_flow, should_trigger_cnn
from nanouav.perception.cnn import VARIANTS, predict_depth, jitter_sample, fits_in_sram
from nanouav.planning.policy import policy_p1, policy_p2, policy_p3

ASSETS = str(paths.results_dir())
os.makedirs(ASSETS, exist_ok=True)

DARK = '#111111'
BG = '#1a1a1a'
_cmap = LinearSegmentedColormap.from_list(
    'depth',
    [(0.22,0.61,0.87),(0.11,0.62,0.46),(0.94,0.62,0.15),(0.88,0.29,0.29)][::-1],
    N=256)
PCOLS = {'P1': '#E24B4A', 'P2': '#EF9F27', 'P3': '#5DCAA5'}


def _style(ax):
    ax.set_facecolor(DARK)
    ax.tick_params(colors='#888', labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor('#333')


# ── 1. Snapshot ───────────────────────────────────────────────────────────
print('[1/4] Snapshot...')
t = 3.5
gray, depth_gt = synthetic_scene(t)
depth_pred = predict_depth(gray, depth_gt, 'M')
tof = tof_8x8(depth_gt)

fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor=BG)
fig.suptitle('Pipeline de percepción · variante M · 9 fps', color='white', fontsize=12)
for ax in axes: _style(ax)

axes[0].imshow(gray, cmap='gray', vmin=0, vmax=255)
axes[0].set_title('Cámara 96×96', color='white', fontsize=10); axes[0].axis('off')

im = axes[1].imshow(depth_pred, cmap=_cmap, vmin=0.2, vmax=4.0)
axes[1].set_title(f'Profundidad CNN  (mín: {depth_pred.min():.2f} m)', color='white', fontsize=10)
axes[1].axis('off'); plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

im2 = axes[2].imshow(tof, cmap=_cmap, vmin=0.2, vmax=4.0)
axes[2].set_title('ToF 8×8 (VL53L5CX)', color='white', fontsize=10)
axes[2].axis('off'); plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

fig.tight_layout()
plt.savefig(os.path.join(ASSETS, 'snapshot.png'), dpi=120, bbox_inches='tight', facecolor=BG)
plt.close(fig)
print('  → assets/snapshot.png')


# ── 2. Flujo óptico + trigger P3 ─────────────────────────────────────────
print('[2/4] Flujo óptico...')
fig, axes = plt.subplots(2, 4, figsize=(14, 6), facecolor=BG)
fig.suptitle('Flujo óptico y trigger P3 · 8 instantes', color='white', fontsize=11)

prev_g = None
for i, ax in enumerate(axes.flat):
    _style(ax)
    t_i = i * 0.15
    gray_i, depth_i = synthetic_scene(t_i)
    flow_pts, div = optical_flow(prev_g, gray_i) if prev_g is not None else ([], 0.0)
    vis = draw_flow(gray_i, flow_pts)
    ax.imshow(vis)
    triggered = should_trigger_cnn(div)
    color = '#E24B4A' if triggered else '#5DCAA5'
    ax.set_title(f't={t_i:.2f}  div={div:.3f}\n{"CNN ACTIVA" if triggered else "CNN reposo"}',
                 color=color, fontsize=7)
    ax.axis('off')
    prev_g = gray_i.copy()

fig.tight_layout()
plt.savefig(os.path.join(ASSETS, 'optical_flow.png'), dpi=120, bbox_inches='tight', facecolor=BG)
plt.close(fig)
print('  → assets/optical_flow.png')


# ── 3. Políticas P1/P2/P3 ────────────────────────────────────────────────
print('[3/4] Comparativa de políticas...')
N_FRAMES = 60
records = {p: {'dists': [], 'cnn_on': []} for p in ['P1','P2','P3']}
prev_g = None

for i in range(N_FRAMES):
    gray_i, depth_i = synthetic_scene(i * 0.04)
    dp = predict_depth(gray_i, depth_i, 'M')
    tof_i = tof_8x8(depth_i)
    _, div_i = optical_flow(prev_g, gray_i) if prev_g is not None else ([], 0.0)
    prev_g = gray_i.copy()

    d1 = policy_p1(dp)
    d2 = policy_p2(tof_i, dp)
    d3 = policy_p3(div_i, tof_i, dp)

    records['P1']['dists'].append(d1.min_dist);  records['P1']['cnn_on'].append(1)
    records['P2']['dists'].append(d2.min_dist);  records['P2']['cnn_on'].append(1 if d2.triggered_by=='cnn' else 0)
    records['P3']['dists'].append(d3.min_dist);  records['P3']['cnn_on'].append(1 if d3.triggered_by=='cnn' else 0)

fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor=BG)
for ax in axes: _style(ax)

axes[0].set_title('Distancia detectada por política', color='white', fontsize=10)
for p, rec in records.items():
    axes[0].plot(rec['dists'], label=p, color=PCOLS[p], lw=1.5, alpha=0.9)
axes[0].axhline(0.5, color='#E24B4A', lw=0.7, ls='--', label='evasión')
axes[0].axhline(1.0, color='#EF9F27', lw=0.7, ls='--', label='precaución')
axes[0].set_ylabel('distancia mínima (m)', color='#888'); axes[0].set_xlabel('frame', color='#888')
axes[0].legend(fontsize=8, facecolor='#222', edgecolor='#333', labelcolor='white')

act_pcts = {p: sum(rec['cnn_on'])/N_FRAMES*100 for p, rec in records.items()}
bars = axes[1].bar(list(act_pcts.keys()), list(act_pcts.values()),
                   color=[PCOLS[p] for p in act_pcts], alpha=0.85)
axes[1].set_title('CNN activa (% frames)', color='white', fontsize=10)
axes[1].set_ylabel('%', color='#888'); axes[1].set_ylim(0, 115)
for bar, (p, pct) in zip(bars, act_pcts.items()):
    axes[1].text(bar.get_x()+bar.get_width()/2, pct+1, f'{pct:.0f}%',
                 color=PCOLS[p], ha='center', fontsize=9)

fig.tight_layout()
plt.savefig(os.path.join(ASSETS, 'policy_comparison.png'), dpi=120, bbox_inches='tight', facecolor=BG)
plt.close(fig)
print('  → assets/policy_comparison.png')


# ── 4. Benchmark variantes ────────────────────────────────────────────────
print('[4/4] Benchmark variantes...')
COLORS = {'XS':'#5DCAA5','S':'#378ADD','M':'#EF9F27','L':'#E24B4A'}
N = 200
results = {}
for vname in VARIANTS:
    j, d = [], []
    prev = None
    for i in range(N):
        g, dg = synthetic_scene(i*0.04)
        dp = predict_depth(g, dg, vname)
        _, div = optical_flow(prev, g) if prev is not None else ([], 0.0)
        prev = g.copy()
        j.append(jitter_sample(vname, div))
        d.append(float(dp.min()))
    results[vname] = {'j': j, 'd': d}

fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor=BG)
for ax in axes: _style(ax)

axes[0].set_title('Jitter PID por variante', color='white', fontsize=10)
for vname, res in results.items():
    axes[0].plot(res['j'], label=vname, color=COLORS[vname], lw=1.2, alpha=0.85)
axes[0].axhline(200, color='white', lw=0.7, ls='--', label='límite 200 µs')
axes[0].set_ylabel('Jitter (µs)', color='#888'); axes[0].set_xlabel('Frame', color='#888')
axes[0].legend(fontsize=8, facecolor='#222', edgecolor='#333', labelcolor='white')

axes[1].set_title('Trade-off jitter p99 vs FPS', color='white', fontsize=10)
for vname, res in results.items():
    p99 = sorted(res['j'])[int(0.99*N)]
    fps = VARIANTS[vname]['fps']
    axes[1].scatter(fps, p99, color=COLORS[vname], s=120, zorder=5)
    axes[1].annotate(vname, (fps, p99), xytext=(5,5), textcoords='offset points',
                     color=COLORS[vname], fontsize=9)
axes[1].axhline(200, color='white', lw=0.7, ls='--')
axes[1].set_xlabel('FPS inferencia', color='#888'); axes[1].set_ylabel('Jitter p99 (µs)', color='#888')

fig.tight_layout()
plt.savefig(os.path.join(ASSETS, 'benchmark.png'), dpi=120, bbox_inches='tight', facecolor=BG)
plt.close(fig)
print('  → assets/benchmark.png')

print(f'\n{"Variante":<8} {"FPS":>5} {"Arena KB":>9} {"SRAM?":>6} {"p99 µs":>8} {"d_safe cm":>10}')
print('─' * 52)
for vname, v in VARIANTS.items():
    p99 = sorted(results[vname]['j'])[int(0.99*N)]
    d_safe = 0.5*(1/v['fps']+0.05)*100
    sram = '✓' if fits_in_sram(vname) else '✗ PSRAM'
    print(f'{vname:<8} {v["fps"]:>5} {v["arena_kb"]:>9} {sram:>6} {p99:>8} {d_safe:>9.0f}')

print('\nHecho. Todas las imágenes en assets/')
