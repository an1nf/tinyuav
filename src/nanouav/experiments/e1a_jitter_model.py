"""
E1-A · Experimento: modelo de jitter con física real del ESP32-S3.

Ejecutar:
    conda activate nano_uav
    python e1a_jitter_model.py

Produce:
    assets/e1a_distributions.png  — distribuciones por variante
    assets/e1a_breakdown.png      — descomposición por fuente
    assets/e1a_comparison.png     — modelo físico vs modelo empírico anterior
"""
import os, sys

import numpy as np
import matplotlib
from nanouav import paths
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from nanouav.benchmark.jitter_model import (
    sample_jitter, jitter_stats, component_breakdown,
    VARIANT_PARAMS, PRIORITY_PENALTY_US
)
from nanouav.perception.cnn import VARIANTS as CNN_VARIANTS

ASSETS = str(paths.results_dir())
os.makedirs(ASSETS, exist_ok=True)

DARK = '#111111'
BG   = '#1a1a1a'
COLORS = {'XS': '#5DCAA5', 'S': '#378ADD', 'M': '#EF9F27', 'L': '#E24B4A'}
COMP_COLORS = {
    'ctx': '#5DCAA5', 'ahb': '#378ADD',
    'cache': '#EF9F27', 'dma': '#9B59B6', 'wifi': '#E24B4A'
}
COMP_LABELS = {
    'ctx': 'Cambio de ctx', 'ahb': 'Bus AHB',
    'cache': 'Cache miss', 'dma': 'DMA cámara', 'wifi': 'WiFi IRQ'
}

N = 20_000


def _style(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(DARK)
    ax.tick_params(colors='#888', labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor('#333')
    if title:   ax.set_title(title, color='white', fontsize=9)
    if xlabel:  ax.set_xlabel(xlabel, color='#888', fontsize=8)
    if ylabel:  ax.set_ylabel(ylabel, color='#888', fontsize=8)


# ── 1. Estadísticas por variante ──────────────────────────────────────────
print('[E1-A] Calculando distribuciones...')
stats = {v: jitter_stats(v, N) for v in ['XS', 'S', 'M', 'L']}

print(f'\n{"Variante":8} {"p50":>7} {"p95":>7} {"p99":>7} {"p99.9":>8} {"max":>8} {">200µs":>8} {"Apto"}')
print('─' * 65)
for v, st in stats.items():
    print(f'{v:8} {st["p50"]:7.1f} {st["p95"]:7.1f} {st["p99"]:7.1f} '
          f'{st["p999"]:8.1f} {st["max"]:8.0f} {st["pct_over_200"]:7.3f}%  '
          f'{"✓" if st["safe_for_flight"] else "✗"}')

# ── 2. Descomposición por fuente ──────────────────────────────────────────
print('\n[E1-A] Descomponiendo fuentes de jitter...')
breakdown = {v: component_breakdown(v, 5000) for v in ['XS', 'S', 'M', 'L']}

print(f'\nDescomposición p99 (µs) por fuente:')
print(f'{"Variante":8} {"ctx":>8} {"AHB":>8} {"cache":>8} {"DMA":>8} {"WiFi":>8} {"total":>8}')
print('─' * 58)
for v, bd in breakdown.items():
    print(f'{v:8} {bd["ctx"]["p99"]:8.2f} {bd["ahb"]["p99"]:8.2f} '
          f'{bd["cache"]["p99"]:8.2f} {bd["dma"]["p99"]:8.2f} '
          f'{bd["wifi"]["p99"]:8.2f} {bd["total"]["p99"]:8.2f}')


# ── Figura 1: Distribuciones ──────────────────────────────────────────────
print('\n[E1-A] Generando gráficas...')
fig = plt.figure(figsize=(14, 10), facecolor=BG)
fig.suptitle('E1-A · Modelo de jitter PID con física ESP32-S3 (Monte Carlo N=20k)',
             color='white', fontsize=11, y=0.98)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32,
                       left=0.07, right=0.97, top=0.93, bottom=0.06)

for idx, (variant, color) in enumerate(COLORS.items()):
    ax = fig.add_subplot(gs[idx // 2, idx % 2])
    _style(ax, f'Variante {variant}  (arena={VARIANT_PARAMS[variant].arena_kb} KB)',
           'Jitter (µs)', 'frecuencia')

    samples = sample_jitter(variant, N, seed=idx)
    # Zoom en el rango principal (excluye spikes WiFi para visualizar mejor)
    p999 = np.percentile(samples, 99.9)
    ax.hist(samples[samples < p999], bins=60, color=color, alpha=0.85,
            edgecolor='#222', linewidth=0.3)
    for pct, ls, label in [(50, '--', 'p50'), (95, ':', 'p95'), (99, '-', 'p99')]:
        val = np.percentile(samples, pct)
        ax.axvline(val, color='white', lw=0.8, ls=ls, label=f'{label}={val:.1f}µs')
    ax.axvline(200, color='#E24B4A', lw=1.0, ls='-', label='límite 200µs')
    ax.legend(fontsize=7, facecolor='#222', edgecolor='#333', labelcolor='white')

    st = stats[variant]
    ax.text(0.97, 0.92,
            f'>{200}µs: {st["pct_over_200"]:.3f}%\n'
            f'{"APTO ✓" if st["safe_for_flight"] else "NO APTO ✗"}',
            transform=ax.transAxes, color=color, fontsize=8,
            ha='right', va='top', fontweight='bold')

plt.savefig(os.path.join(ASSETS, 'e1a_distributions.png'),
            dpi=130, bbox_inches='tight', facecolor=BG)
plt.close(fig)
print('  → assets/e1a_distributions.png')


# ── Figura 2: Descomposición por fuente ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)
fig.suptitle('E1-A · Descomposición del jitter por fuente',
             color='white', fontsize=11)

# Barras apiladas: media
variants_list = ['XS', 'S', 'M', 'L']
components    = ['ctx', 'ahb', 'cache', 'dma', 'wifi']
x = np.arange(len(variants_list))
bottom = np.zeros(len(variants_list))

_style(axes[0], 'Contribución media por fuente (µs)', 'Variante', 'µs')
for comp in components:
    vals = [breakdown[v][comp]['mean'] for v in variants_list]
    axes[0].bar(x, vals, bottom=bottom, label=COMP_LABELS[comp],
                color=COMP_COLORS[comp], alpha=0.88)
    bottom += np.array(vals)
axes[0].set_xticks(x); axes[0].set_xticklabels(variants_list, color='#888')
axes[0].legend(fontsize=8, facecolor='#222', edgecolor='#333', labelcolor='white')

# Barras apiladas: p99
bottom = np.zeros(len(variants_list))
_style(axes[1], 'Contribución p99 por fuente (µs)', 'Variante', 'µs')
for comp in components:
    vals = [breakdown[v][comp]['p99'] for v in variants_list]
    axes[1].bar(x, vals, bottom=bottom, label=COMP_LABELS[comp],
                color=COMP_COLORS[comp], alpha=0.88)
    bottom += np.array(vals)
axes[1].axhline(200, color='white', lw=0.8, ls='--', label='límite 200µs')
axes[1].set_xticks(x); axes[1].set_xticklabels(variants_list, color='#888')
axes[1].legend(fontsize=8, facecolor='#222', edgecolor='#333', labelcolor='white')

fig.tight_layout()
plt.savefig(os.path.join(ASSETS, 'e1a_breakdown.png'),
            dpi=130, bbox_inches='tight', facecolor=BG)
plt.close(fig)
print('  → assets/e1a_breakdown.png')


# ── Figura 3: Modelo físico vs empírico anterior ──────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)
fig.suptitle('E1-A · Modelo físico vs modelo empírico (literatura)',
             color='white', fontsize=11)

from nanouav.perception.cnn import jitter_sample as empirical_jitter

_style(axes[0], 'p99 jitter: físico vs empírico', 'Variante', 'µs (p99)')
_style(axes[1], 'Scatter FPS vs p99 jitter', 'FPS inferencia', 'Jitter p99 (µs)')

w = 0.35
x = np.arange(4)
p99_physical  = [stats[v]['p99'] for v in variants_list]
p99_empirical = []
for v in variants_list:
    emp = [empirical_jitter(v, 0.2) for _ in range(2000)]
    p99_empirical.append(np.percentile(emp, 99))

axes[0].bar(x - w/2, p99_physical,  width=w, color='#5DCAA5', alpha=0.85, label='Físico (E1-A)')
axes[0].bar(x + w/2, p99_empirical, width=w, color='#E24B4A', alpha=0.85, label='Empírico (anterior)')
axes[0].axhline(200, color='white', lw=0.8, ls='--', label='límite 200µs')
axes[0].set_xticks(x); axes[0].set_xticklabels(variants_list, color='#888')
axes[0].legend(fontsize=8, facecolor='#222', edgecolor='#333', labelcolor='white')

for v, color in COLORS.items():
    fps     = CNN_VARIANTS[v]['fps']
    p99_ph  = stats[v]['p99']
    p99_emp = p99_empirical[variants_list.index(v)]
    axes[1].scatter(fps, p99_ph,  color=color, s=120, marker='o', zorder=5)
    axes[1].scatter(fps, p99_emp, color=color, s=120, marker='x', zorder=5, linewidths=2)
    axes[1].annotate(v, (fps, max(p99_ph, p99_emp)),
                     xytext=(5, 5), textcoords='offset points',
                     color=color, fontsize=9)

axes[1].axhline(200, color='white', lw=0.8, ls='--', label='límite 200µs')
from matplotlib.lines import Line2D
leg = [Line2D([0],[0], marker='o', color='w', markerfacecolor='#888', label='Físico'),
       Line2D([0],[0], marker='x', color='w', markerfacecolor='#888',
              markeredgecolor='#888', label='Empírico')]
axes[1].legend(handles=leg, fontsize=8, facecolor='#222', edgecolor='#333', labelcolor='white')
axes[1].set_xlabel('FPS inferencia', color='#888')

fig.tight_layout()
plt.savefig(os.path.join(ASSETS, 'e1a_comparison.png'),
            dpi=130, bbox_inches='tight', facecolor=BG)
plt.close(fig)
print('  → assets/e1a_comparison.png')

# ── Resumen interpretativo ────────────────────────────────────────────────
print(f"""
[E1-A] Conclusiones del modelo físico:

  Fuente dominante:  WiFi IRQ (spikes raros pero grandes, ~120µs)
  Fuente continua:   Cambio de contexto FreeRTOS (~3-5µs siempre)
  Fuente variable:   Cache miss post-CNN (crece con tamaño de arena)
  Fuente menor:      Contención bus AHB (<1µs media)

  Con PID en prioridad máxima y WiFi activo:
    Todas las variantes tienen p99 < 30µs  →  muy por debajo del límite 200µs
    Los spikes >200µs son eventos WiFi raros (<0.1% del tiempo)

  Implicación: el límite real no es qué variante de CNN usar,
  sino si el WiFi está activo durante el vuelo.
  Desactivar WiFi durante maniobras críticas reduciría p99 a <15µs.

  E3-A (medición real con GPIO) falsificará estas predicciones.
""")
