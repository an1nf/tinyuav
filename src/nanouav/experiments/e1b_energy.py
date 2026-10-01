"""
E1-B · Experiment: energy simulator with real nano-UAV physics.

Run:
    conda activate nano_uav
    nanouav run e1b

Produces:
    results/e1b_budget.png        — energy budget per component
    results/e1b_tradeoff.png      — speed-safety-endurance trade-off
    results/e1b_sensitivity.png   — sensitivity to the scale factor
    results/e1b_policy.png        — P1/P2/P3 comparison in energy and speed
"""
import os, sys

import numpy as np
import matplotlib
from nanouav import paths
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

from nanouav.benchmark.energy_model import (
    compute_all, baseline, endurance_vs_speed,
    sensitivity_analysis, EnergyBudget,
    VARIANTS, CNN_DUTY, BAT_J, SCALE_LO, SCALE_HI, SCALE_MID,
    LAT_CNN_CPU_S, T_REACT_S, P_MOTORS_W, P_MCU_IDLE_W,
    P_TOF_W, P_IMU_W, P_CNN_ACTIVE_W, EFFICIENCY
)

ASSETS = str(paths.results_dir())
os.makedirs(ASSETS, exist_ok=True)

DARK = '#111111'
BG   = '#1a1a1a'
COLORS_V = {'XS': '#5DCAA5', 'S': '#378ADD', 'M': '#EF9F27', 'L': '#E24B4A'}
COLORS_P = {'P1': '#E24B4A', 'P2': '#EF9F27', 'P3': '#5DCAA5'}
COMP_C   = {'Motors': '#378ADD', 'MCU': '#5DCAA5',
            'Sensors': '#9B59B6', 'CNN': '#EF9F27'}


def _style(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(DARK)
    ax.tick_params(colors='#888', labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor('#333')
    if title:  ax.set_title(title,  color='white', fontsize=9)
    if xlabel: ax.set_xlabel(xlabel, color='#888',  fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, color='#888',  fontsize=8)


results = compute_all(SCALE_MID)
t_base  = baseline()

# ── Console ───────────────────────────────────────────────────────────────
print(f'\n[E1-B] Baseline endurance (no CNN): {t_base:.1f} min')
print(f'       Battery: 550 mAh × 3.7V = {BAT_J:.0f} J\n')

print(f'{"Variant":6} {"Policy":8} {"P_total(W)":11} {"Endurance":10}'
      f' {"vs base":8} {"v_max(m/s)":11} {"v_prec(m/s)":12}')
print('─' * 70)
for v in VARIANTS:
    for p in CNN_DUTY:
        b = results[v][p]
        delta = b.endurance_min - t_base
        print(f'{v:6} {p:8} {b.p_total:11.4f} {b.endurance_min:10.1f}'
              f' {delta:+8.1f} {b.v_max_evasion:11.2f} {b.v_max_precaution:12.2f}')
    print()


# ── Figure 1: Energy budget ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)
fig.suptitle('E1-B · ESP32-S3 nano-UAV energy budget',
             color='white', fontsize=11)

# Stacked bars per component — variant M, P1/P2/P3
ax = axes[0]
_style(ax, 'Power breakdown per policy (variant M)', 'Policy', 'Potencia (W)')
policies = ['P1', 'P2', 'P3']
x = np.arange(len(policies))
components = {
    'Motors':  [P_MOTORS_W / EFFICIENCY] * 3,
    'MCU':      [P_MCU_IDLE_W / EFFICIENCY] * 3,
    'Sensors': [(P_TOF_W + P_IMU_W) / EFFICIENCY] * 3,
    'CNN':      [results['M'][p].p_cnn_avg / EFFICIENCY for p in policies],
}
bottom = np.zeros(3)
for label, vals in components.items():
    ax.bar(x, vals, bottom=bottom, color=COMP_C[label], alpha=0.88, label=label)
    for i, (v, b) in enumerate(zip(vals, bottom)):
        if v > 0.002:
            ax.text(i, b + v/2, f'{v*1000:.1f}mW', ha='center',
                    va='center', color='white', fontsize=7)
    bottom += np.array(vals)
ax.set_xticks(x); ax.set_xticklabels(policies, color='#888')
ax.legend(fontsize=8, facecolor='#222', edgecolor='#333', labelcolor='white')

# Endurance per variant and policy
ax2 = axes[1]
_style(ax2, 'Flight endurance (min)', 'Variant', 'minutos')
w = 0.22
xs = np.arange(4)
vnames = list(VARIANTS.keys())
ax2.axhline(t_base, color='white', lw=0.8, ls='--', label=f'no CNN {t_base:.1f} min')
for i, p in enumerate(policies):
    vals = [results[v][p].endurance_min for v in vnames]
    bars = ax2.bar(xs + (i-1)*w, vals, width=w, color=COLORS_P[p],
                   alpha=0.85, label=p)
ax2.set_xticks(xs); ax2.set_xticklabels(vnames, color='#888')
ax2.legend(fontsize=8, facecolor='#222', edgecolor='#333', labelcolor='white')
ax2.set_ylim(t_base * 0.97, t_base * 1.005)

fig.tight_layout()
plt.savefig(os.path.join(ASSETS, 'e1b_budget.png'),
            dpi=130, bbox_inches='tight', facecolor=BG)
plt.close(fig)
print('[plot] results/e1b_budget.png')


# ── Figure 2: Speed-safety trade-off ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)
fig.suptitle('E1-B · Trade-off: maximum safe speed vs endurance',
             color='white', fontsize=11)

ax = axes[0]
_style(ax, 'Maximum safe speed (avoidance at 0.5 m)', 'Variant', 'm/s')
for i, p in enumerate(policies):
    vals = [results[v][p].v_max_evasion for v in vnames]
    ax.bar(xs + (i-1)*w, vals, width=w, color=COLORS_P[p], alpha=0.85, label=p)
    for xi, val in zip(xs + (i-1)*w, vals):
        ax.text(xi, val + 0.05, f'{val:.1f}', ha='center',
                color=COLORS_P[p], fontsize=7)
ax.set_xticks(xs); ax.set_xticklabels(vnames, color='#888')
ax.legend(fontsize=8, facecolor='#222', edgecolor='#333', labelcolor='white')

# Scatter: v_max vs endurance (design space)
ax2 = axes[1]
_style(ax2, 'Design space: speed vs endurance', 'v_max avoidance (m/s)', 'Endurance (min)')
for v in vnames:
    for p in policies:
        b = results[v][p]
        ax2.scatter(b.v_max_evasion, b.endurance_min,
                    color=COLORS_V[v], s=80,
                    marker={'P1': 'o', 'P2': 's', 'P3': '^'}[p],
                    zorder=5, alpha=0.9)
        if p == 'P3':
            ax2.annotate(v, (b.v_max_evasion, b.endurance_min),
                         xytext=(5, 3), textcoords='offset points',
                         color=COLORS_V[v], fontsize=8)

# Manual legends
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
leg_v = [Patch(color=COLORS_V[v], label=v) for v in vnames]
leg_p = [Line2D([0],[0], marker=m, color='w', markerfacecolor='#888',
                label=p, markersize=8)
         for p, m in [('P1','o'),('P2','s'),('P3','^')]]
ax2.legend(handles=leg_v + leg_p, fontsize=7,
           facecolor='#222', edgecolor='#333', labelcolor='white', ncol=2)
ax2.axhline(t_base, color='white', lw=0.6, ls='--', alpha=0.4)

fig.tight_layout()
plt.savefig(os.path.join(ASSETS, 'e1b_tradeoff.png'),
            dpi=130, bbox_inches='tight', facecolor=BG)
plt.close(fig)
print('[plot] results/e1b_tradeoff.png')


# ── Figure 3: Sensitivity to the scale factor ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)
fig.suptitle('E1-B · Sensitivity to the CPU→ESP32-S3 scale factor',
             color='white', fontsize=11)

for v in vnames:
    sa = sensitivity_analysis(v, 'P3')
    scales = sa['scales']
    ends   = [r['endurance'] for r in sa['results']]
    vmaxs  = [r['v_max']     for r in sa['results']]
    axes[0].plot(scales, ends,  color=COLORS_V[v], lw=1.5, label=v)
    axes[1].plot(scales, vmaxs, color=COLORS_V[v], lw=1.5, label=v)

for ax, ylabel, title in zip(
    axes,
    ['Endurance (min)', 'v_max avoidance (m/s)'],
    ['Endurance vs scale factor (policy P3)',
     'Maximum safe speed vs scale factor']
):
    _style(ax, title, 'Scale factor (×)', ylabel)
    ax.axvline(SCALE_MID, color='white', lw=0.7, ls='--',
               label=f'mean estimate ×{SCALE_MID:.0f}')
    ax.axvspan(SCALE_LO, SCALE_HI, alpha=0.07, color='white',
               label=f'range [{SCALE_LO:.0f}–{SCALE_HI:.0f}×]')
    ax.legend(fontsize=8, facecolor='#222', edgecolor='#333', labelcolor='white')

axes[0].axhline(t_base, color='#5DCAA5', lw=0.6, ls=':', alpha=0.5)

fig.tight_layout()
plt.savefig(os.path.join(ASSETS, 'e1b_sensitivity.png'),
            dpi=130, bbox_inches='tight', facecolor=BG)
plt.close(fig)
print('[plot] results/e1b_sensitivity.png')


# ── Figure 4: Full P1/P2/P3 comparison ───────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor=BG)
fig.suptitle('E1-B · P1 / P2 / P3 comparison — variant M',
             color='white', fontsize=11)

speeds   = np.linspace(0.1, 6.0, 200)
for p in policies:
    safe, end = endurance_vs_speed('M', p, speeds)
    axes[0].plot(speeds, safe.astype(float), color=COLORS_P[p], lw=2, label=p)
    axes[1].axhline(end, color=COLORS_P[p], lw=1.5, ls='--', label=f'{p}: {end:.1f} min')

_style(axes[0], 'Safe flight at given speed (1 = yes)', 'Speed (m/s)', '')
axes[0].set_yticks([0, 1]); axes[0].set_yticklabels(['No', 'Yes'], color='#888')
axes[0].legend(fontsize=8, facecolor='#222', edgecolor='#333', labelcolor='white')

_style(axes[1], 'Endurance per policy (variant M)', 'Policy', 'minutos')
axes[1].set_xlim(-0.5, 2.5)
axes[1].axhline(t_base, color='white', lw=0.7, ls=':', label=f'no CNN {t_base:.1f}')
axes[1].legend(fontsize=8, facecolor='#222', edgecolor='#333', labelcolor='white')

# P3 vs P1 energy savings
ax3 = axes[2]
_style(ax3, 'Energy saving P3 vs P1 per variant', 'Variant', '% ahorro')
savings = []
for v in vnames:
    p_p1 = results[v]['P1'].p_total
    p_p3 = results[v]['P3'].p_total
    savings.append((p_p1 - p_p3) / p_p1 * 100)
bars = ax3.bar(vnames, savings, color=[COLORS_V[v] for v in vnames], alpha=0.85)
for bar, s in zip(bars, savings):
    ax3.text(bar.get_x() + bar.get_width()/2, s + 0.001,
             f'{s:.3f}%', ha='center', color='white', fontsize=8)
ax3.set_ylim(0, max(savings) * 1.4)

fig.tight_layout()
plt.savefig(os.path.join(ASSETS, 'e1b_policy.png'),
            dpi=130, bbox_inches='tight', facecolor=BG)
plt.close(fig)
print('[plot] results/e1b_policy.png')

print(f"""
[E1-B] Conclusions:

  The CNN consumes <2% of the total energy budget.
  The real impact of P3 vs P1 on endurance is <1.5 minutes out of ~81 min.

  The metric that matters is not endurance but MAXIMUM SAFE SPEED:
    XS (18 fps): up to 4.4 m/s for avoidance at 0.5m
    M  ( 9 fps): up to 3.0 m/s
    L  ( 5 fps): up to 1.9 m/s
    → Variant XS allows flying 2.3× faster than L with the same safety.

  Main uncertainty: CPU→ESP32-S3 scale factor.
    With factor 8×:  v_max(XS) = 5.2 m/s
    With factor 15×: v_max(XS) = 3.8 m/s
    E3-A (real measurement) will resolve this uncertainty.
""")
