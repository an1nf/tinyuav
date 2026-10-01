import sys, os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from nanouav.scene.scene import synthetic_scene
from nanouav.sensors.tof import tof_8x8
from nanouav.sensors.optical_flow import optical_flow
from nanouav.perception.cnn import VARIANTS, predict_depth, jitter_sample
from nanouav.planning.policy import policy_p1, policy_p2, policy_p3

BATTERY_MAH = 550
BATTERY_V = 3.7
P_BASE_W = 1.5
E_FRAME_MJ_REF = 1.2   # for variant M
SPEED_MS = 0.5
T_REACT_S = 0.05
N = 200

DARK = '#111111'
BG   = '#1a1a1a'
COLORS = {'XS': '#5DCAA5', 'S': '#378ADD', 'M': '#EF9F27', 'L': '#E24B4A'}
POLICY_COLORS = {'P1': '#E24B4A', 'P2': '#EF9F27', 'P3': '#5DCAA5'}


def _style(ax):
    ax.set_facecolor(DARK)
    ax.tick_params(colors='#888', labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor('#333')
    ax.title.set_color('white')
    ax.xaxis.label.set_color('#888')
    ax.yaxis.label.set_color('#888')


def run_benchmark(n: int = N):
    results = {}
    for vname, v in VARIANTS.items():
        jitters, dists, divs = [], [], []
        prev = None
        for i in range(n):
            gray, depth_gt = synthetic_scene(i * 0.04)
            depth_pred = predict_depth(gray, depth_gt, vname)
            _, div = optical_flow(prev, gray) if prev is not None else ([], 0.0)
            prev = gray.copy()
            jitters.append(jitter_sample(vname, div))
            dists.append(float(depth_pred.min()))
            divs.append(div)
        results[vname] = {'jitters': jitters, 'dists': dists, 'divs': divs}
    return results


def run_policy_benchmark(variant: str = 'M', n: int = N):
    policy_results = {'P1': [], 'P2': [], 'P3': []}
    cnn_activations = {'P1': 0, 'P2': 0, 'P3': 0}
    prev = None
    for i in range(n):
        gray, depth_gt = synthetic_scene(i * 0.04)
        depth_pred = predict_depth(gray, depth_gt, variant)
        tof = tof_8x8(depth_gt)
        _, div = optical_flow(prev, gray) if prev is not None else ([], 0.0)
        prev = gray.copy()

        d1 = policy_p1(depth_pred)
        d2 = policy_p2(tof, depth_pred)
        d3 = policy_p3(div, tof, depth_pred)

        policy_results['P1'].append(d1.min_dist)
        policy_results['P2'].append(d2.min_dist)
        policy_results['P3'].append(d3.min_dist)

        cnn_activations['P1'] += 1
        if d2.triggered_by == 'cnn':
            cnn_activations['P2'] += 1
        if d3.triggered_by == 'cnn':
            cnn_activations['P3'] += 1

    activation_pct = {k: v / n * 100 for k, v in cnn_activations.items()}
    return policy_results, activation_pct


def plot_jitter(results, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor=BG)
    ax1, ax2 = axes
    for ax in axes:
        _style(ax)

    ax1.set_title('PID jitter per variant (200 frames)')
    for vname, res in results.items():
        ax1.plot(res['jitters'], label=vname, color=COLORS[vname], lw=1.2, alpha=0.85)
    ax1.axhline(200, color='white', lw=0.7, ls='--', label='200 µs limit')
    ax1.set_ylabel('Jitter (µs)')
    ax1.set_xlabel('Frame')
    ax1.legend(fontsize=8, facecolor='#222', edgecolor='#333', labelcolor='white')

    ax2.set_title('Trade-off: jitter p99 vs FPS')
    for vname, res in results.items():
        p99 = sorted(res['jitters'])[int(0.99 * N)]
        fps = VARIANTS[vname]['fps']
        ax2.scatter(fps, p99, color=COLORS[vname], s=120, zorder=5)
        ax2.annotate(vname, (fps, p99), xytext=(5, 5),
                     textcoords='offset points', color=COLORS[vname], fontsize=9)
    ax2.axhline(200, color='white', lw=0.7, ls='--', label='200 µs limit')
    ax2.set_xlabel('Inference FPS')
    ax2.set_ylabel('Jitter p99 (µs)')
    ax2.legend(fontsize=8, facecolor='#222', edgecolor='#333', labelcolor='white')

    fig.tight_layout()
    path = os.path.join(out_dir, 'benchmark_jitter.png')
    plt.savefig(path, dpi=120, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'  → {path}')


def plot_energy(out_dir):
    battery_j = BATTERY_MAH * 1e-3 * BATTERY_V * 3600
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor=BG)
    for ax in axes:
        _style(ax)

    axes[0].set_title('Estimated flight endurance')
    axes[1].set_title(f'Safe detection distance (at {SPEED_MS} m/s)')

    for vname, v in VARIANTS.items():
        fps = v['fps']
        e_frame_j = E_FRAME_MJ_REF * 1e-3 * (fps / 9)
        p_total = P_BASE_W + e_frame_j * fps
        endurance = battery_j / p_total / 60
        d_safe = SPEED_MS * (1 / fps + T_REACT_S) * 100  # cm

        c = COLORS[vname]
        axes[0].bar(vname, endurance, color=c, alpha=0.85)
        axes[0].text(vname, endurance + 0.1, f'{endurance:.1f} min', color=c,
                     ha='center', fontsize=8)
        axes[1].bar(vname, d_safe, color=c, alpha=0.85)
        axes[1].text(vname, d_safe + 0.3, f'{d_safe:.0f} cm', color=c,
                     ha='center', fontsize=8)

    axes[0].set_ylabel('minutes')
    axes[1].set_ylabel('centimeters')
    fig.tight_layout()
    path = os.path.join(out_dir, 'benchmark_energy.png')
    plt.savefig(path, dpi=120, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'  → {path}')


def plot_policy(policy_results, activation_pct, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor=BG)
    for ax in axes:
        _style(ax)

    axes[0].set_title('Detected distance per policy (variant M)')
    for pname, dists in policy_results.items():
        axes[0].plot(dists, label=pname, color=POLICY_COLORS[pname], lw=1.2, alpha=0.85)
    axes[0].axhline(0.5, color='#E24B4A', lw=0.8, ls='--', label='avoidance 0.5 m')
    axes[0].axhline(1.0, color='#EF9F27', lw=0.8, ls='--', label='caution 1.0 m')
    axes[0].set_ylabel('Minimum distance (m)')
    axes[0].set_xlabel('Frame')
    axes[0].legend(fontsize=8, facecolor='#222', edgecolor='#333', labelcolor='white')

    axes[1].set_title('% of frames with the CNN active, per policy')
    bars = axes[1].bar(list(activation_pct.keys()), list(activation_pct.values()),
                       color=[POLICY_COLORS[p] for p in activation_pct], alpha=0.85)
    for bar, (pname, pct) in zip(bars, activation_pct.items()):
        axes[1].text(bar.get_x() + bar.get_width() / 2, pct + 1,
                     f'{pct:.0f}%', color=POLICY_COLORS[pname], ha='center', fontsize=9)
    axes[1].set_ylabel('% CNN activations')
    axes[1].set_ylim(0, 115)

    fig.tight_layout()
    path = os.path.join(out_dir, 'benchmark_policy.png')
    plt.savefig(path, dpi=120, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'  → {path}')


if __name__ == '__main__':
    from nanouav import paths
    out_dir = str(paths.results_dir())
    os.makedirs(out_dir, exist_ok=True)

    print('Running variant benchmark...')
    results = run_benchmark()
    for vname, res in results.items():
        p99 = sorted(res['jitters'])[int(0.99 * N)]
        print(f'  {vname}: jitter p99={p99} µs | dist_min={np.mean(res["dists"]):.2f} m')
    plot_jitter(results, out_dir)
    plot_energy(out_dir)

    print('Running policy benchmark...')
    pol_results, act_pct = run_policy_benchmark()
    for pname, pct in act_pct.items():
        print(f'  {pname}: CNN active {pct:.0f}% of the time')
    plot_policy(pol_results, act_pct, out_dir)
