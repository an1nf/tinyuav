"""
E2-B · Experiment: runtime ToF→CNN calibration.

Simulates a flight of N frames in which the calibrator progressively
adjusts the CNN depth scale using the ToF.

Run:
    conda activate nano_uav
    nanouav run e2b

Produces:
    results/e2b_calibration.png   — convergence of the scale estimator
    results/e2b_depth_compare.png — depth before/after calibration
"""
import os, sys, time
import numpy as np
import matplotlib
from nanouav import paths
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = str(paths.repo_root())

from nanouav.scene.scene import synthetic_scene
from nanouav.scene.renderer import render_scene
from nanouav.sensors.tof import tof_8x8
from nanouav.perception.cnn import predict_depth_norm, is_real_mode, _load_interpreter
from nanouav.config import norm_to_m
from nanouav.perception.tof_calibrator import ToFCalibrator
from nanouav.config import D_MAX as DEPTH_MAX, D_MIN as DEPTH_MIN

ASSETS = str(paths.results_dir())
os.makedirs(ASSETS, exist_ok=True)
DARK, BG = '#111111', '#1a1a1a'
_cmap = LinearSegmentedColormap.from_list(
    'depth',
    [(0.22,0.61,0.87),(0.11,0.62,0.46),(0.94,0.62,0.15),(0.88,0.29,0.29)][::-1],
    N=256)

N_FRAMES = 150
VARIANT  = 'M'


def _style(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(DARK)
    ax.tick_params(colors='#888', labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor('#333')
    if title:  ax.set_title(title,  color='white', fontsize=9)
    if xlabel: ax.set_xlabel(xlabel, color='#888',  fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, color='#888',  fontsize=8)


_load_interpreter()
print(f'[E2-B] CNN mode {"REAL" if is_real_mode() else "FAKE"}')
print(f'       {N_FRAMES} frames · variant {VARIANT}\n')

cal = ToFCalibrator(alpha=0.15)

# Accumulators
frames       = np.arange(N_FRAMES)
gt_min       = np.zeros(N_FRAMES)    # true minimum depth
raw_min      = np.zeros(N_FRAMES)    # uncalibrated CNN
cal_min      = np.zeros(N_FRAMES)    # calibrated CNN
tof_min_arr  = np.zeros(N_FRAMES)    # minimum ToF
scale_arr    = np.zeros(N_FRAMES)    # EMA scale

# Store 4 frames for visualization
preview_frames = [10, 30, 60, 120]
previews = {}

for i in range(N_FRAMES):
    # Scene: alternate between the 3D renderer and the original synthetic scene
    if i % 3 == 0:
        gray, depth_gt = render_scene(seed=i * 7)
    else:
        gray, depth_gt = synthetic_scene(t=i * 0.06)

    # CNN inference (normalized output [0,1])
    depth_pred_norm = predict_depth_norm(gray, VARIANT, depth_gt)
    # Uncalibrated meters (scale learned by the CNN)
    depth_raw_m = norm_to_m(depth_pred_norm)

    # Simulated ToF (absolute meters)
    tof = tof_8x8(depth_gt)

    # Calibrate and get depth in meters
    depth_cal_m = cal.update_and_scale(depth_pred_norm, tof)

    # Record metrics
    gt_min[i]      = float(depth_gt.min())
    raw_min[i]     = float(depth_raw_m.min())
    cal_min[i]     = float(depth_cal_m.min())
    tof_min_arr[i] = float(tof.min())
    scale_arr[i]   = cal.scale_ema

    if i in preview_frames:
        previews[i] = {
            'gray':      gray.copy(),
            'depth_gt':  depth_gt.copy(),
            'depth_raw': depth_raw_m.copy(),
            'depth_cal': depth_cal_m.copy(),
            'tof':       tof.copy(),
            'scale':     cal.scale_ema,
            'n_updates': cal.n_updates,
        }

summary = cal.summary()
print(f'Final EMA scale:   {summary["scale_ema"]:.3f}')
print(f'Updates:           {summary["n_updates"]}')
print(f'Mean scale:        {summary["scale_mean"]:.3f} ± {summary["scale_std"]:.3f}')

# Minimum-depth error: raw CNN vs calibrated CNN vs ground truth
err_raw = np.abs(raw_min - gt_min)
err_cal = np.abs(cal_min - gt_min)
print(f'\nMinimum-distance error (MAE):')
print(f'  Uncalibrated CNN: {err_raw.mean():.3f} m')
print(f'  Calibrated CNN:   {err_cal.mean():.3f} m')
print(f'  Improvement:      {(1 - err_cal.mean()/err_raw.mean())*100:.1f}%')


# ── Figure 1: Calibrator convergence ─────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 8), facecolor=BG)
fig.suptitle('E2-B · ToF→CNN calibration: convergence of the scale estimator',
             color='white', fontsize=11)

ax = axes[0, 0]
_style(ax, 'EMA scale factor along the flight',
       'frame', 'escala')
ax.plot(frames, scale_arr, color='#5DCAA5', lw=1.5)
ax.axhline(1.0, color='white', lw=0.6, ls='--', label='scale=1 (uncalibrated)')
ax.fill_between(frames, scale_arr * 0.9, scale_arr * 1.1,
                alpha=0.1, color='#5DCAA5')
ax.axvline(5, color='#EF9F27', lw=0.8, ls=':', label='calibrated (5 updates)')
ax.legend(fontsize=8, facecolor='#222', edgecolor='#333', labelcolor='white')
ax.text(0.97, 0.05, f'final scale: {summary["scale_ema"]:.3f}',
        transform=ax.transAxes, color='#5DCAA5', fontsize=8, ha='right')

ax = axes[0, 1]
_style(ax, 'Minimum detected distance', 'frame', 'metros')
ax.plot(frames, gt_min,      color='white',    lw=1.0, alpha=0.6, label='ground truth')
ax.plot(frames, tof_min_arr, color='#9B59B6',  lw=1.2, alpha=0.8, label='ToF')
ax.plot(frames, raw_min,     color='#E24B4A',  lw=1.2, alpha=0.8, label='CNN uncal.')
ax.plot(frames, cal_min,     color='#5DCAA5',  lw=1.5, label='calibrated CNN')
ax.axhline(0.5, color='#E24B4A', lw=0.6, ls='--')
ax.legend(fontsize=7, facecolor='#222', edgecolor='#333', labelcolor='white')

ax = axes[1, 0]
_style(ax, 'Absolute error: minimum distance', 'frame', '|error| (m)')
ax.plot(frames, err_raw, color='#E24B4A', lw=1.2, alpha=0.8, label=f'Uncal. MAE={err_raw.mean():.3f}m')
ax.plot(frames, err_cal, color='#5DCAA5', lw=1.5, label=f'Calibrated MAE={err_cal.mean():.3f}m')
ax.fill_between(frames, 0, err_raw - err_cal,
                where=err_raw > err_cal, alpha=0.15, color='#5DCAA5',
                label='improvement')
ax.legend(fontsize=8, facecolor='#222', edgecolor='#333', labelcolor='white')

ax = axes[1, 1]
_style(ax, 'Histograma error (m)', '|error| (m)', 'frequency')
ax.hist(err_raw, bins=30, color='#E24B4A', alpha=0.6, label='Uncalibrated')
ax.hist(err_cal, bins=30, color='#5DCAA5', alpha=0.7, label='Calibrated')
ax.axvline(err_raw.mean(), color='#E24B4A', lw=1, ls='--')
ax.axvline(err_cal.mean(), color='#5DCAA5', lw=1, ls='--')
ax.legend(fontsize=8, facecolor='#222', edgecolor='#333', labelcolor='white')

fig.tight_layout()
out = os.path.join(ASSETS, 'e2b_calibration.png')
plt.savefig(out, dpi=130, bbox_inches='tight', facecolor=BG)
plt.close(fig)
print(f'\n[plot] {out}')


# ── Figure 2: Before/after visual comparison ──────────────────────────────
n_prev = len(previews)
fig, axes = plt.subplots(n_prev, 4, figsize=(13, n_prev * 3), facecolor=BG)
fig.suptitle('E2-B · CNN depth: uncalibrated vs calibrated vs GT',
             color='white', fontsize=11)
cols = ['Camera', 'CNN uncalibrated', 'CNN calibrated', 'ToF 8×8 GT']

for row, (fidx, pv) in enumerate(sorted(previews.items())):
    for col in range(4):
        ax = axes[row, col]
        ax.set_facecolor(DARK); ax.axis('off')
        if row == 0:
            ax.set_title(cols[col], color='white', fontsize=8)
        if col == 0:
            ax.imshow(pv['gray'], cmap='gray')
            ax.set_ylabel(f'f={fidx}\nscale={pv["scale"]:.2f}',
                          color='#888', fontsize=7)
        elif col == 1:
            ax.imshow(pv['depth_raw'], cmap=_cmap, vmin=DEPTH_MIN, vmax=DEPTH_MAX)
            ax.set_xlabel(f'min={pv["depth_raw"].min():.2f}m', color='#E24B4A', fontsize=7)
        elif col == 2:
            ax.imshow(pv['depth_cal'], cmap=_cmap, vmin=DEPTH_MIN, vmax=DEPTH_MAX)
            ax.set_xlabel(f'min={pv["depth_cal"].min():.2f}m', color='#5DCAA5', fontsize=7)
        else:
            ax.imshow(pv['tof'], cmap=_cmap, vmin=DEPTH_MIN, vmax=DEPTH_MAX)
            ax.set_xlabel(f'min={pv["tof"].min():.2f}m', color='#9B59B6', fontsize=7)

fig.tight_layout()
out2 = os.path.join(ASSETS, 'e2b_depth_compare.png')
plt.savefig(out2, dpi=130, bbox_inches='tight', facecolor=BG)
plt.close(fig)
print(f'[plot] {out2}')
print('\n[E2-B] Done.')
