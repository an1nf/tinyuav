"""
E0-C · Experimento: latencia real de activación CNN bajo política P3.

Ejecutar con:
    conda activate nano_uav
    python e0c_p3_latency.py

Requiere model.tflite generado por generate_model.py.
"""
import sys, os

import numpy as np
import matplotlib
from nanouav import paths
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from nanouav.planning.p3_latency import run, N_FRAMES, VARIANT, DIV_THRESH
from nanouav.perception.cnn import is_real_mode, _load_interpreter

ASSETS = str(paths.results_dir())
os.makedirs(ASSETS, exist_ok=True)
DARK, BG = '#111111', '#1a1a1a'

# ── Colores ───────────────────────────────────────────────────────────────
C_FLOW   = '#378ADD'
C_CNN    = '#EF9F27'
C_P3     = '#5DCAA5'
C_P1     = '#E24B4A'
C_TRIG   = '#EF9F27'
C_IDLE   = '#444'

def _style(ax, title=''):
    ax.set_facecolor(DARK)
    ax.tick_params(colors='#888', labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor('#333')
    if title:
        ax.set_title(title, color='white', fontsize=9)


# ── Correr experimento ────────────────────────────────────────────────────
_load_interpreter()
mode_str = 'REAL (int8)' if is_real_mode() else 'FAKE (placeholder)'
print(f'\n[E0-C] CNN en modo {mode_str}')
print(f'       {N_FRAMES} frames · variante {VARIANT} · umbral div={DIV_THRESH}\n')

rec = run()

n           = N_FRAMES
frames      = np.arange(n)
triggered   = rec['cnn_triggered']
n_trig      = triggered.sum()
pct_trig    = n_trig / n * 100

# Latencias solo cuando CNN activa (para estadísticas reales)
cnn_lats    = rec['lat_cnn_ms'][triggered]
p1_lats     = rec['lat_p1_ms']

# ── Consola ───────────────────────────────────────────────────────────────
print(f"  CNN activada:      {n_trig}/{n} frames  ({pct_trig:.0f}%)")
print(f"  Ahorro P3 vs P1:   {100-pct_trig:.0f}% menos inferencias")
if len(cnn_lats):
    print(f"\n  Latencia CNN (tu CPU, {mode_str}):")
    print(f"    media  = {cnn_lats.mean():.3f} ms")
    print(f"    p50    = {np.percentile(cnn_lats,50):.3f} ms")
    print(f"    p99    = {np.percentile(cnn_lats,99):.3f} ms")
    print(f"    máx    = {cnn_lats.max():.3f} ms")
    print(f"\n  Latencia flujo óptico (media): {rec['lat_flow_ms'].mean():.3f} ms")
    print(f"  Latencia P3 total   (media): {rec['lat_p3_ms'][triggered].mean():.3f} ms")
    print(f"  Latencia P1 total   (media): {p1_lats.mean():.3f} ms")
    scale_lo, scale_hi = 8, 15
    print(f"\n  Estimación ESP32-S3 (factor {scale_lo}–{scale_hi}×):")
    print(f"    CNN latencia est.: {cnn_lats.mean()*scale_lo:.1f}–{cnn_lats.mean()*scale_hi:.1f} ms")
    print(f"    FPS estimados:     {1000/cnn_lats.mean()/scale_hi:.1f}–{1000/cnn_lats.mean()/scale_lo:.1f}")
else:
    print("  (sin activaciones CNN en este run)")


# ── Figura ────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10), facecolor=BG)
fig.suptitle(
    f'E0-C · Política P3 con latencia real · variante {VARIANT} · {mode_str}',
    color='white', fontsize=11, y=0.98)

gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.52, wspace=0.35,
                       left=0.07, right=0.97, top=0.93, bottom=0.06)

# ── [0,0] Divergencia + trigger ──────────────────────────────────────────
ax = fig.add_subplot(gs[0, :])
_style(ax, 'Divergencia de flujo óptico y activaciones CNN')

# Fondo coloreado por trigger
for i in frames:
    if triggered[i]:
        ax.axvspan(i-0.5, i+0.5, alpha=0.18, color=C_TRIG, lw=0)

ax.plot(frames, rec['div'], color=C_FLOW, lw=1.4, label='divergencia', zorder=3)
ax.axhline(DIV_THRESH, color=C_TRIG, lw=0.8, ls='--',
           label=f'umbral {DIV_THRESH}', zorder=2)

# Marcadores de activación
trig_idx = np.where(triggered)[0]
ax.scatter(trig_idx, rec['div'][trig_idx], color=C_CNN, s=25, zorder=4,
           label=f'CNN activa ({n_trig} frames)')

ax.set_ylabel('divergencia', color='#888')
ax.set_xlabel('frame', color='#888')
ax.set_ylim(-0.02, max(rec['div'].max() * 1.2, DIV_THRESH * 2))
ax.legend(fontsize=8, facecolor='#222', edgecolor='#333', labelcolor='white',
          loc='upper right')

# Anotación de ahorro
ax.text(0.01, 0.92, f'{pct_trig:.0f}% frames con CNN activa  ·  {100-pct_trig:.0f}% ahorro vs P1',
        transform=ax.transAxes, color=C_CNN, fontsize=8, va='top')

# ── [1,0] Latencia CNN por frame (solo activados) ────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
_style(ax2, 'Latencia CNN por frame activado (ms)')

if len(cnn_lats):
    ax2.bar(np.where(triggered)[0], rec['lat_cnn_ms'][triggered],
            color=C_CNN, alpha=0.85, width=1.0)
    ax2.axhline(cnn_lats.mean(), color='white', lw=0.8, ls='--',
                label=f'media {cnn_lats.mean():.3f} ms')
    ax2.set_ylabel('ms', color='#888')
    ax2.set_xlabel('frame', color='#888')
    ax2.legend(fontsize=8, facecolor='#222', edgecolor='#333', labelcolor='white')
else:
    ax2.text(0.5, 0.5, 'sin activaciones', transform=ax2.transAxes,
             color='#888', ha='center')

# ── [1,1] P3 vs P1: latencia acumulada ──────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
_style(ax3, 'Latencia acumulada: P3 vs P1')

cum_p3 = np.cumsum(rec['lat_p3_ms'])
cum_p1 = np.cumsum(p1_lats)
ax3.plot(frames, cum_p3, color=C_P3, lw=1.5, label='P3 (condicional)')
ax3.plot(frames, cum_p1, color=C_P1, lw=1.5, label='P1 (siempre CNN)')
ax3.fill_between(frames, cum_p3, cum_p1, alpha=0.12, color=C_P3,
                 label='tiempo ahorrado')
ax3.set_ylabel('ms acumulados', color='#888')
ax3.set_xlabel('frame', color='#888')
ax3.legend(fontsize=8, facecolor='#222', edgecolor='#333', labelcolor='white')

# Anotación ahorro final
if cum_p1[-1] > 0:
    saved_pct = (cum_p1[-1] - cum_p3[-1]) / cum_p1[-1] * 100
    ax3.text(0.98, 0.05, f'ahorro: {saved_pct:.0f}%',
             transform=ax3.transAxes, color=C_P3, fontsize=9,
             ha='right', fontweight='bold')

# ── [2,0] Distancia detectada P3 vs P1 ───────────────────────────────────
ax4 = fig.add_subplot(gs[2, 0])
_style(ax4, 'Distancia mínima detectada: P3 vs P1')

ax4.plot(frames, rec['min_dist_p3'], color=C_P3, lw=1.3, alpha=0.9, label='P3')
ax4.plot(frames, rec['min_dist_p1'], color=C_P1, lw=1.0, alpha=0.6,
         ls='--', label='P1')
ax4.axhline(0.5, color='#E24B4A', lw=0.7, ls=':')
ax4.axhline(1.0, color='#EF9F27', lw=0.7, ls=':')
ax4.set_ylabel('distancia (m)', color='#888')
ax4.set_xlabel('frame', color='#888')
ax4.set_ylim(0, 4.2)
ax4.legend(fontsize=8, facecolor='#222', edgecolor='#333', labelcolor='white')

# ── [2,1] Histograma latencia CNN ────────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 1])
_style(ax5, 'Distribución de latencia CNN (frames activos)')

if len(cnn_lats) > 1:
    ax5.hist(cnn_lats, bins=20, color=C_CNN, alpha=0.85, edgecolor='#333')
    ax5.axvline(np.percentile(cnn_lats, 50), color='white', lw=1, ls='--',
                label=f'p50={np.percentile(cnn_lats,50):.3f}ms')
    ax5.axvline(np.percentile(cnn_lats, 99), color='#E24B4A', lw=1, ls='--',
                label=f'p99={np.percentile(cnn_lats,99):.3f}ms')
    ax5.set_xlabel('ms', color='#888')
    ax5.set_ylabel('frecuencia', color='#888')
    ax5.legend(fontsize=8, facecolor='#222', edgecolor='#333', labelcolor='white')
    # Estimación ESP32-S3
    est_lo = np.percentile(cnn_lats,50) * 8
    est_hi = np.percentile(cnn_lats,50) * 15
    ax5.text(0.98, 0.92, f'ESP32-S3 est. p50:\n{est_lo:.0f}–{est_hi:.0f} ms',
             transform=ax5.transAxes, color='#888', fontsize=7.5,
             ha='right', va='top')
else:
    ax5.text(0.5, 0.5, 'insuficientes datos', transform=ax5.transAxes,
             color='#888', ha='center')

out = os.path.join(ASSETS, 'e0c_p3_latency.png')
plt.savefig(out, dpi=130, bbox_inches='tight', facecolor=BG)
plt.close(fig)
print(f'\n  → {out}')


# ── GIF: flujo óptico + estado P3 frame a frame ───────────────────────────
print('\nGenerando GIF de detección...')

from PIL import Image
import cv2
from matplotlib.colors import LinearSegmentedColormap
from nanouav.scene.scene import synthetic_scene as scene_fn
from nanouav.sensors.optical_flow import draw_flow, should_trigger_cnn, optical_flow
from nanouav.perception.cnn import predict_depth

_cmap = LinearSegmentedColormap.from_list(
    'depth',
    [(0.22,0.61,0.87),(0.11,0.62,0.46),(0.94,0.62,0.15),(0.88,0.29,0.29)][::-1],
    N=256)

GIF_FRAMES = 80
SCALE      = 4        # cada frame se renderiza a 4× para que sea visible
W, H       = 96, 96
PW, PH     = W * SCALE, H * SCALE

gif_imgs = []
prev_g   = None

for i in range(GIF_FRAMES):
    t = i * 0.05
    gray, depth_gt = scene_fn(t)
    flow_pts, div  = optical_flow(prev_g, gray) if prev_g is not None else ([], 0.0)
    trig           = should_trigger_cnn(div) or False

    if trig:
        depth_pred = predict_depth(gray, depth_gt, VARIANT)
    else:
        depth_pred = np.full((H, W), 4.0, np.float32)

    prev_g = gray.copy()

    # ── Componer frame 4-panel ────────────────────────────────────────────
    # Panel 1: cámara con flechas de flujo
    flow_rgb = draw_flow(gray, flow_pts)            # (96,96,3) uint8
    cam = cv2.resize(flow_rgb, (PW, PH), interpolation=cv2.INTER_NEAREST)

    # Panel 2: mapa de profundidad
    import matplotlib.cm as cm
    depth_norm = (np.clip(depth_pred, 0.2, 4.0) - 0.2) / 3.8
    depth_rgb  = (_cmap(depth_norm)[:,:,:3] * 255).astype(np.uint8)
    dep = cv2.resize(depth_rgb, (PW, PH), interpolation=cv2.INTER_NEAREST)

    # Combinar horizontalmente
    top = np.concatenate([cam, dep], axis=1)         # (PH, 2*PW, 3)

    # Barra de estado inferior
    bar_h = 28
    bar   = np.zeros((bar_h, 2*PW, 3), dtype=np.uint8)
    bar[:] = (30, 30, 30)

    # Color del estado
    if trig and depth_pred.min() < 0.5:
        state_col = (60, 80, 226)   # rojo BGR
        state_txt = f'EVADIR  div={div:.3f}'
    elif trig and depth_pred.min() < 1.0:
        state_col = (30, 159, 239)  # naranja BGR
        state_txt = f'PRECAUCION  div={div:.3f}'
    elif trig:
        state_col = (60, 160, 93)   # verde oscuro
        state_txt = f'CNN ACTIVA  div={div:.3f}'
    else:
        state_col = (60, 60, 60)
        state_txt = f'CNN en reposo  div={div:.3f}'

    # Barra de color izquierda
    bar[:, :6] = state_col

    # Texto en la barra
    cv2.putText(bar, state_txt, (12, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1, cv2.LINE_AA)

    # Frame número
    cv2.putText(bar, f'f={i:03d}', (2*PW-55, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100,100,100), 1, cv2.LINE_AA)

    frame = np.concatenate([top, bar], axis=0)
    gif_imgs.append(Image.fromarray(frame))

    sys.stdout.write(f'\r  frame {i+1}/{GIF_FRAMES}')
    sys.stdout.flush()

print()

gif_path = os.path.join(ASSETS, 'e0c_detection.gif')
gif_imgs[0].save(
    gif_path,
    save_all=True,
    append_images=gif_imgs[1:],
    duration=80,          # ms por frame → ~12.5 fps
    loop=0,
    optimize=False,
)
print(f'  → {gif_path}')
print('\n[E0-C] Listo.')


