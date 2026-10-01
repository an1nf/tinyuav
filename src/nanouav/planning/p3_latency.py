"""
E0-C · Policy P3 with real measurement of CNN activation latency.

What it measures:
  - CNN inference latency (time.perf_counter) per triggered frame
  - Total P3 pipeline latency (flow + decision + CNN if triggered)
  - Divergence distribution: when it triggers and when it does not
  - Reaction-time comparison: P1 (always) vs P3 (conditional)

The latency measured here is real (your CPU), not estimated.
The scaling factor to the ESP32-S3 is roughly 8–15× slower.
"""
import time
import sys
import os
import numpy as np


from nanouav.scene.scene import synthetic_scene
from nanouav.sensors.tof import tof_8x8
from nanouav.sensors.optical_flow import optical_flow, should_trigger_cnn, DIV_THRESHOLD
from nanouav.perception.cnn import predict_depth, is_real_mode, VARIANTS
from nanouav.planning.policy import policy_p1, policy_p3, Decision

N_FRAMES   = 120
VARIANT    = 'M'
DIV_THRESH = DIV_THRESHOLD


def _now() -> float:
    return time.perf_counter() * 1000  # ms


def run(n: int = N_FRAMES, variant: str = VARIANT) -> dict:
    """
    Runs the full P3 pipeline and collects latency metrics.
    Returns a dict of result arrays.
    """
    records = {
        # per frame
        'div':           np.zeros(n),
        'cnn_triggered': np.zeros(n, dtype=bool),
        'lat_flow_ms':   np.zeros(n),   # optical-flow time only
        'lat_cnn_ms':    np.zeros(n),   # CNN inference time (0 if not triggered)
        'lat_p3_ms':     np.zeros(n),   # total P3 time (flow + CNN if triggered)
        'lat_p1_ms':     np.zeros(n),   # P1 time (always CNN) for comparison
        'min_dist_p3':   np.zeros(n),
        'min_dist_p1':   np.zeros(n),
        'action_p3':     [],
    }

    prev_gray = None

    for i in range(n):
        t = i * 0.05
        gray, depth_gt = synthetic_scene(t)
        tof = tof_8x8(depth_gt)

        # ── Optical flow (always runs) ───────────────────────────────────
        t0 = _now()
        flow_pts, div = optical_flow(prev_gray, gray) if prev_gray is not None else ([], 0.0)
        t_flow = _now() - t0

        # ── P3: conditional CNN ───────────────────────────────────────────
        t0 = _now()
        triggered = should_trigger_cnn(div) or float(tof.min()) < 0.4
        lat_cnn = 0.0
        if triggered:
            t_cnn0 = _now()
            depth_pred = predict_depth(gray, depth_gt, variant)
            lat_cnn = _now() - t_cnn0
            dec_p3 = policy_p3(div, tof, depth_pred)
        else:
            # no CNN: use the ToF or return free
            depth_pred_dummy = np.full((96, 96), 4.0, dtype=np.float32)
            dec_p3 = policy_p3(div, tof, depth_pred_dummy)
        lat_p3 = _now() - t0 + t_flow

        # ── P1: CNN always ────────────────────────────────────────────────
        t0 = _now()
        depth_pred_p1 = predict_depth(gray, depth_gt, variant)
        dec_p1 = policy_p1(depth_pred_p1)
        lat_p1 = _now() - t0

        # ── Record ────────────────────────────────────────────────────────
        records['div'][i]           = div
        records['cnn_triggered'][i] = triggered
        records['lat_flow_ms'][i]   = t_flow
        records['lat_cnn_ms'][i]    = lat_cnn
        records['lat_p3_ms'][i]     = lat_p3
        records['lat_p1_ms'][i]     = lat_p1
        records['min_dist_p3'][i]   = dec_p3.min_dist
        records['min_dist_p1'][i]   = dec_p1.min_dist
        records['action_p3'].append(dec_p3.action)

        prev_gray = gray.copy()

    return records
