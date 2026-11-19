"""Episode videos from the drone's own perception.

perception  camera with the chosen trajectory, SGM depth, local map of its own SGM, network decisions
sgm         full SGM depth with the chosen trajectory, network decisions below

Written with OpenCV and re-encoded to H.264 when ffmpeg is available.
"""
import math
import os

import cv2
import numpy as np

R_DRONE = 0.2
WHITE, ORANGE, CYAN = (255, 255, 255), (0, 140, 255), (230, 200, 0)


def _writer(fn, fps, size):
    w = cv2.VideoWriter(fn, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    if not w.isOpened():
        raise RuntimeError(f'cannot open a video writer for {fn}')
    return w


def _to_h264(fn):
    """Re-encode to H.264 when ffmpeg is installed."""
    import shutil
    import subprocess
    if not shutil.which('ffmpeg'):
        return fn
    tmp = fn[:-4] + '.h264.mp4'
    r = subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', fn, '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                        '-crf', '23', tmp])
    if r.returncode == 0:
        os.replace(tmp, fn)
    return fn


BG = (26, 24, 22)                                   # BGR
PANEL_TXT = (225, 225, 225)
DIM_TXT = (150, 150, 150)
_FONTS = {}


def _font(size):
    """DejaVu Sans via Pillow, or None for OpenCV text."""
    if size not in _FONTS:
        _FONTS[size] = None
        try:
            from PIL import ImageFont
            cands = ['/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']
            try:
                import matplotlib
                cands.append(os.path.join(matplotlib.get_data_path(), 'fonts', 'ttf', 'DejaVuSans.ttf'))
            except ImportError:
                pass
            for c in cands:
                if os.path.exists(c):
                    _FONTS[size] = ImageFont.truetype(c, size)
                    break
        except ImportError:
            pass
    return _FONTS[size]


def _texts(img, items, outline=0):
    """Draw [(text, (x, y), size px, BGR)] in one Pillow pass; outline > 0 adds a dark edge."""
    if not items:
        return img
    if _font(12) is None:                              # no Pillow: OpenCV text
        for t, (x, y), sz, col in items:
            cv2.putText(img, t.replace('ω', 'w'), (int(x), int(y + sz)), cv2.FONT_HERSHEY_SIMPLEX, sz / 30, col, 1,
                        cv2.LINE_AA)
        return img
    from PIL import Image, ImageDraw
    pil = Image.fromarray(img[..., ::-1])
    dr = ImageDraw.Draw(pil)
    for t, (x, y), sz, col in items:
        dr.text((int(x), int(y)), t, font=_font(sz), fill=(col[2], col[1], col[0]),
                stroke_width=outline, stroke_fill=(12, 12, 12))
    img[:] = np.asarray(pil)[..., ::-1]
    return img


def _free_spot(xy, taken, dx=26, dy=14):
    """True if a label at xy clears every placed label (dx x dy boxes)."""
    return all(abs(xy[0] - a) > dx or abs(xy[1] - b) > dy for a, b in taken)


D_VIS = 10.0                                        # m, depth colour range


def _cost_color(c):
    """Predicted collision cost to BGR, green to red."""
    t = float(np.clip(c / 0.3, 0, 1))
    return (int(60 + 0 * t), int(200 * (1 - t) + 60 * t), int(60 * (1 - t) + 230 * t))


def _ribbon(P, half=R_DRONE):
    """Left and right edges (N, 3) of a centre line at +-half."""
    d = np.gradient(P[:, :2], axis=0)
    n = np.stack([-d[:, 1], d[:, 0]], 1)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-6)
    off = np.c_[n * half, np.zeros(len(P))]
    return P + off, P - off


def _blend_poly(img, pts, color, alpha):
    ov = img.copy()
    cv2.fillPoly(ov, [pts], color, cv2.LINE_AA)
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)


KEPT_COL = (235, 235, 235)                          # kept previous plan


def _hypotheses(d):
    """(hypotheses, predicted costs, chosen index; -1 when the previous plan was kept)."""
    W = [np.asarray(w, float) for w in d['W']]
    costs = list(d.get('costs') or [])
    return W, costs, -1 if (d.get('kept') or d.get('fanned')) else int(d.get('k', 0))


def _camera_hud(d, rig, w=800, h=600):
    from tinyuav.follow.estimate import project
    img = d['img']
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img[..., ::-1].copy()
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    sx, sy = w / rig.width, h / rig.height
    pose = (np.array(d['p']), *d['att'])
    W, costs, k_sel = _hypotheses(d)
    p0 = np.array(d['p'], float)
    items, idx = [], []
    for k in sorted(range(len(W)), key=lambda k: k == k_sel):
        P = np.vstack([p0, W[k]])
        priv = not costs
        col = (230, 200, 0) if priv else _cost_color(costs[k])
        if k == k_sel:                                   # chosen trajectory: tunnel
            items += _tunnel(img, rig, pose, P, sx, sy, 0, col)
        else:                                           # other hypotheses: centre lines
            uv, X = project(rig, pose, P)
            ok = X > 0.4
            if ok.sum() >= 2:
                pts = np.array([[int((u + 0.5) * sx), int((v + 0.5) * sy)] for u, v in uv[ok]], np.int32)
                cv2.polylines(img, [pts], False, col, 1, cv2.LINE_AA)
                if 0 <= pts[-1][0] < w and 0 <= pts[-1][1] < h:
                    xy = (int(np.clip(pts[-1][0] + 5, 2, w - 14)), int(np.clip(pts[-1][1] - 9, 2, h - 18)))
                    idx.append((str(k), xy, 13, col))
    if d.get('W_kept'):                                  # kept previous plan
        items += _tunnel(img, rig, pose, np.vstack([p0, d['W_kept']]), sx, sy, 0, KEPT_COL)
    if d.get('target_box'):                              # locked target
        x1, y1, x2, y2 = d['target_box']
        cv2.rectangle(img, (int(x1 * sx) - 4, int(y1 * sy) - 4), (int(x2 * sx) + 4, int(y2 * sy) + 4),
                      (70, 210, 70), 3, cv2.LINE_AA)
    taken = [it[1] for it in items]                     # time labels first, then indices that fit
    for it in idx:
        if _free_spot(it[1], taken, dx=22, dy=16):
            items.append(it)
            taken.append(it[1])
    return _texts(img, items, outline=2)


def _tunnel(img, rig, pose, P, sx, sy, oy, col, half=(R_DRONE, 0.12), steps=(2, 4, 6, 8, 10), labels=True):
    """Chosen trajectory as a tunnel of drone-sized frames every 0.2 s, in perspective; returns the 0.4 s
    and 1.0 s labels as text items.
    """
    from tinyuav.follow.estimate import project
    d = np.gradient(P[:, :2], axis=0)
    n = np.stack([-d[:, 1], d[:, 0]], 1)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-6)
    frames = []
    for j in steps:
        if j >= len(P):
            continue
        c, lat, up = P[j], np.r_[n[j] * half[0], 0.0], np.array([0.0, 0.0, half[1]])
        C = np.array([c + lat + up, c - lat + up, c - lat - up, c + lat - up])
        uv, X = project(rig, pose, C)
        if (X > 0.5).all():
            frames.append((j, np.array([[int((u + 0.5) * sx), int(oy + (v + 0.5) * sy)] for u, v in uv], np.int32)))
    for (j0, a), (j1, b) in zip(frames, frames[1:]):
        for q in range(4):
            cv2.line(img, tuple(a[q]), tuple(b[q]), col, 1, cv2.LINE_AA)
    items, taken = [], []
    H_, W_ = img.shape[:2]
    for i, (j, q) in enumerate(frames):
        if i == 0:
            _blend_poly(img, q, col, 0.25)
        cv2.polylines(img, [q], True, col, 3 if i < 2 else 2, cv2.LINE_AA)
        if labels and j in (4, 10):
            tr = q[np.argmax(q[:, 0] - q[:, 1])]             # top-right corner on screen
            xy = (int(np.clip(tr[0] + 5, 2, W_ - 40)), int(np.clip(tr[1] - 16, 2, H_ - 18)))
            if _free_spot(xy, taken, dx=40, dy=16):
                items.append((f'{j / 10:.1f} s', xy, 12, (235, 235, 235)))
                taken.append(xy)
    uv, X = project(rig, pose, P)
    ok = X > 0.5
    if ok.sum() >= 2:
        pts = np.array([[int((u + 0.5) * sx), int(oy + (v + 0.5) * sy)] for u, v in uv[ok]], np.int32)
        cv2.polylines(img, [pts], False, col, 2, cv2.LINE_AA)
    return items


Z_NEAR = 0.5                                        # m, brightest depth colour


def _depth_t(dep):
    """Depth (m) to a colour coordinate in [0, 1], log scale from D_VIS (dark) to Z_NEAR (bright)."""
    t = np.log(D_VIS / np.clip(dep, Z_NEAR, D_VIS)) / math.log(D_VIS / Z_NEAR)
    return 0.12 + 0.88 * t                            # far end stays dark purple


def _depth_rgb(dep, img=None):
    """SGM depth (m, 0 none) to BGR inferno; holes show the dimmed camera image."""
    col = cv2.applyColorMap((_depth_t(dep) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    if img is not None:
        g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (dep.shape[1], dep.shape[0]), interpolation=cv2.INTER_AREA).astype(np.float32)
        base = np.stack([g * 0.30 + 16, g * 0.26 + 12, g * 0.24 + 10], -1).astype(np.uint8)
    else:
        base = np.full(col.shape, 40, np.uint8)
    hole = dep <= 0
    col[hole] = base[hole]
    return col


def _colorbar(img, x0, y0, hgt, wid=14, size=11):
    """Log depth scale bar, near at the bottom."""
    z = D_VIS * (Z_NEAR / D_VIS) ** np.linspace(0, 1, hgt)       # top: far
    t = (_depth_t(z) * 255).astype(np.uint8)[:, None]
    img[y0:y0 + hgt, x0:x0 + wid] = np.repeat(cv2.applyColorMap(t, cv2.COLORMAP_INFERNO), wid, axis=1)
    frac = lambda m: math.log(D_VIS / m) / math.log(D_VIS / Z_NEAR)
    labels = []
    for m in (0.5, 1, 2, 3, 5, 10):
        y = int(y0 + frac(m) * (hgt - 1))
        cv2.line(img, (x0 + wid, y), (x0 + wid + 3, y), DIM_TXT, 1)
        labels.append((f'{m:g}', (x0 + wid + 5, y - 7), size, DIM_TXT))
    return labels


def _depth_panel(d, rig, w=480, h=360, box=None):
    dep = d['sgm'].astype(np.float32) / 1000.0
    col = _depth_rgb(dep, d.get('gray', d.get('img')))
    k = w / 480
    iw = w - int(44 * k)
    ih = int(iw * rig.height / rig.width)
    img = np.full((h, w, 3), BG, np.uint8)
    y0 = (h - ih) // 2 + int(8 * k)
    img[y0:y0 + ih, :iw] = cv2.resize(col, (iw, ih), interpolation=cv2.INTER_AREA)
    sx, sy = iw / rig.width, ih / rig.height
    W, costs, k_sel = _hypotheses(d)
    if W:
        P = np.vstack([np.array(d['p'], float), d['W_kept'] if d.get('W_kept') else W[k_sel]])
        ov = img.copy()
        _tunnel(ov, rig, (np.array(d['p']), *d['att']), P, sx, sy, y0, (255, 230, 120), labels=False)
        cv2.addWeighted(ov, 0.7, img, 0.3, 0, img)
    if box and d.get('target_box'):
        x1, y1, x2, y2 = d['target_box']
        cv2.rectangle(img, (int(x1 * sx) - 3, y0 + int(y1 * sy) - 3), (int(x2 * sx) + 3, y0 + int(y2 * sy) + 3),
                      (70, 210, 70), 2, cv2.LINE_AA)
    labels = _colorbar(img, iw + int(6 * k), y0, ih, size=int(11 * k))
    valid = float((dep > 0).mean())
    return _texts(img, [('profundidad estéreo (SGM), m', (int(10 * k), int(6 * k)), int(14 * k), PANEL_TXT),
                        (f'{valid:.0%} con dato', (iw - int(90 * k), int(8 * k)), int(12 * k), DIM_TXT)] + labels)


def _sgm_points(d, rig, stride=4):
    """SGM of one frame back-projected to world points."""
    from tinyuav.follow.estimate import camera_pose
    dep = d['sgm'][::stride, ::stride].astype(np.float32) / 1000.0
    v, u = np.nonzero(dep > 0)
    X = dep[v, u]
    u, v = u * stride, v * stride
    O, Rc = camera_pose(rig, (np.array(d['p']), *d['att']))
    dirs = np.stack([np.ones_like(X), -(u + 0.5 - rig.width / 2) / rig.focal, -(v + 0.5 - rig.height / 2) / rig.focal], 1)
    return O + (dirs * X[:, None]) @ Rc.T


OMEGA_COL = (255, 200, 90)


def _arrow(img, a, b, col, label=None, thick=2):
    cv2.arrowedLine(img, a, b, col, thick, cv2.LINE_AA, tipLength=0.18)
    if label:
        cv2.putText(img, label, (b[0] + 4, b[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)


def _local_map(hist, d, rig, w=480, h=360, ahead=10.0, v=None):
    """Top view in the heading frame of its own SGM points of the last second at flight height, with the
    drone margin, field of view, trajectories, commanded direction and velocity.
    """
    from tinyuav.sim import trajectory as TR
    FOV_BG, RING = (40, 37, 34), (70, 66, 62)
    img = np.full((h, w, 3), BG, np.uint8)
    s = (h - 64) / ahead
    p, yaw = np.array(d['p'], float), d['att'][0]
    o = (w / 2, h - 15)
    px = lambda x, y: (int(round(o[0] - y * s)), int(round(o[1] - x * s)))
    half = rig.hfov / 2
    wedge = np.array([px(0, 0)] + [px(ahead * 1.6 * math.cos(a), ahead * 1.6 * math.sin(a))
                                   for a in np.linspace(-half, half, 24)], np.int32)
    cv2.fillPoly(img, [wedge], FOV_BG, cv2.LINE_AA)
    labels = []
    a_lab = -(half + math.radians(7))                  # ring labels right of the field of view
    for r in (2, 4, 6, 8, 10):
        cv2.circle(img, px(0, 0), int(r * s), RING, 1, cv2.LINE_AA)
        x, y = px(r * math.cos(a_lab), r * math.sin(a_lab))
        labels.append((f'{r} m', (min(x + 3, w - 32), y - 13), 11, DIM_TXT))
    # own SGM points of the last second, coloured by distance and fading with age; margin contour
    # around cells seen in >= 3 frames
    cell, hits, n = 0.15, {}, len(hist)
    cloud = []
    for age, P in enumerate(reversed(hist)):
        P = P[np.abs(P[:, 2] - p[2]) < 0.6]
        Q = TR.to_heading_frame(P, p, yaw)
        k = (Q[:, 0] > -1) & (Q[:, 0] < ahead + 1) & (np.abs(Q[:, 1]) < w / 2 / s)
        Q = Q[k]
        cloud.append((age, Q))
        for key in set(map(tuple, np.floor(Q[:, :2] / cell).astype(int))):
            hits[key] = hits.get(key, 0) + 1
    occ = [key for key, c in hits.items() if c >= min(3, n)]
    if occ:
        mask = np.zeros((h, w), np.uint8)
        for kx, ky in occ:
            cv2.rectangle(mask, px(kx * cell, (ky + 1) * cell), px((kx + 1) * cell, ky * cell), 1, -1)
        rr = max(int(round(R_DRONE * s)), 1)
        margin = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rr + 1, 2 * rr + 1)))
        ov = img.copy()
        ov[margin > 0] = (70, 60, 130)
        cv2.addWeighted(ov, 0.14, img, 0.86, 0, img)
        cnt, _ = cv2.findContours(margin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnt, -1, (120, 105, 200), 1, cv2.LINE_AA)
    for age, Q in sorted(cloud, key=lambda x: -x[0]):  # oldest first
        if not len(Q):
            continue
        t = 0.42 + 0.58 * (_depth_t(np.hypot(Q[:, 0], Q[:, 1])) - 0.12) / 0.88     # bright part of inferno
        col = cv2.applyColorMap((t * 255).astype(np.uint8)[:, None], cv2.COLORMAP_INFERNO)[:, 0].astype(np.float32)
        a_ = 0.3 + 0.7 * (1 - age / max(n, 1))
        col = col * a_ + np.array(BG, np.float32) * (1 - a_)
        u = np.round(o[0] - Q[:, 1] * s).astype(int)
        v_ = np.round(o[1] - Q[:, 0] * s).astype(int)
        ok = (u >= 1) & (u < w - 1) & (v_ >= 1) & (v_ < h - 1)
        u, v_, col = u[ok], v_[ok], col[ok].astype(np.uint8)
        for du, dv in ((0, 0), (-1, 0), (0, -1), (-1, -1)):      # 2 x 2 px dots
            img[v_ + dv, u + du] = col

    # trajectories coloured by predicted cost; the chosen one drone-wide;
    # index labels where they fit
    W, costs, k_sel = _hypotheses(d)
    taken = [px(0, 0)]
    if d.get('omega') is not None:
        om = TR.rotate_to_heading(np.asarray(d['omega'], float), yaw)
        nrm = max(float(np.linalg.norm(om[:2])), 1e-6)
        tip_w = px(2.5 * om[0] / nrm, 2.5 * om[1] / nrm)
        taken.append((tip_w[0] + 4, tip_w[1] - 16))
    for k in sorted(range(len(W)), key=lambda k: k == k_sel):
        Q = TR.to_heading_frame(np.vstack([p, W[k]]), p, yaw)
        col = CYAN if not costs else _cost_color(costs[k])
        line = np.array([px(x, y) for x, y, _ in Q], np.int32)
        if k == k_sel:
            L, R = _ribbon(Q)
            poly = np.array([px(x, y) for x, y, _ in L] + [px(x, y) for x, y, _ in R[::-1]], np.int32)
            _blend_poly(img, poly, col, 0.6)
            cv2.polylines(img, [poly], True, col, 1, cv2.LINE_AA)
            cv2.polylines(img, [line], False, WHITE, 1, cv2.LINE_AA)
        else:
            cv2.polylines(img, [line], False, col, 1, cv2.LINE_AA)
        xy = (int(line[-1][0] + 5), int(line[-1][1] - 8))
        if costs and np.hypot(*Q[-1, :2]) > 1.2 and _free_spot(xy, taken, dx=22, dy=16):
            labels.append((str(k), xy, 12, col))
            taken.append(xy)
    if d.get('W_kept'):                                 # kept previous plan
        Q = TR.to_heading_frame(np.vstack([p, d['W_kept']]), p, yaw)
        L, R = _ribbon(Q)
        poly = np.array([px(x, y) for x, y, _ in L] + [px(x, y) for x, y, _ in R[::-1]], np.int32)
        _blend_poly(img, poly, KEPT_COL, 0.45)
        cv2.polylines(img, [poly], True, KEPT_COL, 1, cv2.LINE_AA)
    if d.get('omega') is not None:                     # commanded direction
        _arrow(img, px(0, 0), tip_w, OMEGA_COL, None, 1)
        labels.append(('ω', (tip_w[0] + 4, tip_w[1] - 16), 15, OMEGA_COL))
    if v is not None:
        vh = TR.rotate_to_heading(np.asarray(v, float), yaw)
        _arrow(img, px(0, 0), px(0.4 * vh[0], 0.4 * vh[1]), WHITE, None, 2)
    tip = [px(0.5, 0), px(-0.25, 0.25), px(-0.25, -0.25)]
    cv2.fillPoly(img, [np.array(tip, np.int32)], ORANGE, cv2.LINE_AA)
    labels += [('mapa local', (10, 6), 14, PANEL_TXT),
               ('su propio SGM del último segundo, a su altura', (10, 25), 11, DIM_TXT),
               ('ω  dirección a la meta', (w - 150, 8), 11, OMEGA_COL),
               ('—  margen del dron', (w - 150, 24), 11, (150, 135, 220))]
    return _texts(img, labels)


def _cost_bars(img, d):
    """Predicted collision cost of each student hypothesis, the chosen one outlined."""
    W, costs, k_sel = _hypotheses(d)
    h, w = img.shape[:2]
    n, pad = len(costs), 24
    cw = (w - pad * (n + 1)) // n
    items = []
    for k, c in enumerate(costs):
        x0, y0, bh = pad + k * (cw + pad), h // 2 - 15, 30
        col = _cost_color(c)
        cv2.rectangle(img, (x0, y0), (x0 + cw, y0 + bh), (55, 52, 50), -1)
        L = int(cw * min(c / 0.3, 1.0))
        if L > 0:
            cv2.rectangle(img, (x0, y0), (x0 + L, y0 + bh), col, -1)
        if k == k_sel:
            cv2.rectangle(img, (x0 - 3, y0 - 3), (x0 + cw + 3, y0 + bh + 3), WHITE, 2, cv2.LINE_AA)
        items += [(str(k), (x0, y0 - 24), 15, col if k != k_sel else WHITE),
                  (f'{c:.2f}', (x0 + cw - 44, y0 - 22), 14, PANEL_TXT if k == k_sel else DIM_TXT)]
    return _texts(img, items)


ACC_COL, SPEED_COL = (255, 200, 90), (120, 210, 140)


def _decisions(d, dec, v, v_des, w=800, h=120):
    """Network decisions under the image: diffphys acceleration in the heading frame and speed, or the
    student's predicted costs."""
    from tinyuav.nav.diffphys import A_MAX
    from tinyuav.sim.trajectory import rotate_to_heading
    img = np.full((h, w, 3), BG, np.uint8)
    if dec is None and d.get('costs'):
        return _cost_bars(img, d)
    rows = []
    if dec is not None:
        a = rotate_to_heading(np.asarray(dec['acc'], float), math.radians(dec['att_deg'][0]))
        rows += [('adelante', a[0], -A_MAX, ACC_COL), ('lateral', a[1], -A_MAX, ACC_COL),
                 ('vertical', a[2], -A_MAX, ACC_COL)]
    speed = float(np.linalg.norm(v))
    v_top = max(12.0, 1.25 * (v_des or 0.0))
    rows.append(('velocidad', speed, 0.0, SPEED_COL))
    k = h / (34 * len(rows) + 16)
    rh, fs = int(34 * k), max(int(15 * k), 11)
    x0, x1 = int(118 * k) + 16, w - int(110 * k)
    items = []
    for i, (name, val, lo, col) in enumerate(rows):
        y = int(8 * k) + i * rh + rh // 2
        hi = A_MAX if lo < 0 else v_top
        X = lambda q: int(x0 + (np.clip(q, lo, hi) - lo) / (hi - lo) * (x1 - x0))
        cv2.rectangle(img, (x0, y - rh // 4), (x1, y + rh // 4), (48, 45, 42), -1)
        z = X(0.0)
        cv2.rectangle(img, (min(z, X(val)), y - rh // 4), (max(z, X(val)), y + rh // 4), col, -1)
        cv2.line(img, (z, y - rh // 3), (z, y + rh // 3), DIM_TXT, 1)
        if lo == 0.0 and v_des:
            cv2.line(img, (X(v_des), y - rh // 2 + 2), (X(v_des), y + rh // 2 - 2), WHITE, 2)
        unit, val = ('m/s' if lo == 0.0 else 'm/s²'), round(float(val), 1) + 0.0
        items += [(name, (16, y - fs // 2 - 2), fs, PANEL_TXT),
                  (f'{val:+5.1f} {unit}' if lo < 0 else f'{val:4.1f} {unit}', (x1 + 12, y - fs // 2 - 2), fs, PANEL_TXT)]
    return _texts(img, items)


def _speed_at(L, t):
    return L[min(int(np.searchsorted(L[:, 0], t)), len(L) - 1), 4:7]


def _flush(wr, frame, fps, fn):
    for _ in range(int(fps * 1.5)):
        wr.write(frame)
    wr.release()
    return _to_h264(fn)


def render_perception(log, trace, rig, fn, fps=15.0, result=None, decisions=None, v_des=None):
    from collections import deque
    W, H = 1280, 720
    wr = _writer(fn, fps, (W, H))
    L = np.array([row[:7] for row in log], float)
    dec = {round(x['t'], 3): x for x in decisions or []}
    hist = deque(maxlen=max(1, int(round(1.0 * fps))))
    frame = None
    for i, d in enumerate(trace):
        v = _speed_at(L, d['t'])
        hist.append(_sgm_points(d, rig))
        last = i == len(trace) - 1
        left = np.vstack([_camera_hud(d, rig), _decisions(d, dec.get(round(d['t'], 3)), v, v_des)])
        right = np.vstack([_depth_panel(d, rig), _local_map(hist, d, rig, v=v)])
        frame = np.hstack([left, right])
        frame[:, 799:801] = (12, 12, 12)
        frame[359:361, 800:] = (12, 12, 12)
        frame[599:601, :800] = (12, 12, 12)
        if last and result:                            # outcome: green or red border
            cv2.rectangle(frame, (2, 2), (797, 597), (70, 190, 70) if 'EXITO' in result else (60, 60, 220), 4)
        wr.write(frame)
    return _flush(wr, frame, fps, fn)


def render_sgm(log, trace, rig, fn, fps=15.0, result=None, decisions=None, v_des=None):
    W, H, HB = 960, 720, 200
    wr = _writer(fn, fps, (W, H + HB))
    L = np.array([row[:7] for row in log], float)
    dec = {round(x['t'], 3): x for x in decisions or []}
    frame = None
    for i, d in enumerate(trace):
        top = _depth_panel(d, rig, w=W, h=H, box=True)
        frame = np.vstack([top, _decisions(d, dec.get(round(d['t'], 3)), _speed_at(L, d['t']), v_des, w=W, h=HB)])
        frame[H - 1:H + 1] = (12, 12, 12)
        if i == len(trace) - 1 and result:
            cv2.rectangle(frame, (2, 2), (W - 3, H - 3), (70, 190, 70) if 'EXITO' in result else (60, 60, 220), 4)
        wr.write(frame)
    return _flush(wr, frame, fps, fn)


def render_episode(log, trace, rig, fn, fps=15.0, result=None, view='perception', decisions=None, v_des=None):
    """Write the video of a traced episode: view 'perception' or 'sgm'."""
    if not trace or 'sgm' not in trace[0]:
        raise ValueError('the episode was not traced with camera frames')
    f = render_sgm if view == 'sgm' else render_perception
    return f(log, trace, rig, fn, fps=fps, result=result, decisions=decisions, v_des=v_des)


def run(model, out, world='forest', v=3.0, seed=1000, pilot='student', cam_hz=15.0, opts=None, suffix='',
        view='perception'):
    """Fly one test episode in our simulator and write its video and decisions."""
    import json
    from tinyuav.nav import dagger
    from tinyuav.nav.rollout import Config, Rollout
    from tinyuav.nav.run import apply_opts
    from tinyuav.nav.student import omega_of
    from tinyuav.sim import envs as EV
    env = EV.TEST[world](seed)
    cfg = Config(v_des=v, cam_hz=cam_hz, seed=seed, omega_ref=omega_of(model) if pilot == 'student' and model else 'line')
    R = Rollout(env, cfg, pilot, dagger._policy(os.path.abspath(model)) if pilot == 'student' else
                os.path.abspath(model) if pilot == 'diffphys' else None)
    apply_opts(R, opts or {})
    R.trace, R.trace_depth, R.record_cam = [], True, True
    if pilot == 'diffphys':
        R.dp_trace = []
    r, _, log = R.run()
    res = 'EXITO' if r['success'] else ('CHOQUE' if r['crashed'] else 'SIN LLEGAR')
    d = os.path.join(out, 'video')
    os.makedirs(d, exist_ok=True)
    tag = os.path.splitext(os.path.basename(model))[0] if pilot in ('student', 'diffphys') and model else pilot
    fn = os.path.join(d, f'{tag}{suffix}_{world}_v{v:g}_s{seed}_{view}.mp4')
    render_episode(log, R.trace, cfg.rig, fn, fps=cam_hz, result=f'{res} ({r["time_s"]:.1f} s)', view=view,
                   decisions=R.dp_trace, v_des=v)
    if R.dp_trace is not None:
        with open(fn[:-4] + '.json', 'w') as fh:
            json.dump({'result': r, 'decisions': R.dp_trace}, fh, default=float)
    print(f'{res} · {r["time_s"]} s · {fn}')
    return fn
