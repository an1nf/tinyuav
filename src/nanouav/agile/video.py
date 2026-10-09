"""Video of one evaluation episode (`nanouav run r18 video`): how the student flies.

Three panels at the camera rate (15 fps = real time):

* left — top view that follows the drone: trees (dotted: the drone radius added), the flown path, the
  camera's field of view, and the M hypotheses of the current decision (green: free according to the
  true map, red: collides; thick: the chosen one);
* top right — what the network sees: the SGM depth at the network resolution, with the hypotheses
  projected into the image;
* bottom right — chase camera 3 m behind the drone (the renderer of the simulator; the drone is the
  orange ball), unless --no-chase.

The last frame is held for 1.5 s with the result. Written with OpenCV and re-encoded to H.264 with
ffmpeg if it is installed (plays in a browser or a phone).
"""
import math
import os

import cv2
import numpy as np

R_DRONE = 0.2
GREEN, RED, WHITE, GRAY, ORANGE, CYAN = (60, 190, 60), (60, 60, 230), (255, 255, 255), (150, 150, 150), (0, 140, 255), (230, 200, 0)


def _writer(fn, fps, size):
    w = cv2.VideoWriter(fn, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    if not w.isOpened():
        raise RuntimeError(f'cannot open a video writer for {fn}')
    return w


def _to_h264(fn):
    """MPEG-4 part 2 does not play in browsers or most phones: re-encode with ffmpeg when it is installed."""
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


def _topdown(env, log_xy, d, p, yaw, size=720, span=20.0, result=None, mark=None):
    img = np.full((size, size, 3), 245, np.uint8)
    s = size / span
    cx, cy = p[0] + 0.3 * span * math.cos(yaw), p[1] + 0.3 * span * math.sin(yaw)

    def px(x, y):
        return int(round((x - cx) * s + size / 2)), int(round(size / 2 - (y - cy) * s))
    for g in np.arange(math.floor(cx - span), cx + span, 5.0):          # 5 m grid
        cv2.line(img, px(g, cy - span), px(g, cy + span), (230, 230, 230), 1)
    for g in np.arange(math.floor(cy - span), cy + span, 5.0):
        cv2.line(img, px(cx - span, g), px(cx + span, g), (230, 230, 230), 1)
    gx = env.goal[0]
    cv2.line(img, px(gx, cy - span), px(gx, cy + span), CYAN, 2)
    if hasattr(env.world, 'points'):                       # Flightmare scene: its point cloud near the flight height
        P = env.world.points
        k = (np.abs(P[:, 0] - cx) < span / 2) & (np.abs(P[:, 1] - cy) < span / 2) & (np.abs(P[:, 2] - p[2]) < 0.8)
        Q = P[k][::max(1, int(k.sum() // 30000)), :2]
        U = np.round((Q[:, 0] - cx) * s + size / 2).astype(int)
        V = np.round(size / 2 - (Q[:, 1] - cy) * s).astype(int)
        for du in (0, 1):
            for dv in (0, 1):
                m = (U + du >= 0) & (U + du < size) & (V + dv >= 0) & (V + dv < size)
                img[V[m] + dv, U[m] + du] = (70, 70, 70)
    for o in env.world.objects:
        if hasattr(o, 'half') and abs(o.center[0] - cx) < span and abs(o.center[1] - cy) < span:   # boxes (trunks)
            c, h = o.center, o.half
            cv2.rectangle(img, px(c[0] - h[0], c[1] + h[1]), px(c[0] + h[0], c[1] - h[1]), (90, 90, 90), -1)
            cv2.rectangle(img, px(c[0] - h[0] - R_DRONE, c[1] + h[1] + R_DRONE),
                          px(c[0] + h[0] + R_DRONE, c[1] - h[1] - R_DRONE), (170, 170, 170), 1)
        if hasattr(o, 'cx') and abs(o.cx - cx) < span and abs(o.cy - cy) < span:
            cv2.circle(img, px(o.cx, o.cy), max(int(o.r * s), 1), (90, 90, 90), -1, cv2.LINE_AA)
            cv2.circle(img, px(o.cx, o.cy), int((o.r + R_DRONE) * s), (170, 170, 170), 1, cv2.LINE_AA)
    for a in (-math.pi / 4, math.pi / 4):                                  # camera field of view
        cv2.line(img, px(*p[:2]), px(p[0] + 10 * math.cos(yaw + a), p[1] + 10 * math.sin(yaw + a)), (200, 200, 120), 1)
    if len(log_xy) > 1:
        cv2.polylines(img, [np.array([px(x, y) for x, y in log_xy], np.int32)], False, (0, 0, 0), 2, cv2.LINE_AA)
    if d is not None:
        order = sorted(range(len(d['W'])), key=lambda k: k == d['k'])
        for k in order:
            W = np.array(d['W'][k])
            col = GREEN if d['clear'][k] >= R_DRONE else RED
            pts = np.array([px(*p[:2])] + [px(x, y) for x, y, _ in W], np.int32)
            cv2.polylines(img, [pts], False, col, 3 if k == d['k'] else 1, cv2.LINE_AA)
    tip = [px(p[0] + 0.6 * math.cos(yaw), p[1] + 0.6 * math.sin(yaw)),
           px(p[0] + 0.35 * math.cos(yaw + 2.5), p[1] + 0.35 * math.sin(yaw + 2.5)),
           px(p[0] + 0.35 * math.cos(yaw - 2.5), p[1] + 0.35 * math.sin(yaw - 2.5))]
    cv2.fillPoly(img, [np.array(tip, np.int32)], ORANGE, cv2.LINE_AA)
    cv2.putText(img, '5 m', (size - 60, size - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GRAY, 1)
    if mark is not None:                                   # the obstacle point the drone hit
        q = px(mark[0], mark[1])
        cv2.drawMarker(img, q, (0, 0, 220), cv2.MARKER_TILTED_CROSS, 22, 3, cv2.LINE_AA)
    if result:
        cv2.putText(img, result, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 200) if 'CHOQUE' in result else (0, 150, 0), 3)
    return img


def _onboard(d, rig, net_size, w=480, h=360):
    if d is None or 'depth' not in d:
        img = np.full((h, w, 3), 40, np.uint8)
        cv2.putText(img, '(sin camara: piloto privilegiado)', (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)
        return img
    dep = d['depth'].astype(np.float32) / 1000.0
    v = np.clip(dep / 10.0, 0, 1)
    col = cv2.applyColorMap((255 - v * 255).astype(np.uint8), cv2.COLORMAP_JET)
    col[dep <= 0] = (0, 0, 0)                                            # invalid SGM pixels
    img = cv2.resize(col, (w, h), interpolation=cv2.INTER_NEAREST)
    from nanouav.follow.estimate import project
    pose = (np.array(d['p']), *d['att'])
    sx, sy = w / rig.width, h / rig.height
    for k in sorted(range(len(d['W'])), key=lambda k: k == d['k']):
        uv, X = project(rig, pose, np.array(d['W'][k]))
        pts = [(int((u + 0.5) * sx), int((vv + 0.5) * sy)) for (u, vv), x in zip(uv, X) if x > 0.3]
        col = GREEN if d['clear'][k] >= R_DRONE else RED
        if len(pts) > 1:
            cv2.polylines(img, [np.array(pts, np.int32)], False, col, 3 if k == d['k'] else 1, cv2.LINE_AA)
        if pts:                                    # end of the hypothesis (1 s ahead): at the horizon when level
            cv2.circle(img, pts[-1], 7 if k == d['k'] else 4, col, -1 if k == d['k'] else 1, cv2.LINE_AA)
    label = 'lo que ve la red: profundidad SGM (negro = sin dato)' if d.get('costs') else \
        'profundidad SGM (negro = sin dato) | verde: plan'
    cv2.putText(img, label, (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
    cv2.putText(img, '0 m rojo  -  10 m azul', (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1)
    return img


def _chase(env, p, yaw, w=480, h=360):
    from nanouav.agile.stereo import StereoRig, cull
    from nanouav.scene.renderer import Sphere, Texture, World, render_view
    rig = StereoRig(width=w, height=h, hfov_deg=80.0, render_range=25.0)
    cam = np.array([p[0] - 3.0 * math.cos(yaw), p[1] - 3.0 * math.sin(yaw), p[2] + 1.0])
    world = cull(env.world, cam, yaw, rig)
    drone = Sphere(tuple(float(x) for x in p), 0.2, Texture(albedo=0.95))
    world = World(list(world.objects) + [drone], world.ground, sun=world.sun, ambient=world.ambient,
                  diffuse=world.diffuse, sky=world.sky)
    gray, _ = render_view(world, cam, yaw, 0.0, 0.0, cam_pitch=math.radians(15), cam_offset=(0, 0, 0),
                          hfov=rig.hfov, post=False, shadows=False, size=(h, w), d_max=25.0)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    # paint the drone ball orange: project its center
    f = w / 2 / math.tan(rig.hfov / 2)
    c, s_ = math.cos(yaw), math.sin(yaw)
    dv = p - cam
    x, y, z = c * dv[0] + s_ * dv[1], -s_ * dv[0] + c * dv[1], dv[2]
    pt = math.radians(15)
    xc, zc = x * math.cos(pt) - z * math.sin(pt), x * math.sin(pt) + z * math.cos(pt)
    if xc > 0.3:
        u, v = int(-y / xc * f + w / 2), int(-zc / xc * f + h / 2)
        cv2.circle(img, (u, v), max(int(0.2 / xc * f), 3), ORANGE, -1, cv2.LINE_AA)
    cv2.putText(img, 'camara de seguimiento (el dron es la bola naranja)', (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return img


def render_episode(env, log, trace, rig, net_size, fn, fps=15.0, chase=True, result=None, title='', mark=None,
                   view='perception'):
    """view='perception' (default): only what the drone perceives — camera with the planned trajectories,
    full-resolution SGM, and a local map built from its own SGM of the last 0.5 s (no prior map).
    view='debug': the old panels (top view of the true map, network input, chase camera)."""
    if view == 'perception' and trace and 'sgm' in trace[0]:
        return render_perception(log, trace, rig, fn, fps=fps, result=result, title=title)
    W, H = 720 + 480, 720
    wr = _writer(fn, fps, (W, H))
    times = [d['t'] for d in trace] if trace else list(np.arange(0, log[-1][0], 1 / fps))
    L = np.array([row[:7] for row in log], float)
    frame = None
    for i, t in enumerate(times):
        j = int(np.searchsorted(L[:, 0], t))
        j = min(j, len(L) - 1)
        p = L[j, 1:4]
        v = L[j, 4:7]
        d = trace[i] if trace else None
        yaw = d['att'][0] if d is not None else math.atan2(v[1], v[0])
        last = i == len(times) - 1
        top = _topdown(env, L[:j + 1:2, 1:3], d, p, yaw, result=result if last else None, mark=mark if last else None)
        if d is not None and 'gray' in d:                  # external renderer (Flightmare): its left image
            low = cv2.cvtColor(cv2.resize(d['gray'], (480, 360)), cv2.COLOR_GRAY2BGR)
            cv2.putText(low, 'camara izquierda (Flightmare)', (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            low = _chase(env, p, yaw) if chase else np.full((360, 480, 3), 60, np.uint8)
        right = np.vstack([_onboard(d, rig, net_size), low])
        frame = np.hstack([top, right])
        info = f'{title}  t={t:5.2f} s  |v|={np.linalg.norm(v):4.1f} m/s'
        if d is not None and d.get('costs'):
            info += f"  elegida={d['k']}  costos={['%.2f' % c for c in d['costs']]}  beta={d['beta']:.2f}"
        elif d is not None:
            info += f"  plan de {d.get('pilot', 'privilegiado')} (1 s)"
        cv2.rectangle(frame, (0, 0), (W, 26), (255, 255, 255), -1)
        cv2.putText(frame, info, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        wr.write(frame)
    for _ in range(int(fps * 1.5)):
        wr.write(frame)
    wr.release()
    return _to_h264(fn)


def run(model, out, world='forest', v=3.0, seed=1000, pilot='student', cam_hz=15.0, opts=None, chase=True, suffix='',
        view='perception'):
    from nanouav.agile import dagger, envs as EV
    from nanouav.agile.rollout import Config, Rollout
    env = EV.TEST[world](seed)
    cfg = Config(v_des=v, cam_hz=cam_hz, seed=seed)
    R = Rollout(env, cfg, pilot, dagger._policy(os.path.abspath(model)) if pilot == 'student' else None)
    from nanouav.agile.run import apply_opts
    apply_opts(R, opts or {})
    R.trace, R.trace_depth, R.record_cam = [], True, True
    r, _, log = R.run()
    res = 'EXITO' if r['success'] else ('CHOQUE' if r['crashed'] else 'SIN LLEGAR')
    d = os.path.join(out, 'video')
    os.makedirs(d, exist_ok=True)
    tag = os.path.splitext(os.path.basename(model))[0] if pilot == 'student' else pilot
    fn = os.path.join(d, f'{tag}{suffix}_{world}_v{v:g}_s{seed}.mp4')
    render_episode(env, log, R.trace, cfg.rig, cfg.net_size, fn, fps=cam_hz, chase=chase,
                   result=f'{res} ({r["time_s"]:.1f} s)', title=f'{tag} | {world} | {v:g} m/s | semilla {seed}', view=view)
    print(f'{res} · {r["time_s"]} s · → {fn}')
    return fn


def from_gz(json_path, out, chase=True):
    """Video of a Gazebo flight of `gazebo/nodes/agile_node.py` (same panels; the chase camera and the top
    view use our renderer of the same world; the depth panel needs AGILE_KEEP_DEPTH=true)."""
    import json
    from nanouav.agile import envs as EV
    from nanouav.agile.stereo import StereoRig
    D = json.load(open(json_path))
    r = D['result']
    env = EV.TEST[r['world']](int(r['seed']))
    trace = D['trace']
    dz = json_path[:-5] + '_depth.npz'
    if os.path.exists(dz):
        for d, dep in zip(trace, np.load(dz)['depth']):
            d['depth'] = dep
    d = os.path.join(out, 'video')
    os.makedirs(d, exist_ok=True)
    fn = os.path.join(d, 'gz_' + os.path.basename(json_path)[:-5] + '.mp4')
    res = {'success': 'EXITO', 'crash': 'CHOQUE'}.get(r['status'], r['status'].upper())
    render_episode(env, D['log'], trace, StereoRig(), (120, 160), fn, chase=chase,
                   result=f"{res} ({r['time_s']:.1f} s)", title=f"Gazebo | {r['model']} | {r['world']} | "
                                                              f"{r['v_des']:g} m/s | semilla {r['seed']}")
    print(f"{res} · {r['time_s']} s · → {fn}")
    return fn


# ── perception view: only what the drone has ─────────────────────────────────
D_VIS = 10.0                                        # m, depth colour scale (the SGM's range)


def _cost_color(c):
    """Predicted collision cost → BGR (green free … red collision)."""
    t = float(np.clip(c / 0.3, 0, 1))
    return (int(60 + 0 * t), int(200 * (1 - t) + 60 * t), int(60 * (1 - t) + 230 * t))


def _ribbon(P, half=R_DRONE):
    """Centre line (N, 3) → left and right edges (N, 3) at ±half, horizontal normal."""
    d = np.gradient(P[:, :2], axis=0)
    n = np.stack([-d[:, 1], d[:, 0]], 1)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-6)
    off = np.c_[n * half, np.zeros(len(P))]
    return P + off, P - off


def _blend_poly(img, pts, color, alpha):
    ov = img.copy()
    cv2.fillPoly(ov, [pts], color, cv2.LINE_AA)
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)


def _hypotheses(d):
    W = [np.asarray(w, float) for w in d['W']]
    costs = list(d.get('costs') or [])
    return W, costs, int(d.get('k', 0))


def _camera_hud(d, rig, w=800, h=600):
    from nanouav.follow.estimate import project
    img = d['img']
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img[..., ::-1].copy()
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    sx, sy = w / rig.width, h / rig.height
    pose = (np.array(d['p']), *d['att'])
    W, costs, k_sel = _hypotheses(d)
    p0 = np.array(d['p'], float)
    for k in sorted(range(len(W)), key=lambda k: k == k_sel):
        P = np.vstack([p0, W[k]])
        priv = not costs
        col = (230, 200, 0) if priv else _cost_color(costs[k])
        if k == k_sel:                                   # the chosen trajectory: a tunnel of drone-sized frames
            _tunnel(img, rig, pose, P, sx, sy, 0, col)
        else:                                           # the other hypotheses: thin centre lines
            uv, X = project(rig, pose, P)
            ok = X > 0.4
            if ok.sum() >= 2:
                pts = np.array([[int((u + 0.5) * sx), int((v + 0.5) * sy)] for u, v in uv[ok]], np.int32)
                cv2.polylines(img, [pts], False, col, 1, cv2.LINE_AA)
                cv2.putText(img, f'{k}', tuple(pts[-1] + [4, 0]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
    return img


def _tunnel(img, rig, pose, P, sx, sy, oy, col, half=(R_DRONE, 0.12), steps=(2, 4, 6, 8, 10), labels=True):
    """Chosen trajectory as a 'tunnel in the sky': the drone's cross-section (2·R_DRONE wide) every 0.2 s,
    in perspective, joined at the corners; the nearest frames thicker."""
    from nanouav.follow.estimate import project
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
    for i, (j, q) in enumerate(frames):
        if i == 0:
            _blend_poly(img, q, col, 0.25)
        cv2.polylines(img, [q], True, col, 3 if i < 2 else 2, cv2.LINE_AA)
        if labels and j in (4, 10):
            cv2.putText(img, f'{j / 10:.1f} s', tuple(q[0] + [6, 0]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                        cv2.LINE_AA)
    uv, X = project(rig, pose, P)
    ok = X > 0.5
    if ok.sum() >= 2:
        pts = np.array([[int((u + 0.5) * sx), int(oy + (v + 0.5) * sy)] for u, v in uv[ok]], np.int32)
        cv2.polylines(img, [pts], False, col, 2, cv2.LINE_AA)


def _colorbar(img, x0, y0, hgt, wid=14):
    v = 1 - np.arange(hgt) / (hgt - 1)                     # top: far (10 m)
    col = cv2.applyColorMap(((1 - v) * 255).astype(np.uint8)[:, None], cv2.COLORMAP_TURBO)   # (hgt, 1, 3)
    img[y0:y0 + hgt, x0:x0 + wid] = np.repeat(col, wid, axis=1)
    for m in (0, 2, 4, 6, 8, 10):
        y = int(y0 + (1 - m / D_VIS) * (hgt - 1))
        cv2.putText(img, f'{m}', (x0 + wid + 3, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, WHITE, 1, cv2.LINE_AA)


def _depth_panel(d, rig, w=480, h=360):
    from nanouav.follow.estimate import project
    dep = d['sgm'].astype(np.float32) / 1000.0
    v = np.clip(dep / D_VIS, 0, 1)
    col = cv2.applyColorMap(((1 - v) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    col[dep <= 0] = (45, 45, 45)
    iw, ih = w - 44, int((w - 44) * rig.height / rig.width)
    img = np.full((h, w, 3), 25, np.uint8)
    y0 = (h - ih) // 2 + 8
    img[y0:y0 + ih, :iw] = cv2.resize(col, (iw, ih), interpolation=cv2.INTER_NEAREST)
    pose = (np.array(d['p']), *d['att'])
    W, costs, k_sel = _hypotheses(d)
    if W:
        P = np.vstack([np.array(d['p'], float), W[k_sel]])
        _tunnel(img, rig, pose, P, iw / rig.width, ih / rig.height, y0, WHITE, labels=False)
    _colorbar(img, iw + 6, y0, ih)
    valid = float((dep > 0).mean())
    cv2.putText(img, f'profundidad SGM (m) | {valid:.0%} con dato | gris = sin dato', (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)
    return img


def _sgm_points(d, rig, stride=4):
    """The SGM of one frame back-projected to world points (the drone's own pose estimate)."""
    from nanouav.follow.estimate import camera_pose
    dep = d['sgm'][::stride, ::stride].astype(np.float32) / 1000.0
    v, u = np.nonzero(dep > 0)
    X = dep[v, u]
    u, v = u * stride, v * stride
    O, Rc = camera_pose(rig, (np.array(d['p']), *d['att']))
    dirs = np.stack([np.ones_like(X), -(u + 0.5 - rig.width / 2) / rig.focal, -(v + 0.5 - rig.height / 2) / rig.focal], 1)
    return O + (dirs * X[:, None]) @ Rc.T


def _local_map(hist, d, rig, w=480, h=360, ahead=10.0):
    """Top view in the drone's heading frame (forward up) of its own SGM points of the last frames at its
    flight height (± 0.6 m), with the trajectories. Nothing comes from a prior map."""
    from nanouav.agile import trajectory as TR
    img = np.full((h, w, 3), 238, np.uint8)
    s = (h - 30) / ahead
    p, yaw = np.array(d['p'], float), d['att'][0]
    px = lambda x, y: (int(round(w / 2 - y * s)), int(round(h - 15 - x * s)))
    for r in (2, 4, 6, 8, 10):
        cv2.circle(img, px(0, 0), int(r * s), (215, 215, 215), 1, cv2.LINE_AA)
        cv2.putText(img, f'{r} m', (px(r, 0)[0] + 3, px(r, 0)[1] + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, GRAY, 1)
    for a in (-math.pi / 4, math.pi / 4):
        cv2.line(img, px(0, 0), px(ahead * math.cos(a), ahead * math.sin(a)), (200, 200, 150), 1, cv2.LINE_AA)
    for age, P in enumerate(reversed(hist)):
        P = P[np.abs(P[:, 2] - p[2]) < 0.6]
        Q = TR.to_heading_frame(P, p, yaw)
        k = (Q[:, 0] > -1) & (Q[:, 0] < ahead) & (np.abs(Q[:, 1]) < w / 2 / s)
        g = int(40 + 25 * age)
        for x, y in Q[k][:, :2]:
            c = px(x, y)
            img[max(c[1] - 1, 0):c[1] + 1, max(c[0] - 1, 0):c[0] + 1] = (g, g, g)
    W, costs, k_sel = _hypotheses(d)
    for k in sorted(range(len(W)), key=lambda k: k == k_sel):
        Q = TR.to_heading_frame(np.vstack([p, W[k]]), p, yaw)
        col = (200, 160, 0) if not costs else _cost_color(costs[k])
        if k == k_sel:
            L, R = _ribbon(Q)
            poly = np.array([px(x, y) for x, y, _ in L] + [px(x, y) for x, y, _ in R[::-1]], np.int32)
            _blend_poly(img, poly, col, 0.5)
            cv2.polylines(img, [np.array([px(x, y) for x, y, _ in Q], np.int32)], False, col, 2, cv2.LINE_AA)
        else:
            cv2.polylines(img, [np.array([px(x, y) for x, y, _ in Q], np.int32)], False, col, 1, cv2.LINE_AA)
    tip = [px(0.5, 0), px(-0.25, 0.25), px(-0.25, -0.25)]
    cv2.fillPoly(img, [np.array(tip, np.int32)], ORANGE, cv2.LINE_AA)
    cv2.putText(img, 'mapa local: solo su SGM del ultimo 0.5 s, a su altura', (8, 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (40, 40, 40), 1, cv2.LINE_AA)
    return img


def _info_bar(d, v, t, title, result=None, w=800, h=120):
    img = np.full((h, w, 3), 30, np.uint8)
    W, costs, k_sel = _hypotheses(d)
    cv2.putText(img, title, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)
    cv2.putText(img, f't = {t:5.2f} s   |v| = {np.linalg.norm(v):4.1f} m/s   beta = {d.get("beta", 1):.2f}',
                (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)
    if costs:
        cv2.putText(img, 'costo de choque predicho por la red', (10, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.45, GRAY, 1)
        for k, c in enumerate(costs):
            x0 = 10 + k * 190
            y0 = 86
            L = int(150 * min(c / 0.3, 1.0))
            cv2.rectangle(img, (x0, y0), (x0 + 150, y0 + 18), (70, 70, 70), 1)
            cv2.rectangle(img, (x0, y0), (x0 + max(L, 1), y0 + 18), _cost_color(c), -1)
            cv2.putText(img, f'{k}: {c:.2f}' + ('  <- elegida' if k == k_sel else ''), (x0 + 4, y0 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 1, cv2.LINE_AA)
    else:
        cv2.putText(img, 'piloto privilegiado: su plan de 1 s (cian)', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (230, 200, 0), 1, cv2.LINE_AA)
    if result:
        cv2.putText(img, result, (470, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (60, 60, 230) if 'CHOQUE' in result else (60, 200, 60), 2, cv2.LINE_AA)
    return img


def render_perception(log, trace, rig, fn, fps=15.0, result=None, title=''):
    from collections import deque
    W, H = 1280, 720
    wr = _writer(fn, fps, (W, H))
    L = np.array([row[:7] for row in log], float)
    hist = deque(maxlen=max(1, int(round(0.5 * fps))))
    frame = None
    for i, d in enumerate(trace):
        j = min(int(np.searchsorted(L[:, 0], d['t'])), len(L) - 1)
        v = L[j, 4:7]
        hist.append(_sgm_points(d, rig))
        last = i == len(trace) - 1
        left = np.vstack([_camera_hud(d, rig), _info_bar(d, v, d['t'], title, result if last else None)])
        right = np.vstack([_depth_panel(d, rig), _local_map(hist, d, rig)])
        frame = np.hstack([left, right])
        wr.write(frame)
    for _ in range(int(fps * 1.5)):
        wr.write(frame)
    wr.release()
    return _to_h264(fn)
