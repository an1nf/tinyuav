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


def render_episode(env, log, trace, rig, net_size, fn, fps=15.0, chase=True, result=None, title='', mark=None):
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


def run(model, out, world='forest', v=3.0, seed=1000, pilot='student', cam_hz=15.0, opts=None, chase=True, suffix=''):
    from nanouav.agile import dagger, envs as EV
    from nanouav.agile.rollout import Config, Rollout
    env = EV.TEST[world](seed)
    cfg = Config(v_des=v, cam_hz=cam_hz, seed=seed)
    R = Rollout(env, cfg, pilot, dagger._policy(os.path.abspath(model)) if pilot == 'student' else None)
    from nanouav.agile.run import apply_opts
    apply_opts(R, opts or {})
    R.trace, R.trace_depth = [], True
    r, _, log = R.run()
    res = 'EXITO' if r['success'] else ('CHOQUE' if r['crashed'] else 'SIN LLEGAR')
    d = os.path.join(out, 'video')
    os.makedirs(d, exist_ok=True)
    tag = os.path.splitext(os.path.basename(model))[0] if pilot == 'student' else pilot
    fn = os.path.join(d, f'{tag}{suffix}_{world}_v{v:g}_s{seed}.mp4')
    render_episode(env, log, R.trace, cfg.rig, cfg.net_size, fn, fps=cam_hz, chase=chase,
                   result=f'{res} ({r["time_s"]:.1f} s)', title=f'{tag} | {world} | {v:g} m/s | semilla {seed}')
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
