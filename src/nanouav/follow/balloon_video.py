"""Video of one R20 balloon-following flight (`nanouav run r20 video`), 0.27.9.

Three panels at the camera rate (10 fps = real time):

* left — top view that follows the drone: trunks (thin ring: the drone radius added), the flown path, the
  camera's field of view, the balloon's own path (dotted), where the balloon is (red) and where the drone
  believes it is (cross; a ring while it is lost and only predicted);
* top right — the drone's color camera: every detection of the color detector (thin box) and the locked
  balloon (thick box), with the lock state;
* bottom right — chase camera 3 m behind the drone, in color (the drone is the orange ball).

The last frame is held for 1.5 s with the result. Re-encoded to H.264 when ffmpeg is installed.
"""
import math
import os
import types

import cv2
import numpy as np

from nanouav.agile.video import CYAN, ORANGE, _chase, _to_h264, _topdown, _writer

RED, BLUE, WHITE = (40, 40, 230), (200, 120, 0), (255, 255, 255)


def _chase_rgb(world, p, yaw, w=480, h=360):
    """Chase camera in color (the balloons keep their color)."""
    from nanouav.agile.stereo import StereoRig, cull
    from nanouav.scene.renderer import Sphere, Texture, World, render_view
    rig = StereoRig(width=w, height=h, hfov_deg=80.0, render_range=30.0)
    cam = np.array([p[0] - 3.0 * math.cos(yaw), p[1] - 3.0 * math.sin(yaw), p[2] + 1.0])
    vis = cull(world, cam, yaw, rig)
    drone = Sphere(tuple(float(x) for x in p), 0.2, Texture(albedo=0.95, color=(1.0, 0.55, 0.0)))
    vis = World(list(vis.objects) + [drone], vis.ground, sun=vis.sun, ambient=vis.ambient,
                diffuse=vis.diffuse, sky=vis.sky)
    gray, _, chroma = render_view(vis, cam, yaw, 0.0, 0.0, cam_pitch=math.radians(15), cam_offset=(0, 0, 0),
                                  hfov=rig.hfov, post=False, shadows=False, size=(h, w), d_max=30.0, chroma=True)
    rgb = np.clip(gray.astype(np.float32)[..., None] * chroma, 0, 255).astype(np.uint8)
    img = np.ascontiguousarray(rgb[..., ::-1])
    cv2.putText(img, 'camara de seguimiento (el dron es la bola naranja)', (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 1)
    return img


def _camera(rec, w=480, h=360):
    """The drone's color camera with the detections and the locked balloon."""
    rgb = rec['rgb']
    H, W = rgb.shape[:2]
    img = cv2.resize(np.ascontiguousarray(rgb[..., ::-1]), (w, h))
    sx, sy = w / W, h / H
    for box in rec['dets']:
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (int(x1 * sx), int(y1 * sy)), (int(x2 * sx), int(y2 * sy)), WHITE, 1)
    if rec['lock'] is not None:
        x1, y1, x2, y2 = rec['lock']
        cv2.rectangle(img, (int(x1 * sx) - 3, int(y1 * sy) - 3), (int(x2 * sx) + 3, int(y2 * sy) + 3), (0, 230, 0), 3)
    state = {'locked': 'bloqueado', 'relocked': 'recuperado', 'lost': 'PERDIDO (prediccion)',
             'search': 'buscando'}.get(rec['how'], rec['how'] or '')
    cv2.rectangle(img, (0, 0), (w, 26), (0, 0, 0), -1)
    cv2.putText(img, f'camara del dron | {state}', (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 0, 255) if rec['how'] == 'lost' else WHITE, 1, cv2.LINE_AA)
    return img


def _overlay_balloons(img, rec, p, yaw, trail, size=720, span=20.0):
    """Balloon path (dotted), the true balloon (red) and the drone's estimate (cross, or a ring while lost)."""
    s = size / span
    cx, cy = p[0] + 0.3 * span * math.cos(yaw), p[1] + 0.3 * span * math.sin(yaw)

    def px(x, y):
        return int(round((x - cx) * s + size / 2)), int(round(size / 2 - (y - cy) * s))
    for x, y in trail[::3]:
        cv2.circle(img, px(x, y), 2, (150, 150, 240), -1, cv2.LINE_AA)
    for q in rec['others']:
        cv2.circle(img, px(*q[:2]), max(int(0.3 * s), 3), BLUE, 1, cv2.LINE_AA)
    cv2.circle(img, px(*rec['true'][:2]), max(int(0.3 * s), 4), RED, -1, cv2.LINE_AA)
    if rec['est'] is not None:
        e = px(*rec['est'][:2])
        if rec['how'] == 'lost':
            cv2.circle(img, e, 10, RED, 2, cv2.LINE_AA)
        else:
            cv2.drawMarker(img, e, (0, 0, 0), cv2.MARKER_CROSS, 16, 2, cv2.LINE_AA)
    if rec['goal'] is not None:
        cv2.drawMarker(img, px(*rec['goal'][:2]), CYAN, cv2.MARKER_DIAMOND, 12, 2, cv2.LINE_AA)
    cv2.putText(img, 'rojo: globo | cruz: estimado | rombo: punto a 4 m', (14, size - 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (90, 90, 90), 1, cv2.LINE_AA)
    return img


def run(model, out, seed=3000, speed=7.0, density=0.0625, pilot='diffphys', cam_hz=10.0, standoff=4.0,
        opts=None, chase=True, edge=True, tag=''):
    """Fly one R20 episode and write its video → path of the .mp4."""
    from nanouav.agile import dagger
    from nanouav.agile.rollout import Config, Rollout
    from nanouav.follow.balloon_mission import BalloonMission
    from nanouav.follow.run_balloon import _env
    dagger._worker_init()
    env = _env(dict(seed=seed, density=density, speed=speed))
    M = BalloonMission(env, standoff=standoff, seed=seed)
    for k, v in (opts or {}).items():
        setattr(M, k, v)
    if (opts or {}).get('q_acc') is not None:
        M.flt = M._make_filter()
    recs, last = [], {}
    _detect, _frame = M._detect, M.frame

    def detect(cam):
        dets = _detect(cam)
        last['dets'] = [tuple(d['box']) for d in dets]
        return dets

    def frame(t, p, v, yaw, cam):
        F = _frame(t, p, v, yaw, cam)
        tr = next((k for k in M.tracker.tracks if k.id == M.lock.track_id and k.lost == 0), None)
        recs.append({'t': t, 'p': np.array(p), 'v': float(np.linalg.norm(v)), 'yaw': float(yaw),
                     'rgb': cam['rgb'], 'dets': last.get('dets', []),
                     'lock': tuple(tr.box) if tr is not None and F.get('how') in ('locked', 'relocked') else None,
                     'how': F.get('how'), 'true': np.array(M.target.pos(t)),
                     'others': [np.array(w.pos(t)) for w in M.walkers[1:]],
                     'est': None if M.flt.pos is None else M.flt.pos.copy(),
                     'goal': None if M.flt.pos is None else np.array(p) + _goal_of(M, p),
                     'v_des': float(F.get('v_des', 0.0))})
        return F
    M._detect, M.frame = detect, frame
    pol = dagger._policy(os.path.abspath(model)) if pilot == 'student' else \
        (os.path.abspath(model) if pilot == 'diffphys' else None)
    R = Rollout(env, Config(v_des=2.0, cam_hz=cam_hz, seed=seed), pilot, pol, mission=M)
    R.EDGE = edge
    r, _, log = R.run()
    res = 'EXITO' if r['success'] else ('CHOQUE' if r['crashed'] else 'GLOBO PERDIDO')
    d = os.path.join(out, 'video')
    os.makedirs(d, exist_ok=True)
    name = (os.path.splitext(os.path.basename(model))[0] if model else pilot) + (f'_{tag}' if tag else '')
    fn = os.path.join(d, f'balloon_{name}_d{density:g}_v{speed:g}_s{seed}.mp4')
    L = np.array([row[:7] for row in log], float)
    trail = np.array([M.target.pos(t)[:2] for t in np.arange(0, M.t_max(), 0.1)])
    W, H = 720 + 480, 720 + 30
    wr = _writer(fn, cam_hz, (W, H))
    img = None
    for i, rec in enumerate(recs):
        p, yaw, t = rec['p'], rec['yaw'], rec['t']
        j = min(int(np.searchsorted(L[:, 0], t)), len(L) - 1)
        lastf = i == len(recs) - 1
        top = _topdown(types.SimpleNamespace(world=env.world, goal=(1e4, 0.0)), L[:j + 1:2, 1:3], None, p, yaw,
                       result=f'{res} ({r["time_s"]:.1f} s)' if lastf else None)
        top = _overlay_balloons(top, rec, p, yaw, trail)
        if chase:
            for wk in M.walkers:
                wk.move(t)
            low = _chase_rgb(env.render_world(), p, yaw)
        else:
            low = _chase(env, p, yaw)
        img = np.vstack([np.hstack([top, np.vstack([_camera(rec), low])]), np.full((30, W, 3), 255, np.uint8)])
        dist = float(np.hypot(*(rec['true'][:2] - p[:2])))
        info = (f'{name} | bosque 1/{1 / density:.0f} m-2 | globo {speed:g} m/s | semilla {seed} | t={t:5.2f} s | '
                f'dron {rec["v"]:4.1f} m/s (pedido {rec["v_des"]:3.1f}) | distancia {dist:4.1f} m')
        cv2.putText(img, info, (8, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        wr.write(img)
    for _ in range(int(cam_hz * 1.5)):
        wr.write(img)
    wr.release()
    fn = _to_h264(fn)
    print(f'{res} · {r["time_s"]} s · error de distancia {r.get("dist_err_med")} m → {fn}')
    return fn


def _goal_of(M, p):
    """Offset from the drone to the follow point (the point at `standoff` m from the estimate)."""
    tgt = M.flt.pos
    if tgt is None:
        return np.zeros(3)
    h = np.asarray(p[:2]) - tgt[:2]
    n = max(float(np.hypot(*h)), 1e-6)
    goal = np.r_[tgt[:2] + h / n * M.standoff, tgt[2]]
    return goal - np.asarray(p)
