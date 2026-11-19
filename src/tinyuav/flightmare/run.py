"""Flightmare (Unity) as the renderer of the closed loop: our dynamics, Unity's stereo images.

  export TINYUAV_FLIGHTMARE=/path/RPG_Flightmare.x86_64
  tinyuav flightmare arena                               # scan the empty scene once
  tinyuav flightmare eval --pilots diffphys --model M.keras --v 3,5,7,10 --n 30 --edge
  tinyuav flightmare video --pilot diffphys --model M.keras --v 10 --env 8 --edge --view sgm
  tinyuav flightmare follow --model F.keras --speed 7 --fast-target
  tinyuav flightmare follow-video --model F.keras --speed 7 --fast-target
  tinyuav flightmare fake                                # our renderer behind the same protocol

Binaries: 2022 (agile_flight: 0 warehouse, 1 street, 2 forest, 3 wasteland) and 2020 (GitHub release).
"""
import argparse
import json
import math
import os
import time

import numpy as np

from tinyuav import paths
from tinyuav.nav.run import add_flight_opts, apply_opts, opts_of

SCENES = {'2022': {'forest': (2, (21.0, 217.0, 94.0)), 'warehouse': (0, (0, 0, 0)), 'street': (1, (-9.38, -12.66, 0.1)),
                   'wasteland': (3, (1140.0, 1026.0, 108.0))},
          '2020': {'forest': (3, (0, 0, 0)), 'industrial': (0, (0, 0, 0)), 'warehouse': (1, (0, 0, 0)), 'garage': (2, (0, 0, 0))}}
ARENA_BOX = {'warehouse': ('30,0,5', '100,20,10')}      # center and extents (m) of the scanned box
TILT_V, TILT_DEG = 7.0, 30.0                             # student: camera tilted up at >= 7 m/s
SEEING = ('student', 'reactive', 'diffphys')             # pilots that fly on Unity's images
_UNITY = None


def _out(a):
    d = a.out or os.path.join(str(paths.results_dir()), 'flightmare')
    os.makedirs(d, exist_ok=True)
    return d


def _calib(a):
    fn = os.path.join(_out(a), 'calib.json')
    return json.load(open(fn)) if os.path.exists(fn) else {}


def _kill_players(name):
    """Terminate processes whose argv[0] ends with `name`."""
    import signal
    killed = False
    for d in os.listdir('/proc'):
        if not d.isdigit() or int(d) in (os.getpid(), os.getppid()):
            continue
        try:
            argv0 = open(f'/proc/{d}/cmdline', 'rb').read().split(b'\0')[0].decode(errors='ignore')
        except OSError:
            continue
        if argv0.endswith(name):
            try:
                os.kill(int(d), signal.SIGTERM)
                killed = True
            except OSError:
                pass
    if killed:
        time.sleep(2.0)


def _launch_unity(a):
    """Start a fresh player per command (--unity or $TINYUAV_FLIGHTMARE); a reused one drops reconnects."""
    global _UNITY
    exe = a.unity or os.environ.get('TINYUAV_FLIGHTMARE', '')
    if not exe or _UNITY is not None:
        return
    import atexit
    import subprocess
    exe = os.path.expanduser(exe)
    if not os.path.exists(exe):
        raise SystemExit(f'Unity player not found: {exe}')
    _kill_players(os.path.basename(exe))
    logf = open(os.path.join(_out(a), 'unity.log'), 'w')
    _UNITY = subprocess.Popen([exe, '-input-port', str(a.pub_port), '-output-port', str(a.sub_port),
                               '-client-ip', '127.0.0.1'], stdout=logf, stderr=subprocess.STDOUT)
    print(f'started {exe} (pid {_UNITY.pid}, log {logf.name})')

    def _stop():
        if _UNITY.poll() is None:
            _UNITY.terminate()
            try:
                _UNITY.wait(10)
            except Exception:
                _UNITY.kill()
    atexit.register(_stop)


def rig_of(a):
    from tinyuav.sim.stereo import StereoRig
    return StereoRig(width=640, height=480, num_disp=96, texture_win=11) if a.vga else StereoRig()


def unity_fov(rig):
    """Vertical field of view (deg) that gives Unity's images the focal length of the rig."""
    return math.degrees(2 * math.atan(rig.height / 2 / rig.focal))


def make_client(a, calib=None, stereo=True, n_objects=0, rig=None):
    from tinyuav.flightmare.bridge import FlightmareClient
    from tinyuav.sim.stereo import StereoRig
    os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')
    _launch_unity(a)
    sid, off = SCENES[a.binary][a.scene]
    c = calib or {}
    rig = rig or StereoRig()
    cl = FlightmareClient(scene_id=sid, render_offset=off, fov_deg=unity_fov(rig), depth_mult=c.get('depth_mult'),
                          depth_far=a.depth_far, vehicle_size=a.vehicle_size, stereo=stereo, n_objects=n_objects,
                          box_unit=c.get('box_unit', (0.8104, 0.8102, 0.8124)), width=rig.width, height=rig.height,
                          pub_port=a.pub_port, sub_port=a.sub_port)
    print(f'waiting for Flightmare ({a.scene}, scene {sid}, {a.binary} binary) on ports {a.pub_port}/{a.sub_port}')
    cl.connect()
    print('connected')
    return cl


class FlightmareObserver:
    """Rollout observer: Unity's stereo pair at the drone pose, returns (gray, SGM depth, Unity depth, left image)."""

    def __init__(self, client, rig, calib=None, overlay=None):
        """overlay(o, pose) edits Unity's images before stereo matching."""
        self.c, self.rig, self.calib, self.overlay = client, rig, calib or {}, overlay
        self.t_render = []

    def __call__(self, p, yaw, pitch, roll):
        import cv2
        from tinyuav.sim.stereo import sgm_depth
        from tinyuav.sim.render import body_rot
        t0 = time.time()
        o = self.c.render(np.asarray(p, float), body_rot(yaw, pitch, roll))
        if self.overlay is not None:
            o = self.overlay(o, (np.asarray(p, float), yaw, pitch, roll))
        self.t_render.append(time.time() - t0)
        g = lambda im: cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) if im.ndim == 3 and im.shape[2] == 3 else im[..., 0]
        gl, gr = g(o['left']), g(o['right'])
        sgm = sgm_depth(gl, gr, self.rig)
        d = o['depth_left']
        gt = np.where((d > 0) & (d < self.rig.max_range), d, 0.0).astype(np.float32) if d is not None else np.zeros_like(sgm)
        return gl, sgm, gt, o['left']


def _arena(a):
    from tinyuav.flightmare import arena as A
    fn = os.path.join(_out(a), f'arena_{a.scene}.npz')
    if not os.path.exists(fn):
        raise SystemExit(f'{fn} not found: run `tinyuav flightmare arena --scene {a.scene}` first')
    return A.load(fn)


def _episode(a, env, i, pilot, v, client, calib, model, trace=False, rig=None):
    import dataclasses
    from tinyuav.nav import dagger
    from tinyuav.nav.rollout import Config, Rollout
    from tinyuav.nav.student import omega_of
    rig = dataclasses.replace(rig, tilt_up_deg=TILT_DEG if pilot != 'diffphys' and v >= TILT_V - 1e-6 else 0.0)
    if client is not None:
        client.set_tilt(rig.tilt_up_deg)
    if pilot == 'diffphys' and not model:
        raise SystemExit('the diffphys pilot needs --model')
    om = omega_of(model) if pilot == 'student' and model else 'line'
    cfg = Config(v_des=v, cam_hz=a.cam_hz, seed=1000 + i, omega_ref=om, latency=a.latency, rig=rig)
    R = Rollout(env, cfg, pilot, dagger._policy(model) if pilot == 'student' else model if pilot == 'diffphys' else None)
    apply_opts(R, opts_of(a)[0])
    obs = None
    if pilot in SEEING or trace:
        if client.n_objects:
            client.place(env.world.objects)
        obs = R.observer = FlightmareObserver(client, cfg.rig, calib)
    if trace:
        R.trace, R.trace_depth, R.record_cam = [], True, True
        if pilot == 'diffphys':
            R.dp_trace = []
    r, _, log = R.run()
    r.update(env=i, pilot=pilot, v=v,
             render_ms=round(1000 * float(np.mean(obs.t_render)), 1) if obs and obs.t_render else None)
    if r['crashed'] and log:
        r['crash'] = crash_info(env.world, np.array(log[-1][1:4], float))
    return r, R, log


def crash_info(w, p):
    """What the drone hit: a spawned trunk, the static scene or the floor."""
    from tinyuav.flightmare.arena import box_distance
    P = np.asarray(p, float)[None]
    db = box_distance(P, w.C, w.H)[0] if len(w.C) else np.inf
    ds, df = float(w.sdf(P)[0]), float(p[2] - w.floor)
    k = int(np.argmin([db, ds, df]))
    out = {'at': np.round(p, 2).tolist(), 'what': ['trunk', 'scene', 'floor'][k], 'dist': round(float(min(db, ds, df)), 3)}
    if k == 0:
        j = int(np.argmin(np.linalg.norm(np.maximum(np.abs(w.C - p) - w.H, 0), axis=1)))
        out['obstacle'] = np.round(w.C[j], 2).tolist()
    else:
        out['obstacle'] = np.round(p, 2).tolist()
    out['above_ground'] = round(float(p[2] - w.floor), 2)
    return out


def _save_decisions(fn, r, R):
    if R.dp_trace is not None:
        with open(fn[:-4] + '.json', 'w') as fh:
            json.dump({'result': r, 'decisions': R.dp_trace}, fh, default=float)


def cmd_eval(a):
    from tinyuav.flightmare.arena import arena_env
    out, calib, ar = _out(a), _calib(a), _arena(a)
    pilots = a.pilots.split(',')
    speeds = [float(x) for x in a.v.split(',')]
    rig = rig_of(a)
    client = make_client(a, calib, n_objects=a.n_objects, rig=rig) if set(SEEING) & set(pilots) else None
    model = os.path.abspath(a.model) if a.model else None
    table, t0 = {}, time.time()
    for p in pilots:
        for v in speeds:
            for i in range(a.n):
                r, *_ = _episode(a, arena_env(ar, 1000 + i, kind='test'), i, p, v, client, calib, model, rig=rig)
                table.setdefault(f'{p}_v{v:g}', []).append(r)
                print(f"{p:8s} v={v:g} env {i}: {'ok' if r['success'] else 'CRASH' if r['crashed'] else 'fail'} "
                      f"{r['time_s']:.1f} s" + (f"  hit {r['crash']}" if r.get('crash') else ''), flush=True)
    tag = (os.path.splitext(os.path.basename(model))[0] if model else 'pilots') + opts_of(a)[1] + \
        (f'_{a.tag}' if a.tag else '')
    fn = os.path.join(out, f'eval_{a.scene}_{tag}.json')
    old = json.load(open(fn)) if os.path.exists(fn) else {}
    old.update(table)
    json.dump(old, open(fn, 'w'), indent=1)
    print(f'{"pilot":10s} {"v":>5s} {"success":>8s} {"crash":>6s} {"%/s":>6s} {"%/m":>6s} {"dec_ms":>7s}')
    for p in pilots:
        for v in speeds:
            R = table[f'{p}_v{v:g}']
            k, T = sum(r['crashed'] for r in R), sum(r['time_s'] for r in R)
            D = sum(r['time_s'] * r['mean_speed'] for r in R)
            dm = [r['decide_ms'] for r in R if r.get('decide_ms') is not None]
            print(f'{p:10s} {v:5.1f} {sum(r["success"] for r in R):3d}/{len(R):<4d} {k:6d} '
                  f'{100 * k / max(T, 1e-9):6.1f} {100 * k / max(D, 1e-9):6.2f} '
                  f'{np.median(dm) if dm else float("nan"):7.1f}')
    print(f'{fn}  ({time.time() - t0:.0f} s)')
    if client:
        client.close()


def cmd_compare(a):
    from tinyuav.nav.stats import compare
    compare(a.files)


def cmd_video(a):
    from tinyuav.flightmare.arena import arena_env
    from tinyuav.nav.video import render_episode
    out, calib, ar = _out(a), _calib(a), _arena(a)
    env = arena_env(ar, 1000 + a.env, kind='test')
    rig = rig_of(a)
    client = make_client(a, calib, n_objects=a.n_objects, rig=rig)
    try:
        r, R, log = _episode(a, env, a.env, a.pilot, a.v, client, calib,
                             os.path.abspath(a.model) if a.model else None, trace=True, rig=rig)
    finally:
        client.close()
    os.makedirs(os.path.join(out, 'video'), exist_ok=True)
    tag = os.path.splitext(os.path.basename(a.model))[0] if a.pilot in ('student', 'diffphys') else a.pilot
    fn = os.path.join(out, 'video', f'{tag}_{a.scene}_env{a.env}_v{a.v:g}_{a.view}.mp4')
    res = 'EXITO' if r['success'] else ('CHOQUE' if r['crashed'] else 'SIN LLEGAR')
    render_episode(log, R.trace, R.cfg.rig, fn, fps=a.cam_hz, result=f"{res} ({r['time_s']:.1f} s)", view=a.view,
                   decisions=R.dp_trace, v_des=a.v)
    _save_decisions(fn, r, R)
    print(f"{res} · {r['time_s']} s · {fn}")


def _follow_setup(a):
    from tinyuav.follow.run import lost_opts
    rig = rig_of(a)
    calib = _calib(a)
    return _out(a), _arena(a), rig, calib, make_client(a, calib, n_objects=a.n_objects, rig=rig), lost_opts(a)


def cmd_follow(a):
    """Follow a balloon painted into Unity's images through the crates of the arena."""
    from tinyuav.flightmare.follow import balloon_arena_env, episode
    out, ar, rig, calib, client, opts = _follow_setup(a)
    model = os.path.abspath(a.model) if a.model else None
    pilots = a.pilots.split(',')
    table, t0 = {}, time.time()
    try:
        for p in pilots:
            for s in range(a.seed0, a.seed0 + a.n):
                env = balloon_arena_env(ar, s, density=a.density, speed=a.speed, color=a.color)
                r, *_ = episode(client, rig, calib, env, s, p, model, standoff=a.standoff, cam_hz=a.cam_hz, opts=opts,
                                edge=a.edge)
                table.setdefault(p, []).append(r)
                print(f"{p:8s} seed {s}: {'ok' if r['success'] else 'CRASH' if r['crashed'] else 'lost'} "
                      f"{r['time_s']:.1f} s  d_err {r.get('dist_err_med')}  render {r.get('render_ms')} ms", flush=True)
    finally:
        client.close()
    fn = os.path.join(out, f'balloon_{a.scene}_{a.tag or "follow"}.json')
    json.dump(table, open(fn, 'w'), indent=1)
    from tinyuav.follow.run import print_table
    print_table(table, pilots)
    print(f'{fn}  ({time.time() - t0:.0f} s)')


def cmd_follow_video(a):
    from tinyuav.flightmare.follow import balloon_arena_env, episode
    from tinyuav.nav.video import render_episode
    out, ar, rig, calib, client, opts = _follow_setup(a)
    model = os.path.abspath(a.model) if a.model else None
    try:
        env = balloon_arena_env(ar, a.seed, density=a.density, speed=a.speed, color=a.color)
        r, R, log = episode(client, rig, calib, env, a.seed, a.pilot, model, standoff=a.standoff, cam_hz=a.cam_hz,
                            opts=opts, edge=a.edge, trace=True)
    finally:
        client.close()
    name = (os.path.splitext(os.path.basename(model))[0] if model else a.pilot) + (f'_{a.tag}' if a.tag else '')
    os.makedirs(os.path.join(out, 'video'), exist_ok=True)
    v = env.meta['speed']
    fn = os.path.join(out, 'video', f'balloon_{a.scene}_{name}_d{a.density:g}_v{v:g}_s{a.seed}_{a.view}.mp4')
    res = 'EXITO' if r['success'] else ('CHOQUE' if r['crashed'] else 'GLOBO PERDIDO')
    render_episode(log, R.trace, R.cfg.rig, fn, fps=a.cam_hz, result=f"{res} ({r['time_s']:.1f} s)", view=a.view,
                   decisions=R.dp_trace)
    _save_decisions(fn, r, R)
    print(f"{res} · {r['time_s']} s · d_err {r.get('dist_err_med')} m · {fn}")


def cmd_arena(a):
    """Scan the empty scene, find its floor and free rectangle, and calibrate the spawned box against Unity's
    depth: arena_<scene>.npz and box_unit in calib.json."""
    from tinyuav.flightmare import arena as A
    from tinyuav.flightmare.scan import scan
    from tinyuav.sim.render import Box, body_rot
    out = _out(a)
    calib = _calib(a)
    o_def, s_def = ARENA_BOX.get(a.scene, ('30,0,4', '80,40,10'))
    origin = np.array([float(v) for v in (a.origin or o_def).split(',')])
    size = np.array([float(v) for v in (a.size or s_def).split(',')])
    cl = make_client(a, calib, stereo=False, n_objects=1)
    P, hits, st = scan(cl, origin, size, a.res, a.step, [float(v) for v in a.heights.split(',')], a.yaws,
                       [float(v) for v in a.pitches.split(',')], a.max_range, a.stride, min_hits=2)
    lo, hi = origin - size / 2, origin + size / 2
    sdf, floor, rect = A.build(P, a.res, lo, hi)
    info = {'scan': st, 'floor': round(floor, 3), 'rect': [round(x, 2) for x in rect],
            'free_length': round(rect[1] - rect[0], 1), 'free_width': round(rect[3] - rect[2], 1)}
    # a 0.5 x 0.5 x 3 m box 6 m ahead, measured in Unity's depth
    yc = 0.5 * (rect[2] + rect[3])
    p = np.array([rect[0] + 2.0, yc, floor + 1.5])
    box = Box((p[0] + 6.0, yc, floor + 1.5), (0.25, 0.25, 1.5))
    cl.place([box])
    d = cl.render(p, body_rot(0.0, 0.0, 0.0))['depth_left']
    H, W = d.shape
    f = H / 2 / math.tan(math.radians(cl.fov) / 2)
    front_exp = 6.0 - 0.25 - cl.T_left[0, 3]
    win = d[H // 2 - 2:H // 2 + 3, W // 2 - 2:W // 2 + 3]
    front = float(np.median(win))

    def run_len(x, c):                                   # run of pixels at the box depth through index c
        m = np.abs(x - front) < 0.3
        if not m[c]:
            return 0
        i0 = c
        while i0 > 0 and m[i0 - 1]:
            i0 -= 1
        i1 = c
        while i1 < len(m) - 1 and m[i1 + 1]:
            i1 += 1
        return i1 - i0 + 1
    w_px = run_len(d[H // 2], W // 2)
    # upper half only: the floor touches the bottom of the box
    h_up = run_len(d[:H // 2 + 1, W // 2][::-1], 0)
    w_exp, h_up_exp = f * 0.5 / front_exp, min(f * 1.5 / front_exp, H / 2)
    unit = list(cl.box_unit)
    ok_box = abs(front - front_exp) < 0.15 and abs(w_px / w_exp - 1) < 0.15 and abs(h_up / h_up_exp - 1) < 0.15
    if not ok_box and abs(front - front_exp) < 0.6 and w_px > 3 and h_up > 3:      # rescale the prefab
        unit = [unit[0] * w_px / w_exp, unit[1] * w_px / w_exp, unit[2] * h_up / h_up_exp]
    h_px, h_exp = h_up, h_up_exp
    info['box_check'] = {'front_m': round(front, 3), 'front_expected_m': round(front_exp, 3), 'width_px': w_px,
                         'width_expected_px': round(w_exp, 1), 'upper_height_px': h_px, 'upper_height_expected_px': round(h_exp, 1),
                         'ok': bool(ok_box), 'box_unit': [round(u, 4) for u in unit]}
    cl.close()
    calib['box_unit'] = info['box_check']['box_unit']
    json.dump(calib, open(os.path.join(out, 'calib.json'), 'w'), indent=1)
    fn = os.path.join(out, f'arena_{a.scene}.npz')
    A.save(fn, P, a.res, floor, rect, lo, hi, dict(info, scene=a.scene, binary=a.binary))
    print(json.dumps(info, indent=1))
    warn = []
    if rect[1] - rect[0] < 48:
        warn.append(f'free length {rect[1] - rect[0]:.1f} m < 48 m: flights will be shorter than 40 m')
    if rect[3] - rect[2] < 10:
        warn.append(f'free width {rect[3] - rect[2]:.1f} m is narrow')
    if not ok_box:
        warn.append('spawned box off: see box_check (box_unit rescaled if possible)')
    for w_ in warn:
        print('WARNING:', w_)
    print(('OK: ' if not warn else '') + fn)


def cmd_fake(a):
    from tinyuav.flightmare.fake import FakeUnity, arena_world
    from tinyuav.sim import envs as EV
    world = arena_world(a.seed) if a.world == 'arena' else EV.TEST[a.world](a.seed).world
    f = FakeUnity(world, a.pub_port, a.sub_port).start()
    print(f'fake Flightmare: {a.world} seed {a.seed} on ports {a.pub_port}/{a.sub_port}, Ctrl+C to stop')
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        f.stop()


def main(argv=None):
    from tinyuav.follow.run import add_lost_opts
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('arena', help='scan the empty scene once')
    p.add_argument('--origin', default='', help='center of the scanned box (default per scene)')
    p.add_argument('--size', default='', help='extents x,y,z (default per scene)')
    p.add_argument('--res', type=float, default=0.15)
    p.add_argument('--step', type=float, default=4.0)
    p.add_argument('--heights', default='1.5,3.0,5.0')
    p.add_argument('--yaws', type=int, default=6)
    p.add_argument('--pitches', default='-30,0,30')
    p.add_argument('--max-range', type=float, default=25.0)
    p.add_argument('--stride', type=int, default=2)
    for name in ('eval', 'video'):
        p = sub.add_parser(name)
        p.add_argument('--model', default='')
        add_flight_opts(p, select='sgm', keep=True)
        if name == 'eval':
            p.add_argument('--v', default='3,5,7')
            p.add_argument('--pilots', default='gbl,student')
            p.add_argument('--n', type=int, default=10)
            p.add_argument('--tag', default='')
        else:
            p.add_argument('--v', type=float, default=3.0)
            p.add_argument('--pilot', default='student', choices=['student', 'expert', 'gbl', 'reactive', 'diffphys'])
            p.add_argument('--env', type=int, default=0, help='test environment index')
            p.add_argument('--view', default='perception', choices=['perception', 'sgm'])
    for name in ('follow', 'follow-video'):
        p = sub.add_parser(name)
        p.add_argument('--model', default='', help='diffphys follow model')
        p.add_argument('--density', type=float, default=1 / 16, help='crates per m^2')
        p.add_argument('--speed', type=float, default=None, help='balloon speed (m/s; default 1-4 at random)')
        p.add_argument('--standoff', type=float, default=4.0)
        p.add_argument('--color', default='red')
        p.add_argument('--no-edge', dest='edge', action='store_false', help='diffphys without the edge filter')
        p.add_argument('--tag', default='')
        add_lost_opts(p)
        if name == 'follow':
            p.add_argument('--pilots', default='gbl,direct,diffphys')
            p.add_argument('--n', type=int, default=30)
            p.add_argument('--seed0', type=int, default=3000)
        else:
            p.add_argument('--pilot', default='diffphys')
            p.add_argument('--seed', type=int, default=3000)
            p.add_argument('--view', default='perception', choices=['perception', 'sgm'])
    p = sub.add_parser('compare', help='eval JSONs on the same environments, paired against the first')
    p.add_argument('files', nargs='+')
    p = sub.add_parser('fake', help='our renderer behind the Flightmare protocol')
    p.add_argument('--world', default='forest', choices=['forest', 'gap', 'arena'])
    p.add_argument('--seed', type=int, default=1000)
    for q in sub.choices.values():
        q.add_argument('--results', dest='out', default='', help='results directory (default results/flightmare)')
        q.add_argument('--scene', default='warehouse')
        q.add_argument('--binary', default='2022', choices=['2022', '2020'])
        q.add_argument('--pub-port', type=int, default=10253)
        q.add_argument('--sub-port', type=int, default=10254)
        q.add_argument('--cam-hz', type=float, default=None, help='default 15; follow: 10')
        q.add_argument('--vehicle-size', type=float, default=0.01, help='scale of the drone mesh')
        q.add_argument('--depth-far', type=float, default=100.0, help='far clip of the depth layer (m)')
        q.add_argument('--vga', action=argparse.BooleanOptionalAction, default=None,
                       help='640 x 480 stereo pair (default on, off for follow)')
        q.add_argument('--n-objects', type=int, default=160, help='spawnable crates')
        q.add_argument('--unity', default='', help='RPG_Flightmare.x86_64 to start (default $TINYUAV_FLIGHTMARE)')
    a = ap.parse_args(argv)
    follow = a.cmd in ('follow', 'follow-video')
    a.cam_hz = a.cam_hz or (10.0 if follow else 15.0)
    a.vga = (not follow) if a.vga is None else a.vga
    if a.scene not in SCENES[a.binary]:
        raise SystemExit(f'scene {a.scene!r} is not in the {a.binary} binary: {sorted(SCENES[a.binary])}')
    return {'arena': cmd_arena, 'eval': cmd_eval, 'video': cmd_video, 'follow': cmd_follow,
            'follow-video': cmd_follow_video, 'compare': cmd_compare, 'fake': cmd_fake}[a.cmd](a)


if __name__ == '__main__':
    main()
