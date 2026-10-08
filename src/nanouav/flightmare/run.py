"""R22 · the R18 student flying in Flightmare's Unity scenes (the simulator of the paper).

Start the Unity standalone first (it connects to us), then:

  nanouav run r22 probe                      # one stereo pair + depth + point cloud: checks the conventions → calib.json
  nanouav run r22 cloud --origin 30,0,4 --size 80,40,10      # the privileged map of the flight area → cloud.npz
  nanouav run r22 scan  --origin 30,0,4 --size 80,40,10      # the same map from Unity depth images (no colliders)
  nanouav run r22 survey --size 300          # top-down mosaic of the scene: where to scan and fly
  nanouav run r22 lines --n 10               # start → goal lines (40 m) with a collision-free global path → lines.json
  nanouav run r22 eval --model results/r18/student_train.keras --pilots gbl,student --v 3,5,7
  nanouav run r22 video --model results/r18/student_train.keras --v 3 --line 0
  nanouav run r22 fake                       # without Unity: our renderer behind the same protocol (tests, dry runs)

Scenes: the 2022 binary of agile_flight (download.ifi.uzh.ch/rpg/Flightmare/RPG_Flightmare.zip) has
0 warehouse, 1 street, 2 forest, 3 wasteland and needs the scene's render offset (`--binary 2022`, default);
the 2020 GitHub release has 0 industrial, 1 warehouse, 2 garage, 3 forest (`--binary 2020`).
Results in results/r22. See docs/research/R22_flightmare.md.
"""
import argparse
import json
import math
import os
import time

import numpy as np

from nanouav import paths

SCENES = {'2022': {'forest': (2, (21.0, 217.0, 94.0)), 'warehouse': (0, (0, 0, 0)), 'street': (1, (-9.38, -12.66, 0.1)),
                   'wasteland': (3, (1140.0, 1026.0, 108.0))},
          '2020': {'forest': (3, (0, 0, 0)), 'industrial': (0, (0, 0, 0)), 'warehouse': (1, (0, 0, 0)), 'garage': (2, (0, 0, 0))}}


def _out(a):
    d = a.out or os.path.join(str(paths.results_dir()), 'r22')
    os.makedirs(d, exist_ok=True)
    return d


def _calib(a):
    fn = os.path.join(_out(a), 'calib.json')
    return json.load(open(fn)) if os.path.exists(fn) else {}


_UNITY = None


def _kill_players(name):
    """Terminate running players: processes whose argv[0] ends with `name` (not ours: our own command line
    contains the path too, so no `pkill -f`)."""
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
    """Start the Unity player for this command (--unity or $NANOUAV_FLIGHTMARE): after a client leaves, the
    player waits 10 s, reloads its menu scene and has to reconnect; a fresh player per command avoids that."""
    global _UNITY
    exe = a.unity or os.environ.get('NANOUAV_FLIGHTMARE', '')
    if not exe or _UNITY is not None:
        return
    import atexit
    import subprocess
    exe = os.path.expanduser(exe)
    if not os.path.exists(exe):
        raise SystemExit(f'Unity player not found: {exe}')
    _kill_players(os.path.basename(exe))                                        # a stale player holds the ports
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


def make_client(a, calib=None, stereo=True):
    from nanouav.flightmare.bridge import FlightmareClient
    _launch_unity(a)
    sid, off = SCENES[a.binary][a.scene]
    c = calib or {}
    cl = FlightmareClient(scene_id=sid, render_offset=off, fov_deg=c.get('fov_deg', 73.74),
                          depth_mult=c.get('depth_mult'), depth_far=a.depth_far, vehicle_size=a.vehicle_size,
                          stereo=stereo,
                          pub_port=a.pub_port, sub_port=a.sub_port)
    print(f'waiting for Flightmare ({a.scene}, scene {sid}, {a.binary} binary) on ports {a.pub_port}/{a.sub_port} …')
    cl.connect()
    print('connected')
    return cl


class FlightmareObserver:
    """Rollout.observer: Flightmare's stereo pair at the drone pose → (gray, SGM depth, Unity depth)."""

    def __init__(self, client, rig, calib=None):
        self.c, self.rig, self.calib = client, rig, calib or {}
        self.t_render = []

    def __call__(self, p, yaw, pitch, roll):
        import cv2
        from nanouav.agile.stereo import sgm_depth
        from nanouav.scene.renderer import body_rot
        t0 = time.time()
        o = self.c.render(np.asarray(p, float), body_rot(yaw, pitch, roll))
        self.t_render.append(time.time() - t0)
        g = lambda im: cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) if im.ndim == 3 and im.shape[2] == 3 else im[..., 0]
        gl, gr = g(o['left']), g(o['right'])
        sgm = sgm_depth(gl, gr, self.rig)
        d = o['depth_left']
        gt = np.where((d > 0) & (d < self.rig.max_range), d, 0.0).astype(np.float32) if d is not None else np.zeros_like(sgm)
        return gl, sgm, gt


# ── probe: check the conventions on one frame ───────────────────────────────
def predicted_depth(points, rig, pose, focal):
    """Forward distance seen by the left camera from a point cloud (z-buffer, 2 × 2 px splats)."""
    from nanouav.follow.estimate import camera_pose
    O, Rc = camera_pose(rig, pose)
    Q = (points - O) @ Rc
    X = Q[:, 0]
    k = X > 0.3
    Q, X = Q[k], X[k]
    u = (-Q[:, 1] / X * focal + rig.width / 2 - 0.5).round().astype(int)
    v = (-Q[:, 2] / X * focal + rig.height / 2 - 0.5).round().astype(int)
    D = np.full((rig.height, rig.width), np.inf, np.float32)
    for du in (0, 1):
        for dv in (0, 1):
            uu, vv = u + du, v + dv
            m = (uu >= 0) & (uu < rig.width) & (vv >= 0) & (vv < rig.height)
            np.minimum.at(D, (vv[m], uu[m]), X[m])
    return D


def compare(pred, meas, max_d=10.0):
    m = np.isfinite(pred) & (pred < max_d) & (meas > 0.2) & (meas < max_d * 1.5)
    if m.sum() < 50:
        return float('inf'), int(m.sum())
    return float(np.median(np.abs(meas[m] - pred[m]) / pred[m])), int(m.sum())


def cmd_probe(a):
    """One stereo pair + Unity depth + the point cloud around it, at --pose; checks the conventions → calib.json,
    probe.png (left | right | — / SGM | Unity depth | depth predicted from the cloud), probe.npz (raw data)."""
    import cv2
    from nanouav.agile.stereo import StereoRig, sgm_depth
    from nanouav.scene.renderer import body_rot
    out = _out(a)
    cl = make_client(a)
    rig = StereoRig()
    x, y, z, yaw = (float(v) for v in a.pose.split(','))
    p = np.array([x, y, z])
    o = cl.render(p, body_rot(math.radians(yaw), 0.0, 0.0))
    gl = cv2.cvtColor(o['left'], cv2.COLOR_RGB2GRAY)
    gr = cv2.cvtColor(o['right'], cv2.COLOR_RGB2GRAY)
    sgm = sgm_depth(gl, gr, rig)
    calib = {'scene': a.scene, 'binary': a.binary, 'pose': a.pose, 'vehicle_size': a.vehicle_size}
    warn = []
    # 1 · the two cameras see different images (a baseline), and SGM finds matches
    lr = float(np.abs(gl.astype(int) - gr).mean())
    calib['stereo'] = {'mean_abs_left_right': round(lr, 2), 'sgm_valid': round(float((sgm > 0).mean()), 3)}
    if lr < 1.0:
        warn.append('left and right images are (almost) the same: no stereo baseline')
    # 2 · Unity depth: value × far clip plane = metres along the optical axis? (checked against SGM)
    raw = o['depth_left'] / cl.depth_mult if o['depth_left'] is not None else None
    if raw is not None:
        m = (sgm > 1.0) & (sgm < 15.0) & (raw > 0)
        ratio = float(np.median(sgm[m] / raw[m])) if m.sum() > 200 else float('nan')
        calib['unity_depth'] = {'raw_median': float(np.median(raw)), 'raw_max': float(raw.max()),
                                'sgm_over_raw': round(ratio, 2), 'far_clip': cl.depth_far, 'n': int(m.sum())}
        if np.isfinite(ratio) and abs(math.log(ratio / cl.depth_far)) < math.log(1.3):
            calib['depth_mult'] = cl.depth_far
        elif np.isfinite(ratio):
            calib['depth_mult'] = round(ratio, 3)
            warn.append(f'Unity depth × {cl.depth_far:g} does not match SGM (SGM / raw = {ratio:.2f})')
        else:
            calib['depth_mult'] = cl.depth_far
            warn.append('too few SGM pixels to check the Unity depth scale')
    unity_d = raw * calib['depth_mult'] if raw is not None else None
    # 3 · the point cloud of the box ahead (scene frame) seen from the pose
    size = np.array([40.0, 24.0, 10.0])
    c, s_ = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    origin = p + np.array([15.0 * c, 15.0 * s_, 1.0])
    print(f'asking for the point cloud ({size[0]:g} × {size[1]:g} × {size[2]:g} m, res {a.res}) — Unity freezes '
          'meanwhile …', flush=True)
    P, info = cl.cloud_points(origin, size, resolution=a.res, path=os.path.join(out, 'point_clouds'), name='probe')
    calib['cloud'] = info
    if len(P):
        z_lo, z_hi = np.percentile(P[:, 2], [1, 99])
        calib['cloud'].update(z_range=[round(float(z_lo), 2), round(float(z_hi), 2)])
        if z_hi - z_lo < 1.0:
            warn.append('the point cloud is a flat sheet: the scene has (almost) no colliders here, '
                        'build the map with `r22 scan` instead of `r22 cloud`')
    else:
        warn.append('the point cloud is empty: build the map with `r22 scan`')
    pose = (p, math.radians(yaw), 0.0, 0.0)
    res, preds = {}, {}
    for mirror in (False, True):
        Q = P * np.array([1, -1, 1]) if mirror else P
        for fname, f in (('vertical', rig.focal), ('horizontal', rig.width / 2 / math.tan(math.radians(73.74) / 2))):
            key = f'mirror={mirror} fov={fname}'
            pred = predicted_depth(Q, rig, pose, f) if len(Q) else np.full(sgm.shape, np.inf, np.float32)
            preds[key] = pred
            e_s, n_s = compare(pred, sgm)
            row = {'sgm_rel_err': round(e_s, 3), 'n_sgm': n_s}
            if unity_d is not None:
                e_u, n_u = compare(pred, unity_d)
                row.update(unity_rel_err=round(e_u, 3), n_unity=n_u)
            res[key] = row
    score = lambda k: res[k].get('unity_rel_err', res[k]['sgm_rel_err'])
    best = min(res, key=score)
    calib.update(checks=res, best=best, y_mirror='mirror=True' in best, fov_is=best.split('fov=')[1],
                 fov_deg=73.74 if 'vertical' in best else 90.0)
    if score(best) > 0.15:
        warn.append(f'the cloud does not explain the depth image ({best}: {score(best):.2f} median relative error)')
    calib['warnings'] = warn
    json.dump(calib, open(os.path.join(out, 'calib.json'), 'w'), indent=1)
    np.savez_compressed(os.path.join(out, 'probe.npz'), left=o['left'], right=o['right'], sgm=sgm,
                        raw=raw if raw is not None else np.zeros(0), pose=np.r_[p, yaw],
                        cloud=P[:: max(1, len(P) // 300000)].astype(np.float32))
    v = lambda d: (cv2.applyColorMap((255 - np.clip(d / 20, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
                   * ((d > 0) & np.isfinite(d))[..., None])
    lab = lambda im, t: cv2.putText(im.copy(), t, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    blank = np.zeros_like(o['left'])
    panel = np.vstack([np.hstack([lab(o['left'][..., ::-1], 'left'), lab(o['right'][..., ::-1], 'right'),
                                  lab(blank, 'depth 0-20 m (red near)')]),
                       np.hstack([lab(v(sgm), 'SGM'), lab(v(unity_d) if unity_d is not None else blank, 'Unity depth'),
                                  lab(v(np.where(np.isfinite(preds[best]), preds[best], 0)), 'cloud')])])
    cv2.imwrite(os.path.join(out, 'probe.png'), panel)
    print(json.dumps({k: calib[k] for k in ('stereo', 'unity_depth', 'cloud', 'best') if k in calib}, indent=1))
    print(json.dumps(res, indent=1))
    for w_ in warn:
        print('WARNING:', w_)
    print(('OK: ' if not warn else '') + f'→ {out}/calib.json, probe.png, probe.npz')
    cl.close()


def cmd_cloud(a):
    from nanouav.flightmare.cloud import CloudWorld
    out = _out(a)
    calib = _calib(a)
    cl = make_client(a, calib)
    origin = np.array([float(v) for v in a.origin.split(',')])
    size = np.array([float(v) for v in a.size.split(',')])
    P, info = cl.cloud_points(origin, size, resolution=a.res, path=os.path.join(out, 'point_clouds'),
                              name=f'{a.scene}_area')
    if calib.get('y_mirror'):
        P = P * np.array([1, -1, 1])
    w = CloudWorld(P, a.res, f'{a.scene}')
    dst = os.path.join(out, f'cloud_{a.scene}.npz')
    w.save(dst, dict(info, origin=origin.tolist(), size=size.tolist(), scene=a.scene, binary=a.binary))
    print(f'{len(P)} points ({info}) → {dst}')
    cl.close()


def cmd_scan(a):
    """The map from Unity's depth images on a grid of poses (for scenes without colliders) → cloud_<scene>.npz,
    the same file as `cloud`, so lines / eval / video use it unchanged."""
    from nanouav.agile.stereo import StereoRig
    from nanouav.flightmare.cloud import CloudWorld
    from nanouav.flightmare.scan import scan
    from nanouav.scene.renderer import body_rot
    out = _out(a)
    calib = _calib(a)
    cl = make_client(a, calib, stereo=False)
    origin = np.array([float(v) for v in a.origin.split(',')])
    size = np.array([float(v) for v in a.size.split(',')])
    heights = [float(v) for v in a.heights.split(',')]
    pitches = [float(v) for v in a.pitches.split(',')]
    P, hits, st = scan(cl, origin, size, a.res, a.step, heights, a.yaws, pitches, a.max_range, a.stride,
                       min_hits=a.min_hits)
    # check: a pose off the grid, its Unity depth against the depth the map predicts
    p = np.array([origin[0] + a.step / 4, origin[1] + a.step / 4, float(np.mean(heights))])
    yaw = math.radians(17.0)
    d = cl.render(p, body_rot(yaw, 0.0, 0.0))['depth_left']
    rig = StereoRig()
    pred = predicted_depth(P, rig, (p, yaw, 0.0, 0.0), rig.focal)
    err, n = compare(pred, d, max_d=15.0)
    st.update(check_rel_err=round(err, 3), check_n=n)
    w = CloudWorld(P, a.res, f'{a.scene}', inflate=a.res / 2, hits=hits)
    dst = os.path.join(out, f'cloud_{a.scene}.npz')
    w.save(dst, dict(st, source='scan', origin=origin.tolist(), size=size.tolist(), scene=a.scene, binary=a.binary,
                     step=a.step, heights=heights, yaws=a.yaws, pitches=pitches))
    print(json.dumps(st))
    print(f'{len(P)} voxels → {dst}')
    if not err < 0.1:
        print(f'WARNING: the map predicts the depth of a new view with {err:.0%} median error (n={n}): '
              'try a smaller --step or more --heights')
    cl.close()


def cmd_survey(a):
    """Top-down mosaic of the scene (colour and height of the highest surface per metre) with a 10 m grid, the
    scan box and the lines, to choose a forested, flat area → survey_<scene>.png."""
    import cv2
    from nanouav.flightmare.scan import survey
    out = _out(a)
    cl = make_client(a, _calib(a), stereo=False)
    cx, cy = (float(v) for v in a.center.split(','))
    RGB, G = survey(cl, (cx, cy), a.size, a.alt, a.step, 1.0, log=lambda m: print(m, flush=True))
    cl.close()
    np.savez_compressed(os.path.join(out, f'survey_{a.scene}.npz'), rgb=RGB, height=G, center=[cx, cy], size=a.size)
    n, k = RGB.shape[0], a.scale
    # image: x to the right, y up (row 0 = largest y)
    img = cv2.resize(np.ascontiguousarray(RGB.transpose(1, 0, 2)[::-1, :, ::-1]), (n * k, n * k),
                     interpolation=cv2.INTER_NEAREST)
    g = G.T[::-1]
    lo, hi = np.nanpercentile(G, [2, 98]) if np.isfinite(G).any() else (0, 1)
    hm = cv2.applyColorMap((np.clip((np.nan_to_num(g, nan=lo) - lo) / max(hi - lo, 1e-3), 0, 1) * 255).astype(np.uint8),
                           cv2.COLORMAP_VIRIDIS)
    hm = cv2.resize(hm, (n * k, n * k), interpolation=cv2.INTER_NEAREST)
    x0, y1 = cx - a.size / 2, cy + a.size / 2
    px = lambda x, y: (int((x - x0) * k), int((y1 - y) * k))
    for im in (img, hm):
        for v in np.arange(math.ceil(x0 / 10) * 10, x0 + a.size, 10):
            cv2.line(im, px(v, y1), px(v, y1 - a.size), (255, 255, 255) if v % 50 == 0 else (150, 150, 150), 1)
            if v % 50 == 0:
                cv2.putText(im, f'x={v:g}', (px(v, 0)[0] + 2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        for v in np.arange(math.ceil((cy - a.size / 2) / 10) * 10, y1, 10):
            cv2.line(im, px(x0, v), px(x0 + a.size, v), (255, 255, 255) if v % 50 == 0 else (150, 150, 150), 1)
            if v % 50 == 0:
                cv2.putText(im, f'y={v:g}', (2, px(0, v)[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.drawMarker(im, px(0, 0), (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
        cw = os.path.join(out, f'cloud_{a.scene}.npz')
        if os.path.exists(cw):
            m = json.loads(str(np.load(cw)['meta']))
            if 'origin' in m:
                o, sz = m['origin'], m['size']
                cv2.rectangle(im, px(o[0] - sz[0] / 2, o[1] + sz[1] / 2), px(o[0] + sz[0] / 2, o[1] - sz[1] / 2),
                              (0, 200, 255), 2)
        lf = os.path.join(out, f'lines_{a.scene}.json')
        if os.path.exists(lf):
            for l in json.load(open(lf)):
                cv2.line(im, px(*l['start'][:2]), px(*l['goal'][:2]), (0, 255, 0), 2)
    cv2.putText(hm, f'altura {lo:.1f} (oscuro) - {hi:.1f} m (claro)', (6, n * k - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1)
    fn = os.path.join(out, f'survey_{a.scene}.png')
    cv2.imwrite(fn, np.hstack([img, hm]))
    print(f'→ {fn}  (red cross: scene origin; orange: scanned box; green: lines)')


def _world(a):
    from nanouav.flightmare.cloud import CloudWorld
    return CloudWorld.load(os.path.join(_out(a), f'cloud_{a.scene}.npz'))


def cmd_lines(a):
    import cv2
    from nanouav.flightmare.cloud import find_lines
    out = _out(a)
    w = _world(a)
    L = find_lines(w, n=a.n, length=a.length, z_above=a.z_above, seed=a.seed, heading=math.radians(a.heading),
                   max_terrain=a.max_terrain, check_v=a.check_v, log=lambda m: print(m, flush=True))
    json.dump(L, open(os.path.join(out, f'lines_{a.scene}.json'), 'w'), indent=1)
    # top view: points 0.5–3 m above the ground, and the lines
    P = w.points
    g = w.ground_z(P[:, 0], P[:, 1])
    k = (P[:, 2] > g + 0.5) & (P[:, 2] < g + 3.0)
    lo, hi = P[:, :2].min(0), P[:, :2].max(0)
    s = 10.0
    img = np.full((int((hi[1] - lo[1]) * s) + 1, int((hi[0] - lo[0]) * s) + 1, 3), 255, np.uint8)
    px = lambda x, y: (int((x - lo[0]) * s), int((hi[1] - y) * s))
    for x, y in P[k][:, :2]:
        img[px(x, y)[1], px(x, y)[0]] = (60, 60, 60)
    for i, l in enumerate(L):
        cv2.line(img, px(*l['start'][:2]), px(*l['goal'][:2]), (0, 160, 0), 2)
        cv2.putText(img, str(i), px(*l['start'][:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)
    cv2.imwrite(os.path.join(out, f'lines_{a.scene}.png'), img)
    print(f'{len(L)} lines → {out}/lines_{a.scene}.json, lines_{a.scene}.png')


def _episode(a, w, line, i, pilot, v, client, calib, model, trace=False):
    from nanouav.agile import dagger
    from nanouav.agile.rollout import Config, Rollout
    from nanouav.agile.run import apply_opts
    from nanouav.flightmare.cloud import line_env
    env = line_env(w, line, i)
    cfg = Config(v_des=v, cam_hz=a.cam_hz, seed=1000 + i)
    R = Rollout(env, cfg, pilot, dagger._policy(model) if pilot == 'student' else None)
    apply_opts(R, {'tscale': a.tscale, 'vcap': a.vcap, 'select': a.select})
    obs = None
    if pilot == 'student' or trace:
        obs = FlightmareObserver(client, cfg.rig, calib)
        R.observer = obs
    if trace:
        R.trace, R.trace_depth, R.record_cam = [], True, True
    r, _, log = R.run()
    r.update(line=i, pilot=pilot, v=v, render_ms=round(1000 * float(np.mean(obs.t_render)), 1) if obs and obs.t_render else None)
    if r['crashed'] and log:
        r['crash'] = crash_info(w, np.array(log[-1][1:4], float))
    return r, R, log, env


def crash_info(w, p):
    """What the drone hit: the nearest map point, its height above the ground there, and how many scan
    frames saw it (if the map comes from `scan`)."""
    d, j = w.tree.query(p)
    q = w.points[j]
    g = float(w.ground_z(np.array([q[0]]), np.array([q[1]]))[0])
    out = {'at': np.round(p, 2).tolist(), 'obstacle': np.round(q, 2).tolist(), 'dist': round(float(d), 3),
           'above_ground': round(float(q[2] - g), 2) if np.isfinite(g) else None}
    if getattr(w, 'hits', None) is not None:
        out['seen_by_frames'] = int(w.hits[j])
    return out


def cmd_eval(a):
    out = _out(a)
    calib = _calib(a)
    w = _world(a)
    L = json.load(open(os.path.join(out, f'lines_{a.scene}.json')))[:a.n]
    pilots = a.pilots.split(',')
    speeds = [float(x) for x in a.v.split(',')]
    client = make_client(a, calib) if 'student' in pilots else None
    model = os.path.abspath(a.model) if a.model else None
    table = {}
    t0 = time.time()
    for p in pilots:
        for v in speeds:
            for i, line in enumerate(L):
                r, *_ = _episode(a, w, line, i, p, v, client, calib, model)
                table.setdefault(f'{p}_v{v:g}', []).append(r)
                print(f"{p:8s} v={v:g} line {i}: {'ok' if r['success'] else 'CRASH' if r['crashed'] else 'fail'} "
                      f"{r['time_s']:.1f} s" + (f"  hit {r['crash']}" if r.get('crash') else ''), flush=True)
    tag = (os.path.splitext(os.path.basename(model))[0] if model else 'pilots') + (f'_{a.tag}' if a.tag else '')
    fn = os.path.join(out, f'eval_{a.scene}_{tag}.json')
    json.dump(table, open(fn, 'w'), indent=1)
    print(f'{"pilot":10s} {"v":>5s} {"success":>8s} {"crash":>6s}')
    for p in pilots:
        for v in speeds:
            R = table[f'{p}_v{v:g}']
            print(f'{p:10s} {v:5.1f} {sum(r["success"] for r in R):3d}/{len(R):<4d} {sum(r["crashed"] for r in R):6d}')
    print(f'→ {fn}  ({time.time() - t0:.0f} s)')
    if client:
        client.close()


def cmd_video(a):
    from nanouav.agile.video import render_episode
    out = _out(a)
    calib = _calib(a)
    w = _world(a)
    line = json.load(open(os.path.join(out, f'lines_{a.scene}.json')))[a.line]
    client = make_client(a, calib)
    r, R, log, env = _episode(a, w, line, a.line, a.pilot, a.v, client, calib,
                              os.path.abspath(a.model) if a.model else None, trace=True)
    os.makedirs(os.path.join(out, 'video'), exist_ok=True)
    tag = os.path.splitext(os.path.basename(a.model))[0] if a.pilot == 'student' else a.pilot
    fn = os.path.join(out, 'video', f'{tag}_{a.scene}_line{a.line}_v{a.v:g}.mp4')
    res = 'EXITO' if r['success'] else ('CHOQUE' if r['crashed'] else 'SIN LLEGAR')
    render_episode(env, log, R.trace, R.cfg.rig, R.cfg.net_size, fn, fps=a.cam_hz, chase=False,
                   result=f"{res} ({r['time_s']:.1f} s)", title=f'Flightmare {a.scene} | {tag} | {a.v:g} m/s | linea {a.line}',
                   mark=r['crash']['obstacle'] if r.get('crash') else None)
    print(f'{res} · {r["time_s"]} s · → {fn}')
    if r.get('crash'):
        c = r['crash']
        print(f"hit the map point {c['obstacle']} ({c['dist']} m from the drone, {c['above_ground']} m above the "
              f"ground" + (f", seen by {c['seen_by_frames']} scan frames" if 'seen_by_frames' in c else '') + ')')
    client.close()


def cmd_fake(a):
    from nanouav.agile import envs as EV
    from nanouav.flightmare.fake import FakeUnity
    env = EV.TEST[a.world](a.seed)
    f = FakeUnity(env.world, a.pub_port, a.sub_port).start()
    print(f'fake Flightmare: {a.world} seed {a.seed} (our renderer) on ports {a.pub_port}/{a.sub_port} — Ctrl+C to stop')
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        f.stop()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('probe')
    p.add_argument('--pose', default='0,0,2,0', help='x,y,z,yaw_deg of the probe (ROS frame of the scene)')
    p.add_argument('--res', type=float, default=0.15)
    p = sub.add_parser('cloud')
    p.add_argument('--origin', default='30,0,4', help='center of the box (ROS frame)')
    p.add_argument('--size', default='80,40,10', help='extents x,y,z (m)')
    p.add_argument('--res', type=float, default=0.15)
    p = sub.add_parser('scan')
    p.add_argument('--origin', default='30,0,4', help='center of the box (ROS frame)')
    p.add_argument('--size', default='80,40,10', help='extents x,y,z (m)')
    p.add_argument('--res', type=float, default=0.15)
    p.add_argument('--step', type=float, default=4.0, help='grid spacing of the camera poses (m)')
    p.add_argument('--heights', default='1.5,3.0')
    p.add_argument('--yaws', type=int, default=6)
    p.add_argument('--pitches', default='-30,0,30', help='deg (positive = nose down)')
    p.add_argument('--max-range', type=float, default=20.0)
    p.add_argument('--stride', type=int, default=2, help='pixel stride of the back-projection')
    p.add_argument('--min-hits', type=int, default=2, help='drop voxels seen by fewer scan frames (specks)')
    p = sub.add_parser('survey')
    p.add_argument('--center', default='0,0', help='x,y of the centre of the square (scene frame)')
    p.add_argument('--size', type=float, default=300.0, help='side of the square (m)')
    p.add_argument('--alt', type=float, default=60.0, help='height of the top-down views (m)')
    p.add_argument('--step', type=float, default=50.0, help='spacing of the top-down views (m)')
    p.add_argument('--scale', type=int, default=2, help='pixels per metre of the image')
    p = sub.add_parser('lines')
    p.add_argument('--n', type=int, default=10)
    p.add_argument('--length', type=float, default=40.0)
    p.add_argument('--z-above', type=float, default=1.5)
    p.add_argument('--heading', type=float, default=0.0, help='deg (keep 0: the planner goal is x ≥ goal.x)')
    p.add_argument('--max-terrain', type=float, default=1.0, help='max ground height change along a line (m)')
    p.add_argument('--check-v', type=float, default=3.0, help='keep a line only if τ_gbl flies it at this speed '
                   '(0: no check)')
    p.add_argument('--seed', type=int, default=0)
    for name in ('eval', 'video'):
        p = sub.add_parser(name)
        p.add_argument('--model', default='')
        p.add_argument('--v', default='3,5,7' if name == 'eval' else 3.0, type=str if name == 'eval' else float)
        p.add_argument('--tscale', default='chord', choices=['chord', 'arc'])
        p.add_argument('--vcap', type=float, default=0.0)
        p.add_argument('--select', default='snap', choices=['snap', 'consistent'])
        if name == 'eval':
            p.add_argument('--pilots', default='gbl,student')
            p.add_argument('--n', type=int, default=10)
            p.add_argument('--tag', default='')
        else:
            p.add_argument('--pilot', default='student', choices=['student', 'expert', 'gbl'])
            p.add_argument('--line', type=int, default=0)
    p = sub.add_parser('fake')
    p.add_argument('--world', default='forest', choices=['forest', 'gap'])
    p.add_argument('--seed', type=int, default=1000)
    for q in sub.choices.values():
        q.add_argument('--results', dest='out', default='', help='results directory (default results/r22)')
        q.add_argument('--scene', default='forest')
        q.add_argument('--binary', default='2022', choices=['2022', '2020'])
        q.add_argument('--pub-port', type=int, default=10253)
        q.add_argument('--sub-port', type=int, default=10254)
        q.add_argument('--cam-hz', type=float, default=15.0)
        q.add_argument('--vehicle-size', type=float, default=0.01,
                       help='scale of the drone mesh (1 = full size, in view of the cameras)')
        q.add_argument('--depth-far', type=float, default=100.0, help='far clip plane of the depth layer (m)')
        q.add_argument('--unity', default='', help='path of RPG_Flightmare.x86_64: start (and stop) the player '
                       'for this command (default $NANOUAV_FLIGHTMARE; empty: start it yourself)')
    a = ap.parse_args(argv)
    if a.scene not in SCENES[a.binary]:
        raise SystemExit(f'scene {a.scene!r} is not in the {a.binary} binary: {sorted(SCENES[a.binary])}')
    return {'probe': cmd_probe, 'cloud': cmd_cloud, 'scan': cmd_scan, 'survey': cmd_survey, 'lines': cmd_lines, 'eval': cmd_eval, 'video': cmd_video,
            'fake': cmd_fake}[a.cmd](a)


if __name__ == '__main__':
    main()
