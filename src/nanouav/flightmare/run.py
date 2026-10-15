"""R22 · the R18 student flying in Flightmare's Unity scenes (the simulator of the paper).

Start the Unity standalone first (it connects to us), then:

  nanouav run r22 probe                      # one stereo pair + depth + point cloud: checks the conventions → calib.json
  nanouav run r22 cloud --origin 30,0,4 --size 80,40,10      # the privileged map of the flight area → cloud.npz
  nanouav run r22 scan  --origin 30,0,4 --size 80,40,10      # the same map from Unity depth images (no colliders)
  nanouav run r22 survey --size 300          # top-down mosaic of the scene: where to scan and fly

  # training in Flightmare (the paper's setup: exact spawned trunks in a scanned empty scene, VGA SGM)
  nanouav run r22 arena --scene warehouse    # scan the empty scene, free area, spawned-box check
  nanouav run r22 dagger --scene warehouse --init results/r18/student_spatial_in.keras
  nanouav run r22 eval --arena --scene warehouse --model results/r22/student_fm_warehouse.keras --v 3,5,7,10
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


def rig_of(a):
    """The stereo rig of the r22 commands: VGA (the paper's camera) with --vga, else our half resolution."""
    from nanouav.agile.stereo import StereoRig
    if getattr(a, 'vga', False):
        return StereoRig(width=640, height=480, num_disp=96, texture_win=11)
    return StereoRig()


def unity_fov(rig, calib=None):
    """Field of view sent to Unity (vertical: the 2022 player's convention, measured by `r22 check`) so that its
    images have our rig's focal length. A value in calib.json counts only if `r22 check` measured it; the one
    the old probe wrote was decided on a flat point cloud and came out as 90° (focal 0.75× ours)."""
    from nanouav.agile.stereo import StereoRig
    rig = rig or StereoRig()
    c = calib or {}
    if c.get('fov_from') == 'check' and c.get('fov_deg'):
        return float(c['fov_deg'])
    fov = math.degrees(2 * math.atan(rig.height / 2 / rig.focal))
    if c.get('fov_deg') and abs(float(c['fov_deg']) - fov) > 0.5:
        print(f"calib.json: fov_deg {c['fov_deg']} (old probe) ignored → {fov:.2f}° (focal of the rig)")
    return fov


def make_client(a, calib=None, stereo=True, n_objects=0, rig=None):
    from nanouav.flightmare.bridge import FlightmareClient
    os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')       # the GPU is shared with Unity
    _launch_unity(a)
    sid, off = SCENES[a.binary][a.scene]
    c = calib or {}
    fov = unity_fov(rig, c)
    cl = FlightmareClient(scene_id=sid, render_offset=off, fov_deg=fov,
                          depth_mult=c.get('depth_mult'), depth_far=a.depth_far, vehicle_size=a.vehicle_size,
                          stereo=stereo, n_objects=n_objects,
                          box_unit=c.get('box_unit', (0.8104, 0.8102, 0.8124)),
                          width=(rig.width if rig else 320), height=(rig.height if rig else 240),
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
        return gl, sgm, gt, o['left']


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
                 fov_deg_probe=73.74 if 'vertical' in best else 90.0)    # `r22 check` measures the one used
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


def _episode(a, env, i, pilot, v, client, calib, model, trace=False, rig=None):
    from nanouav.agile import dagger
    from nanouav.agile.rollout import Config, Rollout
    from nanouav.agile.run import apply_opts
    if rig is not None and getattr(a, 'arena', False):         # trained with the paper's tilt at ≥ 7 m/s
        from nanouav.flightmare.dagger import tilted
        rig = tilted(rig, v)
        if client is not None:
            client.set_tilt(rig.tilt_up_deg)
    from nanouav.agile.policy import omega_of
    om = omega_of(model) if pilot == 'student' and model else 'line'
    cfg = Config(v_des=v, cam_hz=a.cam_hz, seed=1000 + i, omega_ref=om, **({'rig': rig} if rig is not None else {}))
    R = Rollout(env, cfg, pilot, dagger._policy(model) if pilot == 'student' else None)
    apply_opts(R, {'tscale': a.tscale, 'vcap': a.vcap, 'select': a.select, 'lookahead': a.lookahead,
                   'keep': getattr(a, 'keep', False)})
    obs = None
    if pilot == 'student' or trace:
        if client.n_objects:
            client.place(env.world.objects)
        obs = FlightmareObserver(client, cfg.rig, calib)
        R.observer = obs
    if trace:
        R.trace, R.trace_depth, R.record_cam = [], True, True
    r, _, log = R.run()
    r.update(line=i, pilot=pilot, v=v, render_ms=round(1000 * float(np.mean(obs.t_render)), 1) if obs and obs.t_render else None)
    if r['crashed'] and log:
        r['crash'] = crash_info(env.world, np.array(log[-1][1:4], float))
    return r, R, log, env


def _envs(a):
    """The evaluation environments: the arena's test forests (--arena) or the lines found in the cloud."""
    out = _out(a)
    if getattr(a, 'arena', False):
        from nanouav.flightmare import arena as A
        ar = A.load(os.path.join(out, f'arena_{a.scene}.npz'))
        return [(i, A.arena_env(ar, 1000 + i, kind='test')) for i in range(a.n)]
    from nanouav.flightmare.cloud import line_env
    w = _world(a)
    L = json.load(open(os.path.join(out, f'lines_{a.scene}.json')))[:a.n]
    return [(i, line_env(w, l, i)) for i, l in enumerate(L)]


def crash_info(w, p):
    """What the drone hit: the nearest map point, its height above the ground there, and how many scan
    frames saw it (if the map comes from `scan`)."""
    if hasattr(w, 'split_ground'):                     # arena: a spawned trunk, the floor or the static scene
        from nanouav.flightmare.arena import box_distance
        P = np.asarray(p, float)[None]
        db = box_distance(P, w.C, w.H)[0] if len(w.C) else np.inf
        ds, df = float(w.sdf(P)[0]), float(p[2] - w.floor)
        k = int(np.argmin([db, ds, df]))
        out = {'at': np.round(p, 2).tolist(), 'what': ['trunk', 'scene', 'floor'][k],
               'dist': round(float(min(db, ds, df)), 3)}
        if k == 0:
            j = int(np.argmin(np.linalg.norm(np.maximum(np.abs(w.C - p) - w.H, 0), axis=1)))
            out['obstacle'] = np.round(w.C[j], 2).tolist()
        else:
            out['obstacle'] = np.round(p, 2).tolist()
        out['above_ground'] = round(float(p[2] - w.floor), 2)
        return out
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
    E = _envs(a)
    pilots = a.pilots.split(',')
    speeds = [float(x) for x in a.v.split(',')]
    rig = rig_of(a)
    client = make_client(a, calib, n_objects=a.n_objects if a.arena else 0, rig=rig) if 'student' in pilots else None
    model = os.path.abspath(a.model) if a.model else None
    table = {}
    t0 = time.time()
    for p in pilots:
        for v in speeds:
            for i, env in E:
                r, *_ = _episode(a, env, i, p, v, client, calib, model, rig=rig)
                table.setdefault(f'{p}_v{v:g}', []).append(r)
                print(f"{p:8s} v={v:g} env {i}: {'ok' if r['success'] else 'CRASH' if r['crashed'] else 'fail'} "
                      f"{r['time_s']:.1f} s" + (f"  hit {r['crash']}" if r.get('crash') else ''), flush=True)
    tag = (os.path.splitext(os.path.basename(model))[0] if model else 'pilots') + (f'_{a.tag}' if a.tag else '') + \
        (f'_{a.select}' if getattr(a, 'select', 'snap') != 'snap' else '') + \
        (f'_la{a.lookahead:g}' if getattr(a, 'lookahead', 0) else '') + ('_keep' if getattr(a, 'keep', False) else '')
    fn = os.path.join(out, f"eval_{'arena_' if a.arena else ''}{a.scene}_{tag}.json")
    json.dump(table, open(fn, 'w'), indent=1)
    # crash rate per second and per metre of flight (R22: constant per second across speeds — the slow
    # flights crash more because they last longer)
    print(f'{"pilot":10s} {"v":>5s} {"success":>8s} {"crash":>6s} {"%/s":>6s} {"%/m":>6s}')
    for p in pilots:
        for v in speeds:
            R = table[f'{p}_v{v:g}']
            k, T = sum(r['crashed'] for r in R), sum(r['time_s'] for r in R)
            D = sum(r['time_s'] * r['mean_speed'] for r in R)
            print(f'{p:10s} {v:5.1f} {sum(r["success"] for r in R):3d}/{len(R):<4d} {k:6d} '
                  f'{100 * k / max(T, 1e-9):6.1f} {100 * k / max(D, 1e-9):6.2f}')
    print(f'→ {fn}  ({time.time() - t0:.0f} s)')
    if client:
        client.close()


def cmd_compare(a):
    """Runs of `eval` over the same environments: arrivals, crash rates, and each run against the first,
    paired by environment (exact McNemar)."""
    from nanouav.agile.stats import compare
    compare(a.files)


def cmd_video(a):
    from nanouav.agile.video import render_episode
    out = _out(a)
    calib = _calib(a)
    a.n = a.line + 1
    i, env = _envs(a)[a.line]
    rig = rig_of(a)
    client = make_client(a, calib, n_objects=a.n_objects if a.arena else 0, rig=rig)
    r, R, log, env = _episode(a, env, i, a.pilot, a.v, client, calib,
                              os.path.abspath(a.model) if a.model else None, trace=True, rig=rig)
    os.makedirs(os.path.join(out, 'video'), exist_ok=True)
    tag = os.path.splitext(os.path.basename(a.model))[0] if a.pilot == 'student' else a.pilot
    fn = os.path.join(out, 'video', f"{tag}_{'arena_' if a.arena else ''}{a.scene}_{'env' if a.arena else 'line'}{a.line}_v{a.v:g}.mp4")
    res = 'EXITO' if r['success'] else ('CHOQUE' if r['crashed'] else 'SIN LLEGAR')
    render_episode(env, log, R.trace, R.cfg.rig, R.cfg.net_size, fn, fps=a.cam_hz, chase=False,
                   result=f"{res} ({r['time_s']:.1f} s)", title=f'Flightmare {a.scene} | {tag} | {a.v:g} m/s | linea {a.line}',
                   mark=r['crash']['obstacle'] if r.get('crash') else None, view='debug' if a.debug_map else 'perception')
    print(f'{res} · {r["time_s"]} s · → {fn}')
    if r.get('crash'):
        c = r['crash']
        print(f"hit {c.get('what', 'the map point')} {c['obstacle']} ({c['dist']} m from the drone, {c['above_ground']} m "
              f"above the ground" + (f", seen by {c['seen_by_frames']} scan frames" if 'seen_by_frames' in c else '') + ')')
    client.close()


ARENA_BOX = {'warehouse': ('30,0,5', '100,20,10')}    # agile_flight's training box: x −20…80, y −10…10, z 0…10


def cmd_arena(a):
    """Map the empty scene once (scan), find its floor and the free rectangle, check the spawned box's
    geometry against Unity's depth → arena_<scene>.npz (+ box_unit in calib.json)."""
    from nanouav.flightmare import arena as A
    from nanouav.flightmare.scan import scan
    from nanouav.scene.renderer import Box, body_rot
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
    # the spawned box: a 0.5 × 0.5 × 3 m trunk 6 m ahead of the camera, measured in Unity's depth
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
    def run_len(x, c):                                   # contiguous pixels at the box's depth through index c
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
    # vertically only above the centre (the floor touches the bottom of the box): top half of the trunk
    h_up = run_len(d[:H // 2 + 1, W // 2][::-1], 0)
    w_exp, h_up_exp = f * 0.5 / front_exp, min(f * 1.5 / front_exp, H / 2)
    unit = list(cl.box_unit)
    ok_box = abs(front - front_exp) < 0.15 and abs(w_px / w_exp - 1) < 0.15 and abs(h_up / h_up_exp - 1) < 0.15
    if not ok_box and abs(front - front_exp) < 0.6 and w_px > 3 and h_up > 3:   # correct the prefab size
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
        warn.append('the spawned box is not where/what it should be: see box_check (box_unit corrected if possible)')
    for w_ in warn:
        print('WARNING:', w_)
    print(('OK: ' if not warn else '') + f'→ {fn}')


def _dagger_tag(a):
    """Default data / model name: different ω or label conventions never share a data directory."""
    return f'fm_{a.scene}' + {'line': '', 'gbl': '_og', 'goal': '_goal'}[a.omega] + ('_ctg' if a.label_ref == 'ctg' else '')


def cmd_dagger(a):
    from nanouav.flightmare import dagger as FD
    out = _out(a)
    calib = _calib(a)
    rig = rig_of(a)
    arena_path = os.path.join(out, f'arena_{a.scene}.npz')
    if not os.path.exists(arena_path):
        raise SystemExit(f'{arena_path} not found: run `nanouav run r22 arena --scene {a.scene}` first')
    cl = make_client(a, calib, n_objects=a.n_objects, rig=rig)
    tk = dict(lr=a.lr, lr_end=a.lr_end, keep_best=a.best, val_by_env=a.val_by_env, head=a.head)
    m = FD.run(cl, arena_path, out, rig, n_envs=a.envs, per_round=a.per_round, workers=a.workers, epochs=a.epochs,
               batch=a.batch, v_min=a.v_min, v_max=a.v_max, cam_hz=a.cam_hz,
               init_model=os.path.abspath(a.init) if a.init else None, xi_start=a.xi_start, xi_step=a.xi_step,
               train_kw=tk, tag=a.tag or _dagger_tag(a), omega_ref=a.omega, label_ref=a.label_ref, select=a.select)
    cl.close()
    print('student →', m)


def _check_panel(rgb, S, D, rig, tilt):
    """left image | SGM | Unity depth (0–max_range m, black: invalid / beyond), one row per tilt."""
    import cv2
    col = lambda d: (cv2.applyColorMap((255 - np.clip(d / rig.max_range, 0, 1) * 255).astype(np.uint8),
                                       cv2.COLORMAP_JET) * ((d > 0) & (d < rig.max_range))[..., None])
    ims = [cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), col(S), col(D)]
    for im, t in zip(ims, (f'left, tilt {tilt:g} deg', f'SGM ({(S > 0).mean():.0%} valid)', 'Unity depth')):
        cv2.putText(im, t, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return np.hstack(ims)


def focal_ratio(D, rig, p, boxes, ks=np.arange(0.5, 1.6001, 0.005)):
    """Focal length of a level Unity depth image relative to our rig's: the scale k (about the image centre)
    that best lays the projected front faces of the boxes over pixels at their depth. NaN if no k fits."""
    import dataclasses
    from nanouav.follow.estimate import project
    rig = dataclasses.replace(rig, tilt_up_deg=0.0)
    H, W = D.shape
    c0 = np.array([(W - 1) / 2, (H - 1) / 2])
    faces = []
    for b in boxes:
        c, h = np.asarray(b.center, float), np.asarray(b.half, float)
        C = np.array([[c[0] - h[0], c[1] + sy * h[1], c[2] + sz * h[2]] for sy in (-1, 1) for sz in (-1, 1)])
        uv, X = project(rig, (p, 0.0, 0.0, 0.0), C)
        faces.append((np.asarray(uv, float), float(np.mean(X))))
    score = []
    for k in ks:
        fr = []
        for uv, X in faces:
            q = c0 + k * (uv - c0)
            (u0, v0), (u1, v1) = q.min(0), q.max(0)
            du, dv = 0.15 * (u1 - u0), 0.15 * (v1 - v0)           # keep off the edges
            u0, u1 = int(max(u0 + du, 0)), int(min(u1 - du, W - 1))
            v0, v1 = int(max(v0 + dv, 0)), int(min(v1 - dv, H - 1))
            if u1 <= u0 or v1 <= v0:
                fr.append(0.0)
                continue
            fr.append(float(np.mean(np.abs(D[v0:v1 + 1, u0:u1 + 1] - X) < 0.25)))
        score.append(np.mean(fr))
    score = np.array(score)
    if score.max() <= 0.6:
        return float('nan')
    return float(np.median(ks[score >= score.max() - 0.02]))        # the middle of the plateau


def cmd_check(a):
    """Geometry of the rendered images against ours, with spawned boxes off the axis (the arena's box check
    was straight ahead, blind to a left/right mirror): position and distance of each box in Unity's depth
    and in the SGM, level and tilted 30°."""
    import cv2
    from nanouav.agile.stereo import sgm_depth
    from nanouav.flightmare import arena as A
    from nanouav.follow.estimate import project
    from nanouav.scene.renderer import Box, body_rot
    out = _out(a)
    calib = _calib(a)
    ar = A.load(os.path.join(out, f'arena_{a.scene}.npz'))
    x0, x1, y0, y1 = ar.meta['rect']
    yc, f0 = 0.5 * (y0 + y1), ar.floor
    rig0 = rig_of(a)
    cl = make_client(a, calib, n_objects=3, rig=rig0)
    p = np.array([x0 + 2.0, yc, f0 + 1.5])
    boxes = [Box((p[0] + 6.0, yc + 2.0, f0 + 1.5), (0.25, 0.25, 1.5)),      # left, 6 m
             Box((p[0] + 9.0, yc - 3.0, f0 + 1.5), (0.3, 0.3, 1.5)),        # right, 9 m
             Box((p[0] + 7.5, yc + 0.0, f0 + 3.6), (0.3, 0.3, 0.3))]        # ahead, above the camera
    cl.place(boxes)
    import dataclasses
    rows, ok_all, panels = [], True, []
    for tilt in (0.0, 30.0):
        rig = dataclasses.replace(rig0, tilt_up_deg=tilt)
        cl.set_tilt(tilt)
        o = cl.render(p, body_rot(0.0, 0.0, 0.0))
        D = o['depth_left']
        if tilt == 0.0:
            D0 = D
        g = lambda im: cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        S = sgm_depth(g(o['left']), g(o['right']), rig)
        for name, b in zip(('left 6 m', 'right 9 m', 'above 7.5 m'), boxes):
            face = np.array([b.center[0] - b.half[0], b.center[1], b.center[2]])
            (u, v), X = project(rig, (p, 0.0, 0.0, 0.0), face[None])[0][0], project(rig, (p, 0.0, 0.0, 0.0), face[None])[1][0]
            H, W = D.shape
            row = {'tilt': tilt, 'box': name, 'u_exp': round(float(u), 1), 'v_exp': round(float(v), 1),
                   'depth_exp': round(float(X), 2)}
            m = np.abs(D - X) < 0.25
            vv, uu = np.nonzero(m)
            near = (np.abs(uu - u) < 40) & (np.abs(vv - v) < 40)
            mir = (np.abs(uu - (W - 1 - u)) < 40) & (np.abs(vv - v) < 40)
            row['found'] = int(near.sum())
            row['mirrored'] = int(mir.sum()) if abs(u - (W - 1 - u)) > 80 else None
            if near.sum():
                row['u_meas'], row['v_meas'] = round(float(uu[near].mean()), 1), round(float(vv[near].mean()), 1)
                k = (S[vv[near], uu[near]] > 0)
                row['sgm_rel_err'] = round(float(np.median(np.abs(S[vv[near], uu[near]][k] - X) / X)), 3) if k.sum() > 10 else None
                row['sgm_valid'] = round(float(k.mean()), 3)                 # on the box
            tol = 8.0 * W / 640                                    # px
            placed = near.sum() > 0 and abs(row['u_meas'] - u) <= tol and abs(row['v_meas'] - v) <= tol
            good = near.sum() > 20 and placed and (row['mirrored'] is None or row['mirrored'] < near.sum() / 4)
            row['ok'] = bool(good)
            ok_all &= good
            rows.append(row)
        fl = (np.arange(D.shape[0])[:, None] > D.shape[0] * 0.8) & (S > 0) & (D > 0)
        near_px = (D > 0.3) & (D < rig.max_range)
        near_px[:, :rig.num_disp] = False                  # SGM has no match on the left band
        rows.append({'tilt': tilt, 'floor_sgm_rel_err': round(float(np.median(np.abs(S[fl] - D[fl]) / D[fl])), 3)
                     if fl.sum() > 50 else None, 'sgm_valid': round(float((S > 0).mean()), 3),
                     'sgm_valid_within_range': round(float((S[near_px] > 0).mean()), 3) if near_px.any() else None,
                     'within_range_frac': round(float(near_px.mean()), 3)})
        panels.append(_check_panel(o['left'], S, D, rig, tilt))
    fov_sent = cl.fov
    cl.set_tilt(0.0)
    cl.close()
    for r in rows:
        print(r)
    ratio = focal_ratio(D0, rig0, p, boxes)
    if not np.isfinite(ratio):
        ok_all = False
    rows.append({'focal_ratio': round(ratio, 3), 'fov_sent': round(fov_sent, 2)})
    json.dump(rows, open(os.path.join(out, 'check_geometry.json'), 'w'), indent=1)
    cv2.imwrite(os.path.join(out, 'check_geometry.png'), np.vstack(panels))
    print(f'focal of the images / ours: {ratio:.3f} (fov sent {fov_sent:.2f}°) → {out}/check_geometry.png')
    if np.isfinite(ratio) and abs(ratio - 1) > 0.03:
        fov_new = math.degrees(2 * math.atan(math.tan(math.radians(fov_sent) / 2) * ratio))
        calib.update(fov_deg=round(fov_new, 2), fov_from='check', focal_ratio=round(ratio, 3))
        json.dump(calib, open(os.path.join(out, 'calib.json'), 'w'), indent=1)
        print(f'calib.json: fov_deg {fov_sent:.2f} → {fov_new:.2f} (measured). Run `r22 check` again.')
    print('OK: the rendered geometry matches ours (no mirror, tilt right, same focal length)' if ok_all else
          'WARNING: a box is not where our camera model puts it — send check_geometry.json')


def cmd_diagnose(a):
    """Why the student crashes in the arena: each crash classified like `r18 diagnose` (from the true
    clearance of its hypotheses, privileged — diagnosis only), plus the lateral drift from the start→goal
    line, which side the chosen hypotheses go to, and the SGM quality of the frames it decided on."""
    from nanouav.agile import diagnose as DG
    out = _out(a)
    calib = _calib(a)
    E = _envs(a)
    rig = rig_of(a)
    client = make_client(a, calib, n_objects=a.n_objects, rig=rig)
    model = os.path.abspath(a.model)
    tag = os.path.splitext(os.path.basename(model))[0] + (f'_{a.select}' if a.select != 'snap' else '') + \
        (f'_la{a.lookahead:g}' if a.lookahead else '') + ('_keep' if a.keep else '')
    d_out = os.path.join(out, f'diag_arena_{tag}')
    os.makedirs(d_out, exist_ok=True)
    res = []
    for v in [float(x) for x in a.v.split(',')]:
        for i, env in E:
            from nanouav.agile import dagger
            from nanouav.agile.rollout import Config, Rollout
            from nanouav.agile.run import apply_opts
            from nanouav.flightmare.dagger import tilted
            rv = tilted(rig, v)
            client.set_tilt(rv.tilt_up_deg)
            client.place(env.world.objects)
            from nanouav.agile.policy import omega_of
            R = Rollout(env, Config(v_des=v, cam_hz=a.cam_hz, seed=1000 + i, rig=rv, omega_ref=omega_of(model)),
                        'student', dagger._policy(model))
            apply_opts(R, {'tscale': a.tscale, 'vcap': a.vcap, 'select': a.select, 'lookahead': a.lookahead,
                   'keep': getattr(a, 'keep', False)})
            R.observer = FlightmareObserver(client, rv, calib)
            R.trace = []
            r, _, log = R.run()
            L = np.array([row[1:4] for row in log], float)
            y_off = L[:, 1] - env.start[1]
            tr = R.trace
            x = {'v': v, 'env': i, 'success': r['success'], 'crashed': r['crashed'], 'time_s': r['time_s'],
                 'drift_mean_m': round(float(y_off.mean()), 2), 'drift_end_m': round(float(y_off[-1]), 2),
                 'chosen_left': round(float(np.mean([d['lat'][d['k']] > 0.3 for d in tr])), 2) if tr else None,
                 'chosen_right': round(float(np.mean([d['lat'][d['k']] < -0.3 for d in tr])), 2) if tr else None,
                 'sgm_valid': np.round(np.mean([d['sgm_valid'] for d in tr], 0), 3).tolist() if tr else None,
                 'sgm_err': round(float(np.nanmedian([d['sgm_err'] if d['sgm_err'] is not None else np.nan for d in tr])), 3) if tr else None}
            pe = DG.perception_errors(tr)
            if pe:
                x['perception'] = pe
            ks = DG.keep_stats(tr)
            if ks:
                x['keep'] = ks
            if r['crashed']:
                dg = DG.classify(env, tr, log, v)
                x['class'] = dg['class']
                x['in_view'] = dg['in_view_0.5s']
                x['keep_1s'] = dg['keep_1s']
                DG.plot(env, tr, log, dg, os.path.join(d_out, f'v{v:g}_env{i}.png'))
            res.append(x)
            print(f"v={v:g} env {i}: {'ok' if x['success'] else 'CRASH ' + x.get('class', '') if x['crashed'] else 'fail'}"
                  f"  drift {x['drift_mean_m']:+.2f} m (end {x['drift_end_m']:+.2f})  chosen left/right "
                  f"{x['chosen_left']}/{x['chosen_right']}  SGM valid {x['sgm_valid']} err {x['sgm_err']}"
                  + (f"  SGM vs truth: missed {pe['missed']} false alarm {pe['false_alarm']} chose colliding "
                     f"{pe['chose_colliding']}" if pe else '')
                  + (f"  last 1 s: none free {x['keep_1s']['none_free']}/{x['keep_1s']['n']}, old plan still free "
                     f"{x['keep_1s']['old_free']} (truly {x['keep_1s']['old_truly_free']}), kept {x['keep_1s']['kept']}"
                     if x.get('keep_1s') else ''), flush=True)
    client.close()
    json.dump(res, open(os.path.join(d_out, 'diag.json'), 'w'), indent=1)
    K = ('todas_chocan', 'cambio_de_plan', 'eligio_mal', 'plan_justo', 'seguimiento')
    print(f'{"v":>4s} {"exito":>6s} ' + ' '.join(f'{k:>14s}' for k in K) + f' {"deriva m":>9s} {"izq/der":>9s} {"SGM ok":>7s} {"SGM err":>7s}')
    for v in sorted({x['v'] for x in res}):
        X = [x for x in res if x['v'] == v]
        c = {k: sum(x.get('class') == k for x in X) for k in K}
        print(f'{v:4.1f} {sum(x["success"] for x in X):3d}/{len(X):<2d} ' + ' '.join(f'{c[k]:14d}' for k in K) +
              f' {np.mean([x["drift_mean_m"] for x in X]):+9.2f} '
              f'{np.mean([x["chosen_left"] for x in X]):4.2f}/{np.mean([x["chosen_right"] for x in X]):4.2f} '
              f'{np.mean([x["sgm_valid"][0] for x in X]):7.2f} {np.nanmedian([x["sgm_err"] for x in X]):7.3f}')
    P = [x['perception'] for x in res if x.get('perception')]
    if P:
        m = lambda k: np.nanmean([p[k] if p[k] is not None else np.nan for p in P])
        print(f"--select sgm, its own SGM against the truth: of the plans that truly collide it saw {m('missed'):.0%} "
              f"as free; of the truly free ones it saw {m('false_alarm'):.0%} as blocked; {m('chose_colliding'):.1%} "
              f"of its decisions flew a colliding plan it saw as free")
    KS = [x['keep_1s'] for x in res if x.get('keep_1s')]
    if KS:
        tot = {k: sum(q[k] for q in KS) for k in ('none_free', 'old_free', 'old_truly_free', 'kept', 'n')}
        print(f"last second before each crash ({len(KS)} crashes, {tot['n']} decisions): none of the new plans free in "
              f"{tot['none_free']}; of those, what was left of the plan being flown still free in {tot['old_free']} "
              f"(truly free {tot['old_truly_free']}); kept (--keep) {tot['kept']}")
    print(f'→ {d_out}: diag.json and one PNG per crash (deriva: + = izquierda)')


def cmd_fake(a):
    from nanouav.agile import envs as EV
    from nanouav.flightmare.fake import FakeUnity, arena_world
    world = arena_world(a.seed) if a.world == 'arena' else EV.TEST[a.world](a.seed).world
    f = FakeUnity(world, a.pub_port, a.sub_port).start()
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
    p = sub.add_parser('check')
    for name in ('eval', 'video', 'diagnose'):
        p = sub.add_parser(name)
        p.add_argument('--model', default='')
        p.add_argument('--v', default='3,5,7' if name != 'video' else 3.0, type=str if name != 'video' else float)
        p.add_argument('--tscale', default='chord', choices=['chord', 'arc'])
        p.add_argument('--vcap', type=float, default=0.0)
        p.add_argument('--select', default='sgm', choices=['snap', 'consistent', 'oracle', 'sgm', 'oracle_cost', 'sgm_cost'],
                       help="sgm (default since 0.22.0): the least snap among the plans its own SGM of the last second "
                            "sees free (not privileged; 62/90 in the warehouse against 38/90 for snap, the paper's "
                            "predicted-cost rule); oracle: the same with the true clearance (privileged, diagnosis "
                            "only); *_cost: the predicted-cost rule among the free ones")
        p.add_argument('--lookahead', type=float, default=0.0,
                       help='sgm/oracle: also the continuation of each plan at its final velocity for this many seconds '
                            'must be free (against plans that end pointing into a trunk)')
        p.add_argument('--keep', action='store_true',
                       help='sgm/oracle: when none of the new hypotheses is free, go on flying the current plan if what '
                            'is left of it (≥ 0.3 s) still is')
        p.add_argument('--arena', action='store_true', help='the arena test forests (r22 arena) instead of the lines')
        if name == 'eval':
            p.add_argument('--pilots', default='gbl,student')
            p.add_argument('--n', type=int, default=10)
            p.add_argument('--tag', default='')
        elif name == 'diagnose':
            p.add_argument('--n', type=int, default=10)
        else:
            p.add_argument('--pilot', default='student', choices=['student', 'expert', 'gbl'])
            p.add_argument('--line', type=int, default=0, help='line (or arena environment) index')
            p.add_argument('--debug-map', action='store_true',
                           help='old panels with the true map (default: only what the drone perceives)')
    p = sub.add_parser('compare', help='eval JSONs over the same environments, paired against the first')
    p.add_argument('files', nargs='+', help='eval_*.json (the first is the reference)')
    p = sub.add_parser('fake')
    p.add_argument('--world', default='forest', choices=['forest', 'gap', 'arena'])
    p.add_argument('--seed', type=int, default=1000)
    p = sub.add_parser('arena')
    p.add_argument('--origin', default='', help='centre of the scanned box (default: per scene; warehouse 30,0,5)')
    p.add_argument('--size', default='', help='extents x,y,z (default: per scene; warehouse 100,20,10)')
    p.add_argument('--res', type=float, default=0.15)
    p.add_argument('--step', type=float, default=4.0)
    p.add_argument('--heights', default='1.5,3.0,5.0')
    p.add_argument('--yaws', type=int, default=6)
    p.add_argument('--pitches', default='-30,0,30')
    p.add_argument('--max-range', type=float, default=25.0)
    p.add_argument('--stride', type=int, default=2)
    p = sub.add_parser('dagger')
    p.add_argument('--init', default='', help='student to start from (e.g. results/r18/student_spatial_in.keras)')
    p.add_argument('--envs', type=int, default=300)
    p.add_argument('--per-round', type=int, default=30)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2), help='expert label processes')
    p.add_argument('--v-min', type=float, default=3.0)
    p.add_argument('--v-max', type=float, default=10.0)
    p.add_argument('--xi-start', type=float, default=1.0, help='ξ of the first round (m; 0: pure imitation)')
    p.add_argument('--xi-step', type=float, default=0.5)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--lr-end', type=float, default=1e-5)
    p.add_argument('--best', action='store_true')
    p.add_argument('--val-by-env', action='store_true')
    p.add_argument('--tag', default='', help='data / model name (default fm_<scene>)')
    p.add_argument('--head', default='spatial', choices=['gap', 'spatial'], help='new model only (no --init)')
    p.add_argument('--omega', default='line', choices=['line', 'gbl', 'goal'],
                   help="what the student's direction input points to: line (R18; R23 keeps it), τ_gbl (training only), goal")
    p.add_argument('--label-ref', default='gbl', choices=['gbl', 'ctg'],
                   help="reference of the expert's labels: τ_gbl from the start (R18) or replanned from the current "
                        "position, keeping to the line (R23)")
    p.add_argument('--select', default='sgm', choices=['snap', 'sgm'],
                   help="how the student picks among its hypotheses while it flies the DAgger rounds (the states it "
                        "visits): sgm (default since 0.22.0, what is evaluated) or snap (≤ 0.21)")
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
        q.add_argument('--vga', action=argparse.BooleanOptionalAction, default=None,
                       help='VGA stereo pair (the paper\'s camera; default on for arena, dagger and --arena runs)')
        q.add_argument('--n-objects', type=int, default=160, help='spawnable trunks (arena runs)')
        q.add_argument('--unity', default='', help='path of RPG_Flightmare.x86_64: start (and stop) the player '
                       'for this command (default $NANOUAV_FLIGHTMARE; empty: start it yourself)')
    a = ap.parse_args(argv)
    if a.cmd == 'diagnose':
        a.arena = True                                   # the arena only (exact obstacles)
    if a.vga is None:
        a.vga = a.cmd in ('arena', 'dagger', 'check') or bool(getattr(a, 'arena', False))
    if not hasattr(a, 'arena'):
        a.arena = False
    if a.scene not in SCENES[a.binary]:
        raise SystemExit(f'scene {a.scene!r} is not in the {a.binary} binary: {sorted(SCENES[a.binary])}')
    return {'probe': cmd_probe, 'cloud': cmd_cloud, 'scan': cmd_scan, 'survey': cmd_survey, 'arena': cmd_arena,
            'check': cmd_check, 'diagnose': cmd_diagnose,
            'dagger': cmd_dagger, 'lines': cmd_lines, 'eval': cmd_eval, 'video': cmd_video,
            'fake': cmd_fake, 'compare': cmd_compare}[a.cmd](a)


if __name__ == '__main__':
    main()
