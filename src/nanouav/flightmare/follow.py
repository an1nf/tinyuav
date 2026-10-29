"""R20 in Flightmare (0.27.10): follow a balloon through Unity's scenes (`nanouav run r22 follow`).

The public Flightmare player cannot spawn a balloon (it only moves and scales the prefabs it ships; the
trunks of R22 are its wooden crate). So the balloons are painted into Unity's images: for both cameras of
the stereo pair, every pixel whose ray hits a balloon closer than Unity's own depth gets the balloon's
shaded color, and that depth. The trunks hide the balloon where they are in front of it, the color
detector sees it, and SGM matches it like any other surface. Everything else is R20: the balloon mission
(detector, tracker, lock, filter, `--fast-target`), the pilots and the success rule; the world is an
arena of R22 (the empty scene scanned once, a Poisson forest of crates) instead of our forest.
"""
import math

import numpy as np

from nanouav.agile.expert import GlobalRef
from nanouav.follow.blob import COLORS
from nanouav.scene.renderer import Box, rot_y

SUN = np.array([-0.3, 0.2, 0.93]) / np.linalg.norm([-0.3, 0.2, 0.93])


def paint_spheres(rgb, depth, O, Rc, focal, spheres, sun=SUN, gain=1.0):
    """Paint spheres into one camera image (in place) where they are in front of `depth` (m along the optical
    axis, 0 or inf = nothing). O, Rc: camera origin and rotation camera → world (axes x forward, y left, z up;
    pixel u = −y/x·f + W/2 − 0.5). spheres: (center, radius, RGB reflectance 0–1). → number of pixels painted."""
    H, W = rgb.shape[:2]
    n = 0
    far = np.where((depth > 0) & np.isfinite(depth), depth, np.inf) if depth is not None else np.full((H, W), np.inf)
    for c, r, col in spheres:
        q = (np.asarray(c, float) - O) @ Rc
        if q[0] < r + 0.05:
            continue
        uc, vc = -q[1] / q[0] * focal + W / 2 - 0.5, -q[2] / q[0] * focal + H / 2 - 0.5
        rp = r / max(q[0] - r, 0.05) * focal + 2
        u0, u1 = int(max(math.floor(uc - rp), 0)), int(min(math.ceil(uc + rp), W - 1))
        v0, v1 = int(max(math.floor(vc - rp), 0)), int(min(math.ceil(vc + rp), H - 1))
        if u0 > u1 or v0 > v1:
            continue
        U, V = np.meshgrid(np.arange(u0, u1 + 1), np.arange(v0, v1 + 1))
        d_cam = np.stack([np.ones(U.shape), -(U + 0.5 - W / 2) / focal, -(V + 0.5 - H / 2) / focal], -1)
        D = d_cam @ Rc.T                                    # world directions (camera x = 1: t is the depth)
        oc = O - np.asarray(c, float)
        a = (D * D).sum(-1)
        b = 2 * (D @ oc)
        disc = b * b - 4 * a * (oc @ oc - r * r)
        hit = disc > 0
        t = np.where(hit, (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a), np.inf)
        m = hit & (t > 0.05) & (t < far[v0:v1 + 1, u0:u1 + 1])
        if not m.any():
            continue
        Dm, tm = D[m], t[m]
        N = (O + Dm * tm[:, None] - np.asarray(c, float)) / r
        lam = np.clip(N @ sun, 0, 1)
        R_ = 2 * lam[:, None] * N - sun                     # reflected sun, toward the camera: a highlight
        spec = np.clip((R_ * -Dm / np.linalg.norm(Dm, axis=-1, keepdims=True)).sum(-1), 0, 1) ** 20
        shade = gain * (0.35 + 0.65 * lam)
        px = np.clip(255 * np.asarray(col, float)[None] * shade[:, None] + 70 * spec[:, None], 0, 255)
        sub = rgb[v0:v1 + 1, u0:u1 + 1]
        sub[m] = px[:, :sub.shape[-1]] if sub.shape[-1] <= 3 else np.c_[px, sub[m][:, 3:]]
        if depth is not None:
            depth[v0:v1 + 1, u0:u1 + 1][m] = t[m]
        far[v0:v1 + 1, u0:u1 + 1][m] = t[m]                  # a nearer balloon hides a farther one
        n += int(m.sum())
    return n


class BalloonPainter:
    """FlightmareObserver.overlay: the balloons (walkers of a balloon env) in both images of the pair, at the
    cameras the client put on the vehicle (offset, baseline, tilt)."""

    def __init__(self, client, walkers, focal, extra=()):
        self.c, self.walkers, self.f, self.extra = client, walkers, float(focal), list(extra)

    def spheres(self):
        return [(w.body.center, w.radius, COLORS[w.color]) for w in self.walkers] + self.extra

    def cameras(self, pose):
        from nanouav.scene.renderer import body_rot
        p, yaw, pitch, roll = pose
        Rb = body_rot(yaw, pitch, roll)
        Rc = Rb @ rot_y(-math.radians(self.c.tilt))
        ox, oy, oz = self.c.cam_offset
        left = np.asarray(p, float) + Rb @ np.array([ox, oy, oz])
        right = np.asarray(p, float) + Rb @ np.array([ox, oy - self.c.baseline, oz])
        return (left, Rc), (right, Rc)

    def __call__(self, o, pose):
        S = self.spheres()
        (Ol, Rc), (Or, _) = self.cameras(pose)
        o = dict(o)
        for key, dk, O in (('left', 'depth_left', Ol), ('right', 'depth_right', Or)):
            if o.get(key) is None:
                continue
            img = np.array(o[key], copy=True)
            dep = None if o.get(dk) is None else np.array(o[dk], np.float32, copy=True)
            paint_spheres(img, dep, O, Rc, self.f, S)
            o[key], o[dk] = img, dep
        return o


class FloorFrame:
    """An arena seen with its floor at z = 0: R20's balloon paths assume flat ground at 0."""
    split_ground = True

    def __init__(self, arena):
        self.ar, self.dz = arena, float(arena.floor)
        self.objects, self.ground = arena.objects, 'floor'
        self.points = np.asarray(arena.points, float) - [0.0, 0.0, self.dz]

    def distance(self, P, ground=True):
        return self.ar.distance(np.asarray(P, float).reshape(-1, 3) + [0.0, 0.0, self.dz], ground)

    def clearance(self, p):
        return float(self.distance(np.asarray(p, float)[None])[0])

    def local(self, center, radius):
        return FloorFrame(self.ar.local(np.asarray(center, float) + [0.0, 0.0, self.dz], radius))


def balloon_arena_env(arena, seed, density=1 / 16, speed=None, n_distractors=None, color='red', p_same=0.25,
                      z_lo=1.2, z_hi=2.6, radius=0.3, length=40.0, n_max=160, r=(0.15, 0.35)):
    """R20's balloon environment in an arena: a Poisson forest of crates (widths 2r, 6 m tall) in the free
    rectangle, the balloon floating `length` m at 1–4 m/s (or `speed`) through a random midway point, 0–2
    other balloons crossing above. Same layout as `follow.balloon.balloon_env`, relative to the start."""
    from nanouav.follow.balloon import Balloon, _seg_dist, fly_path
    from nanouav.follow.people import PeopleEnv
    x0, x1, y0, y1 = arena.meta['rect']
    rng = np.random.default_rng([seed, 0xBA, 0xF1])
    ox, oy = x0 + 3.0, 0.5 * (y0 + y1)                      # the drone's start (R20: the origin)
    L = min(length, x1 - x0 - 12.0)
    lat = min(6.0, 0.5 * (y1 - y0) - 2.5)                   # lateral range of the waypoints
    for sub in range(30):
        area = (x1 - x0) * (y1 - y0)
        n = rng.poisson(density * area)
        xy = np.c_[rng.uniform(x0 + 0.5, x1 - 0.5, n), rng.uniform(y0 + 0.5, y1 - 0.5, n)]
        xy = xy[np.hypot(xy[:, 0] - ox, xy[:, 1] - oy) > 4.0]
        if len(xy) > n_max:
            xy = xy[np.argsort(np.abs(xy[:, 1] - oy))[:n_max]]
        boxes = []
        for x, y in xy:
            hw = float(rng.uniform(*r))
            boxes.append(Box((float(x), float(y), arena.floor + 3.0), (hw, hw, 3.0), 0.0))
        boxes = [b for b in boxes if _seg_dist(b, (ox - 3.0, oy), (ox + 6.0, oy)) > 1.3]
        world = arena.with_boxes(boxes, f'{arena.name}_balloon{seed}')
        ff = FloorFrame(world)
        v = float(speed if speed is not None else rng.uniform(1.0, 4.0))
        wp = [(ox + 5.0, oy), (ox + 5 + L / 2, oy + float(rng.uniform(-lat, lat))),
              (ox + 5 + L, oy + float(rng.uniform(-lat, lat)))]
        try:
            path = fly_path(ff, wp, v, rng, z_lo, z_hi, rq=0.5)
        except RuntimeError:
            continue
        lift = lambda g: GlobalRef(g.pts + [0.0, 0.0, arena.floor], g.v)
        target = Balloon(lift(path), v, t0=0.5, radius=radius, color=color, name='target')
        balloons = [target]
        others = [c for c in COLORS if c != color]
        nd = int(n_distractors if n_distractors is not None else rng.integers(0, 3))
        for k in range(nd):
            xc = float(rng.uniform(ox + 12, ox + 5 + L - 5))
            side = 1 if rng.random() < 0.5 else -1
            half = 0.5 * (y1 - y0) - 1.0
            a, b = (xc + rng.uniform(-3, 3), oy + side * half), (xc + rng.uniform(-3, 3), oy - side * half)
            vd = float(rng.uniform(0.8, 2.0))
            t_meet = (xc - ox - 5) / v + 0.5
            col = color if rng.random() < p_same else str(rng.choice(others))
            try:
                pd = fly_path(ff, [a, b], vd, rng, z_hi + 0.7, z_hi + 1.4)
            except RuntimeError:
                continue
            balloons.append(Balloon(lift(pd), vd, t0=max(0.0, t_meet - half / vd + rng.uniform(-1.5, 1.5)),
                                    radius=radius, color=col, name=f'balloon{k + 1}'))
        for bl in balloons:
            bl.move(0.0)
        start = np.array([ox, oy, arena.floor + 1.5])
        meta = {'density': density, 'speed': v, 'n_distractors': len(balloons) - 1,
                'same_color': sum(bl.color == color for bl in balloons[1:]), 'color': color, 'sub': sub,
                'n_trunks': len(boxes)}
        return PeopleEnv(world.name, world, start, start + [L + 5, 0, 0], meta, False, balloons)
    raise RuntimeError(f'no balloon path in the arena for seed {seed}')


def episode(client, rig, calib, env, seed, pilot, model=None, standoff=4.0, cam_hz=10.0, opts=None, edge=True,
            record=False):
    """One R20 flight on Unity's frames → (result, rollout, log, recs or None)."""
    from nanouav.agile import dagger
    from nanouav.agile.rollout import Config, Rollout
    from nanouav.flightmare.run import FlightmareObserver
    from nanouav.follow.balloon_mission import BalloonMission
    fl = env.world.floor
    M = BalloonMission(env, standoff=standoff, seed=seed, z_min=fl + 1.0, z_max=fl + 3.0)
    for k, v in (opts or {}).items():
        setattr(M, k, v)
    if (opts or {}).get('q_acc') is not None:
        M.flt = M._make_filter()
    recs = None
    if record:
        from nanouav.follow.balloon_video import record as rec_
        recs = rec_(M)
    if client.n_objects:
        client.place(env.world.objects)
    if client.tilt:
        client.set_tilt(0.0)                                 # R25/R26 were trained with a level camera
    pol = dagger._policy(model) if pilot == 'student' else (model if pilot == 'diffphys' else None)
    if pilot == 'diffphys' and not model:
        raise SystemExit('the diffphys pilot needs --model (results/r26/follow_fast.keras)')
    R = Rollout(env, Config(v_des=2.0, cam_hz=cam_hz, seed=seed, rig=rig), pilot, pol, mission=M)
    R.EDGE = edge
    R.observer = FlightmareObserver(client, rig, calib, overlay=BalloonPainter(client, env.walkers, rig.focal))
    r, _, log = R.run()
    r.update(seed=seed, n_trunks=env.meta.get('n_trunks'),
             render_ms=round(1000 * float(np.mean(R.observer.t_render)), 1) if R.observer.t_render else None)
    return r, R, log, recs


def chase_view(client, painter, rec, back=3.5, up=1.2, pitch_deg=14.0):
    """Bottom-right panel of the video: Unity's view from 3.5 m behind the drone, with the balloons and the
    drone (an orange ball) painted in → BGR 480 × 360."""
    import cv2
    from nanouav.scene.renderer import body_rot
    p, yaw = rec['p'], rec['yaw']
    cam = np.array([p[0] - back * math.cos(yaw), p[1] - back * math.sin(yaw), p[2] + up])
    pose = (cam, yaw, math.radians(pitch_deg), 0.0)
    o = client.render(cam, body_rot(*pose[1:]))
    for w in painter.walkers:
        w.move(rec['t'])
    drone = (tuple(p), 0.22, (1.0, 0.55, 0.0))
    painter.extra = [drone]
    (O, Rc), _ = painter.cameras(pose)
    img = np.array(o['left'][..., :3], copy=True)
    dep = None if o.get('depth_left') is None else np.array(o['depth_left'], np.float32, copy=True)
    paint_spheres(img, dep, O, Rc, painter.f, painter.spheres())
    painter.extra = []
    out = cv2.resize(np.ascontiguousarray(img[..., ::-1]), (480, 360))
    cv2.putText(out, 'Flightmare: camara de seguimiento (el dron es la bola naranja)', (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return out
