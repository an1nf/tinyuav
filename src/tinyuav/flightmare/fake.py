"""Stand-in for the Flightmare player: the player's side of the protocol, rendered with our renderer.
Left-handed y-up poses, vertical field of view, bottom-up RGB images, depth / far clip plane.

  tinyuav flightmare fake --world forest --seed 1000
"""
import json
import math
import threading
import time

import numpy as np

from tinyuav.sim.quad import euler_from_rot
from tinyuav.sim.stereo import cull
from tinyuav.sim.render import Box, Texture, World, render_view
from tinyuav.flightmare.bridge import BOX_UNIT

S_OURS = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])   # Flightmare camera axes to ours
M_U = np.array([[1.0, 0, 0, 0], [0, 0, 1.0, 0], [0, 1.0, 0, 0], [0, 0, 0, 1.0]])


class _Rig:
    """Field of view and render range for stereo.cull."""

    def __init__(self, hfov, render_range=40.0):
        self.hfov, self.render_range = hfov, render_range


def _quat_to_rot(w, x, y, z):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class FakeUnity:
    def __init__(self, world, pub_port=10253, sub_port=10254, host='localhost', idle_reset_s=None):
        """idle_reset_s: back to the menu after this long without a message, like the real player."""
        import zmq
        self.world, self.zmq = world, zmq
        self.ctx = zmq.Context()
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.connect(f'tcp://{host}:{pub_port}')
        self.sub.setsockopt(zmq.SUBSCRIBE, b'')
        self.pub = self.ctx.socket(zmq.PUB)
        self.pub.connect(f'tcp://{host}:{sub_port}')
        self.settings, self.running, self.frames = None, False, 0
        self.dyn = {}                                    # dynamic objects by ID
        self.idle_reset_s, self.resets = idle_reset_s, 0

    def start(self):
        self.running = True
        self.th = threading.Thread(target=self.loop, daemon=True)
        self.th.start()
        return self

    def stop(self):
        self.running = False
        self.th.join(timeout=5)
        self.sub.close(0)
        self.pub.close(0)
        self.ctx.term()

    def loop(self):
        last = time.time()
        while self.running:
            if not self.sub.poll(100):
                if self.idle_reset_s and self.settings is not None and time.time() - last > self.idle_reset_s:
                    self.settings, self.dyn = None, {}          # back to the menu
                    self.resets += 1
                continue
            topic, body = self.sub.recv_multipart()[:2]
            last = time.time()
            m = json.loads(body.decode())
            if self.settings is None or 'frame_id' not in m:   # settings
                self.settings = m
                self.dyn = {o['ID']: dict(o) for o in m.get('dynamic_objects', [])}
                self.pub.send_multipart([json.dumps({'ready': True}).encode()])
            else:
                self.pub.send_multipart(self.render(m))
            last = time.time()                               # idle time counts from the reply

    def boxes(self):
        """Dynamic objects as renderer boxes (scene frame)."""
        out = []
        for ID, o in self.dyn.items():
            sx, sy, sz = o['size']
            if min(sx, sy, sz) < 0.05:
                continue
            ux, uy, uz = o['position']
            qx, qy, qz, qw = o['rotation']
            w, x, y, z = qw, -qx, -qz, -qy                         # Unity to ROS quaternion
            yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            half = (0.5 * sx * BOX_UNIT[0], 0.5 * sz * BOX_UNIT[1], 0.5 * sy * BOX_UNIT[2])
            out.append(Box((ux, uz, uy), half, yaw, Texture('noise', albedo=0.55, albedo2=0.22, scale=0.08,
                                                            seed=hash(ID) & 0xFFFF)))
        return out

    def render(self, m):
        for o in m.get('dynamic_objects', []):
            if o['ID'] in self.dyn:
                self.dyn[o['ID']].update(o)
        bx = self.boxes()
        W0 = self.world
        world = W0 if not bx else World(list(W0.objects) + bx, W0.ground, sun=W0.sun, ambient=W0.ambient,
                                        diffuse=W0.diffuse, sky=W0.sky)
        v = m['vehicles'][0]
        cams = v.get('cameras') or self.settings['vehicles'][0]['cameras']
        ux, uy, uz = v['position']
        p = np.array([ux, uz, uy])
        qx, qy, qz, qw = v['rotation']
        Rb = _quat_to_rot(qw, -qx, -qz, -qy)
        parts = [json.dumps({'frame_id': m['frame_id'], 'pub_vehicles': [{'collision': False, 'lidar_ranges': []}]}).encode()]
        for c in cams:
            T = M_U @ np.array(c['T_BC'], float).reshape(4, 4) @ M_U.T          # Unity to ROS
            Rc = Rb @ T[:3, :3] @ S_OURS
            pos = p + Rb @ T[:3, 3]
            roll, pitch, yaw = euler_from_rot(Rc)
            W, H = int(c['width']), int(c['height'])
            hfov = 2 * math.atan(math.tan(math.radians(c['fov']) / 2) * W / H)
            w = cull(world, pos, yaw, _Rig(hfov))
            gray, depth = render_view(w, pos, yaw, pitch, roll, cam_pitch=0.0, cam_offset=(0, 0, 0), hfov=hfov,
                                      shadows=False, size=(H, W), d_max=100.0)
            img = np.repeat(gray[..., None], int(c['channels']), -1)[::-1]
            parts.append(np.ascontiguousarray(img).tobytes())
            if c['enabledLayers'][0]:
                far = float(c['farClipPlane'][1])
                parts.append(np.ascontiguousarray((depth / far).astype(np.float32)[::-1]).tobytes())
        self.frames += 1
        return parts

    def offset(self):
        ux, uy, uz = (self.settings or {}).get('render_offset', [0.0, 0.0, 0.0])
        return np.array([ux, uz, uy], float)                                    # Unity to ROS order

def arena_world(seed=0, x=(-20.0, 80.0), y=(-10.0, 10.0), height=10.0):
    """Warehouse stand-in: textured floor, four walls and a roof."""
    from tinyuav.sim import envs as EV
    rng = np.random.default_rng(seed)
    t = lambda: Texture('noise', albedo=float(rng.uniform(0.4, 0.7)), albedo2=0.2, scale=0.3, seed=int(rng.integers(1 << 30)))
    cx, cy, L, Wd = (x[0] + x[1]) / 2, (y[0] + y[1]) / 2, x[1] - x[0], y[1] - y[0]
    walls = [Box((cx, y[0] - 0.25, height / 2), (L / 2 + 0.5, 0.25, height / 2), 0.0, t()),
             Box((cx, y[1] + 0.25, height / 2), (L / 2 + 0.5, 0.25, height / 2), 0.0, t()),
             Box((x[0] - 0.25, cy, height / 2), (0.25, Wd / 2 + 0.5, height / 2), 0.0, t()),
             Box((x[1] + 0.25, cy, height / 2), (0.25, Wd / 2 + 0.5, height / 2), 0.0, t()),
             Box((cx, cy, height + 0.25), (L / 2 + 0.5, Wd / 2 + 0.5, 0.25), 0.0, t())]
    return World(walls, EV._ground(rng), sun=(-0.4, 0.2, 0.9), ambient=0.45, diffuse=0.6)
