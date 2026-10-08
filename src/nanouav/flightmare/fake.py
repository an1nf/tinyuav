"""A stand-in for the Flightmare Unity player, for tests and for trying the pipeline without Unity.

It speaks the player's side of the protocol (connects to the bridge's sockets, answers the settings with
{"ready": true}, one image reply per pose, writes the point cloud on request) and renders with our own
renderer from an `nanouav.agile.envs` world. It copies what the real player does (flightmare_unity,
dev/agile_flight): left-handed y-up poses, vertical field of view, images bottom-up, RGB, depth along the
optical axis divided by the depth layer's far clip plane, and the point cloud request taken as absolute
(the render offset is not added) in ROS order, a voxel being occupied when a surface is within one
resolution of its centre (Physics.OverlapSphere), written as absolute ROS-order (x, y, z up) centres.
The world is the scene frame: the absolute frame is the scene frame + render_offset.

  nanouav run r22 fake --world forest --seed 1000        # then, in another terminal, any r22 command
"""
import json
import math
import threading

import numpy as np

from nanouav.control.quad_sim import euler_from_rot
from nanouav.agile.stereo import cull
from nanouav.scene.renderer import render_view

S_OURS = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])   # Flightmare camera axes → ours (x fwd, y left)
M_U = np.array([[1.0, 0, 0, 0], [0, 0, 1.0, 0], [0, 1.0, 0, 0], [0, 0, 0, 1.0]])


class _Rig:
    """What stereo.cull needs: the field of view and the render range."""

    def __init__(self, hfov, render_range=40.0):
        self.hfov, self.render_range = hfov, render_range


def _quat_to_rot(w, x, y, z):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class FakeUnity:
    def __init__(self, world, pub_port=10253, sub_port=10254, host='localhost'):
        import zmq
        self.world, self.zmq = world, zmq
        self.ctx = zmq.Context()
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.connect(f'tcp://{host}:{pub_port}')
        self.sub.setsockopt(zmq.SUBSCRIBE, b'')
        self.pub = self.ctx.socket(zmq.PUB)
        self.pub.connect(f'tcp://{host}:{sub_port}')
        self.settings, self.running, self.frames = None, False, 0

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
        while self.running:
            if not self.sub.poll(100):
                continue
            topic, body = self.sub.recv_multipart()[:2]
            m = json.loads(body.decode())
            if topic == b'PointCloud':
                self.point_cloud(m)
            elif 'frame_id' not in m:                       # settings
                self.settings = m
                self.pub.send_multipart([json.dumps({'ready': True}).encode()])
            elif self.settings is not None:
                self.pub.send_multipart(self.render(m))

    def render(self, m):
        v = m['vehicles'][0]
        cams = self.settings['vehicles'][0]['cameras']
        ux, uy, uz = v['position']
        p = np.array([ux, uz, uy])
        qx, qy, qz, qw = v['rotation']
        Rb = _quat_to_rot(qw, -qx, -qz, -qy)
        parts = [json.dumps({'frame_id': m['frame_id'], 'pub_vehicles': [{'collision': False, 'lidar_ranges': []}]}).encode()]
        for c in cams:
            T = M_U @ np.array(c['T_BC'], float).reshape(4, 4) @ M_U.T          # Unity → ROS (camera → body)
            Rc = Rb @ T[:3, :3] @ S_OURS
            pos = p + Rb @ T[:3, 3]
            roll, pitch, yaw = euler_from_rot(Rc)
            W, H = int(c['width']), int(c['height'])
            hfov = 2 * math.atan(math.tan(math.radians(c['fov']) / 2) * W / H)
            w = cull(self.world, pos, yaw, _Rig(hfov))
            gray, depth = render_view(w, pos, yaw, pitch, roll, cam_pitch=0.0, cam_offset=(0, 0, 0), hfov=hfov,
                                      post=False, shadows=False, size=(H, W), d_max=100.0)
            img = np.repeat(gray[..., None], int(c['channels']), -1)[::-1]
            parts.append(np.ascontiguousarray(img).tobytes())
            if c['enabledLayers'][0]:
                far = float(c['farClipPlane'][1])
                parts.append(np.ascontiguousarray((depth / far).astype(np.float32)[::-1]).tobytes())
        self.frames += 1
        return parts

    def offset(self):
        ux, uy, uz = (self.settings or {}).get('render_offset', [0.0, 0.0, 0.0])
        return np.array([ux, uz, uy], float)                                    # Unity → ROS order

    def point_cloud(self, m):
        """SavePointCloud.cs: voxel centres of the (absolute, ROS-order) box with a surface within one
        resolution, binary .ply of absolute ROS-order centres."""
        from nanouav.agile.geometry import Field
        off = self.offset()
        o, s, r = np.array(m['origin'], float) - off, np.array(m['range'], float), float(m['resolution'])
        ax = [np.arange(o[k] - s[k] / 2 + r / 2, o[k] + s[k] / 2 + 1e-5, r) for k in range(3)]
        F = Field(self.world).local(o, float(np.linalg.norm(s)) / 2 + 1.0)
        X, Y = np.meshgrid(ax[0], ax[1], indexing='ij')
        pts = []
        for z in ax[2]:
            P = np.c_[X.ravel(), Y.ravel(), np.full(X.size, z)]
            pts.append(P[np.abs(F(P)) < r])
        P = np.concatenate(pts) if pts else np.zeros((0, 3))
        U = (P + off).astype('<f4')
        fn = m['path'] + m['file_name'] + '.ply'
        with open(fn + '.tmp', 'wb') as f:
            f.write((f'ply\nformat binary_little_endian 1.0\nelement vertex {len(U)}\nproperty float x\n'
                     'property float y\nproperty float z\nend_header\n').encode())
            f.write(U.tobytes())
        import os
        os.replace(fn + '.tmp', fn)
