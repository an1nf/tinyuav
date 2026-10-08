"""Python client of the Flightmare Unity standalone (the protocol of flightlib's UnityBridge).

Like flightlib, this side BINDS two ZMQ sockets and the Unity player connects to them:
  PUB tcp://*:10253 → multipart ["Pose", json]        (settings while connecting, then one pose per frame)
  SUB tcp://*:10254 ← multipart [json, image layers…]  ({"ready": true} once, then one reply per frame)
  PUB ["PointCloud", json] → Unity writes <path><file_name>.ply

Messages carry both the 2020 (`objects`) and the 2022 (`dynamic_objects`, `static_objects`,
`render_offset`, `object_csv`) fields, so it works with the GitHub release binary and with the one of
agile_flight. Unity is left-handed with y up; the conversions are flightlib's (common/math.cpp).

A camera is attached to the vehicle with T_BC (4 × 4, camera → body, ROS frames). In Flightmare's camera
frame the optical axis is +y and up is +z (a Unity camera looks along its +z, which is ROS y): a forward
camera has R_BC = Rz(−90°), as in agile_flight's config (r_BC = [0, 0, −90]).

What the Unity player does (flightmare_unity, branch dev/agile_flight, CameraController.cs, SavePointCloud.cs,
UberReplacement.shader), and what this client relies on:
  · vehicle and camera poses get `render_offset` added; the cameras are placed from the pose and T_BC with
    scale 1, so the vehicle `size` only scales the drone mesh. The mesh is in view of a camera near the body
    centre: size 0.01, as in Loquercio et al.'s own bridge (agile_autonomy, flightmare_bridge.cpp), hides it;
  · the depth layer is depth along the optical axis divided by that layer's far clip plane (Linear01Depth,
    "DepthRaw" mode): metres = value × farClipPlane[1];
  · the point cloud request is NOT offset: origin and range are absolute and in ROS order (x, y, z up), each
    voxel centre is occupied when Physics.OverlapSphere(centre, resolution) touches a collider, and the .ply
    holds the occupied centres in the same absolute ROS order. `cloud_points` adds the offset to the request
    and removes it from the points.
"""
import json
import math
import os
import time

import numpy as np

R_FWD = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])    # Rz(−90°): camera → body, forward camera


def rot_to_quat(R):
    """Rotation matrix → quaternion (w, x, y, z)."""
    R = np.asarray(R, float)
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
    q = np.zeros(4)
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + i] = 0.25 * s
    q[1 + j] = (R[j, i] + R[i, j]) / s
    q[1 + k] = (R[k, i] + R[i, k]) / s
    return q


def pos_ros2unity(p):
    return [float(p[0]), float(p[2]), float(p[1])]


def quat_ros2unity(q):
    """(w, x, y, z) ROS → Unity [x, y, z, w] (flightlib: Quaternion(w, −x, −z, −y))."""
    w, x, y, z = q
    return [float(-x), float(-z), float(-y), float(w)]


def tf_ros2unity(T):
    M = np.zeros((4, 4))
    M[0, 0] = M[1, 2] = M[2, 1] = M[3, 3] = 1.0
    return [float(v) for v in (M @ np.asarray(T, float) @ M.T).ravel()]


def camera_T(t, R=R_FWD, pitch_up_deg=0.0):
    """T_BC (camera → body) of a forward camera at body position t, tilted up by pitch_up_deg."""
    a = math.radians(pitch_up_deg)
    Ry = np.array([[math.cos(a), 0, -math.sin(a)], [0, 1, 0], [math.sin(a), 0, math.cos(a)]])   # nose-up tilt
    T = np.eye(4)
    T[:3, :3] = Ry @ R
    T[:3, 3] = t
    return T


class FlightmareClient:
    def __init__(self, scene_id=2, render_offset=(0.0, 0.0, 0.0), width=320, height=240, fov_deg=73.74,
                 baseline=0.10, cam_offset=(0.10, 0.0, 0.0), depth=True, channels=3, pub_port=10253,
                 sub_port=10254, address='tcp://*', object_csv=None, depth_mult=None, depth_far=100.0,
                 vehicle_size=0.01, timeout_s=60.0):
        import zmq
        self.zmq = zmq
        self.ctx = zmq.Context.instance()
        self.pub = self.ctx.socket(zmq.PUB)
        self.pub.setsockopt(zmq.SNDHWM, 6)
        self.pub.bind(f'{address}:{pub_port}')
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.setsockopt(zmq.RCVHWM, 6)
        self.sub.bind(f'{address}:{sub_port}')
        self.sub.setsockopt(zmq.SUBSCRIBE, b'')
        self.W, self.H, self.C, self.fov = int(width), int(height), int(channels), float(fov_deg)
        self.depth, self.depth_far, self.timeout = bool(depth), float(depth_far), timeout_s
        self.depth_mult = float(depth_mult) if depth_mult else self.depth_far     # Unity sends depth / far
        self.offset = np.asarray(render_offset, float)
        ox, oy, oz = cam_offset
        self.T_left = camera_T([ox, oy, oz])
        self.T_right = camera_T([ox, oy - baseline, oz])
        if object_csv is None:                                 # the 2022 binary reads this file: give it an empty one
            object_csv = os.path.abspath(os.path.join(os.path.expanduser('~'), '.nanouav_flightmare_empty.csv'))
            open(object_csv, 'a').close()
        cams = [self._camera('quadrotor0_0', self.T_left, 0), self._camera('quadrotor0_1', self.T_right, 1)]
        self.vehicle = {'ID': 'quadrotor0', 'position': [0.0, 0.0, 0.0], 'rotation': [0.0, 0.0, 0.0, 1.0],
                        'size': [float(vehicle_size)] * 3, 'cameras': cams, 'lidars': [], 'hasCollisionCheck': False}
        self.settings = {'scene_id': int(scene_id), 'vehicles': [self.vehicle], 'objects': [], 'dynamic_objects': [],
                         'static_objects': [], 'render_offset': pos_ros2unity(render_offset), 'object_csv': object_csv}
        self.frame = 0
        self.ready = False

    def _camera(self, ID, T, idx):
        return {'ID': ID, 'channels': self.C, 'width': self.W, 'height': self.H, 'fov': self.fov,
                'nearClipPlane': [0.01] * 4, 'farClipPlane': [1000.0, self.depth_far, 1000.0, 1000.0], 'T_BC': tf_ros2unity(T),
                'isDepth': False, 'enabledLayers': [self.depth, False, False], 'depthScale': 0.2, 'outputIndex': idx}

    def _send(self, topic, obj):
        self.pub.send_multipart([topic.encode(), json.dumps(obj).encode()])

    def connect(self):
        """Send the settings until the Unity player answers {"ready": true} (start it first)."""
        t0 = time.time()
        while time.time() - t0 < self.timeout:
            self._send('Pose', self.settings)
            if self.sub.poll(500):
                parts = self.sub.recv_multipart()
                try:
                    m = json.loads(parts[0].decode())
                except ValueError:
                    continue
                if isinstance(m, dict) and len(m) <= 1 and m.get('ready'):
                    self.ready = True
                    return True
        raise TimeoutError('Flightmare did not answer: is the Unity standalone (RPG_Flightmare.x86_64) running?')

    def render(self, p, R):
        """Body position p (3,) and rotation R (3 × 3, body → world, ROS) → dict(left, right uint8 (H, W, C),
        depth_left, depth_right float32 m or None, collision)."""
        self.frame += 1
        v = dict(self.vehicle, position=pos_ros2unity(p), rotation=quat_ros2unity(rot_to_quat(R)))
        self._send('Pose', {'frame_id': self.frame, 'vehicles': [v], 'objects': [], 'dynamic_objects': [],
                            'static_objects': []})
        for _ in range(100):
            if not self.sub.poll(int(self.timeout * 1000)):
                raise TimeoutError('no frame from Flightmare')
            parts = self.sub.recv_multipart()
            m = json.loads(parts[0].decode())
            if int(m.get('frame_id', -1)) == self.frame:
                break
        imgs, i = [], 1
        for cam in self.vehicle['cameras']:
            rgb = np.frombuffer(parts[i], np.uint8)[: self.W * self.H * self.C].reshape(self.H, self.W, self.C)[::-1].copy()
            i += 1
            dep = None
            if self.depth:
                dep = np.frombuffer(parts[i], np.float32)[: self.W * self.H].reshape(self.H, self.W)[::-1] * self.depth_mult
                i += 1
            imgs.append((rgb, dep))
        coll = bool(m.get('pub_vehicles', [{}])[0].get('collision', False)) if m.get('pub_vehicles') else False
        return {'left': imgs[0][0], 'right': imgs[1][0], 'depth_left': imgs[0][1], 'depth_right': imgs[1][1],
                'collision': coll}

    def point_cloud(self, origin, size, resolution=0.15, path=None, name='scene', timeout_s=600.0):
        """Ask Unity to write the occupied voxels of a box (origin = centre, size = [x, y, z] extents, both
        ABSOLUTE: Unity does not add the render offset here) to a .ply file. → path of the file.
        `cloud_points` is the scene-frame version."""
        path = os.path.abspath(path or 'results/r22/point_clouds') + '/'
        os.makedirs(path, exist_ok=True)
        fn = os.path.join(path, name + '.ply')
        if os.path.exists(fn):
            os.remove(fn)
        self._send('PointCloud', {'range': [float(x) for x in size], 'origin': [float(x) for x in origin],
                                  'resolution': float(resolution), 'path': path, 'file_name': name})
        t0 = time.time()
        last = -1
        while time.time() - t0 < timeout_s:            # wait until the file exists and stops growing
            if os.path.exists(fn):
                sz = os.path.getsize(fn)
                if sz == last and sz > 0:
                    return fn
                last = sz
            time.sleep(1.0)
        raise TimeoutError(f'Flightmare did not write {fn} in {timeout_s:.0f} s')

    def cloud_points(self, origin, size, resolution=0.15, path=None, name='scene', timeout_s=600.0):
        """Occupied voxels of a box given in the scene frame (the frame of the poses) → (points (N, 3) in the
        scene frame, info). The box is requested at origin + render_offset and the points are moved back."""
        from nanouav.flightmare.cloud import read_ply, to_ros
        origin, size = np.asarray(origin, float), np.asarray(size, float)
        absolute = origin + self.offset
        fn = self.point_cloud(absolute, size, resolution, path, name, timeout_s)
        P = read_ply(fn)
        if len(P) == 0:
            return np.zeros((0, 3)), {'n': 0, 'file': fn, 'requested_abs': absolute.round(3).tolist()}
        P, info = to_ros(P, absolute, size)
        info.update(n=len(P), file=fn, requested_abs=absolute.round(3).tolist(), offset=self.offset.tolist())
        return P - self.offset, info

    def close(self):
        self.pub.close(0)
        self.sub.close(0)
