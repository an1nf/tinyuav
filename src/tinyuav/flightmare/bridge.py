"""Client of the Flightmare Unity player (flightlib's UnityBridge protocol).

This side binds two ZMQ sockets and the player connects to them:
  PUB tcp://*:10253  ["Pose", json]           settings while connecting, then one pose per frame
  SUB tcp://*:10254  [json, image layers...]  {"ready": true} once, then one reply per frame

Messages carry the 2020 and 2022 fields. Unity is left-handed with y up; conversions follow flightlib.
Cameras are attached with T_BC (camera to body, ROS); the optical axis is +y of the Flightmare camera frame.
The depth layer is z-depth divided by the far clip plane. A vehicle size of 0.01 hides the drone mesh.
"""
import json
import math
import os
import threading
import time

import numpy as np

R_FWD = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])    # camera to body, forward camera
# rpg_box02 size per unit of scale, ROS order (x, y, z up)
BOX_PREFAB, BOX_UNIT = 'rpg_box02', (0.8104, 0.8102, 0.8124)
PARK = (0.0, 0.0, -300.0)                      # parked objects


def rot_to_quat(R):
    """Rotation matrix to quaternion (w, x, y, z)."""
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
    """(w, x, y, z) ROS to Unity [x, y, z, w]."""
    w, x, y, z = q
    return [float(-x), float(-z), float(-y), float(w)]


def tf_ros2unity(T):
    M = np.zeros((4, 4))
    M[0, 0] = M[1, 2] = M[2, 1] = M[3, 3] = 1.0
    return [float(v) for v in (M @ np.asarray(T, float) @ M.T).ravel()]


def camera_T(t, R=R_FWD, pitch_up_deg=0.0):
    """T_BC of a forward camera at body position t, tilted up by pitch_up_deg."""
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
                 vehicle_size=0.01, stereo=True, n_objects=0, box_unit=BOX_UNIT, prefab=BOX_PREFAB, timeout_s=60.0):
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
        self.depth_mult = float(depth_mult) if depth_mult else self.depth_far
        self.offset = np.asarray(render_offset, float)
        ox, oy, oz = cam_offset
        self.cam_offset, self.baseline, self.tilt = (ox, oy, oz), float(baseline), 0.0
        self.T_left = camera_T([ox, oy, oz])
        self.T_right = camera_T([ox, oy - baseline, oz])
        if object_csv is None:                                 # the 2022 binary reads this file
            object_csv = os.path.abspath(os.path.join(os.path.expanduser('~'), '.tinyuav_flightmare_empty.csv'))
            open(object_csv, 'a').close()
        cams = [self._camera('quadrotor0_0', self.T_left, 0)]
        if stereo:                                             # False: left camera only
            cams.append(self._camera('quadrotor0_1', self.T_right, 1))
        self.vehicle = {'ID': 'quadrotor0', 'position': [0.0, 0.0, 0.0], 'rotation': [0.0, 0.0, 0.0, 1.0],
                        'size': [float(vehicle_size)] * 3, 'cameras': cams, 'lidars': [], 'hasCollisionCheck': False}
        # dynamic objects: the spawned crates
        self.box_unit, self.n_objects = np.asarray(box_unit, float), int(n_objects)
        self.objects = [self._object(f'nano_obj{i}', prefab) for i in range(self.n_objects)]
        self.settings = {'scene_id': int(scene_id), 'vehicles': [self.vehicle], 'objects': [],
                         'dynamic_objects': self.objects,
                         'static_objects': [], 'render_offset': pos_ros2unity(render_offset), 'object_csv': object_csv}
        self.frame = 0
        self.ready = False
        self._lock = threading.RLock()                 # sockets shared by render, place and keepalive
        self._last_t, self._last_pose, self.keepalives, self.reloads = time.time(), (np.zeros(3), np.eye(3)), 0, 0

    @staticmethod
    def _object(ID, prefab):
        return {'ID': ID, 'prefabID': prefab, 'position': pos_ros2unity(PARK), 'rotation': [0.0, 0.0, 0.0, 1.0],
                'size': [0.01, 0.01, 0.01]}

    def place(self, boxes):
        """Place the dynamic objects at these boxes (scene frame); the rest are parked out of sight."""
        with self._lock:
            self._place(boxes)

    def _place(self, boxes):
        if len(boxes) > self.n_objects:
            raise ValueError(f'{len(boxes)} boxes but the client was made with n_objects={self.n_objects}')
        for o, b in zip(self.objects, list(boxes) + [None] * (self.n_objects - len(boxes))):
            if b is None:
                o.update(position=pos_ros2unity(PARK), rotation=[0.0, 0.0, 0.0, 1.0], size=[0.01, 0.01, 0.01])
                continue
            c, h, yaw = np.asarray(b.center, float), np.asarray(b.half, float), float(getattr(b, 'yaw', 0.0))
            sc = 2 * h / self.box_unit                         # scale per ROS axis
            Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
            o.update(position=pos_ros2unity(c), rotation=quat_ros2unity(rot_to_quat(Rz)),
                     size=[float(sc[0]), float(sc[2]), float(sc[1])])                 # Unity (x, y up, z)

    def set_tilt(self, deg):
        """Tilt both cameras up by `deg`; effective from the next frame."""
        with self._lock:
            ox, oy, oz = self.cam_offset
            self.tilt = float(deg)
            self.T_left = camera_T([ox, oy, oz], pitch_up_deg=self.tilt)
            self.T_right = camera_T([ox, oy - self.baseline, oz], pitch_up_deg=self.tilt)
            for c, T in zip(self.vehicle['cameras'], (self.T_left, self.T_right)):
                c['T_BC'] = tf_ros2unity(T)

    def _camera(self, ID, T, idx):
        return {'ID': ID, 'channels': self.C, 'width': self.W, 'height': self.H, 'fov': self.fov,
                'nearClipPlane': [0.01] * 4, 'farClipPlane': [1000.0, self.depth_far, 1000.0, 1000.0], 'T_BC': tf_ros2unity(T),
                'isDepth': False, 'enabledLayers': [self.depth, False, False], 'depthScale': 0.2, 'outputIndex': idx}

    def _send(self, topic, obj):
        self.pub.send_multipart([topic.encode(), json.dumps(obj).encode()])

    def connect(self, keepalive=2.0):
        """Send the settings until the player answers ready. The player resets after 10 s without messages, so
        with `keepalive` a frame is rendered at least that often.
        """
        ok = self._connect()
        if keepalive and not getattr(self, '_ka', None):
            self._ka_period, self._ka_stop = float(keepalive), threading.Event()
            self._ka = threading.Thread(target=self._keepalive, daemon=True)
            self._ka.start()
        return ok

    def _keepalive(self):
        while not self._ka_stop.wait(0.5):
            if self.ready and time.time() - self._last_t > self._ka_period and self._last_pose is not None:
                try:
                    with self._lock:
                        if time.time() - self._last_t > self._ka_period:
                            self._render(*self._last_pose)
                            self._last_t = time.time()
                            self.keepalives += 1
                except Exception:                      # retried; render() reports errors
                    pass

    def _connect(self):
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
        raise TimeoutError('Flightmare did not answer. Is the Unity player (RPG_Flightmare.x86_64) running? After a '
                           'previous command it returns to its menu and may not reconnect: restart it, or pass '
                           '--unity <path> (or set TINYUAV_FLIGHTMARE) so each command starts its own')

    def render(self, p, R):
        """Images at body position p and rotation R: dict(left, right, depth_left, depth_right, collision)."""
        with self._lock:
            out = self._render(np.asarray(p, float), np.asarray(R, float))
            self._last_t, self._last_pose = time.time(), (np.asarray(p, float).copy(), np.asarray(R, float).copy())
            return out

    def _pose_message(self, v):
        # every pose carries the settings: a reset player reloads from it and the pose is resent
        return {'frame_id': self.frame, 'vehicles': [v], 'objects': [], 'dynamic_objects': self.objects,
                'static_objects': [], 'scene_id': self.settings['scene_id'],
                'render_offset': self.settings['render_offset'], 'object_csv': self.settings['object_csv']}

    def _render(self, p, R):
        self.frame += 1
        v = dict(self.vehicle, position=pos_ros2unity(p), rotation=quat_ros2unity(rot_to_quat(R)))
        msg = self._pose_message(v)
        self._send('Pose', msg)
        for _ in range(100):
            if not self.sub.poll(int(self.timeout * 1000)):
                raise TimeoutError('no frame from Flightmare')
            parts = self.sub.recv_multipart()
            m = json.loads(parts[0].decode())
            if 'frame_id' not in m and m.get('ready'):     # the player reset and reloaded
                self.reloads += 1
                self._send('Pose', msg)
                continue
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
        right = imgs[1] if len(imgs) > 1 else (None, None)
        return {'left': imgs[0][0], 'right': right[0], 'depth_left': imgs[0][1], 'depth_right': right[1],
                'collision': coll}

    def close(self):
        if getattr(self, '_ka', None):
            self._ka_stop.set()
            self._ka.join(timeout=5)
            self._ka = None
        self.pub.close(0)
        self.sub.close(0)
