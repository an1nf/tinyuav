"""Camera geometry of the stereo rig, bounding box + depth → 3D, and the world-frame target filter."""
import math

import numpy as np

from nanouav.scene.renderer import body_rot, rot_y


def camera_pose(rig, pose):
    """→ (camera origin O, rotation camera→world Rc) for a body pose (p, yaw, pitch, roll)."""
    p, yaw, pitch, roll = pose
    Rb = body_rot(yaw, pitch, roll)
    Rc = Rb @ rot_y(-math.radians(rig.tilt_up_deg))
    return np.asarray(p, float) + Rb @ np.asarray(rig.offset, float), Rc


def project(rig, pose, P):
    """World points (N,3) → pixel (N,2) and forward distance X (N,) of the left camera (renderer model:
    camera axes x forward, y left, z up; u = −y/x·f, v = −z/x·f)."""
    O, Rc = camera_pose(rig, pose)
    Q = (np.asarray(P, float) - O) @ Rc
    X = Q[:, 0]
    with np.errstate(divide='ignore', invalid='ignore'):
        u = -Q[:, 1] / X * rig.focal + rig.width / 2 - 0.5
        v = -Q[:, 2] / X * rig.focal + rig.height / 2 - 0.5
    return np.stack([u, v], 1), X


def backproject(rig, pose, px, X):
    """Pixel (u, v) at forward distance X → world point."""
    O, Rc = camera_pose(rig, pose)
    u, v = px
    d = np.array([1.0, -(u + 0.5 - rig.width / 2) / rig.focal, -(v + 0.5 - rig.height / 2) / rig.focal])
    return O + Rc @ (d * X)


def box_distance(depth, box, rig, person_h=1.69):
    """Forward distance of the person in a box from the stereo depth: lower quartile of the valid depths in
    the central part of the box (the person is in front of the background); if too few valid pixels,
    from the box height and a nominal person height. → (X, source)."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    xa, xb = int(max(x1 + 0.3 * w, 0)), int(min(x2 - 0.3 * w, rig.width - 1)) + 1
    ya, yb = int(max(y1 + 0.15 * h, 0)), int(min(y2 - 0.25 * h, rig.height - 1)) + 1
    patch = depth[ya:yb, xa:xb] if (xb > xa and yb > ya) else np.zeros(0)
    vals = patch[patch > 0]
    if vals.size >= max(6, 0.08 * patch.size):
        return float(np.percentile(vals, 25)), 'stereo'
    if h > 4 and y1 > 1 and y2 < rig.height - 2:          # full person visible: pinhole height
        return float(rig.focal * person_h / h), 'height'
    return float('nan'), 'none'


def box_to_world(rig, pose, depth, box):
    """3D position (feet on the ground below the box center) of the person in a box, or None."""
    X, src = box_distance(depth, box, rig)
    if not np.isfinite(X):
        return None, src
    x1, y1, x2, y2 = box
    c = backproject(rig, pose, ((x1 + x2) / 2, (y1 + y2) / 2), X)
    c[2] = 0.0
    return c, src


class TargetFilter:
    """Constant-velocity Kalman filter of the target position on the ground plane (x, y, vx, vy)."""

    def __init__(self, q_acc=1.5):
        self.x, self.P, self.q, self.t = None, None, q_acc, None

    def predict(self, t):
        if self.x is None:
            return None
        dt = max(t - self.t, 0.0)
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        G = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + G @ G.T * self.q ** 2
        self.t = t
        return self.x

    def update(self, t, z, sigma):
        if self.x is None:
            self.x = np.array([z[0], z[1], 0.0, 0.0])
            self.P = np.diag([sigma ** 2, sigma ** 2, 1.0, 1.0])
            self.t = t
            return self.x
        self.predict(t)
        H = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
        S = H @ self.P @ H.T + np.eye(2) * sigma ** 2
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (np.asarray(z[:2]) - H @ self.x)
        self.P = (np.eye(4) - K @ H) @ self.P
        return self.x

    def gate(self, z, t, n_sigma=3.0, floor=1.0):
        """Is z (x, y) compatible with the prediction at t?"""
        if self.x is None:
            return True
        dt = max(t - self.t, 0.0)
        pred = self.x[:2] + self.x[2:] * dt
        s = math.sqrt(max(self.P[0, 0], self.P[1, 1])) + 1.5 * dt
        return float(np.hypot(*(np.asarray(z[:2]) - pred))) < max(n_sigma * s, floor)

    @property
    def pos(self):
        return None if self.x is None else np.array([self.x[0], self.x[1], 0.0])

    @property
    def vel(self):
        return None if self.x is None else np.array([self.x[2], self.x[3], 0.0])
