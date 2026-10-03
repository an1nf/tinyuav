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


def rotation_warp(rig, pose_prev, pose_cur):
    """Pixel motion caused by the rotation of the camera between two frames (translation ignored: exact for
    far points, and the tracker's box velocity absorbs the rest). → f(u, v) → (u', v')."""
    _, R0 = camera_pose(rig, pose_prev)
    _, R1 = camera_pose(rig, pose_cur)
    R = R1.T @ R0

    def f(u, v):
        d = np.array([1.0, -(u + 0.5 - rig.width / 2) / rig.focal, -(v + 0.5 - rig.height / 2) / rig.focal])
        q = R @ d
        if q[0] < 1e-3:
            return u, v
        return -q[1] / q[0] * rig.focal + rig.width / 2 - 0.5, -q[2] / q[0] * rig.focal + rig.height / 2 - 0.5
    return f


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


def blob_to_world(rig, pose, depth, box, radius):
    """3D center of a sphere of known radius (the balloon) seen in a box, or None. Distance from the apparent
    diameter (the box height: a trunk hiding part of the balloon narrows the box but does not shorten it)
    when the box does not touch the image border; otherwise from the stereo depth (median of
    the valid pixels in the central part of the box + radius). A balloon has no texture, so SGM mostly
    finds the background through it: the size is the better cue whenever it is available. → (point, source)."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    if max(w, h) > 2 and x1 > 1 and y1 > 1 and x2 < rig.width - 2 and y2 < rig.height - 2:
        # trunks are vertical: they narrow the box, so the height is the diameter; a box clearly wider
        # than tall is two balloons side by side that the detector could not split: its height is the one
        diam = h if w > 1.25 * h else max(w, h)
        X, src = float(rig.focal * 2 * radius / diam), 'size'
    else:
        r = 0.3 * max(w, h)
        xa, xb = int(max(cx - r, 0)), int(min(cx + r, rig.width - 1)) + 1
        ya, yb = int(max(cy - r, 0)), int(min(cy + r, rig.height - 1)) + 1
        patch = depth[ya:yb, xa:xb] if (xb > xa and yb > ya) else np.zeros(0)
        vals = patch[patch > 0]
        if vals.size < max(6, 0.15 * patch.size):
            return None, 'none'
        X, src = float(np.percentile(vals, 25)) + radius, 'stereo'
    return backproject(rig, pose, (cx, cy), X), src


class TargetFilter:
    """Constant-velocity Kalman filter of the target position: on the ground plane (dims = 2: x, y, vx, vy;
    a person) or in 3D (dims = 3; a balloon)."""

    def __init__(self, q_acc=1.5, dims=2):
        self.x, self.P, self.q, self.t, self.n = None, None, q_acc, None, dims

    def predict(self, t):
        if self.x is None:
            return None
        n = self.n
        dt = max(t - self.t, 0.0)
        F = np.eye(2 * n)
        F[:n, n:] = np.eye(n) * dt
        G = np.vstack([np.eye(n) * dt * dt / 2, np.eye(n) * dt])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + G @ G.T * self.q ** 2
        self.t = t
        return self.x

    def update(self, t, z, sigma):
        n = self.n
        if self.x is None:
            self.x = np.r_[np.asarray(z[:n], float), np.zeros(n)]
            self.P = np.diag([sigma ** 2] * n + [1.0] * n)
            self.t = t
            return self.x
        self.predict(t)
        H = np.hstack([np.eye(n), np.zeros((n, n))])
        S = H @ self.P @ H.T + np.eye(n) * sigma ** 2
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (np.asarray(z[:n], float) - H @ self.x)
        self.P = (np.eye(2 * n) - K @ H) @ self.P
        return self.x

    def gate(self, z, t, n_sigma=3.0, floor=1.0):
        """Is z compatible with the prediction at t? (x, y; and z in 3D)"""
        if self.x is None:
            return True
        n = self.n
        dt = max(t - self.t, 0.0)
        pred = self.x[:n] + self.x[n:] * dt
        s = math.sqrt(max(np.diag(self.P)[:n])) + 1.5 * dt
        return float(np.linalg.norm(np.asarray(z[:n], float) - pred)) < max(n_sigma * s, floor)

    def distance(self, z, t):
        """Distance from z to the predicted position at t."""
        if self.x is None:
            return 0.0
        n = self.n
        pred = self.x[:n] + self.x[n:] * max(t - self.t, 0.0)
        return float(np.linalg.norm(np.asarray(z[:n], float) - pred))

    @property
    def pos(self):
        return None if self.x is None else np.r_[self.x[:self.n], np.zeros(3 - self.n)]

    @property
    def vel(self):
        return None if self.x is None else np.r_[self.x[self.n:], np.zeros(3 - self.n)]
