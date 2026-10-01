"""
nanouav/control/quad_sim.py — 6-DOF quadrotor dynamics with motors.

  * Asymmetric first-order motors (τ spin-up/spin-down, like MulticopterMotorModel)
  * Thrust kf·ω², reaction torque km·f, X allocation
  * Linear drag (rotor drag), external force/torque disturbances
  * Simple ground contact
  * Semi-implicit integration at 1 kHz, renormalized quaternion
"""
import numpy as np
from .vehicle import G


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                     w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2,
                     w1*z2 + x1*y2 - y1*x2 + z1*w2])


def quat_to_rot(q):
    w, x, y, z = q
    return np.array([[1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
                     [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
                     [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]])


def euler_from_rot(R):
    """(roll, pitch, yaw) ZYX; pitch>0 = nose down (FLU)."""
    pitch = -np.arcsin(np.clip(R[2, 0], -1, 1))
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


def quat_from_euler(roll, pitch, yaw):
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([cr*cp*cy + sr*sp*sy, sr*cp*cy - cr*sp*sy,
                     cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy])


class QuadSim:
    def __init__(self, P, pos=(0, 0, 0), yaw=0.0, dt=0.001, gyro_noise=0.0, seed=0):
        self.P = P
        self.dt = dt
        self.p = np.array(pos, float)
        self.v = np.zeros(3)
        self.q = quat_from_euler(0, 0, yaw)
        self.w = np.zeros(3)
        self.wm = np.full(4, np.sqrt(P.hover_f / P.kf)) if pos[2] > 0.01 else np.zeros(4)
        self.acc = np.zeros(3)
        self.gyro_noise = gyro_noise
        self.rng = np.random.default_rng(seed)
        self.F_ext = np.zeros(3)
        self.T_ext = np.zeros(3)

    @property
    def R(self):
        return quat_to_rot(self.q)

    def step(self, w_cmd, duration):
        P = self.P
        n = max(1, int(round(duration / self.dt)))
        w_cmd = np.clip(np.asarray(w_cmd, float), 0, P.w_max)
        for _ in range(n):
            tau = np.where(w_cmd > self.wm, P.tau_up, P.tau_down)
            self.wm += (w_cmd - self.wm) * (1 - np.exp(-self.dt / tau))
            f = P.kf * self.wm ** 2
            u = P.A @ f                       # [T, τx, τy, τz]
            R = self.R
            acc = (R[:, 2] * u[0] - P.lin_drag * self.v + self.F_ext) / P.mass
            acc[2] -= G
            Jw = P.Jv * self.w
            wdot = (u[1:] + self.T_ext - np.cross(self.w, Jw)) / P.Jv
            self.v += acc * self.dt
            self.p += self.v * self.dt
            self.w += wdot * self.dt
            if self.p[2] <= 0.0:              # ground
                self.p[2] = 0.0
                if self.v[2] < 0:
                    self.v[:] = 0.0
                    self.w *= 0.5
            th = self.w * self.dt
            ang = np.linalg.norm(th)
            if ang > 1e-12:
                dq = np.concatenate([[np.cos(ang / 2)], np.sin(ang / 2) * th / ang])
                self.q = quat_mul(self.q, dq)
                self.q /= np.linalg.norm(self.q)
            self.acc = acc

    def state(self):
        """'Measured' state (ground truth + optional gyro noise)."""
        w = self.w + (self.rng.normal(0, self.gyro_noise, 3) if self.gyro_noise else 0)
        return {'p': self.p.copy(), 'v': self.v.copy(), 'R': self.R, 'w': w,
                'acc': self.acc.copy(), 'motors': self.wm.copy()}
