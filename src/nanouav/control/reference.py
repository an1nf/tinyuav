"""
nanouav/control/reference.py — Smooth reference generation.

JerkLimited: velocity profile with bounded acceleration and jerk (near
time-optimal S-curve, no overshoot): the desired acceleration follows the braking
curve a_des = sign(e)·min(a_max, sqrt(2·j_max·|e|)), and its change is limited to j_max·dt.
Returns (v_ref, a_ref) → a_ref is used as feed-forward in the velocity loop.

YawShaper: same principle applied to the yaw angle (bounded rate and angular
acceleration), wrapped to ±π. Returns (ψ_ref, r_ref).
"""
import numpy as np


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class JerkLimited:
    def __init__(self, a_max, j_max, n=3):
        self.a_max = np.broadcast_to(np.asarray(a_max, float), (n,)).copy()
        self.j_max = np.broadcast_to(np.asarray(j_max, float), (n,)).copy()
        self.v = np.zeros(n)
        self.a = np.zeros(n)

    def reset(self, v, a=None):
        self.v = np.array(v, float)
        self.a = np.zeros_like(self.v) if a is None else np.array(a, float)

    def set_limits(self, a_max=None, j_max=None):
        if a_max is not None:
            self.a_max[:] = a_max
        if j_max is not None:
            self.j_max[:] = j_max

    def update(self, v_target, dt):
        e = np.asarray(v_target, float) - self.v
        # braking consistent with the current acceleration (avoids overshoot)
        e_brake = e - self.a * np.abs(self.a) / (2 * self.j_max)
        a_des = np.sign(e_brake) * np.minimum(self.a_max, np.sqrt(2 * self.j_max * np.abs(e_brake)))
        self.a += np.clip(a_des - self.a, -self.j_max * dt, self.j_max * dt)
        self.v += self.a * dt
        done = (np.abs(e) < 2e-3) & (np.abs(self.a) < self.j_max * dt * 2)
        self.v = np.where(done, v_target, self.v)
        self.a = np.where(done, 0.0, self.a)
        return self.v.copy(), self.a.copy()


class YawShaper:
    def __init__(self, r_max, a_max, psi0=0.0):
        self.r_max, self.a_max = r_max, a_max
        self.psi, self.r = psi0, 0.0

    def reset(self, psi, r=0.0):
        self.psi, self.r = psi, r

    def update(self, psi_target, dt):
        e = wrap(psi_target - self.psi) - self.r * abs(self.r) / (2 * self.a_max)
        r_des = np.sign(e) * min(self.r_max, np.sqrt(2 * self.a_max * abs(e)))
        self.r += np.clip(r_des - self.r, -self.a_max * dt, self.a_max * dt)
        self.psi = wrap(self.psi + self.r * dt)
        return self.psi, self.r
