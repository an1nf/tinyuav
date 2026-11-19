"""Quadrotor parameters and allocation. FLU frames; rotor dir +1 is CCW from above; tau_z = -dir km f."""
from dataclasses import dataclass, field

import numpy as np

G = 9.81


@dataclass
class VehicleParams:
    name: str
    mass: float
    J: tuple
    rotor_xy: tuple            # m
    rotor_dir: tuple           # +1 CCW, -1 CW
    kf: float                  # N/(rad/s)^2
    km: float                  # m (tau = km f)
    w_max: float               # rad/s
    tau_up: float              # s
    tau_down: float            # s
    lin_drag: float            # N/(m/s)
    radius: float              # m, collision radius
    A: np.ndarray = field(init=False, repr=False)
    A_inv: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        A = np.zeros((4, 4))
        for i, ((x, y), d) in enumerate(zip(self.rotor_xy, self.rotor_dir)):
            A[:, i] = [1.0, y, -x, -d * self.km]
        self.A = A
        self.A_inv = np.linalg.inv(A)
        self.Jv = np.asarray(self.J, float)

    @property
    def f_max(self):
        return self.kf * self.w_max ** 2

    @property
    def hover_f(self):
        return self.mass * G / 4.0

    @property
    def thrust_to_weight(self):
        return 4 * self.f_max / (self.mass * G)

    @property
    def tau_motor(self):
        return 0.5 * (self.tau_up + self.tau_down)


MASS = 0.80
C_MAX = 35.3                         # m/s^2, thrust-to-weight 3.6
W_MAX = 3000.0                       # rad/s
ARM = 0.12
_R = ARM / np.sqrt(2)

AGILE = VehicleParams(                # racing quadrotor, 0.2 m radius
    name='agile', mass=MASS, J=(0.007, 0.007, 0.012),
    rotor_xy=((_R, -_R), (-_R, _R), (_R, _R), (-_R, -_R)), rotor_dir=(+1, +1, -1, -1),
    kf=MASS * C_MAX / 4 / W_MAX ** 2, km=0.015, w_max=W_MAX, tau_up=0.015, tau_down=0.025,
    lin_drag=0.15, radius=0.2)

TILT_MAX_DEG = 65.0                  # cos(65 deg) c_max ~ 1.5 g vertical
RATE_MAX = np.array([10.0, 10.0, 4.0])
