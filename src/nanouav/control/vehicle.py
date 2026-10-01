"""
nanouav/control/vehicle.py — Vehicle physical parameters and allocation matrix.

Profiles:
  'x3'   : Gazebo X3 (values copied from gazebo/models/X3_UAV/6/model.sdf and from
           the world's MulticopterMotorModel plugins) → the controller validated
           in Python uses exactly the plant that runs in Gazebo.
  'nano' : target ESP32-S3 nano-UAV (65 g, 8520 coreless motors, T/W ≈ 2).

Convention: FLU world and body frames (X forward, Y left, Z up).
Rotor dir = +1 CCW seen from above, −1 CW. Yaw reaction torque:
τz_i = −dir_i · km · f_i  (the body turns opposite to the propeller).
"""
from dataclasses import dataclass, field
import numpy as np

G = 9.81


@dataclass
class VehicleParams:
    name: str
    mass: float
    J: tuple
    rotor_xy: tuple            # ((x, y), ...) m
    rotor_dir: tuple           # +1 CCW, −1 CW
    kf: float                  # N/(rad/s)^2
    km: float                  # m  (τ = km · f)
    w_max: float               # rad/s
    tau_up: float              # s
    tau_down: float            # s
    lin_drag: float            # N/(m/s)
    radius: float              # m, bounding radius for collisions
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


X3 = VehicleParams(
    name='x3', mass=1.5 + 4 * 0.005, J=(0.0347563, 0.07, 0.0977),
    rotor_xy=((0.13, -0.22), (-0.13, 0.20), (0.13, 0.22), (-0.13, -0.20)),
    rotor_dir=(+1, +1, -1, -1),
    kf=8.54858e-06, km=0.016, w_max=800.0, tau_up=0.0125, tau_down=0.025,
    lin_drag=4 * 8.06428e-05 * 660.0, radius=0.36)   # arm 0.255 + propeller 0.1

NANO = VehicleParams(
    name='nano', mass=0.065, J=(3.55e-5, 4.23e-5, 7.47e-5),
    rotor_xy=((0.055, -0.055), (-0.055, 0.055), (0.055, 0.055), (-0.055, -0.055)),
    rotor_dir=(+1, +1, -1, -1),
    kf=4.08e-08, km=0.006, w_max=2800.0, tau_up=0.025, tau_down=0.035,
    lin_drag=0.02, radius=0.11)

PROFILES = {'x3': X3, 'nano': NANO}
