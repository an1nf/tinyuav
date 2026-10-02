"""Quadrotor of the paper (Loquercio et al. 2021, §S5): sphere of 0.2 m radius, maximum mass-normalized
thrust c_max = 35.3 m/s² (thrust-to-weight 3.6) and J = 0.007 kg·m². The mass, the arm length and the motor
constants are not given in the paper: they are chosen for a 5-inch racing quadrotor and only need to
reproduce c_max, J and the size."""
import numpy as np

from nanouav.control.vehicle import VehicleParams, G

MASS = 0.80
C_MAX = 35.3                         # m/s², paper §S5
W_MAX = 3000.0                       # rad/s
KF = MASS * C_MAX / 4 / W_MAX ** 2   # → 4·kf·w_max² = m·c_max
ARM = 0.12

AGILE = VehicleParams(
    name='agile', mass=MASS, J=(0.007, 0.007, 0.012),
    rotor_xy=((ARM / np.sqrt(2), -ARM / np.sqrt(2)), (-ARM / np.sqrt(2), ARM / np.sqrt(2)),
              (ARM / np.sqrt(2), ARM / np.sqrt(2)), (-ARM / np.sqrt(2), -ARM / np.sqrt(2))),
    rotor_dir=(+1, +1, -1, -1),
    kf=KF, km=0.015, w_max=W_MAX, tau_up=0.015, tau_down=0.025,
    lin_drag=0.15, radius=0.2)

TILT_MAX_DEG = 65.0                  # cos(65°)·c_max ≈ 1.5 g of vertical thrust left
RATE_MAX = np.array([10.0, 10.0, 4.0])


def check():
    """c_max and thrust-to-weight of the profile (sanity)."""
    return {'c_max': 4 * AGILE.f_max / AGILE.mass, 'thrust_to_weight': AGILE.thrust_to_weight, 'g': G}
