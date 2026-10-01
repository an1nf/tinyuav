#!/usr/bin/env python3
"""
flight_controller_node.py — Control propio hasta velocidad de motores (modo 'motors').

Corre el mismo FlightStack validado en e3_closed_loop.py (perfil X3 con los
parámetros exactos del SDF) a 250 Hz:
  /nano_uav/setpoint (Twist: linear = v mundo objetivo, angular.z = yaw objetivo)
  + odometría (+ IMU si está puenteada)
  → referencia suave → cascada vel/actitud/tasa → mezclador → ω motores
  → /nano_uav/gazebo/command/motor_speed (actuator_msgs/Actuators)

El MulticopterVelocityControl del X3 debe estar DESHABILITADO (el perception_node
publica enable=False en este modo).

⚠ No probado en Gazebo desde este entorno. Verificar primero con el dron en el
  suelo y luego despegue. Si el yaw diverge (gira cada vez más rápido), el
  convenio de signo del torque de reacción de gz es el opuesto: yaw_sign:=-1.
"""
import os, sys, time
ROOT = os.environ.get('NANOUAV_HOME', os.environ.get('NANO_UAV_ROOT',
                      os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))))
sys.path.insert(0, os.path.join(ROOT, 'src'))         # paquete nanouav sin instalar (contenedor)
os.environ.setdefault('NANOUAV_HOME', ROOT)

import dataclasses
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from actuator_msgs.msg import Actuators

from nanouav.control.vehicle import PROFILES, VehicleParams
from nanouav.control.flight_stack import FlightStack
from nanouav.control.quad_sim import quat_to_rot, euler_from_rot


class FlightControllerNode(Node):
    def __init__(self):
        super().__init__('nano_uav_flight_controller')
        self.declare_parameter('profile', 'x3')
        self.declare_parameter('yaw_sign', 1)
        self.declare_parameter('rate_hz', 250.0)
        self.declare_parameter('odom_twist_frame', 'body')
        P0 = PROFILES[self.get_parameter('profile').value]
        ys = int(self.get_parameter('yaw_sign').value)
        fields = {f.name: getattr(P0, f.name) for f in dataclasses.fields(P0) if f.init}
        fields['rotor_dir'] = tuple(ys * d for d in P0.rotor_dir)
        self.P = VehicleParams(**fields)
        self.fs = FlightStack(self.P)
        self.twist_body = self.get_parameter('odom_twist_frame').value == 'body'
        self.st, self.gyro, self.t0, self.last = None, None, time.monotonic(), None
        self.pub = self.create_publisher(Actuators, '/nano_uav/gazebo/command/motor_speed', 10)
        self.create_subscription(Odometry, '/model/nano_uav/odometry', self.odom_cb, 50)
        self.create_subscription(Imu, '/nano_uav/imu', self.imu_cb, 50)
        self.create_subscription(Twist, '/nano_uav/setpoint', self.sp_cb, 10)
        self.create_timer(1.0 / self.get_parameter('rate_hz').value, self.loop)
        self.get_logger().info(f'Flight controller · {self.P.name} · T/W {self.P.thrust_to_weight:.2f} · yaw_sign {ys}')

    def odom_cb(self, m):
        q = m.pose.pose.orientation
        R = quat_to_rot((q.w, q.x, q.y, q.z))
        v = np.array([m.twist.twist.linear.x, m.twist.twist.linear.y, m.twist.twist.linear.z])
        w = np.array([m.twist.twist.angular.x, m.twist.twist.angular.y, m.twist.twist.angular.z])
        p = m.pose.pose.position
        self.st = {'p': np.array([p.x, p.y, p.z]), 'v': R @ v if self.twist_body else v, 'R': R, 'w': w}

    def imu_cb(self, m):
        self.gyro = np.array([m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z])

    def sp_cb(self, m):
        self.fs.set_target(np.array([m.linear.x, m.linear.y, m.linear.z]), m.angular.z)

    def loop(self):
        if self.st is None:
            return
        now = time.monotonic()
        dt = 0.004 if self.last is None else float(np.clip(now - self.last, 1e-3, 0.02))
        self.last = now
        st = dict(self.st)
        if self.gyro is not None:
            st['w'] = self.gyro
        w_cmd, info = self.fs.update(st, now - self.t0, dt)
        msg = Actuators()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.velocity = [float(x) for x in w_cmd]
        self.pub.publish(msg)


def main():
    rclpy.init()
    n = FlightControllerNode()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
