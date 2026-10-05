#!/usr/bin/env python3
"""
agile_node.py — the R18 student (Loquercio et al. 2021 recreation) flying the agile X3 in Gazebo (R21).

  stereo pair (/agile/left/image, /agile/right/image, L8 320×240 15 Hz)
    → SGM depth → student (.tflite, LiteRT) → plan          (nanouav.agile.gz_core.AgileCore)
  odometry (/model/nano_uav/odometry) → v_cmd = v_ref + τ·a_ref + k_p·(p_ref − p), yaw along the velocity
    → Twist (body frame) to the X3's MulticopterVelocityControl at 100 Hz

Phases: takeoff (climb at A to the flight altitude and settle) → mission (the student flies toward the goal
line 40 m ahead) → done (hover; result + trace in results/gz_runs/agile/<tag>.json; exits with
exit_on_done). The world geometry is rebuilt from (world, seed) with nanouav.agile.envs, so the crash check
and the trace's true clearances use exactly the obstacles of the generated SDF.

⚠ Written and tested offline (`tests/test_gz_agile.py` runs AgileCore with our renderer and a
  velocity-lag drone); the first Gazebo runs should be at 3 m/s.
"""
import json
import math
import os
import sys
import threading
import time

ROOT = os.environ.get('NANOUAV_HOME', os.environ.get('NANO_UAV_ROOT',
                      os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))))
sys.path.insert(0, os.path.join(ROOT, 'src'))
os.environ.setdefault('NANOUAV_HOME', ROOT)

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image

from nanouav.agile import envs as EV
from nanouav.agile.gz_core import AgileCore
from nanouav.agile.policy import load_policy
from nanouav.control.quad_sim import quat_to_rot, euler_from_rot


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class AgileNode(Node):
    def __init__(self):
        super().__init__('agile_student')
        for k, v in (('world', 'forest'), ('seed', 1000), ('v_des', 3.0), ('model', ''), ('tscale', 'chord'),
                     ('vcap', 0.0), ('select', 'snap'), ('gain', 5.0), ('kp', 1.0), ('t_max', 0.0),
                     ('exit_on_done', False), ('run_tag', ''), ('keep_depth', False), ('twist_frame', 'body')):
            self.declare_parameter(k, v)
        gp = lambda n: self.get_parameter(n).value
        self.world, self.seed, self.v = gp('world'), int(gp('seed')), float(gp('v_des'))
        model = gp('model')
        if not model or not os.path.exists(model):
            raise SystemExit(f'model not found: {model!r} (export it on the host: nanouav run r18 export --model ...)')
        self.env = EV.TEST[self.world](self.seed)
        opts = {'tscale': gp('tscale'), 'vcap': float(gp('vcap')) or None, 'select': gp('select')}
        self.core = AgileCore(self.env, load_policy(model), self.v, opts=opts, tau=1.0 / float(gp('gain')),
                              kp=float(gp('kp')), t_max=float(gp('t_max')) or None, keep_depth=bool(gp('keep_depth')))
        self.opts, self.model, self.gain = opts, model, float(gp('gain'))
        self.exit_on_done, self.twist_frame = bool(gp('exit_on_done')), gp('twist_frame')
        tag = gp('run_tag') or f"{os.path.splitext(os.path.basename(model))[0]}_{self.world}_s{self.seed}_v{self.v:g}"
        self.out = os.path.join(ROOT, 'results', 'gz_runs', 'agile', tag + '.json')
        self.A = np.array([self.env.start[0], self.env.start[1], self.env.start[2]], float)
        d0 = self.core.line.pos(1.0) - self.A
        self.yaw0 = math.atan2(d0[1], d0[0])
        self.phase, self.t_settle, self.t_mission = 'takeoff', None, None
        self.st, self.left, self.right, self._lock = None, None, None, threading.Lock()
        self.enable_pub = self.create_publisher(Bool, '/nano_uav/enable', 10)
        self.cmd_pub = self.create_publisher(Twist, '/nano_uav/gazebo/command/twist', 10)
        self.create_subscription(Odometry, '/model/nano_uav/odometry', self.odom_cb, 50)
        self.create_subscription(Image, '/agile/left/image', lambda m: self.img_cb(m, 'left'), 5)
        self.create_subscription(Image, '/agile/right/image', lambda m: self.img_cb(m, 'right'), 5)
        self.create_timer(0.5, lambda: self.enable_pub.publish(Bool(data=True)))
        self.create_timer(0.01, self.control)
        self.get_logger().info(f'R18 student in Gazebo · {self.world} seed {self.seed} · v_des {self.v} m/s · '
                               f'{os.path.basename(model)} · {opts} · gain {self.gain}')

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def odom_cb(self, m):
        q = m.pose.pose.orientation
        R = quat_to_rot((q.w, q.x, q.y, q.z))
        v = np.array([m.twist.twist.linear.x, m.twist.twist.linear.y, m.twist.twist.linear.z])
        p = m.pose.pose.position
        roll, pitch, yaw = euler_from_rot(R)
        with self._lock:
            self.st = {'p': np.array([p.x, p.y, p.z]), 'v': R @ v if self.twist_frame == 'body' else v, 'R': R,
                       'yaw': yaw, 'pitch': pitch, 'roll': roll}

    def img_cb(self, m, side):
        step = getattr(m, 'step', 0) or m.width
        img = np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, step)[:, :m.width]
        hdr = getattr(m, 'header', None)
        stamp = hdr.stamp.sec + hdr.stamp.nanosec * 1e-9 if hdr is not None and hasattr(hdr, 'stamp') else self.now()
        with self._lock:
            setattr(self, side, (stamp, img.copy(), None if self.st is None else dict(self.st)))
            L, Rr = self.left, self.right
        if self.phase != 'mission' or L is None or Rr is None or abs(L[0] - Rr[0]) > 1e-3:
            return
        with self._lock:
            self.left = self.right = None                      # one decision per pair
        st = L[2] or Rr[2]
        if st is not None:
            t0 = time.monotonic()
            self.core.frame(L[0], L[1], Rr[1], st)
            if len(self.core.trace) % 30 == 1:
                d = self.core.trace[-1]
                self.get_logger().info(f"t={d['t']:.1f} s  p=({d['p'][0]:.1f}, {d['p'][1]:.1f}, {d['p'][2]:.1f})  "
                                       f"|v|={np.linalg.norm(d['v']):.1f}  β={d['beta']:.2f}  "
                                       f"decision {1000 * (time.monotonic() - t0):.0f} ms")

    def send(self, v_world, yaw_rate, yaw):
        tw = Twist()
        if self.twist_frame == 'body':
            c, s = math.cos(yaw), math.sin(yaw)
            tw.linear.x, tw.linear.y = c * v_world[0] + s * v_world[1], -s * v_world[0] + c * v_world[1]
        else:
            tw.linear.x, tw.linear.y = float(v_world[0]), float(v_world[1])
        tw.linear.z = float(v_world[2])
        tw.angular.z = float(yaw_rate)
        self.cmd_pub.publish(tw)

    def control(self):
        with self._lock:
            st = None if self.st is None else dict(self.st)
        if st is None:
            return
        t = self.now()
        if self.phase == 'takeoff':                            # climb at A and settle, facing the goal
            e = self.A - st['p']
            v = np.clip(1.2 * e, -1.5, 1.5)
            self.send(v, float(np.clip(1.5 * _wrap(self.yaw0 - st['yaw']), -1, 1)), st['yaw'])
            ok = np.linalg.norm(e) < 0.15 and np.linalg.norm(st['v']) < 0.2 and abs(_wrap(self.yaw0 - st['yaw'])) < 0.05
            self.t_settle = (self.t_settle or t) if ok else None
            if self.t_settle is not None and t - self.t_settle > 1.5:
                self.phase, self.t_mission = 'mission', t
                self.get_logger().info('takeoff done → mission')
            return
        if self.phase == 'mission':
            v_cmd, yaw_cmd = self.core.command(t, st)
            self.send(v_cmd, float(np.clip(3.0 * _wrap(yaw_cmd - st['yaw']), -2.5, 2.5)), st['yaw'])
            status = self.core.status(t, st)
            if status != 'flying':
                self.phase = 'done'
                self.finish()
            return
        self.send(np.zeros(3), 0.0, st['yaw'])                 # done: hover

    def finish(self):
        r = dict(self.core.result, world=self.world, seed=self.seed, model=os.path.basename(self.model),
                 opts=self.opts, gain=self.gain, sim='gazebo')
        self.get_logger().info(f"{r['status'].upper()} · {r['time_s']} s · min clearance {r['min_clearance_m']} m · "
                               f"mean {r['mean_speed']} / max {r['max_speed']} m/s → {self.out}")
        os.makedirs(os.path.dirname(self.out), exist_ok=True)
        trace = self.core.trace
        if self.core.keep_depth:
            np.savez_compressed(self.out[:-5] + '_depth.npz', depth=np.stack([d.pop('depth') for d in trace]))
        json.dump({'result': r, 'log': self.core.log, 'trace': trace}, open(self.out, 'w'))
        if self.exit_on_done:
            raise SystemExit(0)


def main():
    rclpy.init()
    n = AgileNode()
    try:
        rclpy.spin(n)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        n.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
