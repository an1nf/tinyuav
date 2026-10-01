#!/usr/bin/env python3
"""
dyn_obstacles_node.py — moves the dynamic obstacles of a generated world (`nanouav gazebo gen-world`).

Each entry of scn['dynamic'] = {name, p0, axis, amp, v} moves back and forth along p0 + s·axis, s ∈ [0, amp],
at speed v, via the model's VelocityControl plugin (/model/<name>/cmd_vel) and its odometry
(/model/<name>/odometry). Same motion as in our own simulator (`nanouav run r12`, triangle wave).
Usage: python3 dyn_obstacles_node.py --ros-args -p world_json:=/sim/nano_uav/gazebo/worlds/gen/dinamico_s0.json
"""
import json
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


class DynNode(Node):
    def __init__(self):
        super().__init__('nano_uav_dyn_obstacles')
        self.declare_parameter('world_json', '')
        self.declare_parameter('delay', 8.0)           # s after the first odometry before starting to move
        scn = json.load(open(self.get_parameter('world_json').value))
        self.delay = float(self.get_parameter('delay').value)
        self.items = []
        for d in scn['dynamic']:
            it = {'d': d, 'pub': self.create_publisher(Twist, f"/model/{d['name']}/cmd_vel", 10),
                  'p': None, 'dir': 1.0, 't0': None}
            self.create_subscription(Odometry, f"/model/{d['name']}/odometry",
                                     lambda m, it=it: it.__setitem__('p', np.array([m.pose.pose.position.x,
                                                                                    m.pose.pose.position.y,
                                                                                    m.pose.pose.position.z])), 10)
            self.items.append(it)
        self.create_timer(0.05, self.step)
        self.get_logger().info(f'{len(self.items)} dynamic obstacles: '
                               + ', '.join(f"{it['d']['name']} v={it['d']['v']} m/s" for it in self.items))

    def step(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        for it in self.items:
            d, p = it['d'], it['p']
            tw = Twist()
            if p is None:
                continue
            if it['t0'] is None:
                it['t0'] = now
            if now - it['t0'] > self.delay:
                ax = np.array([d['axis'][0], d['axis'][1], 0.0])
                s = float((p - np.asarray(d['p0'])) @ ax)
                if s > d['amp']:
                    it['dir'] = -1.0
                elif s < 0.0:
                    it['dir'] = 1.0
                v = ax * d['v'] * it['dir']
                v[2] = 1.5 * (d['p0'][2] - p[2])             # hold the height
                # the ball does not rotate (VelocityControl fixes ω = 0) → model frame = world frame
                tw.linear.x, tw.linear.y, tw.linear.z = map(float, v)
            it['pub'].publish(tw)


def main():
    rclpy.init()
    n = DynNode()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
