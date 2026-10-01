#!/usr/bin/env python3
"""
dyn_obstacles_node.py — mueve los obstáculos dinámicos de un mundo generado (gen_world.py).

Cada entrada de scn['dynamic'] = {name, p0, axis, amp, v} va y viene sobre p0 + s·axis, s ∈ [0, amp],
a velocidad v, mediante el plugin VelocityControl del modelo (/model/<name>/cmd_vel) y su odometría
(/model/<name>/odometry). Mismo movimiento que el simulador propio (g1_gazebo_world.py, onda triangular).
Uso: python3 dyn_obstacles_node.py --ros-args -p world_json:=/sim/nano_uav/gazebo/worlds/gen/dinamico_s0.json
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
        self.declare_parameter('delay', 8.0)           # s tras la primera odometría antes de mover
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
        self.get_logger().info(f'{len(self.items)} obstáculos dinámicos: '
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
                v[2] = 1.5 * (d['p0'][2] - p[2])             # mantener la altura
                # el balón no rota (VelocityControl fija ω = 0) → marco del modelo = mundo
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
