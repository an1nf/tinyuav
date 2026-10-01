"""Gazebo ROS 2 nodes driven with fake rclpy / message modules (``tests/stubs``), no ROS needed.

Covers what can break when the package changes underneath the nodes: imports, odometry and image
decoding, perception → planner → reference → Twist, the episode monitor (clearance, collisions,
timeout, results JSON) and the dynamic-obstacle motion.
"""
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
STUBS = Path(__file__).parent / 'stubs'
NODES = ROOT / 'gazebo' / 'nodes'
FAKE_MODS = ('rclpy', 'rclpy.node', '_msg', 'geometry_msgs', 'geometry_msgs.msg', 'std_msgs', 'std_msgs.msg',
             'nav_msgs', 'nav_msgs.msg', 'sensor_msgs', 'sensor_msgs.msg', 'rosgraph_msgs', 'rosgraph_msgs.msg',
             'actuator_msgs', 'actuator_msgs.msg')


@pytest.fixture
def ros(monkeypatch):
    """Put the fake ROS modules first on sys.path; remove them (and the nodes) afterwards."""
    saved = {m: sys.modules.pop(m) for m in list(sys.modules) if m in FAKE_MODS}
    monkeypatch.syspath_prepend(str(STUBS))
    monkeypatch.setenv('NANOUAV_HOME', str(ROOT))
    monkeypatch.setenv('NANOUAV_FAKE_CNN', '1')
    import rclpy
    from rclpy.node import Node
    rclpy.init()
    Node.OVERRIDES = {}
    yield rclpy, Node
    rclpy.shutdown()                      # stops the perception thread (while rclpy.ok())
    time.sleep(0.6)
    Node.OVERRIDES = {}
    for m in FAKE_MODS + ('perception_node', 'dyn_obstacles_node', 'flight_controller_node'):
        sys.modules.pop(m, None)
    sys.modules.update(saved)


def load_node(name):
    spec = importlib.util.spec_from_file_location(name, NODES / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def odom(p, yaw=0.0, v=(0, 0, 0)):
    from nav_msgs.msg import Odometry
    m = Odometry()
    m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z = map(float, p)
    m.pose.pose.orientation.w, m.pose.pose.orientation.z = math.cos(yaw / 2), math.sin(yaw / 2)
    m.twist.twist.linear.x, m.twist.twist.linear.y, m.twist.twist.linear.z = map(float, v)
    return m


def clock(t):
    from rosgraph_msgs.msg import Clock
    c = Clock()
    c.clock.sec, c.clock.nanosec = int(t), int((t % 1) * 1e9)
    return c


def image(gray):
    from sensor_msgs.msg import Image
    return Image(height=gray.shape[0], width=gray.shape[1], encoding='mono8', data=gray.tobytes())


@pytest.fixture
def world_json(tmp_path):
    from nanouav.sim import scenarios as SC
    s = SC.make('estrechos', 0)
    f = tmp_path / 'estrechos_s0.json'
    f.write_text(json.dumps(s))
    return s, f


def make_perception(ros, world, **over):
    _, Node = ros
    s, f = world
    Node.OVERRIDES = dict(world_json=str(f), planner='mh', hud=False, goal_x=s['B'][0], goal_y=s['B'][1],
                          t_max=5.0, exit_on_done=True, run_tag='pytest', **over)
    return load_node('perception_node').PerceptionNode(), s


def test_perception_pipeline_publishes_twist(ros, world_json):
    node, s = make_perception(ros, world_json)
    assert '/nano_uav/gazebo/command/twist' in node.publishers
    A = (*s['A'], 0.65)
    node.subscriptions['/model/nano_uav/odometry'](odom(A, s['yaw0'], v=(0.2, 0, 0)))
    assert node.state['yaw'] == pytest.approx(s['yaw0'])
    assert node.state['v'][:2] == pytest.approx([0.2 * math.cos(s['yaw0']), 0.2 * math.sin(s['yaw0'])])   # body → world
    # one camera frame (rendered from the real scenario) → perception thread → planner
    from nanouav.scene import renderer
    from nanouav.sim import scenarios as SC
    g, _ = renderer.render_view(SC.to_world(s), A, s['yaw0'], 0, 0, rng=np.random.default_rng(0))
    node.subscriptions['/nano_uav/camera/image'](image(g))
    t0 = time.time()
    while not node.hud['info'] and time.time() - t0 < 20:
        time.sleep(0.05)
    assert node.hud['info'], 'perception thread produced no plan'
    assert node.A is not None and node.A == pytest.approx(s['A'])
    node.control_loop()
    tw = node.publishers['/nano_uav/gazebo/command/twist'].msgs[-1]
    assert all(np.isfinite([tw.linear.x, tw.linear.y, tw.linear.z, tw.angular.z]))


def test_episode_timeout_writes_result(ros, world_json, tmp_path):
    from nanouav import paths
    node, s = make_perception(ros, world_json)
    node.subscriptions['/clock'](clock(10.0))
    node.subscriptions['/model/nano_uav/odometry'](odom((*s['A'], 0.65)))
    node.monitor()                                            # take-off detected at t = 10 s
    node.subscriptions['/clock'](clock(11.0))
    node.monitor()                                            # first sample of the path
    assert not node.done
    node.subscriptions['/model/nano_uav/odometry'](odom((s['A'][0] + 0.5, s['A'][1], 0.65)))
    node.subscriptions['/clock'](clock(16.0))                 # 6 s > t_max = 5 s
    node.monitor()
    assert node.done
    files = list((paths.results_dir() / 'gz_runs').glob('*_pytest.json'))
    assert len(files) == 1
    r = json.loads(files[0].read_text())
    assert r['result'] == 'timeout' and not r['success'] and r['world'] == 'estrechos' and r['sim_time']
    assert r['collisions'] == 0 and r['path_m'] == pytest.approx(0.5, abs=0.01)
    assert any('=== EPISODIO: timeout' in m for _, m in node.get_logger().lines)   # batch_eval.sh greps this


def test_episode_collision(ros, world_json):
    node, s = make_perception(ros, world_json)
    o = next(o for o in s['objects'] if o['type'] == 'box')
    node.subscriptions['/clock'](clock(1.0))
    node.subscriptions['/model/nano_uav/odometry'](odom((*s['A'], 0.65)))
    node.monitor()
    node.subscriptions['/model/nano_uav/odometry'](odom(o['center']))   # inside an obstacle
    node.subscriptions['/clock'](clock(2.0))
    node.monitor()
    assert node.ep['collisions'] == 1 and node.done
    assert any('COLLISION #1' in m for _, m in node.get_logger().lines)


def test_dynamic_obstacles_triangle_wave(ros):
    _, Node = ros
    from nanouav.sim import scenarios as SC
    s = SC.make('dinamico', 0)
    assert s['dynamic']
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as fh:
        json.dump(s, fh)
    Node.OVERRIDES = {'world_json': fh.name, 'delay': 1.0}
    n = load_node('dyn_obstacles_node').DynNode()
    d = s['dynamic'][0]
    pub = n.publishers[f"/model/{d['name']}/cmd_vel"]
    cb = n.subscriptions[f"/model/{d['name']}/odometry"]
    cb(odom(d['p0']))
    n._fake_time_ns = 0
    n.step()                                                    # before the delay: zero velocity
    assert pub.msgs[-1].linear.x == 0 and pub.msgs[-1].linear.y == 0
    n._fake_time_ns = int(2e9)
    n.step()
    v = np.array([pub.msgs[-1].linear.x, pub.msgs[-1].linear.y])
    assert np.linalg.norm(v) == pytest.approx(d['v'])
    assert v @ np.asarray(d['axis']) > 0                        # moving forward along the axis
    far = np.asarray(d['p0'], float) + np.r_[d['axis'], 0.0] * (d['amp'] + 0.1)
    cb(odom(far))
    n.step()
    v = np.array([pub.msgs[-1].linear.x, pub.msgs[-1].linear.y])
    assert v @ np.asarray(d['axis']) < 0                        # bounced back
    Path(fh.name).unlink()


def test_flight_controller_outputs_motor_speeds(ros):
    _, Node = ros
    Node.OVERRIDES = {'profile': 'x3'}
    n = load_node('flight_controller_node').FlightControllerNode()
    n.subscriptions['/model/nano_uav/odometry'](odom((0, 0, 0.65)))
    from geometry_msgs.msg import Twist
    sp = Twist()
    sp.linear.x = 0.3
    n.subscriptions['/nano_uav/setpoint'](sp)
    for _ in range(5):
        n.loop()
    msg = n.publishers['/nano_uav/gazebo/command/motor_speed'].msgs[-1]
    assert len(msg.velocity) == 4 and all(np.isfinite(msg.velocity)) and min(msg.velocity) > 0


def test_clock_going_backwards_falls_back_to_wall_time(ros, world_json):
    """Two simulations on the same /clock (shared partition/domain) must not freeze the episode monitor."""
    node, s = make_perception(ros, world_json)
    node.subscriptions['/clock'](clock(50.0))
    node.subscriptions['/model/nano_uav/odometry'](odom((*s['A'], 0.65)))
    node.monitor()                                            # take-off at sim 50 s
    node.subscriptions['/clock'](clock(52.0))
    node.subscriptions['/clock'](clock(3.0))                  # another simulation's clock
    assert node._clock_bad
    el = node._now() - node.ep['t_takeoff']
    assert 1.5 < el < 3.0                                     # elapsed time preserved across the switch
    node.ep['t_takeoff'] -= 10.0                              # pretend 10 more (wall) seconds went by
    node.monitor()
    assert node.done                                          # t_max = 5 s → timeout fires on wall time
