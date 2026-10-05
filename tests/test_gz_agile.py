"""R21 · the R18 student in Gazebo, without Gazebo: the decision/control core with our renderer and a
velocity-lag drone, the world generator, the TFLite export and the ROS node on fake rclpy (tests/stubs)."""
import importlib.util
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from nanouav.agile import envs as EV
from nanouav.agile.gz_core import GlobalRefPolicy, fly_offline

ROOT = Path(__file__).resolve().parents[1]
STUBS = Path(__file__).parent / 'stubs'
FAKE = ('rclpy', 'rclpy.node', '_msg', 'geometry_msgs', 'geometry_msgs.msg', 'std_msgs', 'std_msgs.msg', 'nav_msgs',
        'nav_msgs.msg', 'sensor_msgs', 'sensor_msgs.msg', 'rosgraph_msgs', 'rosgraph_msgs.msg', 'actuator_msgs',
        'actuator_msgs.msg')


@pytest.mark.parametrize('v', [3.0, 7.0])
def test_core_loop_with_perfect_predictions_and_velocity_lag(v):
    """The plumbing (frames, plans, lead compensation of the velocity controller, status) flies the
    global path: any crash would be the loop's fault."""
    env = EV.TEST['forest'](1000)
    r, core = fly_offline(env, GlobalRefPolicy(env, v), v, seed=0, render=False)
    assert r['status'] == 'success' and r['min_clearance_m'] > 0.3
    assert abs(r['mean_speed'] - v) < 0.2 * v
    assert {'t', 'p', 'k', 'clear', 'W', 'att'} <= set(core.trace[0])


@pytest.mark.slow
def test_core_loop_with_rendered_stereo():
    env = EV.TEST['forest'](1001)
    r, core = fly_offline(env, GlobalRefPolicy(env, 7.0), 7.0, seed=0, render=True)
    assert r['status'] == 'success' and len(core.trace) > 30


def test_agile_world_and_drone_model():
    sys.path.insert(0, str(ROOT / 'gazebo' / 'tools'))
    import gen_world as G
    G.ensure_model_config()
    d = G.ensure_agile_model()
    m = ET.parse(Path(d) / 'model.sdf').getroot()
    cams = {s.get('name'): s for s in m.iter('sensor') if s.get('type') == 'camera'}
    assert set(cams) == {'left', 'right'}
    y = [float(c.find('pose').text.split()[1]) for c in (cams['left'], cams['right'])]
    assert y[0] - y[1] == pytest.approx(0.1)                       # baseline of StereoRig
    assert cams['left'].find('camera/image/width').text == '320'
    assert [c.get('name') for c in m.iter('collision')] == ['X3/base_link_inertia_collision']
    s = G.agile_scenario('forest', 1000)
    env = EV.TEST['forest'](1000)
    assert len(s['objects']) == len(env.world.objects) and s['A'] == [float(env.start[0]), float(env.start[1])]
    path = Path(G.build(s, 'agile_forest_s1000_pytest'))
    sdf = ET.fromstring(path.read_text())
    for f in [path, path.with_suffix('.json')] + list(Path(G.TEX_DIR).glob('agile_forest_s1000_pytest_*.png')):
        f.unlink()
    w = sdf.find('world')
    assert w.find('include/uri').text == 'model://agile_x3'
    assert '1250.0' in ET.tostring(w.find('include')).decode()


@pytest.mark.slow
def test_tflite_export_matches_keras(tmp_path):
    from nanouav.agile.policy import TFLitePolicy, build_model, export_tflite, load_policy
    m = build_model((120, 160))
    p = tmp_path / 'm.keras'
    m.save(p)
    out = export_tflite(str(p))
    rng = np.random.default_rng(1)
    d, v = rng.uniform(0, 10, (120, 160)).astype(np.float32), rng.normal(0, 3, 3)
    A, B = load_policy(str(p))(d, v, np.eye(3).ravel(), [1, 0, 0]), TFLitePolicy(out)(d, v, np.eye(3).ravel(), [1, 0, 0])
    assert isinstance(load_policy(out), TFLitePolicy)
    assert np.abs(A[0] - B[0]).max() < 1e-4 and np.abs(A[1] - B[1]).max() < 1e-4


@pytest.fixture
def ros(monkeypatch):
    saved = {m: sys.modules.pop(m) for m in list(sys.modules) if m in FAKE}
    monkeypatch.syspath_prepend(str(STUBS))
    monkeypatch.setenv('NANOUAV_HOME', str(ROOT))
    import rclpy
    from rclpy.node import Node
    rclpy.init()
    Node.OVERRIDES = {}
    yield Node
    rclpy.shutdown()
    Node.OVERRIDES = {}
    for m in FAKE + ('agile_node',):
        sys.modules.pop(m, None)
    sys.modules.update(saved)


class _StraightPolicy:
    """Three hypotheses straight ahead at 3 m/s (heading frame), the first one cheapest."""

    def __call__(self, depth, v_h, R, omega_h):
        x = np.arange(1, 11) * 0.3
        H = np.stack([x, np.zeros(10), np.zeros(10)], 1)
        return np.repeat(H[None], 3, 0), np.array([0.1, 0.5, 0.5])


def test_agile_node_takeoff_mission_and_crash(ros, tmp_path):
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import Image
    Node = ros
    model = tmp_path / 'fake.tflite'
    model.write_bytes(b'x')
    Node.OVERRIDES = dict(world='forest', seed=1000, v_des=3.0, model=str(model), run_tag='pytest_agile')
    spec = importlib.util.spec_from_file_location('agile_node', ROOT / 'gazebo' / 'nodes' / 'agile_node.py')
    mod = importlib.util.module_from_spec(spec)
    sys.modules['agile_node'] = mod
    spec.loader.exec_module(mod)
    mod.load_policy = lambda path: _StraightPolicy()
    n = mod.AgileNode()
    control = [cb for per, cb in n.timers if per == 0.01][0]
    env = EV.TEST['forest'](1000)

    def odom(p, v=(0, 0, 0), yaw=0.0):
        m = Odometry()
        m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z = map(float, p)
        m.pose.pose.orientation.w, m.pose.pose.orientation.z = math.cos(yaw / 2), math.sin(yaw / 2)
        m.twist.twist.linear.x, m.twist.twist.linear.y, m.twist.twist.linear.z = map(float, v)
        n.subscriptions['/model/nano_uav/odometry'](m)

    t = 100.0
    odom((env.start[0], env.start[1], 0.25))                        # on the ground: climb
    n._fake_time_ns = int(t * 1e9)
    control()
    tw = n.publishers['/nano_uav/gazebo/command/twist'].msgs[-1]
    assert tw.linear.z > 0.5 and n.phase == 'takeoff'
    odom(env.start)                                                 # at the flight altitude, still
    for _ in range(200):
        t += 0.01
        n._fake_time_ns = int(t * 1e9)
        control()
    assert n.phase == 'mission'
    gray = np.random.default_rng(0).integers(0, 255, (240, 320)).astype(np.uint8)
    for side in ('left', 'right'):
        n.subscriptions[f'/agile/{side}/image'](Image(height=240, width=320, encoding='mono8', data=gray.tobytes()))
    assert n.core.plan is not None and len(n.core.trace) == 1
    t += 0.05
    n._fake_time_ns = int(t * 1e9)
    control()
    assert n.publishers['/nano_uav/gazebo/command/twist'].msgs[-1].linear.x > 0.5     # forward
    tree = env.world.objects[0]
    odom((tree.cx, tree.cy, 1.5), v=(3, 0, 0))                     # inside a trunk
    t += 0.01
    n._fake_time_ns = int(t * 1e9)
    control()
    assert n.phase == 'done'
    out = ROOT / 'results' / 'gz_runs' / 'agile' / 'pytest_agile.json'
    D = json.load(open(out))
    out.unlink()
    assert D['result']['status'] == 'crash' and D['result']['world'] == 'forest' and len(D['trace']) == 1
