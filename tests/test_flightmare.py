"""R22 · Flightmare bridge against a fake Unity player (our renderer behind the same ZMQ protocol)."""
import math

import numpy as np
import pytest

pytest.importorskip('zmq')

from nanouav.agile import envs as EV                       # noqa: E402
from nanouav.flightmare import bridge as B                 # noqa: E402
from nanouav.flightmare.cloud import CloudWorld, find_lines, read_ply, to_ros   # noqa: E402
from nanouav.flightmare.fake import FakeUnity              # noqa: E402

PORTS = iter(range(16253, 16353, 2))


@pytest.fixture
def unity():
    pub, sub = next(PORTS), next(PORTS)
    env = EV.TEST['forest'](1000)
    cl = B.FlightmareClient(scene_id=2, pub_port=pub, sub_port=sub, timeout_s=20)
    fake = FakeUnity(env.world, pub, sub).start()
    cl.connect()
    yield cl, env
    cl.close()
    fake.stop()


def test_conversions_round_trip():
    R = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])        # yaw 90°
    q = B.rot_to_quat(R)
    assert q == pytest.approx([math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)])
    assert B.pos_ros2unity([1, 2, 3]) == [1, 3, 2]
    T = B.camera_T([0.1, 0, 0])
    assert T[:3, :3] @ np.array([0, 1, 0]) == pytest.approx([1, 0, 0])   # Flightmare camera +y → body forward


def test_render_matches_our_stereo_rig(unity):
    from nanouav.agile.stereo import StereoRig, render_stereo
    cl, env = unity
    p = np.array([0.0, 0.0, 1.5])
    o = cl.render(p, np.eye(3))
    assert o['left'].shape == (240, 320, 3) and o['depth_left'].shape == (240, 320)
    gl, gr, d = render_stereo(env.world, StereoRig(), p, 0.0, rng=np.random.default_rng(0))
    m = (d < 15) & (o['depth_left'] < 15)
    assert m.mean() > 0.3 and np.median(np.abs(o['depth_left'][m] - d[m])) < 0.05   # same view as our camera
    assert np.abs(o['left'][..., 0].astype(int) - o['right'][..., 0]).mean() > 1     # two different cameras


def test_point_cloud_frame_detection(unity, tmp_path):
    cl, env = unity
    origin, size = np.array([15.0, 0.0, 3.0]), np.array([30.0, 20.0, 6.0])
    P, info = to_ros(read_ply(cl.point_cloud(origin, size, 0.2, path=str(tmp_path), name='t', timeout_s=30)), origin, size)
    assert info['frame'] == 'unity'
    w = CloudWorld(P, 0.2)
    tree = next(o for o in env.world.objects if abs(o.cx - 15) < 12 and abs(o.cy) < 8)
    assert w.clearance(np.array([tree.cx, tree.cy, 1.5])) < 0.25                     # a trunk is where it should be
    assert w.clearance(np.array([tree.cx + tree.r + 1.0, tree.cy, 1.5])) == pytest.approx(
        env.world.clearance(np.array([tree.cx + tree.r + 1.0, tree.cy, 1.5])), abs=0.25)
    assert np.nanmedian(w.ground_z(P[:, 0], P[:, 1])) == pytest.approx(0.0, abs=0.05)


def test_episode_in_flightmare_with_cloud_map(unity, tmp_path):
    """gbl on a line found in the cloud; the student pipeline (observer → SGM → policy) on a few frames."""
    from nanouav.agile.rollout import Config, Rollout
    from nanouav.flightmare.cloud import line_env
    from nanouav.flightmare.run import FlightmareObserver
    cl, env = unity
    origin, size = np.array([22.0, 0.0, 3.5]), np.array([56.0, 24.0, 7.0])
    P, _ = to_ros(read_ply(cl.point_cloud(origin, size, 0.2, path=str(tmp_path), name='c', timeout_s=30)), origin, size)
    w = CloudWorld(P, 0.2)
    L = find_lines(w, n=1, length=25.0, seed=0)
    assert L
    e = line_env(w, L[0])
    r, _, _ = Rollout(e, Config(v_des=4.0, seed=0), 'gbl').run()
    assert r['success']
    obs = FlightmareObserver(cl, Config().rig)
    g, sgm, gt = obs(e.start, 0.0, 0.0, 0.0)
    assert g.shape == (240, 320) and (sgm > 0).mean() > 0.2 and (gt > 0).mean() > 0.2
