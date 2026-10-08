"""R22 · Flightmare bridge against a fake Unity player (our renderer behind the same ZMQ protocol)."""
import math

import numpy as np
import pytest

pytest.importorskip('zmq')

from nanouav.agile import envs as EV                       # noqa: E402
from nanouav.flightmare import bridge as B                 # noqa: E402
from nanouav.flightmare.cloud import CloudWorld, find_lines, read_ply   # noqa: E402
from nanouav.flightmare.fake import FakeUnity              # noqa: E402

PORTS = iter(range(16253, 16353, 2))


@pytest.fixture
def unity():
    pub, sub = next(PORTS), next(PORTS)
    env = EV.TEST['forest'](1000)
    cl = B.FlightmareClient(scene_id=2, render_offset=(21.0, 217.0, 94.0), pub_port=pub, sub_port=sub, timeout_s=20)
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
    P, info = cl.cloud_points(origin, size, 0.2, path=str(tmp_path), name='t', timeout_s=60)
    assert info['frame'] == 'ros' and info['shift'] == [0.0, 0.0, 0.0]         # absolute, ROS order (SavePointCloud.cs)
    raw = read_ply(info['file'])
    assert raw[:, 1].mean() > 150                                               # the file is in the offset (absolute) frame
    w = CloudWorld(P, 0.2)
    tree = next(o for o in env.world.objects if abs(o.cx - 15) < 12 and abs(o.cy) < 8)
    assert w.clearance(np.array([tree.cx, tree.cy, 1.5])) < 0.25                     # a trunk is where it should be
    assert w.clearance(np.array([tree.cx + tree.r + 1.0, tree.cy, 1.5])) == pytest.approx(
        env.world.clearance(np.array([tree.cx + tree.r + 1.0, tree.cy, 1.5])), abs=0.25)
    assert np.nanmedian(w.ground_z(P[:, 0], P[:, 1])) == pytest.approx(0.0, abs=0.15)


def test_episode_in_flightmare_with_cloud_map(unity, tmp_path):
    """gbl on a line found in the cloud; the student pipeline (observer → SGM → policy) on a few frames."""
    from nanouav.agile.rollout import Config, Rollout
    from nanouav.flightmare.cloud import line_env
    from nanouav.flightmare.run import FlightmareObserver
    cl, env = unity
    origin, size = np.array([22.0, 0.0, 3.5]), np.array([56.0, 24.0, 7.0])
    P, _ = cl.cloud_points(origin, size, 0.2, path=str(tmp_path), name='c', timeout_s=60)
    w = CloudWorld(P, 0.2)
    L = find_lines(w, n=1, length=25.0, seed=0)
    assert L
    e = line_env(w, L[0])
    r, _, _ = Rollout(e, Config(v_des=4.0, seed=0), 'gbl').run()
    assert r['success']
    obs = FlightmareObserver(cl, Config().rig)
    g, sgm, gt = obs(e.start, 0.0, 0.0, 0.0)
    assert g.shape == (240, 320) and (sgm > 0).mean() > 0.2 and (gt > 0).mean() > 0.2


def test_unity_depth_is_divided_by_the_far_plane(unity):
    """The player sends depth / farClipPlane[1]; the client multiplies it back (depth_mult defaults to far)."""
    cl, env = unity
    assert cl.depth_mult == cl.depth_far == 100.0
    assert cl.vehicle['size'] == [0.01] * 3                                       # drone mesh out of the view
    o = cl.render(np.array([0.0, 0.0, 1.5]), np.eye(3))
    d = o['depth_left']
    assert np.median(d[(d > 0) & (d < 99)]) > 2.0                                 # metres, not fractions


def test_probe_with_fake_player(tmp_path):
    from nanouav.flightmare import run as RUN
    pub, sub = next(PORTS), next(PORTS)
    env = EV.TEST['forest'](1000)
    fake = None
    import threading
    def start_fake():
        nonlocal fake
        import time
        time.sleep(0.5)
        fake = FakeUnity(env.world, pub, sub).start()
    th = threading.Thread(target=start_fake)
    th.start()
    try:
        RUN.main(['probe', '--pose', '0,0,1.5,0', '--res', '0.25', '--results', str(tmp_path),
                  '--pub-port', str(pub), '--sub-port', str(sub)])
    finally:
        th.join()
        if fake:
            fake.stop()
    import json
    c = json.load(open(tmp_path / 'calib.json'))
    assert not c['warnings'], c['warnings']
    assert c['depth_mult'] == 100.0 and c['fov_is'] == 'vertical' and not c['y_mirror']
    assert c['checks'][c['best']]['unity_rel_err'] < 0.1
    assert (tmp_path / 'probe.png').exists() and (tmp_path / 'probe.npz').exists()


def test_backprojection_lands_on_the_surfaces(unity):
    from nanouav.agile.geometry import Field
    from nanouav.flightmare.scan import backproject
    from nanouav.scene.renderer import body_rot
    cl, env = unity
    F = Field(env.world)
    for yaw, pitch in ((0.0, 0.0), (0.7, 0.4), (-1.2, -0.3)):
        p, R = np.array([2.0, -1.0, 1.6]), body_rot(yaw, pitch, 0.0)
        P = backproject(cl.render(p, R)['depth_left'], R, p, cl.T_left, cl.fov, 15.0)
        assert len(P) > 1000 and np.median(np.abs(F(P))) < 0.05


def test_scan_map_and_lines(tmp_path):
    from nanouav.agile.rollout import Config, Rollout
    from nanouav.flightmare.cloud import line_env
    from nanouav.flightmare.scan import scan
    pub, sub = next(PORTS), next(PORTS)
    env = EV.TEST['forest'](1000)
    cl = B.FlightmareClient(scene_id=2, render_offset=(21.0, 217.0, 94.0), pub_port=pub, sub_port=sub,
                            stereo=False, timeout_s=20)
    fake = FakeUnity(env.world, pub, sub).start()
    try:
        cl.connect()
        origin, size = np.array([16.0, 0.0, 3.0]), np.array([32.0, 16.0, 6.0])
        P, st = scan(cl, origin, size, 0.15, step=4.0, heights=(1.5,), yaws=4, pitches=(0.0,), log=lambda *_: None)
    finally:
        cl.close()
        fake.stop()
    assert st['frames'] == 8 * 4 * 4
    w = CloudWorld(P, 0.15, inflate=0.075)
    rng = np.random.default_rng(0)
    Q = np.c_[rng.uniform(4, 28, 400), rng.uniform(-6, 6, 400), rng.uniform(0.5, 3, 400)]
    from nanouav.agile.geometry import Field
    true = Field(env.world)(Q)
    near = true < 3.0
    assert np.median(np.abs(w.distance(Q[near]) - true[near])) < 0.15     # the map matches the world
    L = find_lines(w, n=1, length=20.0, seed=0)
    assert L and Rollout(line_env(w, L[0]), Config(v_des=4.0, seed=0), 'gbl').run()[0]['success']
