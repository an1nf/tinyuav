"""Flightmare bridge against a fake player: our renderer behind the same ZMQ protocol."""
import math

import numpy as np
import pytest

pytest.importorskip('zmq')

from tinyuav.sim import envs as EV                       # noqa: E402
from tinyuav.flightmare import bridge as B                 # noqa: E402
from tinyuav.flightmare.fake import FakeUnity              # noqa: E402

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
    R = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])        # yaw 90 deg
    q = B.rot_to_quat(R)
    assert q == pytest.approx([math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)])
    assert B.pos_ros2unity([1, 2, 3]) == [1, 3, 2]
    T = B.camera_T([0.1, 0, 0])
    assert T[:3, :3] @ np.array([0, 1, 0]) == pytest.approx([1, 0, 0])   # Flightmare camera +y is body forward


def test_render_matches_our_stereo_rig(unity):
    from tinyuav.sim.stereo import StereoRig, render_stereo
    cl, env = unity
    p = np.array([0.0, 0.0, 1.5])
    o = cl.render(p, np.eye(3))
    assert o['left'].shape == (240, 320, 3) and o['depth_left'].shape == (240, 320)
    gl, gr, d = render_stereo(env.world, StereoRig(), p, 0.0, rng=np.random.default_rng(0))
    m = (d < 15) & (o['depth_left'] < 15)
    assert m.mean() > 0.3 and np.median(np.abs(o['depth_left'][m] - d[m])) < 0.05   # same view as our camera
    assert np.abs(o['left'][..., 0].astype(int) - o['right'][..., 0]).mean() > 1     # two different cameras


def test_unity_depth_is_divided_by_the_far_plane(unity):
    """The player sends depth / far plane; the client scales it back."""
    cl, env = unity
    assert cl.depth_mult == cl.depth_far == 100.0
    assert cl.vehicle['size'] == [0.01] * 3
    o = cl.render(np.array([0.0, 0.0, 1.5]), np.eye(3))
    d = o['depth_left']
    assert np.median(d[(d > 0) & (d < 99)]) > 2.0


def test_backprojection_lands_on_the_surfaces(unity):
    from tinyuav.sim.geometry import Field
    from tinyuav.flightmare.scan import backproject
    from tinyuav.sim.render import body_rot
    cl, env = unity
    F = Field(env.world)
    for yaw, pitch in ((0.0, 0.0), (0.7, 0.4), (-1.2, -0.3)):
        p, R = np.array([2.0, -1.0, 1.6]), body_rot(yaw, pitch, 0.0)
        P = backproject(cl.render(p, R)['depth_left'], R, p, cl.T_left, cl.fov, 15.0)
        assert len(P) > 1000 and np.median(np.abs(F(P))) < 0.05


def _arena_player(pub, sub):
    from tinyuav.flightmare.fake import arena_world
    return FakeUnity(arena_world(0, x=(-5.0, 45.0), y=(-6.0, 6.0), height=6.0), pub, sub).start()


def test_spawned_boxes_render_and_map(tmp_path):
    from tinyuav.sim.geometry import Field
    from tinyuav.flightmare import arena as A
    from tinyuav.flightmare.scan import scan
    from tinyuav.sim.render import Box
    pub, sub = next(PORTS), next(PORTS)
    fake = _arena_player(pub, sub)
    cl = B.FlightmareClient(scene_id=0, pub_port=pub, sub_port=sub, stereo=False, n_objects=3, timeout_s=20)
    try:
        cl.connect()
        P, hits, st = scan(cl, np.array([20.0, 0.0, 3.0]), np.array([50.0, 12.0, 6.0]), 0.15, step=5.0,
                           heights=(1.5, 3.0), yaws=4, pitches=(0.0, 30.0), log=lambda *_: None)
        sdf, floor, rect = A.build(P, 0.15, np.array([-5.0, -6, 0]), np.array([45.0, 6, 6]))
        assert abs(floor) < 0.15 and rect[1] - rect[0] > 40 and rect[3] - rect[2] > 8
        cl.place([Box((8.0, 0.0, 1.5), (0.25, 0.25, 1.5))])
        d = cl.render(np.array([2.0, 0.0, 1.5]), np.eye(3))['depth_left']
        assert abs(float(np.median(d[118:123, 158:163])) - (8.0 - 0.25 - 2.0 - 0.1)) < 0.05
    finally:
        cl.close()
        fake.stop()
    ar = A.Arena(sdf, floor, meta={'rect': rect})
    e = A.arena_env(ar, 0, kind='test')
    w = e.world
    b = w.objects[0]
    c = np.array(b.center)
    assert w.clearance(c + [b.half[0] + 0.5, 0, 0]) == pytest.approx(0.5, abs=1e-6)
    assert Field(w, ground=False)(np.array([[c[0], c[1] + b.half[1] + 0.3, floor + 0.2]]))[0] == pytest.approx(0.3, abs=1e-6)
    assert Field(w, ground=True)(np.array([[c[0] + 3, c[1] + 3, floor + 0.2]]))[0] <= 0.2 + 1e-6
    from tinyuav.nav.expert import RQ, CostToGo
    r = CostToGo(e, w_line=0.5).ref_from(np.asarray(e.start, float), 5.0)
    assert r is not None and Field(w, ground=False)(r.pos(np.arange(0.3, r.length, 0.1))).min() > RQ


def test_player_reset_and_keepalive():
    """An idle player resets to its menu; keepalive prevents it, and the next pose reloads it otherwise."""
    import time
    from tinyuav.sim.render import Box
    env = EV.TEST['forest'](1000)
    for keepalive, expect_resets in ((None, True), (0.3, False)):
        pub, sub = next(PORTS), next(PORTS)
        fake = FakeUnity(env.world, pub, sub, idle_reset_s=1.5).start()
        cl = B.FlightmareClient(scene_id=2, pub_port=pub, sub_port=sub, n_objects=2, timeout_s=20)
        try:
            cl.connect(keepalive=keepalive)
            cl.place([Box((6.0, 0.0, 1.5), (0.3, 0.3, 1.5))])
            d0 = cl.render(np.array([0.0, 0.0, 1.5]), np.eye(3))['depth_left']
            time.sleep(3.5)
            d1 = cl.render(np.array([0.0, 0.0, 1.5]), np.eye(3))['depth_left']
        finally:
            cl.close()
            fake.stop()
        assert np.allclose(d0, d1, atol=1e-3)
        assert (fake.resets > 0) == expect_resets and (cl.reloads > 0) == expect_resets


def test_camera_tilt_matches_our_rig(unity):
    """set_tilt(30) renders what our tilted rig sees."""
    from tinyuav.sim.stereo import StereoRig, render_stereo
    cl, env = unity
    cl.set_tilt(30.0)
    p = np.array([0.0, 0.0, 1.5])
    o = cl.render(p, np.eye(3))
    gl, gr, d = render_stereo(env.world, StereoRig(tilt_up_deg=30.0), p, 0.0, rng=np.random.default_rng(0))
    m = (d < 15) & (o['depth_left'] < 15)
    assert m.mean() > 0.2 and np.median(np.abs(o['depth_left'][m] - d[m])) < 0.05
    cl.set_tilt(0.0)


def test_reactive_pilot_flies_on_flightmare_frames(unity):
    from tinyuav.nav.rollout import Config, Rollout
    from tinyuav.flightmare.run import FlightmareObserver
    cl, env = unity
    R = Rollout(env, Config(v_des=3.0, seed=0, t_max=2.0), 'reactive')
    R.observer = FlightmareObserver(cl, R.cfg.rig)
    R.KEEP, R.EDGE = True, True
    r, _, log = R.run()
    assert not r['crashed'] and r['decide_ms'] is not None and len(R._perceived.cells()) > 0


def test_diffphys_pilot_flies_on_flightmare_frames(unity, tmp_path):
    pytest.importorskip('tensorflow')
    from tinyuav.nav import diffphys as DP
    from tinyuav.nav.rollout import Config, Rollout
    from tinyuav.flightmare.run import FlightmareObserver
    cl, env = unity
    path = str(tmp_path / 'dp.keras')
    DP.build_policy(hidden=16).save(path)
    R = Rollout(env, Config(v_des=3.0, seed=0, t_max=1.0), 'diffphys', path)
    R.observer = FlightmareObserver(cl, R.cfg.rig)
    r, _, log = R.run()
    assert 'error' not in r and r['decide_ms'] is not None and len(log) > 5


def test_paint_spheres_in_front_of_unity_depth_only():
    """A balloon 5 m ahead is painted where Unity sees farther, not behind a wall."""
    from tinyuav.flightmare.follow import paint_spheres
    H, W, f = 240, 320, 160.0
    img = np.full((H, W, 3), 128, np.uint8)
    dep = np.full((H, W), 20.0, np.float32)
    dep[:, :150] = 3.0                                      # wall on the left half, 3 m
    n = paint_spheres(img, dep, np.zeros(3), np.eye(3), f, [((5.0, 0.0, 0.0), 0.3, (0.85, 0.07, 0.07))])
    assert n > 0 and dep[120, 160] == pytest.approx(4.7, abs=0.02)
    assert img[120, 162, 0] > 90 and img[120, 162, 0] > 5 * img[120, 162, 1]
    assert (img[:, :150] == 128).all()
    assert paint_spheres(img, dep, np.zeros(3), np.eye(3), f, [((-5.0, 0.0, 0.0), 0.3, (1, 0, 0))]) == 0


def _arena_file(tmp_path, pub, sub):
    from tinyuav.flightmare import arena as A
    from tinyuav.flightmare.scan import scan
    cl = B.FlightmareClient(scene_id=0, pub_port=pub, sub_port=sub, stereo=False, timeout_s=20)
    try:
        cl.connect()
        lo, hi = np.array([-5.0, -6, 0]), np.array([45.0, 6, 6])
        P, _, _ = scan(cl, (lo + hi) / 2, hi - lo, 0.15, step=5.0, heights=(1.5,), yaws=4, pitches=(0.0,),
                       log=lambda *_: None)
        sdf, floor, rect = A.build(P, 0.15, lo, hi)
        A.save(str(tmp_path / 'arena_warehouse.npz'), P, 0.15, floor, rect, lo, hi, {'scene': 'warehouse'})
    finally:
        cl.close()
    return A.load(str(tmp_path / 'arena_warehouse.npz'))


def test_eval_and_sgm_video_in_the_arena(tmp_path):
    import json
    from tinyuav.flightmare import run as RUN
    pub, sub = next(PORTS), next(PORTS)
    fake = _arena_player(pub, sub)
    _arena_file(tmp_path, pub, sub)
    ports = ['--results', str(tmp_path), '--pub-port', str(pub), '--sub-port', str(sub), '--no-vga']
    try:
        RUN.main(['eval', '--pilots', 'gbl', '--n', '1', '--v', '4'] + ports)
        RUN.main(['video', '--pilot', 'gbl', '--v', '4', '--view', 'sgm'] + ports)
    finally:
        fake.stop()
    r = json.load(open(next(tmp_path.glob('eval_warehouse_pilots*.json'))))['gbl_v4'][0]
    assert 'success' in r
    assert list((tmp_path / 'video').glob('gbl_warehouse_env0_v4_sgm.mp4'))


def test_follow_a_balloon_in_flightmare(tmp_path):
    """Balloon mission on the fake player's frames: the privileged pilot keeps the lock."""
    import json
    from tinyuav.sim.geometry import Field
    from tinyuav.flightmare import run as RUN
    from tinyuav.flightmare.follow import balloon_arena_env
    pub, sub = next(PORTS), next(PORTS)
    fake = _arena_player(pub, sub)
    ar = _arena_file(tmp_path, pub, sub)
    e = balloon_arena_env(ar, 3001, density=1 / 16, speed=4.0)
    Q = e.balloons[0].path.pts
    assert Field(e.world, ground=False)(Q).min() > 0.4 and (Q[:, 2] - ar.floor).min() >= 1.2 - 1e-6
    assert len(e.world.objects) > 20
    ports = ['--results', str(tmp_path), '--pub-port', str(pub), '--sub-port', str(sub)]
    try:
        RUN.main(['follow', '--pilots', 'gbl', '--n', '1', '--seed0', '3001', '--speed', '4', '--density', '0.03'] + ports)
        RUN.main(['follow-video', '--pilot', 'gbl', '--seed', '3001', '--speed', '4', '--density', '0.03'] + ports)
    finally:
        fake.stop()
    r = json.load(open(tmp_path / 'balloon_warehouse_follow.json'))['gbl'][0]
    assert not r['crashed'] and r['locked_frac'] > 0.8 and r['true_frac'] > 0.9
    assert list((tmp_path / 'video').glob('balloon_warehouse_gbl_d0.03_v4_s3001_perception.mp4'))
