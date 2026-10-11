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
    g, sgm, gt, rgb = obs(e.start, 0.0, 0.0, 0.0)
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
        P, hits, st = scan(cl, origin, size, 0.15, step=4.0, heights=(1.5,), yaws=4, pitches=(0.0,), log=lambda *_: None)
    finally:
        cl.close()
        fake.stop()
    assert st['frames'] == 8 * 4 * 4 and len(hits) == len(P) and hits.min() >= 2
    w = CloudWorld(P, 0.15, inflate=0.075, hits=hits)
    rng = np.random.default_rng(0)
    Q = np.c_[rng.uniform(4, 28, 400), rng.uniform(-6, 6, 400), rng.uniform(0.5, 3, 400)]
    from nanouav.agile.geometry import Field
    true = Field(env.world)(Q)
    near = true < 3.0
    assert np.median(np.abs(w.distance(Q[near]) - true[near])) < 0.15     # the map matches the world
    L = find_lines(w, n=1, length=20.0, seed=0)
    assert L and Rollout(line_env(w, L[0]), Config(v_des=4.0, seed=0), 'gbl').run()[0]['success']


def test_privileged_pilot_video_has_camera_frames_and_crash_info(unity, tmp_path):
    """r22 video for gbl: the trace records the Flightmare frames; a crash names the map point hit."""
    from nanouav.agile.rollout import Config, Rollout
    from nanouav.flightmare.cloud import line_env
    from nanouav.flightmare.run import FlightmareObserver, crash_info
    cl, env = unity
    origin, size = np.array([16.0, 0.0, 3.0]), np.array([32.0, 16.0, 6.0])
    P, _ = cl.cloud_points(origin, size, 0.2, path=str(tmp_path), name='v', timeout_s=60)
    w = CloudWorld(P, 0.2)
    L = find_lines(w, n=1, length=20.0, seed=0)
    R = Rollout(line_env(w, L[0]), Config(v_des=4.0, seed=0), 'gbl')
    R.observer = FlightmareObserver(cl, R.cfg.rig)
    R.trace, R.trace_depth, R.record_cam = [], True, True
    r, _, log = R.run()
    assert r['success'] and len(R.trace) > 20
    d = R.trace[5]
    assert d['gray'].shape == (240, 320) and d['depth'].dtype == np.uint16 and len(d['W']) == 1 and d['costs'] == []
    tree = next(o for o in env.world.objects if 4 < o.cx < 28 and abs(o.cy) < 6)
    c = crash_info(w, np.array([tree.cx + tree.r + 0.1, tree.cy, 1.5]))
    assert c['dist'] < 0.4 and 0.5 < c['above_ground'] < 2.5


# ── arena: scanned static scene + spawned trunks; DAgger in Flightmare ──────
def _arena_player(pub, sub):
    from nanouav.flightmare.fake import arena_world
    return FakeUnity(arena_world(0, x=(-5.0, 45.0), y=(-6.0, 6.0), height=6.0), pub, sub).start()


def test_spawned_boxes_render_and_map(tmp_path):
    from nanouav.agile.geometry import Field
    from nanouav.flightmare import arena as A
    from nanouav.flightmare.scan import scan
    from nanouav.scene.renderer import Box
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
        assert abs(float(np.median(d[118:123, 158:163])) - (8.0 - 0.25 - 2.0 - 0.1)) < 0.05   # the box is there
    finally:
        cl.close()
        fake.stop()
    ar = A.Arena(sdf, floor, meta={'rect': rect})
    e = A.arena_env(ar, 0, kind='test')
    w = e.world
    b = w.objects[0]
    c = np.array(b.center)
    assert w.clearance(c + [b.half[0] + 0.5, 0, 0]) == pytest.approx(0.5, abs=1e-6)          # exact trunks
    assert Field(w, ground=False)(np.array([[c[0], c[1] + b.half[1] + 0.3, floor + 0.2]]))[0] == pytest.approx(0.3, abs=1e-6)
    assert Field(w, ground=True)(np.array([[c[0] + 3, c[1] + 3, floor + 0.2]]))[0] <= 0.2 + 1e-6   # the floor
    from nanouav.agile.expert import RQ, CostToGo                     # R23: replanned references in an arena
    r = CostToGo(e, w_line=0.5).ref_from(np.asarray(e.start, float), 5.0)
    assert r is not None and Field(w, ground=False)(r.pos(np.arange(0.3, r.length, 0.1))).min() > RQ


def test_dagger_in_flightmare_one_round(tmp_path):
    """Two arena environments flown with Unity frames, labels in worker processes, one training round."""
    pytest.importorskip('tensorflow')
    from nanouav.agile.stereo import StereoRig
    from nanouav.flightmare import arena as A
    from nanouav.flightmare import dagger as FD
    from nanouav.flightmare.scan import scan
    pub, sub = next(PORTS), next(PORTS)
    fake = _arena_player(pub, sub)
    cl = B.FlightmareClient(scene_id=0, pub_port=pub, sub_port=sub, n_objects=120, timeout_s=20)
    try:
        cl.connect()
        lo, hi = np.array([-5.0, -6, 0]), np.array([45.0, 6, 6])
        P, _, _ = scan(cl, (lo + hi) / 2, hi - lo, 0.15, step=5.0, heights=(1.5, 3.0), yaws=4, pitches=(0.0, 30.0),
                       log=lambda *_: None)
        sdf, floor, rect = A.build(P, 0.15, lo, hi)
        fn = str(tmp_path / 'arena_t.npz')
        A.save(fn, P, 0.15, floor, rect, lo, hi, {'scene': 't'})
        m = FD.run(cl, fn, str(tmp_path), StereoRig(), n_envs=2, per_round=2, workers=2, epochs=1, batch=4,
                   v_min=3.0, v_max=3.0, init_model=None, xi_start=0.0, tag='t', label_ref='ctg',
                   log=lambda *_: None)
    finally:
        cl.close()
        fake.stop()
    import json, os
    st = json.load(open(tmp_path / 'dagger_t.json'))['rounds'][0]
    assert st['samples'] > 50 and st['success'] == 2 and os.path.exists(m)
    assert json.load(open(m + '.json')) == {'omega_ref': 'line', 'label_ref': 'ctg'}   # R23 sidecar
    z = np.load(tmp_path / 'data_t' / 'env_0000.npz')
    assert z['depth'].shape[1:] == (120, 160) and z['trajs'].shape[1:] == (3, 10, 3) and z['esdf'].shape[1:] == (64, 64)
    assert 1.0 < float(np.median(z['z'])) < 3.0                           # height above the floor


def test_player_reset_and_keepalive():
    """The player goes back to its menu after an idle spell; the client keeps it alive, and if it reset
    anyway the next pose (which carries the settings) reloads it."""
    import time
    from nanouav.scene.renderer import Box
    env = EV.TEST['forest'](1000)
    for keepalive, expect_resets in ((None, True), (0.3, False)):
        pub, sub = next(PORTS), next(PORTS)
        fake = FakeUnity(env.world, pub, sub, idle_reset_s=1.5).start()
        cl = B.FlightmareClient(scene_id=2, pub_port=pub, sub_port=sub, n_objects=2, timeout_s=20)
        try:
            cl.connect(keepalive=keepalive)
            cl.place([Box((6.0, 0.0, 1.5), (0.3, 0.3, 1.5))])
            d0 = cl.render(np.array([0.0, 0.0, 1.5]), np.eye(3))['depth_left']
            time.sleep(3.5)                                         # e.g. a training round
            d1 = cl.render(np.array([0.0, 0.0, 1.5]), np.eye(3))['depth_left']
        finally:
            cl.close()
            fake.stop()
        assert np.allclose(d0, d1, atol=1e-3)                       # same view, the box still there
        assert (fake.resets > 0) == expect_resets and (cl.reloads > 0) == expect_resets


def test_camera_tilt_matches_our_rig(unity):
    """set_tilt(30) (the paper's camera at ≥ 7 m/s) renders what our tilted rig sees."""
    from nanouav.agile.stereo import StereoRig, render_stereo
    cl, env = unity
    cl.set_tilt(30.0)
    p = np.array([0.0, 0.0, 1.5])
    o = cl.render(p, np.eye(3))
    gl, gr, d = render_stereo(env.world, StereoRig(tilt_up_deg=30.0), p, 0.0, rng=np.random.default_rng(0))
    m = (d < 15) & (o['depth_left'] < 15)
    assert m.mean() > 0.2 and np.median(np.abs(o['depth_left'][m] - d[m])) < 0.05
    cl.set_tilt(0.0)


def test_geometry_check_command(tmp_path):
    """r22 check: off-axis boxes appear where our camera model puts them, level and tilted."""
    import json
    from nanouav.flightmare import arena as A
    from nanouav.flightmare import run as RUN
    from nanouav.flightmare.scan import scan
    pub, sub = next(PORTS), next(PORTS)
    fake = _arena_player(pub, sub)
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
    try:
        RUN.main(['check', '--scene', 'warehouse', '--results', str(tmp_path), '--no-vga',
                  '--pub-port', str(pub), '--sub-port', str(sub)])
    finally:
        fake.stop()
    rows = json.load(open(tmp_path / 'check_geometry.json'))
    boxes = [r for r in rows if 'box' in r]
    assert len(boxes) == 6 and all(r['ok'] for r in boxes)
    assert all(abs(r['u_meas'] - r['u_exp']) < 4 and abs(r['v_meas'] - r['v_exp']) < 6 for r in boxes)
