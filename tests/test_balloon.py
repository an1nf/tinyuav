"""R20 · follow a balloon (nanouav.follow.balloon*): RGB rendering, color-blob detector, 3D range from the
apparent size, 3D target filter, camera-motion compensation, balloon worlds and a short closed loop."""
import math

import numpy as np
import pytest

from nanouav.agile.stereo import StereoRig, observe_full, render_stereo
from nanouav.follow.balloon import balloon_env
from nanouav.follow.blob import BlobDetector
from nanouav.follow.estimate import TargetFilter, blob_to_world, project, rotation_warp

RIG = StereoRig(width=160, height=120, num_disp=32)


def _behind(env, dist, dz=0.0):
    c = np.array(env.walkers[0].body.center)
    return c, (c - [dist, 0.0, dz], 0.0, 0.0, 0.0)


def test_rgb_keeps_the_gray_image_and_colors_the_balloon():
    e = balloon_env(1, density=0.0, n_distractors=0)
    c, pose = _behind(e, 4.0)
    g0, _, d0 = render_stereo(e.render_world(), RIG, *pose, rng=np.random.default_rng(3))
    g1, _, d1, rgb = render_stereo(e.render_world(), RIG, *pose, rng=np.random.default_rng(3), rgb=True)
    assert (g0 == g1).all() and (d0 == d1).all()
    (u, v), = project(RIG, pose, c[None])[0]
    r, g, b = rgb[int(v), int(u)].astype(int)
    assert r > 2 * g and r > 2 * b                     # the balloon is red in the color image


@pytest.mark.parametrize('dist', [4.0, 8.0])
def test_blob_detector_finds_balloon_and_range_from_size(dist):
    e = balloon_env(2, density=0.0, n_distractors=0)
    c, pose = _behind(e, dist, dz=0.2)
    _, sgm, _, rgb = observe_full(e.render_world(), RIG, *pose, rng=np.random.default_rng(0), rgb=True)
    D = BlobDetector('red').detect(rgb)
    assert len(D) == 1
    q, src = blob_to_world(RIG, pose, sgm, D[0]['box'], e.walkers[0].radius)
    assert src == 'size' and np.linalg.norm(q - c) < 0.1 * dist


def test_blob_split_by_a_trunk_is_one_detection():
    img = np.full((120, 160, 3), (90, 110, 60), np.uint8)
    yy, xx = np.mgrid[:120, :160]
    img[(xx - 80) ** 2 + (yy - 60) ** 2 < 15 ** 2] = (210, 20, 20)
    img[:, 77:83] = (70, 50, 40)                        # a thin trunk in front of the balloon
    D = BlobDetector('red').detect(img)
    assert len(D) == 1
    x1, y1, x2, y2 = D[0]['box']
    assert x2 - x1 == pytest.approx(30, abs=3) and y2 - y1 == pytest.approx(30, abs=3)
    assert BlobDetector('blue').detect(img) == []


def test_two_overlapping_balloons_are_split_and_range_uses_the_height():
    img = np.full((120, 160, 3), (90, 110, 60), np.uint8)
    yy, xx = np.mgrid[:120, :160]
    for cx, cy, r in ((70, 60, 12), (88, 62, 10)):
        img[(xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2] = (210, 20, 20)
    D = sorted(BlobDetector('red').detect(img), key=lambda d: d['box'][0])
    assert len(D) == 2
    assert [d['box'][3] - d['box'][1] for d in D] == pytest.approx([23, 19], abs=2)
    pose = (np.zeros(3), 0.0, 0.0, 0.0)
    q, _ = blob_to_world(RIG, pose, np.zeros((120, 160)), (40, 50, 80, 70), 0.3)   # wide box: two merged
    assert q[0] == pytest.approx(RIG.focal * 0.6 / 20, rel=0.05)


def test_target_filter_3d_and_rotation_warp():
    flt = TargetFilter(dims=3)
    for k in range(30):
        t = 0.1 * k
        flt.update(t, np.array([1.0 + 1.5 * t, 2.0, 1.5 + 0.5 * t]), 0.1)
    assert flt.vel == pytest.approx([1.5, 0.0, 0.5], abs=0.1)
    assert flt.gate(np.array([5.65, 2.0, 3.05]), 3.1) and not flt.gate(np.array([5.65, 2.0, 8.0]), 3.1)
    p = np.array([0.0, 0.0, 1.5])
    a, b = (p, 0.0, 0.0, 0.0), (p, 0.2, 0.05, -0.03)    # yaw, pitch, roll change between frames
    far = np.array([[60.0, 8.0, 4.0]])
    (u0, v0), = project(RIG, a, far)[0]
    (u1, v1), = project(RIG, b, far)[0]
    assert rotation_warp(RIG, a, b)(u0, v0) == pytest.approx((u1, v1), abs=0.5)


def test_balloon_env_paths_are_free_and_within_heights():
    for s in range(4):
        e = balloon_env(s, density=1 / 25)
        for k, w in enumerate(e.walkers):
            P = w.path.pts
            lo, hi = (1.2, 2.6) if k == 0 else (3.3, 4.0)       # the others float above the drone's band
            assert P[:, 2].min() >= lo - 1e-6 and P[:, 2].max() <= hi + 1e-6
            assert min(e.world.clearance(q) for q in P[::5]) > w.radius
        # corridor from the drone start to the balloon start
        assert min(e.world.clearance(np.array([x, 0.0, 1.5])) for x in np.linspace(-3, 6, 19)) > 1.0


@pytest.mark.slow
def test_balloon_mission_short_closed_loop():
    from nanouav.agile.rollout import Config, Rollout
    from nanouav.follow.balloon_mission import BalloonMission
    e = balloon_env(2, density=0.0, n_distractors=0, speed=2.0)
    M = BalloonMission(e, seed=0)
    r, _, log = Rollout(e, Config(v_des=2.0, cam_hz=5.0, seed=0, t_max=8.0), 'gbl', mission=M).run()
    assert not r['crashed'] and r['locked_frac'] > 0.6 and r['true_frac'] > 0.95
    p = np.array(log[-1][1:4])
    tgt = e.walkers[0].pos(log[-1][0])
    assert math.hypot(*(p[:2] - tgt[:2])) == pytest.approx(4.0, abs=1.5)
    # height: the privileged pilot flies the balloon's own trail (its height where it was ~4 m ago)
    assert 1.0 <= p[2] <= 3.0 and abs(p[2] - M.trail.pos(M.trail.closest(p)[0])[2]) < 0.4


def test_lost_balloon_yaw_and_gate_options():
    """0.27.5: target lost → by default the camera looks at the filter's prediction; with yaw_ext it looks at
    the last estimate moved on at most yaw_ext s, and with gate_v the re-lock gate grows with the target speed."""
    from nanouav.follow.balloon_mission import BalloonMission
    from nanouav.follow.estimate import TargetFilter
    e = balloon_env(2, density=0.0, n_distractors=0, speed=2.0)
    yaws = {}
    for ext in (None, 0.25):
        M = BalloonMission(e, seed=0)
        M.cmc, M.yaw_ext, M.gate_v = False, ext, 1.0
        M._detect, M._image = (lambda cam: []), (lambda cam: np.zeros((8, 8, 3), np.uint8))
        M.lock.track_id, M.lock.state = 1, 'locked'
        M.flt = TargetFilter(q_acc=2.0, dims=3)
        M.flt.x, M.flt.P, M.flt.t = np.r_[10.0, 0, 2, 5, 0, 0], np.eye(6) * 0.01, 1.0
        M.t_last_seen, M._seen = 1.0, (1.0, M.flt.pos.copy(), M.flt.vel.copy())
        F = M.frame(2.0, np.array([0.0, 5.0, 2.0]), np.zeros(3), 0.0, {'pose': None})
        yaws[ext] = math.degrees(F['yaw'])
        assert M.lock.gate_growth == pytest.approx(6.0)
    assert yaws[None] == pytest.approx(math.degrees(math.atan2(-5, 15)), abs=0.5)        # prediction at t = 2 s
    assert yaws[0.25] == pytest.approx(math.degrees(math.atan2(-5, 11.25)), abs=0.5)     # 10 m + 5 m/s · 0.25 s


def test_diffphys_pilot_follows_the_balloon_law(tmp_path):
    """R26: in a follow mission the diffphys pilot gets the follow law's direction and speed (omega × v_des),
    not the env goal, and faces the balloon (the mission's yaw)."""
    pytest.importorskip('tensorflow')
    from nanouav.agile import diffphys as DP
    from nanouav.agile.rollout import Config, Rollout
    from nanouav.follow.balloon_mission import BalloonMission
    path = str(tmp_path / 'dp.keras')
    DP.build_policy(hidden=8).save(path)
    e = balloon_env(2, density=0.0, n_distractors=0, speed=2.0)
    M = BalloonMission(e, seed=0)
    R = Rollout(e, Config(v_des=2.0, cam_hz=5.0, seed=0, t_max=3.0, rig=RIG), 'diffphys', path, mission=M)
    R.dp_trace = []
    seen = []
    orig = DP.DiffPhysPilot.__call__
    DP.DiffPhysPilot.__call__ = lambda self, d, p, v, Rm, g, yaw, vd: seen.append((np.asarray(g), vd)) or \
        orig(self, d, p, v, Rm, g, yaw, vd)
    try:
        r, _, log = R.run()
    finally:
        DP.DiffPhysPilot.__call__ = orig
    assert 'error' not in r and seen
    g, vd = seen[-1]
    assert np.linalg.norm(g) == pytest.approx(1.0, abs=1e-3) and 0.3 <= vd <= M.v_max


def test_fast_target_options_of_r20_sim():
    """0.27.5: r20 sim --fast-target sets the four lost-target options; each one can be given alone."""
    import argparse
    from nanouav.follow.run_balloon import FAST_TARGET, _lost_opts
    ns = lambda **k: argparse.Namespace(**{'fast_target': False, 'q_acc': None, 'yaw_ext': None, 'gate_v': None,
                                          'relock_free': False, **k})
    assert _lost_opts(ns()) == {}
    assert _lost_opts(ns(fast_target=True)) == FAST_TARGET
    assert _lost_opts(ns(fast_target=True, q_acc=3.0))['q_acc'] == 3.0
    assert _lost_opts(ns(yaw_ext=0.5)) == {'yaw_ext': 0.5}


def test_turn_predict_keeps_turning_then_goes_straight():
    """0.27.6 (--turn-pred): a target at 5 m/s turning at 90°/s; w = 0 is constant velocity."""
    from nanouav.follow.mission import turn_predict
    q, v = np.zeros(3), np.array([5.0, 0.0, 0.0])
    assert turn_predict(q, v, 0.0, 1.0) == pytest.approx([5.0, 0, 0])
    r = 5.0 / (math.pi / 2)                                    # radius of the turn
    assert turn_predict(q, v, math.pi / 2, 1.0) == pytest.approx([r, r, 0], abs=1e-6)       # a quarter circle
    assert turn_predict(q, v, math.pi / 2, 2.0) == pytest.approx([r, r + 5.0, 0], abs=1e-6)  # then straight


def test_turn_rate_from_the_filter_velocities():
    """0.27.6: the turn rate is the slope of the heading of the last 0.6 s of filter velocities (0 when slow)."""
    from nanouav.follow.balloon_mission import BalloonMission
    M = BalloonMission(balloon_env(2, density=0.0, n_distractors=0, speed=2.0), seed=0)
    w = math.radians(45)
    M._hist = [(t, 4.0 * np.array([math.cos(w * t), math.sin(w * t)])) for t in np.arange(0.0, 0.6, 0.1)]
    assert M._turn_rate() == pytest.approx(w, rel=1e-3)
    M._hist = [(t, 0.5 * h) for t, h in [(t, v / 4.0) for t, v in M._hist]]        # 0.5 m/s: too slow to tell
    assert M._turn_rate() == 0.0
