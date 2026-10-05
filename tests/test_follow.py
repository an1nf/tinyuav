"""R19 · person following (nanouav.follow): walkers, projection, simulated detector, tracker + target lock,
3D estimate and the follow mission in closed loop (short)."""
import math

import numpy as np
import pytest

from nanouav.agile.stereo import StereoRig, observe_full
from nanouav.follow.detector import SimDetector
from nanouav.follow.estimate import TargetFilter, backproject, box_to_world, project
from nanouav.follow.people import HEIGHT, people_env
from nanouav.follow.tracker import TargetLock, Tracker, appearance, iou, similarity

RIG = StereoRig(width=160, height=120, num_disp=32)


def test_project_backproject_round_trip():
    pose = (np.array([0.0, 0.0, 2.0]), 0.3, 0.05, -0.02)
    P = np.array([[5.0, 1.0, 0.5], [3.0, 2.0, 1.8]])
    uv, X = project(RIG, pose, P)
    for k in range(2):
        assert backproject(RIG, pose, uv[k], X[k]) == pytest.approx(P[k], abs=1e-9)


def test_walker_moves_on_its_path_and_env_is_valid():
    e = people_env(1, n_distractors=1)
    w = e.walkers[0]
    assert w.pos(0.0)[:2] == pytest.approx(w.path.pos(0.0)[:2])
    w.move(w.t0 + 2.0)
    assert np.hypot(w.body.cx - w.path.pos(0)[0], w.body.cy - w.path.pos(0)[1]) == pytest.approx(2 * w.speed, rel=0.05)
    assert len(e.render_world().objects) == len(e.world.objects) + 2 * len(e.walkers)
    assert w.clearance(np.array([w.body.cx, w.body.cy, 1.0])) < 0          # inside the body


def test_sim_detector_box_matches_rendered_person():
    e = people_env(4, n_distractors=0, density=0.0)
    w = e.walkers[0]
    w.move(0.0)
    pose = (np.array([w.body.cx - 4.0, w.body.cy, 1.5]), 0.0, 0.0, 0.0)
    gray, depth, dgt = observe_full(e.render_world(), RIG, *pose, rng=np.random.default_rng(0))
    det = SimDetector(jitter=0.0, fp_rate=0.0, p_max=1.0, seed=0)
    D = det.detect({'rig': RIG, 'pose': pose, 'depth_gt': dgt, 'depth': depth, 'gray': gray}, e.walkers)
    assert len(D) == 1 and D[0]['gt'] == 0
    x1, y1, x2, y2 = D[0]['box']
    _, dg0 = observe_full(e.world, RIG, *pose, rng=np.random.default_rng(0))[1:]
    person = np.abs(dgt - dg0) > 0.3                                     # pixels that the person changes
    ys, xs = np.nonzero(person)
    assert x1 - 2 <= xs.min() and xs.max() <= x2 + 2 and y1 - 2 <= ys.min() and ys.max() <= y2 + 2
    assert (y2 - y1) == pytest.approx(RIG.focal * HEIGHT / 3.9, rel=0.15)   # pinhole height at ~3.9 m
    q, src = box_to_world(RIG, pose, depth, D[0]['box'])
    assert q is not None and np.hypot(*(q[:2] - [w.body.cx, w.body.cy])) < 0.5


def _det(box, gt=0, s=0.9):
    return {'box': box, 'score': s, 'gt': gt}


def test_tracker_keeps_identity_and_lock_survives_crossing():
    img = np.zeros((120, 160), np.uint8)
    img[:, :60] = 200                      # target wears "light" clothes on the left side of the image
    img[:, 60:] = 40
    trk, lock, flt = Tracker(min_hits=1), TargetLock(), TargetFilter()
    pos = lambda tr: np.array([(tr.box[0] + tr.box[2]) / 2 / 10, 0.0, 0.0])
    ids = []
    for k in range(10):                    # target walks right, distractor (dark) walks left; they cross
        a = 10 + 4 * k
        b = 120 - 4 * k
        img2 = img.copy()
        img2[20:100, a:a + 20] = 220
        img2[20:100, b:b + 20] = 30
        dets = [_det((a, 20, a + 20, 100), 0)]
        if abs(a - b) > 14:                # while overlapping, the distractor is hidden
            dets.append(_det((b, 20, b + 20, 100), 1))
        tracks = trk.update(dets, img2)
        if k == 0:
            tr = lock.select_initial(tracks, (160, 120), prefer=(20, 60))
        else:
            tr, _ = lock.update(tracks, pos, flt, 0.1 * k)
        if tr is not None:
            flt.update(0.1 * k, pos(tr), 0.2)
            ids.append(tr.gt)
    assert ids and all(g == 0 for g in ids)
    assert iou([(0, 0, 10, 10)], [(5, 0, 15, 10)])[0, 0] == pytest.approx(1 / 3)
    f = appearance(img, (0, 0, 40, 100))
    assert similarity(f, f) == pytest.approx(1.0) and similarity(f, appearance(img, (100, 0, 140, 100))) < 0.2


def test_target_filter_tracks_constant_velocity():
    flt = TargetFilter()
    for k in range(30):
        t = 0.1 * k
        flt.update(t, np.array([1.0 + 1.5 * t, 2.0, 0.0]), 0.1)
    assert flt.vel[:2] == pytest.approx([1.5, 0.0], abs=0.1)
    assert flt.gate(np.array([1.0 + 1.5 * 3.1, 2.0]), 3.1) and not flt.gate(np.array([20.0, 2.0]), 3.1)


@pytest.mark.slow
def test_follow_mission_short_closed_loop():
    from nanouav.agile.rollout import Config, Rollout
    from nanouav.follow.mission import FollowMission
    e = people_env(0, n_distractors=0, density=0.0)
    M = FollowMission(e, seed=0)
    r, _, log = Rollout(e, Config(v_des=2.0, cam_hz=10.0, seed=0, t_max=8.0), 'gbl', mission=M).run()
    assert not r['crashed'] and r['locked_frac'] > 0.6 and r['true_frac'] > 0.95        # includes acquisition
    p = np.array(log[-1][1:3])
    assert math.hypot(*(p - e.walkers[0].pos(log[-1][0])[:2])) == pytest.approx(4.0, abs=1.5)
