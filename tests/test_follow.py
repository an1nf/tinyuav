import numpy as np
import pytest

from tinyuav.follow.estimate import TargetFilter, backproject, project
from tinyuav.sim.stereo import StereoRig
from tinyuav.follow.tracker import TargetLock, Tracker, appearance, iou, similarity

RIG = StereoRig(width=160, height=120, num_disp=32)


def test_project_backproject_round_trip():
    pose = (np.array([0.0, 0.0, 2.0]), 0.3, 0.05, -0.02)
    P = np.array([[5.0, 1.0, 0.5], [3.0, 2.0, 1.8]])
    uv, X = project(RIG, pose, P)
    for k in range(2):
        assert backproject(RIG, pose, uv[k], X[k]) == pytest.approx(P[k], abs=1e-9)


def _det(box, gt=0, s=0.9):
    return {'box': box, 'score': s, 'gt': gt}


def test_tracker_keeps_identity_and_lock_survives_crossing():
    img = np.zeros((120, 160), np.uint8)
    img[:, :60] = 200
    img[:, 60:] = 40
    trk, lock, flt = Tracker(min_hits=1), TargetLock(), TargetFilter()
    pos = lambda tr: np.array([(tr.box[0] + tr.box[2]) / 2 / 10, 0.0, 0.0])
    ids = []
    for k in range(10):                    # target moves right, a darker distractor left; they cross
        a = 10 + 4 * k
        b = 120 - 4 * k
        img2 = img.copy()
        img2[20:100, a:a + 20] = 220
        img2[20:100, b:b + 20] = 30
        dets = [_det((a, 20, a + 20, 100), 0)]
        if abs(a - b) > 14:                # hidden while overlapping
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
