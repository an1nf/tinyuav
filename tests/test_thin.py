"""Thin structures (R14): training-data option, thin-pole forest scenario, model lookup guards."""
import numpy as np
import pytest

from nanouav.scene.renderer import Box, Cylinder, render_scene, sample_world, add_thin_structures
from nanouav.sim import scenarios as SC


def test_thin_prob_zero_keeps_the_original_dataset():
    for seed in (0, 7, 123):
        g0, d0 = render_scene(seed)
        g1, d1 = render_scene(seed, thin_prob=0.0)
        np.testing.assert_array_equal(g0, g1)
        np.testing.assert_array_equal(d0, d1)


def test_thin_structures_are_thin_and_in_front():
    rng = np.random.default_rng(3)
    for _ in range(20):
        w = sample_world(np.random.default_rng(int(rng.integers(1 << 30))))
        n0 = len(w.objects)
        pos = np.array([0.0, 0.0, 0.7])
        add_thin_structures(w, pos, rng)
        new = w.objects[n0:]
        assert new
        for o in new:
            if isinstance(o, Cylinder):
                assert 0.01 <= o.r <= 0.05 and 0.6 <= o.cx - pos[0] <= 3.0
            else:
                assert isinstance(o, Box) and max(o.half[0], o.half[2]) <= 0.0101 and 0.8 <= o.center[0] <= 3.0


def test_thin_scenes_add_near_depth():
    """With thin_prob = 1 the same seed renders the same base scene plus thin objects in front."""
    changed = 0
    for seed in range(10):
        _, d0 = render_scene(seed)
        _, d1 = render_scene(seed, thin_prob=1.0)
        assert (d1 <= d0 + 1e-6).all()                     # objects are only added, never removed
        changed += int((d1 < d0 - 0.05).any())
    assert changed >= 6                                    # thin objects can hide behind others or miss pixels


def test_bosque_radius_range_and_default_unchanged():
    thin = SC.make('bosque', 0, r_min=0.02, r_max=0.05)
    rs = [o['r'] for o in thin['objects']]
    assert rs and min(rs) >= 0.02 and max(rs) <= 0.05
    base = SC.make('bosque', 0)
    assert [(o['c'], o['z1']) for o in base['objects']] == [(o['c'], o['z1']) for o in thin['objects']]
    assert min(o['r'] for o in base['objects']) >= 0.06


def test_missing_variant_never_falls_back_to_M(tmp_path, monkeypatch):
    from nanouav.perception import cnn
    (tmp_path / 'model.tflite').write_bytes(b'x')
    (tmp_path / 'model_M.tflite').write_bytes(b'x')
    monkeypatch.setenv('NANOUAV_MODELS', str(tmp_path))
    with pytest.raises(FileNotFoundError, match='model_M_ToF_thin'):
        cnn._model_path('M_ToF_thin')
    assert cnn._model_path('M').endswith('model_M.tflite')


def test_fake_mode_needs_no_model_file(tmp_path, monkeypatch):
    from nanouav.perception import cnn
    monkeypatch.setenv('NANOUAV_MODELS', str(tmp_path))
    monkeypatch.setenv('NANOUAV_FAKE_CNN', '1')
    cnn._interps.clear()
    d = np.full((96, 96), 2.0, np.float32)
    assert cnn.predict_depth_norm(np.zeros((96, 96), np.uint8), 'M_ToF_thin', depth_gt=d).shape == (96, 96)
    cnn._interps.clear()
