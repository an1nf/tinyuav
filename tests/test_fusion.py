"""CNN ↔ ToF fusion (fuse_tof, fuse_tof2) and sensor models."""
import numpy as np
import pytest

from nanouav import config
from nanouav.perception import tof_calibrator as F
from nanouav.sensors import tof as T


@pytest.fixture
def scene():
    from nanouav.scene import renderer
    from nanouav.sim import scenarios as SC
    s = SC.make('bosque', 1)
    _, d = renderer.render_view(SC.to_world(s), (*s['A'], 0.65), s['yaw0'], 0, 0, rng=np.random.default_rng(0))
    return d, T.tof_8x8(d, rng=np.random.default_rng(1))


def test_tof_shape_and_range(scene):
    d, tof = scene
    assert tof.shape == (8, 8)
    assert np.isfinite(tof).all() and tof.min() > 0 and tof.max() <= F.D_MAX + 1e-6


def test_fuse_tof_only_raises_phantoms(scene):
    d, tof = scene
    cnn = d * 0.3                                        # a CNN that badly underestimates (phantoms)
    out = F.fuse_tof(cnn, tof)
    assert out.shape == cnn.shape
    assert (out >= cnn - 1e-9).all()                     # never lowers depth
    assert (out > cnn + 1e-6).any()


def test_fuse_tof2_bounds(scene):
    d, tof = scene
    cnn = d * 2.0                                        # a CNN that overestimates (dangerous)
    a, b = F.fuse_tof(cnn, tof), F.fuse_tof2(cnn, tof, lo=0.5)
    assert (b <= a + 1e-9).all()                         # only corrects downwards on top of fuse_tof
    assert (b >= 0.5 * a - 1e-9).all()                   # by at most the factor lo
    assert (b < a - 1e-6).any()


def test_fuse_tof2_noop_when_consistent(scene):
    d, tof = scene
    np.testing.assert_allclose(F.fuse_tof2(d, tof), F.fuse_tof(d, tof), atol=0.15 * F.D_MAX)


def test_fuse_switch_by_setting(scene, monkeypatch):
    d, tof = scene
    cnn = d * 2.0
    np.testing.assert_array_equal(F.fuse(cnn, tof), F.fuse_tof(cnn, tof))
    monkeypatch.setenv('NANOUAV_FUSE2', '1')
    np.testing.assert_array_equal(F.fuse(cnn, tof), F.fuse_tof2(cnn, tof))


def test_calibrator_rescales_towards_tof(scene):
    d, tof = scene
    cal = F.ToFCalibrator()
    dn = config.m_to_norm(d * 0.5)                       # scale error ×0.5
    for _ in range(10):
        out = cal.update_and_scale(dn, tof)
    ratio = np.median(out) / np.median(d)
    assert 0.7 < ratio < 1.4


def test_norm_roundtrip():
    x = np.linspace(0.2, config.D_MAX if hasattr(config, 'D_MAX') else 4.0, 50)
    np.testing.assert_allclose(config.norm_to_m(config.m_to_norm(x)), x, rtol=1e-6, atol=1e-6)
