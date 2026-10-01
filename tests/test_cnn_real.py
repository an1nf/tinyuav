"""Real int8 CNN (needs models/model_*.tflite and a TFLite runtime; skipped otherwise).

Import the models with ``nanouav models import <dir>`` or train them (``nanouav run train``).
"""
import numpy as np
import pytest

from conftest import has_tflite_model

pytestmark = [pytest.mark.tflite,
              pytest.mark.skipif(not has_tflite_model(), reason='no models/model_M.tflite or no TFLite runtime')]


@pytest.fixture
def cnn():
    from nanouav.perception import cnn as m
    m._interps.clear()
    yield m
    m._interps.clear()


def views():
    from nanouav.scene import renderer
    from nanouav.sim import scenarios as SC
    for name in ('bosque', 'interior', 'corredor'):
        for seed in (0, 1):
            s = SC.make(name, seed)
            yield renderer.render_view(SC.to_world(s), (*s['A'], 0.65), s['yaw0'], 0, 0,
                                       rng=np.random.default_rng(seed))


def test_real_mode_and_output_range(cnn):
    assert cnn.is_real_mode('M')
    g, _ = next(views())
    y = cnn.predict_depth_norm(g, 'M')
    assert y.shape == (96, 96) and y.dtype == np.float32
    assert 0.0 <= y.min() and y.max() <= 1.0 and y.std() > 0.01


def test_fake_flag_disables_the_model(cnn, monkeypatch):
    monkeypatch.setenv('NANOUAV_FAKE_CNN', '1')
    assert not cnn.is_real_mode('M')
    g, d = next(views())
    with pytest.raises(RuntimeError):
        cnn.predict_depth_norm(g, 'M')                     # fake mode needs ground truth
    assert cnn.predict_depth_norm(g, 'M', depth_gt=d).shape == (96, 96)


def test_depth_accuracy_on_scenarios(cnn):
    """Mean AbsRel on 6 scenario views (≈0.19 for the shipped M model; generous bound)."""
    from nanouav import config
    ar = [float(np.mean(np.abs(config.norm_to_m(cnn.predict_depth_norm(g, 'M')) - d) / d)) for g, d in views()]
    assert np.mean(ar) < 0.35, ar
