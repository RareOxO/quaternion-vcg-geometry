import pytest
import torch

from qdg.config import validate_config
from qdg.experiments import EXPERIMENTS, experiment_config
from qdg.models import build_model, model_settings

STATS = {"sampling_rate": 500, "signal_length": 5000, "vcg_std": [0.2, 0.3, 0.2]}


def _model(tiny_config, name):
    config = experiment_config(tiny_config, name)
    validate_config(config)
    return build_model(config["model"], STATS)


@pytest.mark.parametrize("name", sorted(EXPERIMENTS))
def test_every_registered_experiment_builds_and_runs(tiny_config, name):
    model = _model(tiny_config, name).eval()
    with torch.inference_mode():
        logits = model(torch.randn(2, 12, 5000))
    assert logits.shape == (2, 5)
    assert torch.isfinite(logits).all()


def test_variant_feature_and_algebra_assignment(tiny_config):
    expected = {
        "M0": ("raw", [], "real", 3),
        "M1": ("first", [20], "real", 4),
        "M2": ("first", [20], "quaternion", 4),
        "M3": ("first", [10, 20, 40, 80], "quaternion", 16),
        "M4": ("first_second", [10, 20, 40, 80], "quaternion", 32),
    }
    for name, (feature, scales, algebra, channels) in expected.items():
        model = _model(tiny_config, name)
        assert model.settings["feature"] == feature
        assert list(model.settings["scales_ms"]) == scales
        assert model.settings["algebra"] == algebra
        assert model.frontend.out_channels == channels


def test_m1_and_m2_receive_identical_input(tiny_config):
    """方案 §7.1: the only difference between M1 and M2 is the interaction rule."""
    ecg = torch.randn(2, 12, 5000)
    m1, m2 = _model(tiny_config, "M1"), _model(tiny_config, "M2")
    torch.testing.assert_close(m1.frontend(ecg), m2.frontend(ecg))


def test_m1_and_m2_encoder_parameter_counts_match(tiny_config):
    """real_width = 2 * quaternions equalizes the conv weight counts exactly.

    Per-channel terms (bias, norm gain) cannot also match, because the quaternion
    encoder carries 4Q channels against the real control's 2Q; those terms are a
    small remainder, so the totals stay close rather than identical (方案 §7.1).
    """
    m1, m2 = _model(tiny_config, "M1"), _model(tiny_config, "M2")
    weights = lambda model: sum(  # noqa: E731
        conv.weight.numel() for block in model.encoder.blocks for conv in block.convs
    )
    assert weights(m1) == weights(m2)
    totals = [sum(p.numel() for p in model.parameters()) for model in (m1, m2)]
    assert abs(totals[0] - totals[1]) / max(totals) < 0.1


def test_m4_extends_m3_with_second_order_channels(tiny_config):
    ecg = torch.randn(2, 12, 5000)
    m3, m4 = _model(tiny_config, "M3"), _model(tiny_config, "M4")
    first_m4 = m4.frontend(ecg).reshape(2, 4, 8, 5000)[:, :, :4]
    torch.testing.assert_close(m3.frontend(ecg), first_m4.flatten(1, 2))


def test_scale_ablations_change_only_the_lag(tiny_config):
    for name, lag in (("M2_s10", 5), ("M2_s40", 20), ("M2_s80", 40)):
        model = _model(tiny_config, name)
        assert model.frontend.lags == (lag,)
        assert model.settings["algebra"] == "quaternion"


def test_mlp_control_shares_the_representation(tiny_config):
    ecg = torch.randn(2, 12, 5000)
    m1, mlp = _model(tiny_config, "M1"), _model(tiny_config, "M1_mlp")
    torch.testing.assert_close(m1.frontend(ecg), mlp.frontend(ecg))
    assert mlp.settings["operator"] == "mlp"


def test_quaternion_raw_input_is_rejected(tiny_config):
    config = dict(tiny_config["model"], variant="M2", feature="raw", scales_ms=[])
    with pytest.raises(ValueError):
        model_settings(config)
