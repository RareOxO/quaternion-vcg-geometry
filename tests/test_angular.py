"""The six sanity checks of the Angular temporal 指导书 §5.

The comparison is only interpretable if the two models differ in exactly one thing:
the operator inside the angular encoder. Everything here defends that.
"""

from pathlib import Path

import pytest
import torch

from qdg.config import load_config, validate_config
from qdg.experiments import ANGULAR_NEW, NOISE_BAND, experiment_config, interpret_angular
from qdg.geometry import hamilton_product
from qdg.models import BRANCH_BLOCKS, BranchedRLANet, build_model
from qdg.quaternion_nn import QuaternionConv1d, hamilton_kernel

STATS = {"sampling_rate": 500, "signal_length": 5000, "vcg_std": [0.19716, 0.12212, 0.17387]}


@pytest.fixture
def full_config():
    return load_config(Path(__file__).resolve().parents[1] / "configs/base.yaml")


@pytest.fixture
def ecg():
    return torch.randn(2, 12, 5000, generator=torch.manual_seed(19))


def _model(config, name):
    current = experiment_config(config, name)
    validate_config(current)
    return build_model(current["model"], STATS)


def _pair(config):
    return _model(config, "RLA_standard"), _model(config, "RLA_quaternion")


# 1. Shapes line up: same input length, same embedding dimension, same output.
def test_shapes_align(full_config, ecg):
    for model in _pair(full_config):
        assert isinstance(model, BranchedRLANet)
        with torch.inference_mode():
            assert model.eval()(ecg).shape == (2, 5)
            assert model.forward_features(ecg).shape[-1] == model.norm.normalized_shape[0]
    standard, quaternion = _pair(full_config)
    with torch.inference_mode():
        assert standard.eval().forward_features(ecg).shape == (
            quaternion.eval().forward_features(ecg).shape
        )


# 2. The A tensor entering both angular encoders is identical.
def test_angular_input_is_identical(full_config, ecg):
    standard, quaternion = _pair(full_config)
    a_standard = standard.frontends["angular"](ecg)
    a_quaternion = quaternion.frontends["angular"](ecg)
    torch.testing.assert_close(a_standard, a_quaternion, atol=0, rtol=0)
    assert a_standard.shape == (2, 4, 5000)
    # And it is still M1's 20 ms descriptor, unchanged (§1).
    reference = _model(full_config, "RLA").frontend(ecg)[:, 4:8]
    torch.testing.assert_close(a_standard, reference)


def test_radial_and_linear_branches_are_identical(full_config, ecg):
    standard, quaternion = _pair(full_config)
    for block in ("radial", "linear"):
        torch.testing.assert_close(
            standard.frontends[block](ecg), quaternion.frontends[block](ecg), atol=0, rtol=0
        )
        assert standard.encoders[block].width == quaternion.encoders[block].width, block
        assert not standard.encoders[block].quaternion
        assert not quaternion.encoders[block].quaternion


# 3. Both angular encoders have the same receptive field.
def test_receptive_fields_match(full_config):
    standard, quaternion = _pair(full_config)
    fields = {
        "standard": standard.encoders["angular"].receptive_field,
        "quaternion": quaternion.encoders["angular"].receptive_field,
    }
    assert fields["standard"] == fields["quaternion"] == 640
    # Every branch shares it, which describe() also enforces.
    for model in (standard, quaternion):
        assert len({e.receptive_field for e in model.encoders.values()}) == 1
        assert model.describe()["receptive_field_samples"] == 640


# 4. Hamilton product: a hand-computed case, signs and component order.
def test_hamilton_product_hand_computed():
    """(1 + 2i + 3j + 4k) (x) (5 + 6i + 7j + 8k) = -60 + 12i + 30j + 24k."""
    a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    b = torch.tensor([5.0, 6.0, 7.0, 8.0])
    torch.testing.assert_close(hamilton_product(a, b), torch.tensor([-60.0, 12.0, 30.0, 24.0]))
    # Non-commutative, and the reverse differs in the vector part only.
    torch.testing.assert_close(hamilton_product(b, a), torch.tensor([-60.0, 20.0, 14.0, 32.0]))
    one, i, j, k = torch.eye(4)
    torch.testing.assert_close(hamilton_product(i, j), k)
    torch.testing.assert_close(hamilton_product(j, i), -k)
    torch.testing.assert_close(hamilton_product(k, k), -one)


def test_quaternion_conv_is_a_real_hamilton_operator():
    """§4: the weights must act through Hamilton coupling, not be renamed channels."""
    layer = QuaternionConv1d(1, 1, 1, bias=False)
    q = torch.randn(1, 4, 6)
    weight = layer.weight.reshape(4)
    expected = hamilton_product(weight.expand(6, 4), q[0].transpose(0, 1)).transpose(0, 1)
    torch.testing.assert_close(layer(q)[0], expected, atol=1e-6, rtol=1e-5)
    # The block kernel carries the Hamilton signs, so an impulse on one component
    # reaches all four outputs.
    kernel = hamilton_kernel(layer.weight)
    assert kernel.shape == (4, 4, 1)
    assert (kernel.abs() > 0).all()


# 5. All four quaternion components receive finite, nonzero gradients.
def test_quaternion_components_all_get_gradients(full_config, ecg):
    quaternion = _model(full_config, "RLA_quaternion")
    quaternion(ecg).square().mean().backward()
    for module in quaternion.encoders["angular"].modules():
        if isinstance(module, QuaternionConv1d):
            grad = module.weight.grad
            assert grad is not None and torch.isfinite(grad).all()
            # weight is (4, out, in, k): every component must move.
            assert (grad.flatten(1).abs().sum(-1) > 0).all()


# 6. Parameter audit: angular encoders matched, totals reported.
def test_parameter_audit(full_config):
    standard, quaternion = _pair(full_config)
    angular = (standard.angular_parameters(), quaternion.angular_parameters())
    totals = tuple(sum(p.numel() for p in m.parameters()) for m in (standard, quaternion))
    assert abs(angular[0] - angular[1]) / angular[0] < 0.02, angular
    assert abs(totals[0] - totals[1]) / totals[0] < 0.02, totals
    # The quaternion encoder is 4Q channels wide and the standard one 2Q, which is what
    # makes the weight counts match rather than the channel counts.
    assert quaternion.encoders["angular"].width == 2 * standard.encoders["angular"].width


def test_forward_backward_smoke(full_config, ecg):
    for model in _pair(full_config):
        logits = model(ecg)
        assert logits.shape == (2, 5) and torch.isfinite(logits).all()
        logits.square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_only_the_angular_operator_differs(full_config):
    """§9: the lag, fusion, head, depth and RF must not move with the operator."""
    standard, quaternion = _pair(full_config)
    for key in ("branch_width", "fusion_dim", "scales_ms", "kernel", "depth", "stem_stride"):
        assert standard.settings[key] == quaternion.settings[key], key
    assert standard.settings["scales_ms"] == [20]
    assert standard.settings["angular_algebra"] == "standard"
    assert quaternion.settings["angular_algebra"] == "quaternion"
    assert type(standard.norm) is type(quaternion.norm)
    assert standard.head.in_features == quaternion.head.in_features
    assert set(standard.frontends) == set(quaternion.frontends) == set(BRANCH_BLOCKS)


def test_angular_verdicts():
    big = 4 * NOISE_BAND
    assert "worth more seeds" in interpret_angular(big)
    assert "Standard is ahead" in interpret_angular(-big)
    assert "Indistinguishable" in interpret_angular(0.001)
    assert "Stop here" in interpret_angular(-0.001)


def test_earlier_variants_are_untouched(full_config, ecg):
    for name, channels in (("M0", 3), ("M1", 4), ("M4", 32), ("RLA", 8)):
        assert _model(full_config, name).frontend.out_channels == channels
    assert _model(full_config, "RLA").encoder.receptive_field == 640
    assert ANGULAR_NEW == ("RLA_standard", "RLA_quaternion")
