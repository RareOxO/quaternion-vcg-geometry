"""The nine sanity tests required by v2 指导书 §11, plus the stage-one constraints.

The point of this stage is attribution of *information loss*, so every test here
either pins a block to its mathematical definition or proves a block is identical
to the existing model it is meant to reuse.
"""

import pytest
import torch

from qdg.config import validate_config
from qdg.experiments import DIAGNOSTIC_NEW, NOISE_BAND, experiment_config, interpret
from qdg.geometry import (
    KORS,
    kors_transform,
    linear_velocity,
    radial_magnitude,
    unit_direction,
)
from qdg.models import DIAGNOSTIC_VARIANTS, build_model

STATS = {"sampling_rate": 500, "signal_length": 5000, "vcg_std": [0.19716, 0.12212, 0.17387]}
EXPECTED_CHANNELS = {"R": 1, "RA": 5, "RU": 4, "RLA": 8}


def _model(tiny_config, name):
    config = experiment_config(tiny_config, name)
    validate_config(config)
    return build_model(config["model"], STATS)


@pytest.fixture
def ecg():
    return torch.randn(2, 12, 5000, generator=torch.manual_seed(11))


# 1. r of [3, 4, 0] is 5.
def test_radial_magnitude_of_a_known_vector():
    torch.testing.assert_close(radial_magnitude(torch.tensor([3.0, 4.0, 0.0])), torch.tensor([5.0]))
    v = torch.tensor([[1.0, 2.0, 2.0], [0.0, 0.0, 7.0]])
    torch.testing.assert_close(radial_magnitude(v).squeeze(-1), torch.tensor([3.0, 7.0]))


# 2. ||u|| == 1 for nonzero V; zero/near-zero stays finite.
def test_direction_norm_and_degenerate_vectors():
    v = torch.randn(4, 50, 3) * 5
    torch.testing.assert_close(
        torch.linalg.vector_norm(unit_direction(v), dim=-1), torch.ones(4, 50), atol=1e-5, rtol=0
    )
    for degenerate in (torch.zeros(3, 3), torch.full((3, 3), 1e-12)):
        out = unit_direction(degenerate)
        assert torch.isfinite(out).all()


# 3. r * u reconstructs V.
def test_radial_times_direction_reconstructs_the_vector():
    v = torch.randn(3, 40, 3) * 2 + 1
    torch.testing.assert_close(
        radial_magnitude(v) * unit_direction(v, eps=0.0), v, atol=1e-5, rtol=1e-4
    )


# 4. RA's angular block is bit-identical to M1's input.
def test_angular_block_matches_m1_exactly(tiny_config, ecg):
    m1 = _model(tiny_config, "M1")
    for name in ("RA", "RLA"):
        model = _model(tiny_config, name)
        start, stop = model.frontend.block_slices["angular"]
        torch.testing.assert_close(model.frontend(ecg)[:, start:stop], m1.frontend(ecg))


# 5. Constant XYZ gives zero linear velocity.
def test_linear_velocity_of_a_constant_signal_is_zero():
    constant = torch.tensor([0.3, -0.2, 0.5]).expand(30, 3).contiguous()
    torch.testing.assert_close(linear_velocity(constant), torch.zeros(30, 3))


# 6. A linear ramp gives a constant velocity.
def test_linear_velocity_of_a_ramp_is_constant():
    step = torch.tensor([0.1, -0.2, 0.05])
    ramp = torch.arange(30.0).unsqueeze(-1) * step
    velocity = linear_velocity(ramp)
    torch.testing.assert_close(velocity, step.expand(30, 3).contiguous(), atol=1e-6, rtol=1e-5)
    # Edge replication, never a circular wrap (§6).
    torch.testing.assert_close(velocity[-1], velocity[-2])


# 7. Every RLA sub-block equals the standalone constructor for that block.
def test_rla_blocks_match_their_standalone_constructors(tiny_config, ecg):
    rla = _model(tiny_config, "RLA")
    features = rla.frontend(ecg)
    for name, block in (("R", "radial"), ("RA", "angular")):
        reference = _model(tiny_config, name)
        start, stop = rla.frontend.block_slices[block]
        ref_start, ref_stop = reference.frontend.block_slices[block]
        torch.testing.assert_close(
            features[:, start:stop], reference.frontend(ecg)[:, ref_start:ref_stop]
        )
    # The linear block is the raw forward difference, scaled by the training-fold RMS.
    vcg = kors_transform(ecg, torch.tensor(KORS))
    expected = linear_velocity(vcg.transpose(1, 2)).transpose(1, 2) / torch.tensor(
        STATS["vcg_std"]
    ).view(1, 3, 1)
    start, stop = rla.frontend.block_slices["linear"]
    torch.testing.assert_close(features[:, start:stop], expected)


def test_linear_velocity_is_built_on_raw_xyz_not_on_u(tiny_config, ecg):
    """§2: '不要把 linear velocity 建在归一化 u 上'."""
    rla = _model(tiny_config, "RLA")
    start, stop = rla.frontend.block_slices["linear"]
    actual = rla.frontend(ecg)[:, start:stop]
    vcg = kors_transform(ecg, torch.tensor(KORS))
    on_u = linear_velocity(unit_direction(vcg.transpose(1, 2))).transpose(1, 2)
    assert not torch.allclose(actual, on_u, atol=1e-3)


# 8. Shapes are [B, C, T] with T unchanged.
@pytest.mark.parametrize("name", DIAGNOSTIC_VARIANTS)
def test_feature_shapes_and_channel_counts(tiny_config, ecg, name):
    model = _model(tiny_config, name)
    features = model.frontend(ecg)
    assert features.shape == (2, EXPECTED_CHANNELS[name], 5000)
    assert model.frontend.out_channels == EXPECTED_CHANNELS[name]
    with torch.inference_mode():
        assert model.eval()(ecg).shape == (2, 5)


# 9. No NaN/Inf on a real-shaped batch, for every variant.
@pytest.mark.parametrize("name", DIAGNOSTIC_VARIANTS)
def test_features_and_gradients_are_finite(tiny_config, ecg, name):
    model = _model(tiny_config, name)
    features = model.frontend(ecg)
    assert torch.isfinite(features).all()
    model(ecg).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_radial_is_not_renormalized_per_record(tiny_config):
    """§6: per-record unit-norm on r would erase amplitude again."""
    model = _model(tiny_config, "R")
    quiet = torch.randn(1, 12, 5000, generator=torch.manual_seed(3)) * 0.1
    loud = quiet * 4
    # A four-fold louder record must produce a four-fold larger radial channel.
    torch.testing.assert_close(
        model.frontend(loud), model.frontend(quiet) * 4, atol=1e-4, rtol=1e-4
    )


def test_radial_scale_comes_only_from_training_fold_statistics(tiny_config):
    """r is divided by sqrt(sum of the per-axis training RMS), nothing batch-derived."""
    model = _model(tiny_config, "R")
    expected = torch.tensor(STATS["vcg_std"]).square().sum().sqrt()
    torch.testing.assert_close(model.frontend.radial_scale.reshape(()), expected)


@pytest.mark.parametrize("name", DIAGNOSTIC_VARIANTS)
def test_stage_one_is_real_only(tiny_config, name):
    """§1/§12: no QuaternionConv, no multi-scale, no second order."""
    model = _model(tiny_config, name)
    assert model.settings["algebra"] == "real"
    assert not model.encoder.quaternion
    assert list(model.settings["scales_ms"]) == [20]
    assert model.frontend.quaternion_channels == 0
    assert "second" not in model.settings["feature"]


def test_quaternion_composite_is_rejected(tiny_config):
    config = {**tiny_config["model"], "variant": "RA", "algebra": "quaternion"}
    with pytest.raises(ValueError, match="Real only"):
        build_model(config, STATS)


def test_m0_to_f4_are_untouched(tiny_config, ecg):
    """The new stage must not perturb any earlier variant."""
    for name, channels in (("M0", 3), ("M1", 4), ("M2", 4), ("M3", 16), ("M4", 32)):
        model = _model(tiny_config, name)
        assert model.frontend.out_channels == channels
        assert model.frontend.blocks == ()
    for name in ("F1", "F4"):
        fusion = _model(tiny_config, name)
        assert fusion.raw.frontend.out_channels == 3


def test_interpret_covers_the_decision_table():
    """§9 patterns map to a verdict; an incomplete stage never claims one."""
    assert interpret({})[0] == "incomplete"
    big = 4 * NOISE_BAND
    supported = interpret({"RA - M1": big, "RU - M0": 0.0, "RA - RU": 0.0, "RLA - RA": 0.0})
    assert supported[0] == "supported"
    audit = interpret({"RA - M1": 0.0, "RU - M0": -big, "RA - RU": 0.0, "RLA - RA": 0.0})
    assert audit[0] == "not supported" and "normaliz" in audit[2].lower()
    flat = interpret({"RA - M1": 0.0, "RU - M0": 0.0, "RA - RU": 0.0, "RLA - RA": 0.0})
    assert flat[0] == "not supported"
    compression = interpret({"RA - M1": big, "RU - M0": 0.0, "RA - RU": -big, "RLA - RA": 0.0})
    assert compression[0] == "partially supported"
    linear = interpret({"RA - M1": 0.0, "RU - M0": 0.0, "RA - RU": 0.0, "RLA - RA": big})
    assert linear[0] == "partially supported" and "linear" in linear[1].lower()


def test_registry_exposes_only_the_four_new_variants():
    assert DIAGNOSTIC_NEW == ("R", "RA", "RU", "RLA")


def test_new_buffers_never_enter_state_dict(tiny_config):
    """§1: existing checkpoints must keep loading with strict=True.

    radial_scale is derived from vcg_scale, so it is registered non-persistently; a
    persistent buffer would add a key and break every checkpoint trained before it.
    """
    for name in ("M0", "M1", "M2", "M3", "M4", "R", "RA", "RU", "RLA"):
        keys = _model(tiny_config, name).state_dict()
        assert "frontend.radial_scale" not in keys
        assert {"frontend.kors", "frontend.vcg_scale"} <= set(keys)
    # An M0 built today loads an M0 state_dict captured before the diagnostic existed.
    old = _model(tiny_config, "M0").state_dict()
    _model(tiny_config, "M0").load_state_dict(old, strict=True)
