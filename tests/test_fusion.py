"""The eight sanity tests required by 双分支方案 §17, plus the capacity control.

The point of every test here is attribution: a fusion result is only interpretable
if the raw branch is provably still M0 and the geometry branch is provably still
M1-M4, unchanged and unperturbed by the other branch.
"""

import pytest
import torch

from qdg.config import validate_config
from qdg.experiments import EXPERIMENTS, FUSION_GAINS, experiment_config
from qdg.models import FUSION_PAIRS, FusionNet, branch_configs, build_model

STATS = {"sampling_rate": 500, "signal_length": 5000, "vcg_std": [0.2, 0.3, 0.2]}


def _model(tiny_config, name):
    config = experiment_config(tiny_config, name)
    validate_config(config)
    return build_model(config["model"], STATS)


@pytest.fixture
def ecg():
    return torch.randn(2, 12, 5000, generator=torch.manual_seed(7))


# 1. The raw branch input is exactly M0's input.
@pytest.mark.parametrize("name", sorted(FUSION_PAIRS))
def test_raw_branch_input_matches_m0(tiny_config, ecg, name):
    fusion, m0 = _model(tiny_config, name), _model(tiny_config, "M0")
    torch.testing.assert_close(fusion.raw.frontend(ecg), m0.frontend(ecg))
    assert fusion.raw.settings["feature"] == "raw"
    assert fusion.raw.frontend.out_channels == 3


# 2. With the same weights, the fusion raw encoder reproduces M0's embedding.
@pytest.mark.parametrize("name", sorted(FUSION_PAIRS))
def test_raw_branch_reproduces_m0_features(tiny_config, ecg, name):
    fusion, m0 = _model(tiny_config, name), _model(tiny_config, "M0")
    fusion.raw.load_state_dict(
        {k: v for k, v in m0.state_dict().items() if not k.startswith("head.")}, strict=False
    )
    fusion.eval(), m0.eval()
    with torch.inference_mode():
        torch.testing.assert_close(fusion.raw.forward_features(ecg), m0.forward_features(ecg))


# 3-6. Each geometry branch input is bit-identical to the model it came from.
@pytest.mark.parametrize("fusion_name,single", sorted(FUSION_PAIRS.items()))
def test_geometry_branch_input_matches_single_model(tiny_config, ecg, fusion_name, single):
    fusion, reference = _model(tiny_config, fusion_name), _model(tiny_config, single)
    torch.testing.assert_close(fusion.geometry.frontend(ecg), reference.frontend(ecg))
    assert fusion.geometry.settings == reference.settings


def test_f4_second_order_construction_is_unchanged(tiny_config, ecg):
    """F4 must keep the difference form, never inverse(q_prev) (x) q_next (§8)."""
    fusion = _model(tiny_config, "F4")
    assert fusion.geometry.settings["feature"] == "first_second"
    features = fusion.geometry.frontend(ecg).reshape(2, 4, 8, 5000)
    first, second = features[:, :, :4], features[:, :, 4:]
    m3 = _model(tiny_config, "F3").geometry.frontend(ecg)
    torch.testing.assert_close(first.flatten(1, 2), m3)
    assert second.abs().max() > 0


# 7. The geometry branch cannot modify the raw feature.
@pytest.mark.parametrize("name", sorted(FUSION_PAIRS))
def test_raw_feature_is_untouched_by_the_geometry_branch(tiny_config, ecg, name):
    fusion = _model(tiny_config, name).eval()
    with torch.inference_mode():
        before = fusion.raw.forward_features(ecg).clone()
        fusion(ecg)
        after = fusion.raw.forward_features(ecg)
    torch.testing.assert_close(before, after)
    # No parameter is shared between the two branches.
    raw_ids = {id(p) for p in fusion.raw.parameters()}
    assert not raw_ids & {id(p) for p in fusion.geometry.parameters()}


# 8. Zeroing the geometry branch still leaves a working forward pass through raw.
@pytest.mark.parametrize("name", sorted(FUSION_PAIRS))
def test_forward_survives_a_zeroed_geometry_branch(tiny_config, ecg, name):
    fusion = _model(tiny_config, name).eval()
    with torch.inference_mode():
        reference = fusion(ecg)
        for parameter in fusion.geometry.parameters():
            parameter.zero_()
        fusion.geometry_project.weight.zero_()
        fusion.geometry_project.bias.zero_()
        logits = fusion(ecg)
    assert logits.shape == (2, 5)
    assert torch.isfinite(logits).all()
    assert not torch.allclose(logits, reference)


def test_all_fusion_models_share_one_fusion_mechanism(tiny_config):
    """双分支方案 §9: identical join for F1-F4, or the comparison is uninterpretable."""
    shapes = {}
    for name in sorted(FUSION_PAIRS):
        fusion = _model(tiny_config, name)
        assert isinstance(fusion, FusionNet)
        assert isinstance(fusion.norm, torch.nn.LayerNorm)
        assert isinstance(fusion.head, torch.nn.Linear)
        # No attention, gating or hidden MLP anywhere in the join.
        joiners = (fusion.raw_project, fusion.geometry_project, fusion.norm, fusion.head)
        assert all(isinstance(m, (torch.nn.Linear, torch.nn.LayerNorm)) for m in joiners)
        shapes[name] = (fusion.raw_project.out_features, fusion.head.in_features)
    assert len(set(shapes.values())) == 1


def test_branch_configs_pin_the_raw_branch(tiny_config):
    """A geometry-side override must never leak into the raw branch (§4)."""
    config = {**tiny_config["model"], "variant": "F3", "scales_ms": [40], "operator": "mlp"}
    raw, geometry = branch_configs(config)
    assert raw["feature"] == "raw" and raw["scales_ms"] == [] and raw["operator"] == "conv"
    assert geometry["variant"] == "M3" and geometry["scales_ms"] == [40]


@pytest.mark.parametrize("name", sorted(FUSION_PAIRS))
def test_fusion_runs_and_trains(tiny_config, ecg, name):
    fusion = _model(tiny_config, name)
    logits = fusion(ecg)
    assert logits.shape == (2, 5)
    logits.square().mean().backward()
    for branch in (fusion.raw, fusion.geometry):
        grads = [p.grad for p in branch.parameters() if p.requires_grad]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)


def test_m0_wide_is_a_raw_only_capacity_control(tiny_config):
    """§10: same input as M0, wider, never any geometry."""
    wide = _model(tiny_config, "M0_wide")
    assert wide.settings["feature"] == "raw"
    assert wide.frontend.out_channels == 3
    assert wide.settings["real_width"] > _model(tiny_config, "M0").settings["real_width"]


def test_m0_wide_matches_the_fusion_parameter_count():
    """§10 asks for under 5% deviation; the registry width is solved for that."""
    base = {
        "quaternions": 32,
        "real_width": None,
        "kernel": 5,
        "depth": 4,
        "stem_stride": 5,
        "dropout": 0.1,
        "renormalize": False,
    }
    count = lambda cfg: sum(  # noqa: E731
        p.numel() for p in build_model({**base, **cfg}, STATS).parameters()
    )
    wide = count(EXPERIMENTS["M0_wide"])
    for name in FUSION_PAIRS:
        assert abs(wide - count(EXPERIMENTS[name])) / count(EXPERIMENTS[name]) < 0.05


def test_m0_to_m4_are_untouched(tiny_config, ecg):
    """The addendum forbids redefining the original ladder; guard it explicitly."""
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
        assert model.head is not None
        # forward is still head(forward_features), i.e. numerically what it always was.
        model.eval()
        with torch.inference_mode():
            torch.testing.assert_close(model(ecg), model.head(model.forward_features(ecg)))


def test_component_gain_definitions_use_fusion_not_m1_minus_m0():
    """§16: M1 - M0 is a replacement gap and must not appear as an added-geometry gain."""
    pairs = {(new, old) for _, _, new, old in FUSION_GAINS}
    assert ("F1", "M0") in pairs and ("F1", "M0_wide") in pairs
    assert ("M1", "M0") not in pairs
