"""The §6 sanity checks of the RLA temporal 指导书.

The claim this stage can make rests on one thing: the RLA input is identical across
the three models and the receptive field is the only systematic variable. Every test
here defends exactly that.
"""

from pathlib import Path

import pytest
import torch

from qdg.config import validate_config
from qdg.experiments import (
    EXPERIMENTS,
    NOISE_BAND,
    TEMPORAL,
    TEMPORAL_NEW,
    experiment_config,
    interpret_temporal,
)
from qdg.config import load_config
from qdg.models import build_model
from qdg.quaternion_nn import TCNEncoder, receptive_field_samples

STATS = {"sampling_rate": 500, "signal_length": 5000, "vcg_std": [0.19716, 0.12212, 0.17387]}
NAMES = [name for _, name in TEMPORAL]


@pytest.fixture
def full_config():
    """The production config. The temporal variants pin depth/kernel/width, so the
    tiny_config overrides used elsewhere would fight them and hide the real numbers."""
    return load_config(Path(__file__).resolve().parents[1] / "configs/base.yaml")


def _model(config, name):
    current = experiment_config(config, name)
    validate_config(current)
    return build_model(current["model"], STATS)


@pytest.fixture
def ecg():
    return torch.randn(2, 12, 5000, generator=torch.manual_seed(5))


# 1. The RLA input tensor is identical for all three variants.
def test_all_three_share_one_rla_input(full_config, ecg):
    reference = _model(full_config, "RLA").frontend(ecg)
    for name in NAMES:
        torch.testing.assert_close(_model(full_config, name).frontend(ecg), reference)
        assert _model(full_config, name).frontend.blocks == ("radial", "linear", "angular")
        assert _model(full_config, name).frontend.out_channels == 8


# 2 & 3. Actual receptive fields, strictly increasing.
def test_receptive_fields_are_ordered_and_on_target(full_config):
    fields = {name: _model(full_config, name).encoder.receptive_field for name in NAMES}
    short, medium, long_ = (fields[name] for name in NAMES)
    assert short < medium < long_
    milliseconds = {name: 1000 * value / 500 for name, value in fields.items()}
    # Targets are ~100 / ~400 / ~1200 ms; allow the order-of-magnitude tolerance §3 grants.
    assert 60 <= milliseconds["RLA_short"] <= 160
    assert 280 <= milliseconds["RLA_medium"] <= 520
    assert 1000 <= milliseconds["RLA"] <= 1500


def test_receptive_field_counts_the_layers_actually_applied():
    """§5: measured from the real kernel/stride/dilation sequence, not asserted."""
    assert receptive_field_samples([(1, 1, 1)]) == 1
    assert receptive_field_samples([(3, 1, 1)]) == 3
    assert receptive_field_samples([(5, 5, 1)]) == 5
    # Stem 5, two k=5 convs: 5 + 4*5 + 4*5.
    assert receptive_field_samples([(5, 5, 1), (5, 1, 1), (5, 1, 1)]) == 45
    encoder = TCNEncoder(8, 64, kernel=5, depth=4, stem_stride=5)
    assert encoder.receptive_field == receptive_field_samples(encoder.layer_spec)
    # The pooling layers are counted; the old closed form dropped them and gave 605.
    assert encoder.receptive_field == 640
    assert sum(1 for k, s, _ in encoder.layer_spec if s == 2) == 3


def test_layer_spec_matches_the_modules_in_order():
    encoder = TCNEncoder(8, 32, kernel=7, depth=2, stem_stride=5)
    convs = [(7, 1, 1), (7, 1, 1)]
    assert encoder.layer_spec == ((5, 5, 1), *convs, (2, 2, 1), *convs)


# 4. Parameter counts are reported, and here they are also close.
def test_parameter_counts_stay_close(full_config):
    counts = {
        name: sum(p.numel() for p in _model(full_config, name).parameters()) for name in NAMES
    }
    spread = (max(counts.values()) - min(counts.values())) / max(counts.values())
    assert spread < 0.05, counts


# 5. Same output shape; forward and backward both work.
@pytest.mark.parametrize("name", NAMES)
def test_forward_backward_smoke(full_config, ecg, name):
    model = _model(full_config, name)
    logits = model(ecg)
    assert logits.shape == (2, 5)
    assert torch.isfinite(logits).all()
    logits.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_long_is_exactly_the_existing_rla(full_config):
    """§7 step 5: Long must be the existing RLA so its seed42 run is reused, not retrained."""
    assert TEMPORAL[-1] == ("RLA-Long", "RLA")
    assert EXPERIMENTS["RLA"] == {"variant": "RLA"}
    assert "RLA" not in TEMPORAL_NEW


def test_only_the_receptive_field_varies(full_config):
    """§11: feature, fusion and algebra must not move together with the RF."""
    settings = [_model(full_config, name).settings for name in NAMES]
    for key in ("feature", "blocks", "algebra", "operator", "stem_stride", "renormalize"):
        assert len({str(s[key]) for s in settings}) == 1, key
    for s in settings:
        assert list(s["scales_ms"]) == [20]
        assert s["algebra"] == "real"


def test_temporal_verdicts():
    big = 4 * NOISE_BAND
    assert "Incomplete" in interpret_temporal({})
    rising = interpret_temporal(
        {"Medium - Short": big / 2, "Long - Medium": big / 2, "Long - Short": big}
    )
    assert "Longer temporal context helps" in rising
    flat = interpret_temporal({"Medium - Short": 0.0, "Long - Medium": 0.0, "Long - Short": 0.0})
    assert "Not supported" in flat
    saturating = interpret_temporal(
        {"Medium - Short": big, "Long - Medium": -0.001, "Long - Short": big}
    )
    assert "bounded useful range" in saturating


def test_earlier_variants_are_untouched(full_config):
    for name, channels in (("M0", 3), ("M1", 4), ("M4", 32), ("RU", 4), ("RLA", 8)):
        assert _model(full_config, name).frontend.out_channels == channels
    assert _model(full_config, "RLA").encoder.receptive_field == 640
