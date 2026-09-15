"""Experiment 4: accessible temporal-context ablation.

E* is untouched; what varies is how much time the recurrence may integrate, set by
resetting the recurrent state every N steps. The tests below defend the three things
that make the sweep a controlled comparison: the parameter count never moves, the
restriction really does block integration across a boundary, and the zero padding used
to make the chunking divide evenly cannot influence a real step.
"""

from pathlib import Path

import pytest
import torch

from qdg.config import load_config, validate_config
from qdg.encoders import build_encoder
from qdg.experiments import (
    CONTEXT,
    CONTEXT_LEVELS,
    CONTEXT_SCALE,
    CONTEXT_STEM,
    CONTEXT_STEP_MS,
    NOISE_BAND,
    experiment_config,
    interpret_context,
)
from qdg.models import build_model

STATS = {"sampling_rate": 500, "signal_length": 5000, "vcg_std": [0.19716, 0.12212, 0.17387]}


@pytest.fixture
def full_config():
    return load_config(Path(__file__).resolve().parents[1] / "configs/base.yaml")


@pytest.fixture
def ecg():
    return torch.randn(2, 12, 5000, generator=torch.manual_seed(53))


def _model(config, name):
    current = experiment_config(config, name)
    validate_config(current)
    return build_model(current["model"], STATS)


def test_every_level_has_the_same_parameter_count(full_config):
    """The whole point: context is the only thing that changes across the sweep."""
    counts = {
        name: sum(p.numel() for p in _model(full_config, name).parameters()) for name in CONTEXT
    }
    assert len(set(counts.values())) == 1, counts


def test_context_maps_to_whole_stem_steps(full_config):
    """20 ms per step, so every level is a whole number of resets."""
    for label, context in CONTEXT_LEVELS:
        model = _model(full_config, f"E4_{label}")
        steps = {branch: encoder.context_steps for branch, encoder in model.encoders.items()}
        # Every branch carries the same restriction, so they stay temporally aligned.
        assert len(set(steps.values())) == 1, steps
        assert steps["angular"] == (None if context is None else context // CONTEXT_STEP_MS)
        assert model.settings["context_ms"] == context


def test_a_context_that_is_not_a_whole_step_is_rejected(full_config):
    config = experiment_config(full_config, "E4_80ms")["model"]
    with pytest.raises(ValueError, match="whole number"):
        build_model({**config, "context_ms": 30}, STATS)


def test_restriction_blocks_integration_across_a_boundary():
    """A perturbation must move steps inside its own chunk and no step in the next one."""
    encoder = build_encoder(
        "lstm_attention", 8, 44, stem={"stride": 5, "poolings": 1}, context_steps=4
    ).eval()
    x = torch.randn(1, 8, 5000, generator=torch.manual_seed(3))
    with torch.inference_mode():
        base = encoder.down(x).transpose(1, 2)
        perturbed = base.clone()
        perturbed[:, 1] += 10.0  # a step inside the first chunk
        before, after = encoder._chunked(base), encoder._chunked(perturbed)
    moved = (before - after).abs().sum(-1)[0]
    assert moved[1] > 0 and moved[3] > 0, "later steps of the same chunk must move"
    assert moved[4:].abs().max() < 1e-6, "nothing after the boundary may move"
    assert moved[0].abs() < 1e-6, "the recurrence is causal, so earlier steps stay put"


def test_unrestricted_does_integrate_across_the_same_boundary():
    """The control for the test above: without the reset the effect crosses freely."""
    encoder = build_encoder("lstm_attention", 8, 44, stem={"stride": 5, "poolings": 1}).eval()
    x = torch.randn(1, 8, 5000, generator=torch.manual_seed(3))
    with torch.inference_mode():
        base = encoder.down(x).transpose(1, 2)
        perturbed = base.clone()
        perturbed[:, 1] += 10.0
        before, after = encoder._recur(base), encoder._recur(perturbed)
    assert (before - after).abs().sum(-1)[0, 4:].max() > 1e-6


def test_tail_padding_cannot_reach_a_real_step():
    """T is not a multiple of every window, so the tail is padded; it must be inert."""
    encoder = build_encoder(
        "lstm_attention", 8, 44, stem={"stride": 5, "poolings": 1}, context_steps=64
    ).eval()
    x = torch.randn(1, 8, 5000, generator=torch.manual_seed(7))
    with torch.inference_mode():
        folded = encoder.down(x).transpose(1, 2)
        assert folded.shape[1] % 64, "this test only means anything when padding happens"
        chunked = encoder._chunked(folded)
        # Running the last partial chunk on its own gives the same states.
        tail_start = (folded.shape[1] // 64) * 64
        direct = encoder._recur(folded[:, tail_start:])
    assert chunked.shape[1] == folded.shape[1]
    torch.testing.assert_close(chunked[:, tail_start:], direct, atol=1e-5, rtol=1e-4)


def test_restriction_is_off_outside_experiment_4(full_config):
    """Experiments 1-3 must be unaffected: no context key, no reset, original stem."""
    for name in ("E1_lstm_attention", "E2_RLQ", "E3_U"):
        model = _model(full_config, name)
        assert model.settings["context_ms"] is None
        assert model.settings["stem"] is None
        for encoder in model.encoders.values():
            assert encoder.context_steps is None


@pytest.mark.parametrize("name", CONTEXT)
def test_levels_run_and_train(full_config, ecg, name):
    model = _model(full_config, name)
    logits = model(ecg)
    assert logits.shape == (2, 5) and torch.isfinite(logits).all()
    logits.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_scale_grouping_is_pre_specified():
    """The plan's grouping, fixed in code so it cannot be fitted to the result."""
    assert CONTEXT_SCALE[20] == CONTEXT_SCALE[40] == CONTEXT_SCALE[80] == "local"
    assert CONTEXT_SCALE[160] == CONTEXT_SCALE[320] == "phase-scale"
    assert CONTEXT_SCALE[640] == CONTEXT_SCALE[1280] == "cycle-scale"
    assert CONTEXT_STEM == {"stride": 5, "poolings": 1}
    assert [context for _, context in CONTEXT_LEVELS] == [20, 40, 80, 160, 320, 640, 1280, None]


def test_context_verdicts():
    # The trend is read between SCALE GROUP means, not between the extreme levels, so a
    # rising case has to clear the band on the group difference.
    rising = {c: 0.90 + 0.003 * i for i, c in enumerate([20, 40, 80, 160, 320, 640, 1280])}
    rising[None] = 0.918
    verdict = interpret_context(rising)
    assert "Longer accessible context helps" in verdict
    assert "Cycle-scale minus local: +0.0135" in verdict
    flat = {c: 0.90 for c in [20, 40, 80, 160, 320, 640, 1280, None]}
    assert "Not supported" in interpret_context(flat)
    assert "Incomplete" in interpret_context({20: 0.9})
    bounded = dict.fromkeys([20, 40, 80, 160, 640, 1280, None], 0.90)
    bounded[320] = 0.90 + 4 * NOISE_BAND
    assert "bounded useful range" in interpret_context(bounded)
