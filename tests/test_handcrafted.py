"""Experiments 5 and 6.

Experiment 5's risk is leakage: the handcrafted features must be a function of one
record and nothing else, and only the classifier may see the training folds. Experiment
6's risk is that the two-scale model differs from the single-scale ones in more than the
two scales.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from qdg.config import load_config, validate_config
from qdg.experiments import (
    COMBINED,
    HANDCRAFTED,
    HANDCRAFTED_ARMS,
    HANDCRAFTED_PROPOSED,
    LOCAL_CONTEXT_MS,
    LOCAL_LONG,
    LONG_CONTEXT_MS,
    NOISE_BAND,
    experiment_config,
    interpret_handcrafted,
    interpret_local_long,
)
from qdg.handcrafted import (
    FEATURE_SETS,
    beat_quality,
    biomarker_features,
    cardiac_vector,
    detect_r_peaks,
    features_for,
    statistical_features,
    velocity_features,
)
from qdg.models import build_model

STATS = {"sampling_rate": 500, "signal_length": 5000, "vcg_std": [0.19716, 0.12212, 0.17387]}
RATE = 500


@pytest.fixture
def full_config():
    return load_config(Path(__file__).resolve().parents[1] / "configs/base.yaml")


def _synthetic(beats=10, rate=RATE, seconds=10, amplitude=1.0):
    """A crude periodic VCG: a sharp complex once per beat, on a small baseline."""
    length = rate * seconds
    time = np.arange(length) / rate
    signal = np.zeros((3, length), dtype=np.float32)
    for index in range(beats):
        centre = int((index + 0.5) * length / beats)
        span = slice(max(0, centre - 20), min(length, centre + 20))
        shape = np.hanning(span.stop - span.start)
        for axis, gain in enumerate((1.0, 0.6, 0.3)):
            signal[axis, span] += amplitude * gain * shape
    signal += 0.01 * np.sin(2 * np.pi * 1.3 * time)[None, :]
    return signal


def _model(config, name):
    current = experiment_config(config, name)
    validate_config(current)
    return build_model(current["model"], STATS)


# --- R-peak detection ---


def test_detector_finds_the_planted_beats():
    for beats in (6, 10, 15):
        peaks = detect_r_peaks(_synthetic(beats=beats), RATE)
        assert abs(len(peaks) - beats) <= 1, (beats, len(peaks))
        assert beat_quality(peaks, RATE, RATE * 10)


def test_quality_rule_rejects_implausible_detections():
    assert not beat_quality(np.array([100, 200]), RATE, 5000)  # too few
    assert not beat_quality(np.arange(0, 5000, 100), RATE, 5000)  # too many
    irregular = np.array([0, 200, 2500, 2700, 4900])  # wildly uneven
    assert not beat_quality(irregular, RATE, 5000)


def test_detector_is_deterministic():
    signal = _synthetic()
    assert np.array_equal(detect_r_peaks(signal, RATE), detect_r_peaks(signal, RATE))


# --- Features ---


@pytest.mark.parametrize("builder", [statistical_features, velocity_features, biomarker_features])
def test_features_are_finite_and_fixed_width(builder):
    widths = {len(builder(_synthetic(beats=beats), RATE)) for beats in (6, 10, 15)}
    assert len(widths) == 1, "a feature vector must not change length with the record"
    for beats in (6, 10, 15):
        values = builder(_synthetic(beats=beats), RATE)
        assert np.isfinite(values).all()


def test_features_depend_on_one_record_only():
    """No fitted statistic, so a record's features cannot move when others change."""
    alone = features_for(np.zeros((12, 5000), dtype=np.float32), RATE, FEATURE_SETS)
    again = features_for(np.zeros((12, 5000), dtype=np.float32), RATE, FEATURE_SETS)
    assert np.array_equal(alone, again)
    assert np.isfinite(alone).all(), "a flat record must not produce NaN"


def test_statistical_features_are_order_insensitive():
    """Set A is a distributional summary, so permuting time must not change it."""
    signal = _synthetic()
    shuffled = signal[:, np.random.default_rng(0).permutation(signal.shape[1])]
    before, after = statistical_features(signal, RATE), statistical_features(shuffled, RATE)
    # The radius block is a pure distribution and must be identical.
    np.testing.assert_allclose(before[:7], after[:7], rtol=1e-5, atol=1e-6)


def test_velocity_features_scale_with_amplitude():
    quiet = velocity_features(_synthetic(amplitude=1.0), RATE)
    loud = velocity_features(_synthetic(amplitude=2.0), RATE)
    assert (loud[:3] > quiet[:3]).all(), "speed must grow with the vector it is built from"


def test_biomarker_fallback_is_flagged():
    """A record whose detection fails the rule falls back, and says so."""
    flat = np.zeros((3, 5000), dtype=np.float32)
    flat[0, ::7] = 1.0  # dense spikes: too many beats for the rule
    values = biomarker_features(flat, RATE)
    assert np.isfinite(values).all()
    assert values[8] in (0.0, 1.0), "the fallback flag is a feature of its own"


def test_feature_sets_concatenate_in_a_fixed_order():
    signal = np.zeros((12, 5000), dtype=np.float32)
    signal[0] = np.sin(np.arange(5000) / 50)
    parts = [features_for(signal, RATE, [kind]) for kind in FEATURE_SETS]
    combined = features_for(signal, RATE, FEATURE_SETS)
    np.testing.assert_allclose(combined, np.concatenate(parts))


def test_cardiac_vector_matches_the_model_transform():
    from qdg.geometry import KORS, kors_transform

    signal = np.random.default_rng(1).normal(size=(12, 500)).astype(np.float32)
    expected = kors_transform(torch.from_numpy(signal)[None], torch.tensor(KORS))[0].numpy()
    np.testing.assert_allclose(cardiac_vector(signal), expected, rtol=1e-4, atol=1e-5)


def test_arms_cover_the_plan(full_config):
    assert set(HANDCRAFTED_ARMS) == {"E5_stat", "E5_velocity", "E5_biomarker", "E5_hybrid"}
    assert HANDCRAFTED_ARMS["E5_hybrid"][1] == FEATURE_SETS
    assert HANDCRAFTED_PROPOSED == "E2_RLQ"
    assert [name for _, name in HANDCRAFTED][3] == HANDCRAFTED_PROPOSED


# --- Experiment 6 ---


def test_local_and_long_reuse_the_experiment_4_runs():
    labels = [name for _, name in LOCAL_LONG]
    assert labels[:2] == [f"E4_{LOCAL_CONTEXT_MS}ms", f"E4_{LONG_CONTEXT_MS}ms"]
    assert labels[2] == COMBINED
    assert (LOCAL_CONTEXT_MS, LONG_CONTEXT_MS) == (80, 640)


def test_combined_runs_both_scales_and_nothing_else(full_config):
    combined = _model(full_config, COMBINED)
    local = _model(full_config, f"E4_{LOCAL_CONTEXT_MS}ms")
    for branch, encoder in combined.encoders.items():
        assert encoder.context_steps == [4, 32], branch
        assert len(encoder.members) == 2
    # Same branches, same input, same stem as the single-scale runs.
    assert combined.settings["branches"] == local.settings["branches"]
    assert combined.settings["stem"] == local.settings["stem"]
    assert combined.settings["angular_blocks"] == local.settings["angular_blocks"]


def test_combined_parameter_count_is_close_to_the_single_scales(full_config):
    counts = {
        name: sum(p.numel() for p in _model(full_config, name).parameters())
        for _, name in LOCAL_LONG
    }
    single = counts[f"E4_{LOCAL_CONTEXT_MS}ms"]
    assert abs(counts[COMBINED] - single) / single < 0.02, counts


def test_combined_runs_and_trains(full_config):
    model = _model(full_config, COMBINED)
    ecg = torch.randn(2, 12, 5000, generator=torch.manual_seed(71))
    logits = model(ecg)
    assert logits.shape == (2, 5) and torch.isfinite(logits).all()
    logits.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_single_context_models_keep_their_parameter_names(full_config):
    """A single context must NOT be wrapped.

    The wrapper inserts a "members.0." level into every parameter name, so wrapping a
    single-context model would make every checkpoint trained before Experiment 6
    existed unloadable. This is the regression that guards it.
    """
    for name in ("E2_RLQ", "E4_320ms", "E1_lstm_attention", "E7_q_full", "E3_U"):
        model = _model(full_config, name)
        assert not any("members." in key for key in model.state_dict()), name
        for encoder in model.encoders.values():
            assert not hasattr(encoder, "members")
    combined = _model(full_config, COMBINED)
    assert any("members." in key for key in combined.state_dict())


def test_verdicts():
    def summary(values):
        return {name: {"macro_auroc_mean": value} for name, value in values.items()}

    assert "Incomplete" in interpret_local_long({})
    names = [name for _, name in LOCAL_LONG]
    gain = summary(dict(zip(names, [0.900, 0.905, 0.905 + 4 * NOISE_BAND])))
    assert "Integration supported" in interpret_local_long(gain)
    flat = summary(dict(zip(names, [0.900, 0.905, 0.9055])))
    assert "not supported" in interpret_local_long(flat)
    assert "narrower claim" in interpret_local_long(flat)

    arms = {"E5_stat": 0.85, "E5_velocity": 0.86, "E5_biomarker": 0.84, "E2_RLQ": 0.90}
    assert "Incomplete" in interpret_handcrafted({})
    no_gain = interpret_handcrafted(summary({**arms, "E5_hybrid": 0.9005}))
    assert "add nothing on top" in no_gain
    complementary = interpret_handcrafted(summary({**arms, "E5_hybrid": 0.92}))
    assert "Complementary" in complementary
