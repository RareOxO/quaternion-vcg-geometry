"""Experiment 7: temporal ordering.

The claim rests on the permutation destroying order and nothing else: the features
themselves must be built from the intact recording, a block shuffle must keep local
order, and the joint scope must use one shared permutation so cross-component
alignment survives.
"""

from pathlib import Path

import pytest
import torch

from qdg.config import load_config, validate_config
from qdg.experiments import (
    ORDERING,
    ORDERING_ORIGINAL,
    SHUFFLE_LEVELS,
    SHUFFLE_SCOPES,
    experiment_config,
    interpret_ordering,
)
from qdg.models import BRANCH_BLOCKS, TemporalShuffle, build_model

STATS = {"sampling_rate": 500, "signal_length": 5000, "vcg_std": [0.19716, 0.12212, 0.17387]}


@pytest.fixture
def full_config():
    return load_config(Path(__file__).resolve().parents[1] / "configs/base.yaml")


@pytest.fixture
def ecg():
    return torch.randn(2, 12, 5000, generator=torch.manual_seed(61))


def _model(config, name):
    current = experiment_config(config, name)
    validate_config(current)
    return build_model(current["model"], STATS)


def test_block_shuffle_keeps_order_inside_a_block():
    x = torch.arange(24).float().reshape(1, 1, 24)
    for block in (2, 4, 6):
        out = TemporalShuffle(block=block, seed=1).eval()(x)[0, 0].tolist()
        for start in range(0, 24, block):
            piece = out[start : start + block]
            assert piece == list(range(int(piece[0]), int(piece[0]) + block)), piece
        assert sorted(out) == list(range(24)), "a permutation, not a resample"


def test_point_wise_shuffle_breaks_local_order():
    x = torch.arange(200).float().reshape(1, 1, 200)
    out = TemporalShuffle(block=1, seed=1).eval()(x)[0, 0]
    assert sorted(out.tolist()) == list(range(200))
    consecutive = (out[1:] - out[:-1] == 1).float().mean()
    assert consecutive < 0.05, "almost no pair should still be adjacent"


def test_remainder_stays_at_the_end():
    """5000 is not a whole number of 160 ms blocks; the tail must be left in place."""
    x = torch.arange(5000).float().reshape(1, 1, 5000)
    block = 80  # 160 ms at 500 Hz
    out = TemporalShuffle(block=block, seed=3).eval()(x)[0, 0]
    kept = (5000 // block) * block
    assert 5000 - kept == 40
    torch.testing.assert_close(out[kept:], torch.arange(kept, 5000).float())


def test_evaluation_is_reproducible_but_training_is_not():
    x = torch.randn(4, 3, 500, generator=torch.manual_seed(5))
    shuffle = TemporalShuffle(block=10, seed=7)
    shuffle.eval()
    assert torch.equal(shuffle(x), shuffle(x))
    shuffle.train()
    assert not torch.equal(shuffle(x), shuffle(x)), "a fixed draw could be inverted"
    # Each record in the batch gets its own permutation.
    index = shuffle.permutation(4, 500, x.device)
    assert not torch.equal(index[0], index[1])


def test_features_are_built_before_the_permutation(full_config, ecg):
    """The frontend must see the intact recording; only its output is permuted."""
    shuffled = _model(full_config, "E7_q_full")
    original = _model(full_config, ORDERING_ORIGINAL)
    for branch in BRANCH_BLOCKS:
        torch.testing.assert_close(
            shuffled.frontends[branch](ecg), original.frontends[branch](ecg), atol=0, rtol=0
        )


def test_q_scope_permutes_only_the_angular_branch(full_config, ecg):
    model = _model(full_config, "E7_q_block80").eval()
    assert model.shuffle_scope == "angular"
    features = {b: model.frontends[b](ecg) for b in BRANCH_BLOCKS}
    index = model.shuffle.permutation(2, 5000, ecg.device)
    moved = model.shuffle(features["angular"], index)
    assert not torch.equal(moved, features["angular"])
    # R and L are untouched under this scope.
    assert model.settings["shuffle"]["scope"] == "angular"


def test_joint_scope_uses_one_shared_permutation(full_config, ecg):
    """Per-branch draws would destroy cross-component alignment as well."""
    model = _model(full_config, "E7_joint_block80").eval()
    assert model.shuffle_scope == "all"
    index = model.shuffle.permutation(2, 5000, ecg.device)
    features = {b: model.frontends[b](ecg) for b in BRANCH_BLOCKS}
    moved = {b: model.shuffle(f, index) for b, f in features.items()}
    # The same time step lands in the same place in every branch.
    for branch in BRANCH_BLOCKS:
        torch.testing.assert_close(
            moved[branch],
            features[branch].gather(-1, index.unsqueeze(1).expand_as(features[branch])),
        )


@pytest.mark.parametrize("name", ORDERING)
def test_conditions_run_and_train(full_config, ecg, name):
    model = _model(full_config, name)
    logits = model(ecg)
    assert logits.shape == (2, 5) and torch.isfinite(logits).all()
    logits.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_every_condition_matches_the_original_in_size(full_config):
    """Only the ordering changes, so the parameter count must not move."""
    counts = {
        name: sum(p.numel() for p in _model(full_config, name).parameters())
        for name in (ORDERING_ORIGINAL, *ORDERING)
    }
    assert len(set(counts.values())) == 1, counts


def test_shuffle_is_off_outside_experiment_7(full_config):
    for name in ("E2_RLQ", "E3_U", "E4_320ms", "E1_lstm_attention"):
        model = _model(full_config, name)
        assert model.shuffle is None
        assert model.settings["shuffle"] is None


def test_block_sizes_are_the_pre_specified_ones(full_config):
    assert [block for _, block in SHUFFLE_LEVELS] == [40, 80, 160, None]
    assert [scope for scope, _ in SHUFFLE_SCOPES] == ["q", "joint"]
    for label, block in SHUFFLE_LEVELS:
        model = _model(full_config, f"E7_q_{label}")
        assert model.settings["shuffle_block_samples"] == (block // 2 if block else 1)


def test_ordering_verdicts():
    def rows(original, block, point):
        summary = {ORDERING_ORIGINAL: {"macro_auroc_mean": original}}
        for scope, _ in SHUFFLE_SCOPES:
            for label, _ in SHUFFLE_LEVELS:
                value = point if label == "full" else block
                summary[f"E7_{scope}_{label}"] = {"macro_auroc_mean": value}
        return summary

    assert "Incomplete" in interpret_ordering({})
    hierarchy = interpret_ordering(rows(0.90, 0.88, 0.85))
    assert "temporal organisation carries information" in hierarchy
    flat = interpret_ordering(rows(0.90, 0.90, 0.899))
    assert "not distinguishable from a bag of local states" in flat
    local = interpret_ordering(rows(0.90, 0.899, 0.85))
    assert "local dynamics matter" in local
