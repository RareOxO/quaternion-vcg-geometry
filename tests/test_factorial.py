"""Experiments 2 and 3: the R/L/Q factorial and the angular representation ablation.

Experiment 2 varies which branches exist; Experiment 3 holds the branches fixed and
varies only what the angular one carries. Both freeze E*, so the tests here are about
the branch wiring being exactly what each design claims.
"""

from pathlib import Path

import pytest
import torch

from qdg.config import load_config, validate_config
from qdg.encoders import QUATERNION_ENCODERS
from qdg.experiments import (
    EXPERIMENTS,
    FACTORIAL,
    FACTORIAL_CONTRIBUTIONS,
    FACTORIAL_REFERENCE,
    REPRESENTATION_ENCODER,
    REPRESENTATION_NEW,
    REPRESENTATION_REUSES_FACTORIAL,
    REPRESENTATIONS,
    SELECTED_ENCODER,
    experiment_config,
)
from qdg.models import ALL_BRANCHES, build_model
from qdg.quaternion_nn import QuaternionConv1d, QuaternionLinear

STATS = {"sampling_rate": 500, "signal_length": 5000, "vcg_std": [0.19716, 0.12212, 0.17387]}
EXPECTED_BRANCHES = {
    "E2_R": ["radial"],
    "E2_L": ["linear"],
    "E2_Q": ["angular"],
    "E2_RL": ["radial", "linear"],
    "E2_RQ": ["radial", "angular"],
    "E2_LQ": ["linear", "angular"],
    "E2_RLQ": ["radial", "linear", "angular"],
    "E2_raw": ["raw"],
}


@pytest.fixture
def full_config():
    return load_config(Path(__file__).resolve().parents[1] / "configs/base.yaml")


@pytest.fixture
def ecg():
    return torch.randn(2, 12, 5000, generator=torch.manual_seed(41))


def _model(config, name):
    current = experiment_config(config, name)
    validate_config(current)
    return build_model(current["model"], STATS)


# --- Experiment 2 ---


@pytest.mark.parametrize("name", [*FACTORIAL, FACTORIAL_REFERENCE])
def test_factorial_variants_build_and_train(full_config, ecg, name):
    model = _model(full_config, name)
    assert model.settings["branches"] == EXPECTED_BRANCHES[name]
    logits = model(ecg)
    assert logits.shape == (2, 5) and torch.isfinite(logits).all()
    logits.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_factorial_is_the_full_two_cubed_minus_one(full_config):
    """Seven non-empty subsets of {R, L, Q}, each present exactly once."""
    subsets = {frozenset(_model(full_config, name).settings["branches"]) for name in FACTORIAL}
    assert len(subsets) == len(FACTORIAL) == 7
    assert frozenset(["radial", "linear", "angular"]) in subsets
    for single in ("radial", "linear", "angular"):
        assert frozenset([single]) in subsets


def test_each_branch_carries_only_its_own_feature(full_config, ecg):
    """A branch's input must be identical whichever subset it appears in."""
    full = _model(full_config, "E2_RLQ")
    for name, branches in EXPECTED_BRANCHES.items():
        if name == FACTORIAL_REFERENCE:
            continue
        model = _model(full_config, name)
        for branch in branches:
            torch.testing.assert_close(
                model.frontends[branch](ecg), full.frontends[branch](ecg), atol=0, rtol=0
            )


def test_raw_reference_is_a_single_unnormalised_branch(full_config, ecg):
    """Raw XYZ is a reference, not a member of the factorial: one branch, magnitude kept."""
    raw = _model(full_config, FACTORIAL_REFERENCE)
    assert raw.settings["branches"] == ["raw"]
    assert raw.frontends["raw"].out_channels == 3
    quiet = torch.randn(1, 12, 5000, generator=torch.manual_seed(2)) * 0.1
    # Scaling the record scales the raw branch: amplitude is not divided away.
    torch.testing.assert_close(
        raw.frontends["raw"](quiet * 3), raw.frontends["raw"](quiet) * 3, atol=1e-4, rtol=1e-4
    )
    assert FACTORIAL_REFERENCE not in FACTORIAL


def test_conditional_contributions_are_leave_one_out(full_config):
    """Each contribution compares the full model against the model missing that branch."""
    for letter, full, without, _ in FACTORIAL_CONTRIBUTIONS:
        present = set(EXPECTED_BRANCHES[full]) - set(EXPECTED_BRANCHES[without])
        assert len(present) == 1
        assert present.pop().startswith({"R": "rad", "L": "lin", "Q": "ang"}[letter])


def test_quaternion_layers_appear_only_on_the_angular_branch(full_config):
    """Plan section 2.1: R, L and raw stay real whatever E* is.

    With a generic E* no branch is quaternion at all; with a quaternion E* exactly the
    angular branch is. Either way, R/L/raw must never be quaternionized.
    """
    quaternion_star = SELECTED_ENCODER in QUATERNION_ENCODERS
    for name, branches in EXPECTED_BRANCHES.items():
        model = _model(full_config, name)
        for branch in branches:
            has_quaternion = any(
                isinstance(m, (QuaternionLinear, QuaternionConv1d))
                for m in model.encoders[branch].modules()
            )
            assert has_quaternion == (quaternion_star and branch == "angular"), (name, branch)


def test_invalid_branch_sets_are_rejected(full_config):
    base = experiment_config(full_config, "E2_RLQ")["model"]
    for branches in ([], ["radial", "radial"], ["nonsense"], ["raw", "radial"]):
        with pytest.raises(ValueError):
            build_model({**base, "branches": branches}, STATS)
    assert set(ALL_BRANCHES) == {"raw", "radial", "linear", "angular"}


# --- Experiment 3 ---


@pytest.mark.parametrize("name", [name for _, name, _ in REPRESENTATIONS])
def test_representation_variants_build_and_train(full_config, ecg, name):
    model = _model(full_config, name)
    assert model.settings["branches"] == ["radial", "linear", "angular"]
    logits = model(ecg)
    assert logits.shape == (2, 5) and torch.isfinite(logits).all()
    logits.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_representation_ablation_changes_only_the_angular_input(full_config, ecg):
    models = {name: _model(full_config, name) for _, name, _ in REPRESENTATIONS}
    rotation = REPRESENTATIONS[-1][1]
    reference = models[rotation]
    for name, model in models.items():
        # R and L are bit-identical across all three.
        for branch in ("radial", "linear"):
            torch.testing.assert_close(
                model.frontends[branch](ecg), reference.frontends[branch](ecg), atol=0, rtol=0
            )
        # Every branch uses the generic operator, so the representation is the only change.
        assert model.settings["branch_encoders"]["angular"] == REPRESENTATION_ENCODER
        assert model.settings["branch_encoders"]["angular"] not in QUATERNION_ENCODERS, name
    channels = {name: model.frontends["angular"].out_channels for name, model in models.items()}
    assert channels == {"E3_U": 3, "E3_D": 4, rotation: 4}
    # U is a 3-vector, which is exactly why a quaternion operator cannot take it.
    assert channels["E3_U"] % 4 != 0


def test_d_and_q_encode_the_same_angle(full_config, ecg):
    """Plan section 4: neither carries more raw information; they are parameterizations."""
    d = _model(full_config, "E3_D").frontends["angular"](ecg)
    q = _model(full_config, REPRESENTATIONS[-1][1]).frontends["angular"](ecg)
    torch.testing.assert_close(
        q[:, 0].clamp(-1, 1).arccos() * 2, d[:, 0].clamp(-1, 1).arccos(), atol=1e-3, rtol=1e-3
    )
    assert not torch.allclose(d, q, atol=1e-2)


def test_representation_q_row_is_reused_when_e_star_is_generic(full_config):
    """A generic E* needs no operator deviation, so Exp 3's Q row IS the factorial run."""
    generic = SELECTED_ENCODER not in QUATERNION_ENCODERS
    assert REPRESENTATION_REUSES_FACTORIAL == generic
    assert REPRESENTATION_ENCODER == (
        SELECTED_ENCODER if generic else {"qgnn": "tcn"}.get(SELECTED_ENCODER, SELECTED_ENCODER)
    )
    if generic:
        assert REPRESENTATIONS[-1][1] == "E2_RLQ"
        assert "E3_Q" not in EXPERIMENTS
        assert REPRESENTATION_NEW == ("E3_U", "E3_D")
    # Whatever E* is, the three rows are three distinct experiments.
    assert len({name for _, name, _ in REPRESENTATIONS}) == 3


def test_earlier_experiments_are_untouched(full_config):
    for name, channels in (("M0", 3), ("M1", 4), ("M4", 32), ("RLA", 8)):
        assert _model(full_config, name).frontend.out_channels == channels
    assert _model(full_config, "E1_qgnn").settings["branches"] == ["radial", "linear", "angular"]
