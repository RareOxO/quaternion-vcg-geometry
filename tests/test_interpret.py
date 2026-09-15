"""The Disease x R/L/Q x R-peak-aligned-time contribution heatmap (plan section 9).

The figure is only readable if the perturbation removes information from one branch at
one phase and changes nothing else. These tests pin that down, and pin the quaternion
filler to the sphere -- a component-wise interpolation would leave it and stop being a
rotation.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from qdg.config import load_config, validate_config
from qdg.experiments import HANDCRAFTED_PROPOSED, experiment_config
from qdg.interpret import (
    BRANCHES,
    DISEASES,
    RELATIVE_MS,
    _slerp,
    _windows,
    aggregate,
    perturb,
    relative_times,
)
from qdg.data import CLASSES
from qdg.models import build_model

STATS = {"sampling_rate": 500, "signal_length": 5000, "vcg_std": [0.19716, 0.12212, 0.17387]}


@pytest.fixture
def full_config():
    return load_config(Path(__file__).resolve().parents[1] / "configs/base.yaml")


# --- SLERP ---


def test_slerp_stays_on_the_sphere_and_moves_at_constant_rate():
    start = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
    end = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]])
    weights = torch.linspace(0, 1, 5).view(1, 1, 5)
    out = _slerp(start, end, weights)[0, 0]
    torch.testing.assert_close(out.norm(dim=-1), torch.ones(5), atol=1e-5, rtol=0)
    angles = 2 * out[:, 0].clamp(-1, 1).arccos()
    steps = angles[1:] - angles[:-1]
    torch.testing.assert_close(steps, steps[0].expand(4), atol=1e-4, rtol=1e-3)


def test_slerp_takes_the_shorter_arc():
    """q and -q are the same rotation; the path must not go the long way round."""
    start = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
    nearby = torch.tensor([[[-0.9999, 0.0, 0.0, 0.0141]]])
    middle = _slerp(start, nearby, torch.tensor([[[0.5]]]))[0, 0, 0]
    assert middle[0] > 0.99, middle


def test_slerp_endpoints_are_the_inputs():
    start = torch.nn.functional.normalize(torch.randn(2, 3, 4), dim=-1)
    end = torch.nn.functional.normalize(torch.randn(2, 3, 4), dim=-1)
    out = _slerp(start, end, torch.tensor([0.0, 1.0]).expand(2, 3, 2))
    torch.testing.assert_close(out[..., 0, :].abs(), start.abs(), atol=1e-4, rtol=1e-3)
    torch.testing.assert_close(out[..., 1, :].abs(), end.abs(), atol=1e-4, rtol=1e-3)


# --- Perturbation ---


def test_real_branch_is_interpolated_between_the_boundaries():
    values = torch.arange(20.0).view(1, 1, 20)
    positions = torch.tensor([[[8, 9, 10, 11]]])
    out = perturb(values, positions, torch.ones(1, 1, dtype=torch.bool), quaternion=False)
    # A ramp interpolates back to itself, which is what "no artificial edge" means.
    torch.testing.assert_close(out, values, atol=1e-5, rtol=0)


def test_perturbation_touches_only_the_window():
    values = torch.randn(1, 3, 40, generator=torch.manual_seed(2))
    positions = torch.tensor([[[10, 11, 12, 13]]])
    out = perturb(values, positions, torch.ones(1, 1, dtype=torch.bool), quaternion=False)
    torch.testing.assert_close(out[..., :10], values[..., :10])
    torch.testing.assert_close(out[..., 14:], values[..., 14:])
    assert not torch.allclose(out[..., 10:14], values[..., 10:14])


def test_invalid_beats_are_skipped():
    values = torch.randn(1, 2, 30, generator=torch.manual_seed(3))
    positions = torch.tensor([[[5, 6, 7, 8]]])
    out = perturb(values, positions, torch.zeros(1, 1, dtype=torch.bool), quaternion=False)
    torch.testing.assert_close(out, values)


def test_quaternion_filler_stays_a_rotation():
    quaternions = torch.nn.functional.normalize(
        torch.randn(2, 4, 40, generator=torch.manual_seed(4)), dim=1
    )
    positions = torch.tensor([[[10, 11, 12, 13]], [[20, 21, 22, 23]]])
    valid = torch.ones(2, 1, dtype=torch.bool)
    for mode in ("interpolate", "identity"):
        out = perturb(quaternions, positions, valid, quaternion=True, mode=mode)
        assert (out.norm(dim=1) - 1).abs().max() < 1e-5, mode
    identity = perturb(quaternions, positions, valid, quaternion=True, mode="identity")
    torch.testing.assert_close(identity[0, :, 11], torch.tensor([1.0, 0.0, 0.0, 0.0]))


def test_windows_mark_out_of_range_beats_invalid():
    peaks = [np.array([5, 2500, 4995])]
    positions, valid = _windows(peaks, 0, 10, 5000, torch.device("cpu"), 3)
    assert valid.tolist() == [[False, True, False]], "beats at the edges cannot be windowed"
    assert positions.min() >= 1 and positions.max() <= 4998


# --- Grid and aggregation ---


def test_time_grid_matches_the_plan():
    times = relative_times()
    assert times[0] == RELATIVE_MS[0] == -300
    assert times[-1] == RELATIVE_MS[1] == 600
    assert np.all(np.diff(times) == 10)
    assert 0 in times, "the R peak itself must be a sampled phase"


def test_aggregation_weights_patients_not_records():
    """A patient with several records must not count several times."""
    times = relative_times(300)
    records, steps = 4, len(times)
    scores = np.zeros((records, steps, len(BRANCHES), len(CLASSES)), dtype=np.float32)
    column = CLASSES.index("MI")
    scores[:, :, 0, column] = np.array([1.0, 1.0, 1.0, 5.0])[:, None]
    labels = np.zeros((records, len(CLASSES)), dtype=np.float32)
    labels[:, column] = 1
    # Three records belong to one patient, the fourth to another.
    patients = np.array([7, 7, 7, 9])
    summary = aggregate(
        {"contributions": scores, "labels": labels, "patients": patients, "times_ms": times},
        bootstrap=32,
    )
    panel = summary["panels"][DISEASES.index("MI"), 0]
    # Patient means are 1 and 5, so the population mean is 3, not (1+1+1+5)/4 = 2.
    np.testing.assert_allclose(panel, np.full(steps, 3.0), rtol=1e-5)
    assert summary["counts"]["MI"] == {"records": 4, "patients": 2}


def test_aggregation_shape_and_bounds():
    times = relative_times(100)
    records = 6
    rng = np.random.default_rng(0)
    scores = rng.normal(size=(records, len(times), len(BRANCHES), len(CLASSES))).astype(np.float32)
    labels = np.ones((records, len(CLASSES)), dtype=np.float32)
    summary = aggregate(
        {
            "contributions": scores,
            "labels": labels,
            "patients": np.arange(records),
            "times_ms": times,
        },
        bootstrap=64,
    )
    assert summary["panels"].shape == (len(DISEASES), len(BRANCHES), len(times))
    assert (summary["lower"] <= summary["panels"] + 1e-5).all()
    assert (summary["upper"] >= summary["panels"] - 1e-5).all()


def test_analysis_needs_all_three_branches(full_config):
    """Only a full R+L+Q model can produce three rows per disease panel."""
    partial = experiment_config(full_config, "E2_RQ")
    validate_config(partial)
    assert set(build_model(partial["model"], STATS).branches) != set(BRANCHES)
    full = experiment_config(full_config, HANDCRAFTED_PROPOSED)
    assert set(build_model(full["model"], STATS).branches) == set(BRANCHES)


def test_fuse_matches_forward_features(full_config):
    """Perturbation rejoins through fuse(), so it must equal the normal path."""
    model = build_model(experiment_config(full_config, "E2_RLQ")["model"], STATS).eval()
    ecg = torch.randn(2, 12, 5000, generator=torch.manual_seed(9))
    with torch.inference_mode():
        features = {branch: model.frontends[branch](ecg) for branch in model.branches}
        torch.testing.assert_close(model.fuse(features), model.forward_features(ecg))
