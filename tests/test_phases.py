"""Cardiac-phase-resolved attribution.

Two things can quietly ruin this analysis. The delineator can repair a boundary instead
of flagging it, which would put an invented QRS into the numbers; and the mapping from
perturbation windows to phases can assign a window to whichever phase its centre lands
in, which at a 40 ms window silently moves a third of the mass across QRS_off. The tests
below pin both, plus the patient-level aggregation and the quaternion's half-lag
reference.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from qdg.data import CLASSES
from qdg.delineate import (
    QRS_MS,
    delineate_beats,
    delineate_record,
    quality_flags,
    spatial_velocity,
)
from qdg.geometry import INDEPENDENT_LEADS, KORS
from qdg.interpret import BRANCHES, DISEASES, _windows
from qdg.phases import (
    DELTA_Q_MS,
    PHASES,
    PHASES_ALL,
    branch_shares,
    disease_summary,
    overlap_fraction,
    patient_table,
    record_weights,
    window_bounds,
)

RATE = 500


def _read(path):
    import csv

    with Path(path).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _hanning(length, centre, width, into):
    span = slice(max(0, centre - width // 2), min(length, centre + width // 2))
    size = span.stop - span.start
    if size > 2:
        into[span] += np.hanning(size)


def _vcg(beats=10, rate=RATE, seconds=10, qrs_ms=80, t_delay_ms=240, t_ms=200, t_gain=0.25):
    """A crude periodic cardiac vector: a sharp complex and a broad T wave per beat."""
    length = rate * seconds
    shape = np.zeros(length)
    tail = np.zeros(length)
    peaks = []
    for index in range(beats):
        centre = int((index + 0.5) * length / beats)
        peaks.append(centre)
        _hanning(length, centre, int(qrs_ms * rate / 1000), shape)
        if t_gain:
            _hanning(length, centre + int(t_delay_ms * rate / 1000), int(t_ms * rate / 1000), tail)
    signal = np.zeros((3, length))
    for axis, gain in enumerate((1.0, 0.6, 0.3)):
        signal[axis] = gain * (shape + t_gain * tail)
    return signal, np.array(peaks)


def _ecg(vcg):
    """12 leads whose Kors transform reproduces the shape of `vcg`."""
    ecg = np.zeros((12, vcg.shape[-1]), dtype=np.float32)
    ecg[list(INDEPENDENT_LEADS)] = np.asarray(KORS, dtype=np.float64).T @ vcg
    return ecg


# --- Delineation ---


def test_delineation_recovers_the_planted_boundaries():
    vcg, peaks = _vcg()
    beats = delineate_beats(vcg, peaks, RATE)
    beats.update(quality_flags(beats))
    assert beats["valid_full_beat"].all(), beats
    per_ms = RATE / 1000
    assert np.allclose(beats["qrs_on"], peaks - 40 * per_ms, atol=8 * per_ms)
    assert np.allclose(beats["qrs_off"], peaks + 40 * per_ms, atol=12 * per_ms)
    assert np.allclose(beats["t_peak"], peaks + 240 * per_ms, atol=25 * per_ms)
    assert (beats["t_end"] > beats["t_peak"]).all()
    assert (beats["t_end"] < peaks + 460 * per_ms).all()


def test_temporal_ordering_holds_for_every_valid_beat():
    vcg, peaks = _vcg()
    beats = delineate_record(_ecg(vcg), RATE, peaks)
    keep = beats["valid_full_beat"]
    for first, second in zip(
        ("qrs_on", "r_peak", "qrs_off", "t_peak"), ("r_peak", "qrs_off", "t_peak", "t_end")
    ):
        assert (beats[first][keep] < beats[second][keep]).all(), (first, second)


def test_a_missing_t_wave_keeps_the_qrs_and_drops_repolarisation():
    """The two flags are separate exactly so this beat still counts towards DEP."""
    vcg, peaks = _vcg(t_gain=0.0)
    beats = delineate_beats(vcg, peaks, RATE)
    beats.update(quality_flags(beats))
    assert beats["valid_qrs"].all()
    assert not beats["valid_twave"].any()


def test_an_implausible_duration_is_flagged_and_not_repaired():
    vcg, peaks = _vcg(qrs_ms=220, t_gain=0.0)
    beats = delineate_beats(vcg, peaks, RATE)
    beats.update(quality_flags(beats))
    assert not beats["valid_qrs"].any(), beats["qrs_duration_ms"]
    # The boundary is still the one that was found; nothing was moved to fit the range.
    assert (beats["qrs_on"] >= 0).all() and (beats["qrs_duration_ms"] > QRS_MS[1]).all()


def test_a_complex_wider_than_the_search_span_reports_failure():
    """Not a boundary clamped to the edge of the search: no boundary at all."""
    vcg, peaks = _vcg(qrs_ms=460, t_gain=0.0)
    beats = delineate_beats(vcg, peaks, RATE)
    assert (beats["qrs_on"] == -1).all() and (beats["qrs_off"] == -1).all()
    assert np.isnan(beats["qrs_duration_ms"]).all()


def test_spatial_velocity_uses_every_lead():
    """One lead going quiet must not silence the delineation front end."""
    vcg, _ = _vcg()
    quiet = vcg.copy()
    quiet[0] = 0.0
    assert spatial_velocity(quiet, RATE).max() > 0.3 * spatial_velocity(vcg, RATE).max()


# --- Window to phase mapping ---


def test_window_bounds_match_the_perturbation_grid():
    """The phase mapping must describe the windows that were actually perturbed."""
    times = np.array([-100, 0, 130])
    peaks = [np.array([1000])]
    starts, ends, width = window_bounds(times, 40, RATE, "radial")
    for step, offset_ms in enumerate(times):
        offset = int(round(offset_ms * RATE / 1000))
        positions, _ = _windows(peaks, offset, width, 5000, torch.device("cpu"), 1)
        first, last = int(positions[0, 0, 0]), int(positions[0, 0, -1])
        assert (first - 1000) * 1000 / RATE == pytest.approx(starts[step])
        assert (last + 1 - 1000) * 1000 / RATE == pytest.approx(ends[step])


def test_quaternion_windows_carry_the_half_lag_reference():
    """q_t spans [t, t + 20 ms], so its physiological instant is the midpoint."""
    times = np.array([0, 50])
    real, _, _ = window_bounds(times, 20, RATE, "radial")
    angular, _, _ = window_bounds(times, 20, RATE, "angular")
    assert np.allclose(angular - real, DELTA_Q_MS / 2)


def test_a_window_is_split_across_a_boundary_by_overlap():
    assert overlap_fraction(-30.0, 10.0, -100.0, 0.0) == pytest.approx(0.75)
    assert overlap_fraction(-30.0, 10.0, 0.0, 300.0) == pytest.approx(0.25)
    # Centre-based assignment would give this window entirely to the first phase.
    assert overlap_fraction(-30.0, 10.0, -100.0, 0.0) != 1.0


def _one_beat(
    qrs_on=-40, qrs_off=40, t_peak=240, t_end=400, valid_qrs=True, valid_twave=True, peak=2500
):
    per_ms = RATE / 1000
    return {
        "r_peak": np.array([peak]),
        "qrs_on": np.array([peak + int(qrs_on * per_ms)]),
        "qrs_off": np.array([peak + int(qrs_off * per_ms)]),
        "t_peak": np.array([peak + int(t_peak * per_ms)]),
        "t_end": np.array([peak + int(t_end * per_ms)]),
        "valid_qrs": np.array([valid_qrs]),
        "valid_twave": np.array([valid_twave]),
    }


def test_repolarisation_is_exactly_its_two_halves():
    """The optional split at the T peak must never disagree with the primary REP."""
    beats = _one_beat()
    times = np.array([100, 200, 300])
    alpha = record_weights(beats, beats["r_peak"], times, 20, RATE, 5000)
    total = alpha[:, :, PHASES_ALL.index("EARLY_REP")] + alpha[:, :, PHASES_ALL.index("LATE_REP")]
    assert np.allclose(alpha[:, :, PHASES_ALL.index("REP")], total)
    assert total.max() > 0


def test_weights_split_a_straddling_window_between_the_phases():
    beats = _one_beat()
    times = np.array([40])
    alpha = record_weights(beats, beats["r_peak"], times, 40, RATE, 5000)
    radial = alpha[BRANCHES.index("radial"), 0]
    primary = [radial[PHASES_ALL.index(phase)] for phase in PHASES]
    assert min(primary) > 0, "the window straddles QRS_off, so both phases take a share"
    assert sum(primary) == pytest.approx(1.0, abs=1e-6)


def test_repolarisation_ignores_beats_whose_t_wave_failed_qc():
    beats = _one_beat(valid_twave=False)
    times = np.array([200])
    alpha = record_weights(beats, beats["r_peak"], times, 20, RATE, 5000)
    assert alpha[:, 0, PHASES.index("REP")].max() == 0.0
    assert (
        record_weights(_one_beat(), beats["r_peak"], times, 20, RATE, 5000)[
            :, 0, PHASES.index("REP")
        ].min()
        > 0.0
    )


def test_a_window_falling_outside_the_record_carries_no_weight():
    beats = _one_beat(peak=10)
    times = np.array([-300])
    alpha = record_weights(beats, beats["r_peak"], times, 20, RATE, 5000)
    assert alpha.max() == 0.0


# --- Aggregation ---


def _run(records, patients, disease="MI", times=np.array([-40, 0, 40, 200])):
    labels = np.zeros((records, len(CLASSES)), dtype=np.float32)
    labels[:, CLASSES.index(disease)] = 1
    peaks = np.tile(np.array([1000, 2000, 3000, 4000]), records)
    return {
        "contributions": np.ones((records, len(times), len(BRANCHES), len(CLASSES)), np.float32),
        "labels": labels,
        "patients": np.asarray(patients),
        "ecg_ids": np.arange(records),
        "cache_rows": np.arange(records),
        "peaks": peaks,
        "peak_offsets": np.arange(records + 1) * 4,
        "times_ms": times,
        "window_ms": 20,
        "quaternion_mode": "interpolate",
    }


def _beats_for(run, **kwargs):
    return [
        {
            key: np.repeat(value, 4)
            for key, value in _one_beat(peak=0, **kwargs).items()
            if key != "r_peak"
        }
        | {"r_peak": np.array([1000, 2000, 3000, 4000])}
        for _ in range(len(run["cache_rows"]))
    ]


def _fix(beats):
    for current in beats:
        for name in ("qrs_on", "qrs_off", "t_peak", "t_end"):
            current[name] = current["r_peak"] + current[name]
    return beats


def test_a_patient_with_many_records_counts_once():
    """Five recordings of one patient must not outvote a patient with one."""
    run = _run(6, [1, 1, 1, 1, 1, 2])
    beats = _fix(_beats_for(run))
    run["contributions"][5] *= 3.0
    table = patient_table(run, beats, RATE, 5000)
    summary = disease_summary(table)
    cell = summary[("MI", "R", "DEP", "absolute_density")]
    assert cell["patients"] == 2
    assert cell["mean"] == pytest.approx(2.0, rel=1e-6), "pooling records would give ~1.33"


def test_branch_shares_sum_to_one():
    run = _run(4, [1, 2, 3, 4])
    table = patient_table(run, _fix(_beats_for(run)), RATE, 5000)
    shares = branch_shares(table)
    for disease in DISEASES[:1]:
        for phase in PHASES:
            total = sum(shares[(disease, branch, phase)]["share"] for branch in ("R", "L", "Q"))
            assert total == pytest.approx(1.0, abs=1e-6), (disease, phase)


def test_density_is_normalised_by_phase_length_but_mass_is_not():
    """DEP is short and REP is long; only mass may reflect that."""
    run = _run(3, [1, 2, 3])
    table = patient_table(run, _fix(_beats_for(run)), RATE, 5000)
    summary = disease_summary(table)
    dep = summary[("MI", "R", "DEP", "absolute_density")]["mean"]
    rep = summary[("MI", "R", "REP", "absolute_density")]["mean"]
    assert dep == pytest.approx(rep, rel=1e-6), "a constant contribution has equal density"


# --- End to end ---


def test_every_required_output_is_written(synthetic_cache, tmp_path):
    from qdg.phases import run as run_phases

    config = synthetic_cache[0]
    cache = Path(config["data"]["cache"])
    rate = 100
    signals = np.load(cache / "signals.npy", allow_pickle=False)
    vcg, peaks = _vcg(beats=10, rate=rate, seconds=10)
    for index in range(len(signals)):
        signals[index] = _ecg(vcg * (1 + 0.05 * index))
    # Same shape and dtype, so the cache manifest's size check still passes.
    np.save(cache / "signals.npy", signals, allow_pickle=False)
    patients = np.load(cache / "patient_ids.npy", allow_pickle=False)
    ecg_ids = np.load(cache / "ecg_ids.npy", allow_pickle=False)
    labels = np.load(cache / "labels.npy", allow_pickle=False)
    root = tmp_path / "attribution"
    times = np.arange(-300, 601, 50)
    rng = np.random.default_rng(0)
    for window, mode in ((20, "interpolate"), (10, "interpolate"), (20, "identity")):
        folder = root / f"w{window}_{mode}"
        folder.mkdir(parents=True)
        np.savez_compressed(
            folder / f"attribution_records_{mode}.npz",
            contributions=rng.normal(size=(len(signals), len(times), 3, len(CLASSES))).astype(
                np.float32
            ),
            labels=labels,
            patients=patients,
            ecg_ids=ecg_ids,
            cache_rows=np.arange(len(signals)),
            peaks=np.tile(peaks, len(signals)),
            peak_offsets=np.arange(len(signals) + 1) * len(peaks),
            times_ms=times,
            window_ms=np.int64(window),
            stride_ms=np.int64(50),
            sampling_rate=np.int64(rate),
            signal_length=np.int64(signals.shape[-1]),
            quaternion_mode=np.array(mode),
        )
    out = tmp_path / "phases"
    result = run_phases(config, root, output=out, bootstrap=50)
    expected = {
        "beat_boundaries.csv",
        "delineation_qc.csv",
        "phase_contribution_patient.csv",
        "phase_contribution_disease_summary.csv",
        "phase_branch_share.csv",
        "phase_robustness_window.csv",
        "phase_robustness_q_perturbation.csv",
    }
    figures = {
        "figure_interpretability_main",
        "figure_delineation_qc",
        "figure_identity_robustness",
        "figure_window_robustness",
    }
    written = {path.name for path in out.iterdir()}
    assert expected <= written, expected - written
    assert {f"{name}{suffix}" for name in figures for suffix in (".png", ".pdf")} <= written
    assert "phase_stability.csv" in written
    varied = {row["varied"] for row in result["stability"]}
    assert varied == {"window", "perturbation"}, result["stability"]
    quality = {row["delineation_quality"] for row in _read(out / "beat_boundaries.csv")}
    assert quality <= {"full", "qrs_only", "failed"} and "full" in quality
    groups = [row["group"] for row in result["delineation"]]
    assert groups[0] == "ALL" and set(DISEASES) <= set(groups), groups
    for phase in PHASES:
        total = sum(result["shares"][("MI", branch, phase)]["share"] for branch in ("R", "L", "Q"))
        assert total == pytest.approx(1.0, abs=1e-6)
