"""Cardiac-phase-resolved attribution: from R-peak-relative time to DEP and REP.

The R-peak-relative heatmap says *when*, relative to one landmark, each of R, L and Q
contributes. It cannot say whether that time was inside the patient's own depolarisation
or already in repolarisation, because a fixed window such as [-60, +60] ms is nobody's
QRS in particular. This module carries the same contributions -- unchanged, not
recomputed -- onto each patient's own intervals from `delineate`.

    DEP = [QRS_on, QRS_off]        ventricular depolarisation
    REP = (QRS_off, T_end]         post-QRS ventricular repolarisation

Three things decide whether the mapping is honest:

* Overlap weighting. A perturbation window is 10-40 ms wide and frequently straddles
  QRS_off, so it is split between the phases by the fraction of its duration on each
  side, never assigned whole to the phase its centre happens to fall in.
* The quaternion's temporal reference. Q_t is the rotation from u_t to u_{t+dQ} with
  dQ = 20 ms, so it is interval-valued, and its physiological instant is the interval
  midpoint t + dQ/2. The Q windows are shifted by that half lag; R and L keep the
  existing convention. Without this the Q attribution sits ~10 ms early at every
  boundary.
* Phase length. DEP is roughly 80 ms and REP roughly 250 ms, so the accumulated mass is
  larger in REP for no interesting reason. The phase comparison is therefore made on
  density -- mass per unit of overlap weight -- with mass reported alongside it.

Nothing here assumes which branch or which phase should come out on top.
"""

import csv
import json
import warnings
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .data import CLASSES, load_manifest
from .delineate import DURATIONS, FIDUCIALS, delineate_record
from .interpret import BRANCH_LABELS, BRANCHES, CLIP_QUANTILE, DISEASES

# The primary comparison, and what the figure shows.
PHASES = ("DEP", "REP")
# What the tables carry. The plan allows repolarisation to be split at the T peak when
# that fiducial is reliable, so the split is always computed and written; REP is the sum
# of its two halves by construction rather than a separate estimate, so the subdivision
# can never disagree with the primary comparison.
PHASES_ALL = ("DEP", "REP", "EARLY_REP", "LATE_REP")
# The lag the angular frontend uses, in milliseconds. Not a free parameter here: it has
# to match the one the model was trained with.
DELTA_Q_MS = 20
BOOTSTRAP = 1000
SEED = 0
PRIMARY = {"window_ms": 20, "quaternion_mode": "interpolate"}
METRICS = ("signed_density", "absolute_density", "absolute_mass")


def load_runs(root):
    """Every saved per-record attribution under `root`, newest layout first."""
    runs = []
    for path in sorted(Path(root).rglob("attribution_records_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            runs.append({key: data[key] for key in data.files} | {"path": str(path)})
        runs[-1]["window_ms"] = int(runs[-1]["window_ms"])
        runs[-1]["quaternion_mode"] = str(runs[-1]["quaternion_mode"].item())
    if not runs:
        raise ValueError(f"No attribution_records_*.npz under {root}; run `qdg interpret` first")
    return runs


def _ragged(flat, offsets, row):
    return flat[offsets[row] : offsets[row + 1]]


def delineation(config, run):
    """Fiducials for every record of the attribution run, in cache order."""
    cache = Path(config["data"]["cache"])
    rate = load_manifest(config["data"])["stats"]["sampling_rate"]
    signals = np.load(cache / "signals.npy", mmap_mode="r", allow_pickle=False)
    rows = []
    for record, index in enumerate(tqdm(run["cache_rows"], desc="Delineation")):
        peaks = _ragged(run["peaks"], run["peak_offsets"], record)
        rows.append(delineate_record(np.asarray(signals[index]), rate, peaks))
    return rows, rate


def window_bounds(times_ms, window_ms, sampling_rate, branch):
    """Start and end of each perturbation window, in milliseconds relative to the R peak.

    Mirrors `interpret._windows`: the window is centred on the offset and is
    `max(2, round(window_ms * rate / 1000))` samples wide, so the bounds are derived from
    the sample grid rather than from `window_ms` directly.
    """
    per_ms = sampling_rate / 1000.0
    width = max(2, int(round(window_ms * per_ms)))
    offsets = np.round(np.asarray(times_ms) * per_ms).astype(int)
    first = offsets - width // 2
    shift = DELTA_Q_MS / 2 if BRANCH_LABELS[branch] == "Q" else 0.0
    return first / per_ms + shift, (first + width) / per_ms + shift, width


def overlap_fraction(window_start, window_end, phase_start, phase_end):
    """|W ∩ P| / |W|, broadcasting over any shape."""
    inner = np.minimum(window_end, phase_end) - np.maximum(window_start, phase_start)
    return np.clip(inner, 0.0, None) / (window_end - window_start)


def record_weights(beats, peaks, times_ms, window_ms, sampling_rate, steps):
    """Overlap weight alpha[branch, time, phase] for one record.

    A beat counts towards a phase only if that phase's delineation passed QC, and only
    at offsets where its perturbation window actually fell inside the record -- the same
    condition `interpret._windows` applies when it decides which beats to perturb.
    """
    per_ms = sampling_rate / 1000.0
    alpha = np.zeros((len(BRANCHES), len(times_ms), len(PHASES_ALL)), dtype=np.float64)
    if not len(peaks):
        return alpha
    relative = {
        "DEP": (beats["qrs_on"] - beats["r_peak"], beats["qrs_off"] - beats["r_peak"]),
        "EARLY_REP": (beats["qrs_off"] - beats["r_peak"], beats["t_peak"] - beats["r_peak"]),
        "LATE_REP": (beats["t_peak"] - beats["r_peak"], beats["t_end"] - beats["r_peak"]),
    }
    eligible = {
        "DEP": beats["valid_qrs"],
        "EARLY_REP": beats["valid_twave"],
        "LATE_REP": beats["valid_twave"],
    }
    for axis, branch in enumerate(BRANCHES):
        starts, ends, width = window_bounds(times_ms, window_ms, sampling_rate, branch)
        # (time, beat) -- which beats were really perturbed at this offset.
        offsets = np.round(np.asarray(times_ms) * per_ms).astype(int)
        first = peaks[None, :] + offsets[:, None] - width // 2
        applied = (first >= 1) & (first + width - 1 < steps - 1)
        for phase, (low_ms, high_ms) in relative.items():
            keep = eligible[phase]
            if not keep.any():
                continue
            low, high = (np.asarray(v, dtype=float)[keep] / per_ms for v in (low_ms, high_ms))
            share = overlap_fraction(starts[:, None], ends[:, None], low[None, :], high[None, :])
            alpha[axis, :, PHASES_ALL.index(phase)] = np.where(applied[:, keep], share, 0.0).mean(
                axis=1
            )
    # REP is exactly its two halves: the same beats are eligible and the intervals meet
    # at the T peak without overlapping.
    alpha[:, :, PHASES_ALL.index("REP")] = (
        alpha[:, :, PHASES_ALL.index("EARLY_REP")] + alpha[:, :, PHASES_ALL.index("LATE_REP")]
    )
    return alpha


def record_metrics(scores, alpha):
    """S, D and A for one record: (metric, disease, branch, phase), plus the weight sum.

    scores is (time, branch, class); alpha is (branch, time, phase).
    """
    columns = [CLASSES.index(disease) for disease in DISEASES]
    signed = np.asarray(scores)[:, :, columns]  # (time, branch, disease)
    weight = np.transpose(alpha, (1, 0, 2))  # (time, branch, phase)
    total = weight.sum(axis=0)  # (branch, phase)
    mass_signed = np.einsum("tbd,tbp->dbp", signed, weight)
    mass_absolute = np.einsum("tbd,tbp->dbp", np.abs(signed), weight)
    with np.errstate(invalid="ignore", divide="ignore"):
        density_signed = np.where(total > 0, mass_signed / total, np.nan)
        density_absolute = np.where(total > 0, mass_absolute / total, np.nan)
    mass = np.where(total > 0, mass_absolute, np.nan)
    return np.stack([density_signed, density_absolute, mass]), total


def patient_table(run, beats, rate, signal_length):
    """Patient-level values: dict (disease, branch, phase, metric) -> {patient: value}.

    Records collapse to their patient first, so a patient with several recordings does
    not count several times downstream.
    """
    times = run["times_ms"]
    per_record = np.full(
        (len(run["cache_rows"]), len(METRICS), len(DISEASES), len(BRANCHES), len(PHASES_ALL)),
        np.nan,
    )
    for record in range(len(run["cache_rows"])):
        peaks = _ragged(run["peaks"], run["peak_offsets"], record)
        alpha = record_weights(
            beats[record], peaks, times, run["window_ms"], rate, int(signal_length)
        )
        per_record[record] = record_metrics(run["contributions"][record], alpha)[0]
    # Every cell exists even when a disease has no positive record here, so the tables
    # and the figure always have the same shape and a missing group reads as "no
    # patients" rather than as a crash.
    table = {
        (disease, BRANCH_LABELS[branch], phase, metric): {}
        for disease in DISEASES
        for branch in BRANCHES
        for phase in PHASES_ALL
        for metric in METRICS
    }
    labels, patients = run["labels"], run["patients"]
    for d, disease in enumerate(DISEASES):
        positive = np.flatnonzero(labels[:, CLASSES.index(disease)] == 1)
        for unique in np.unique(patients[positive]):
            rows = positive[patients[positive] == unique]
            # All-NaN cells are ordinary here: they are the phases this patient had no
            # eligible beat for, and they stay NaN rather than becoming a zero.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                values = np.nanmean(per_record[rows], axis=0)
            for m, metric in enumerate(METRICS):
                for b, branch in enumerate(BRANCHES):
                    for p, phase in enumerate(PHASES_ALL):
                        key = (disease, BRANCH_LABELS[branch], phase, metric)
                        table.setdefault(key, {})[int(unique)] = float(values[m, d, b, p])
    return table


def _bootstrap(values, rng, iterations=BOOTSTRAP):
    values = np.asarray([v for v in values if np.isfinite(v)])
    if not len(values):
        return float("nan"), float("nan"), float("nan"), 0
    draws = rng.integers(0, len(values), size=(iterations, len(values)))
    samples = values[draws].mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(samples, 0.025)),
        float(np.quantile(samples, 0.975)),
        int(len(values)),
    )


def disease_summary(table, seed=SEED):
    """Patient mean and 95% patient-bootstrap interval for every cell."""
    rng = np.random.default_rng(seed)
    summary = {}
    for key, per_patient in sorted(table.items()):
        mean, low, high, count = _bootstrap(list(per_patient.values()), rng)
        summary[key] = {"mean": mean, "low": low, "high": high, "patients": count}
    return summary


def branch_shares(table, seed=SEED):
    """R/L/Q composition of absolute mass within each disease and phase.

    Resampling the same patients for all three branches keeps the three shares summing
    to one inside every bootstrap draw.
    """
    rng = np.random.default_rng(seed)
    shares = {}
    for disease in DISEASES:
        for phase in PHASES_ALL:
            columns = [table[(disease, BRANCH_LABELS[b], phase, "absolute_mass")] for b in BRANCHES]
            common = sorted(set.intersection(*(set(column) for column in columns)))
            values = np.array(
                [[column[patient] for column in columns] for patient in common], dtype=float
            ).reshape(len(common), len(BRANCHES))
            values = values[np.isfinite(values).all(axis=1) & (values.sum(axis=1) > 0)]
            if not len(values):
                for branch in BRANCHES:
                    shares[(disease, BRANCH_LABELS[branch], phase)] = dict.fromkeys(
                        ("share", "low", "high"), float("nan")
                    ) | {"patients": 0}
                continue
            draws = rng.integers(0, len(values), size=(BOOTSTRAP, len(values)))
            totals = values[draws].mean(axis=1)
            fractions = totals / totals.sum(axis=1, keepdims=True)
            mean = values.mean(axis=0)
            mean = mean / mean.sum()
            for axis, branch in enumerate(BRANCHES):
                shares[(disease, BRANCH_LABELS[branch], phase)] = {
                    "share": float(mean[axis]),
                    "low": float(np.quantile(fractions[:, axis], 0.025)),
                    "high": float(np.quantile(fractions[:, axis], 0.975)),
                    "patients": int(len(values)),
                }
    return shares


def delineation_report(beats, run):
    """Success rates and duration distributions, overall and per disease."""
    labels = run["labels"]
    groups = {"ALL": np.arange(len(beats))}
    for disease in DISEASES:
        groups[disease] = np.flatnonzero(labels[:, CLASSES.index(disease)] == 1)
    rows = []
    for name, records in groups.items():
        flags = {
            key: np.concatenate([beats[r][key] for r in records])
            for key in ("valid_qrs", "valid_twave")
        }
        durations = {
            key: np.concatenate([beats[r][key] for r in records])
            for key in ("qrs_duration_ms", "qt_duration_ms")
        }
        qrs = durations["qrs_duration_ms"][flags["valid_qrs"]]
        qt = durations["qt_duration_ms"][flags["valid_twave"]]
        rows.append(
            {
                "group": name,
                "records": len(records),
                "records_with_valid_qrs": int(sum(beats[r]["valid_qrs"].any() for r in records)),
                "records_with_valid_twave": int(
                    sum(beats[r]["valid_twave"].any() for r in records)
                ),
                "beats": int(len(flags["valid_qrs"])),
                "valid_qrs_rate": float(flags["valid_qrs"].mean()),
                "valid_twave_rate": float(flags["valid_twave"].mean()),
                **_spread("qrs_ms", qrs),
                **_spread("qt_ms", qt),
            }
        )
    return rows


def _spread(prefix, values):
    if not len(values):
        return {f"{prefix}_{k}": float("nan") for k in ("median", "p05", "p95")}
    return {
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_p05": float(np.percentile(values, 5)),
        f"{prefix}_p95": float(np.percentile(values, 95)),
    }


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)


def beat_rows(run, beats, rate):
    rows = []
    for record, (ecg_id, patient) in enumerate(zip(run["ecg_ids"], run["patients"])):
        current = beats[record]
        for beat in range(len(current["r_peak"])):
            row = {"record_id": int(ecg_id), "patient_id": int(patient), "beat_id": beat}
            for name in FIDUCIALS:
                sample = int(current[name][beat])
                row[name] = sample
                row[f"{name}_ms"] = round(sample * 1000.0 / rate, 2) if sample >= 0 else ""
            for name in DURATIONS:
                value = current[name][beat]
                row[name] = round(float(value), 2) if np.isfinite(value) else ""
            for name in ("valid_qrs", "valid_twave", "valid_full_beat"):
                row[name] = int(current[name][beat])
            row["delineation_quality"] = (
                "full"
                if current["valid_full_beat"][beat]
                else "qrs_only"
                if current["valid_qrs"][beat]
                else "failed"
            )
            rows.append(row)
    return rows


def summary_rows(summary, run):
    return [
        {
            "disease": disease,
            "branch": branch,
            "phase": phase,
            "metric": metric,
            "window_ms": run["window_ms"],
            "quaternion_perturbation": run["quaternion_mode"],
            "mean": round(values["mean"], 6),
            "ci_low": round(values["low"], 6),
            "ci_high": round(values["high"], 6),
            "patients": values["patients"],
        }
        for (disease, branch, phase, metric), values in sorted(summary.items())
    ]


def _axes_grid(figure, spec, columns):
    return [figure.add_subplot(spec[0, column]) for column in range(columns)]


def main_figure(panels, times, summary, shares, path, clip):
    """Panel A: R-peak-relative contribution. Panel B: DEP vs REP density. Panel C: composition."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(10, 13))
    grid = figure.add_gridspec(6, 1, height_ratios=[4.0, 0.25, 2.6, 0.25, 1.9, 0.05], hspace=0.35)
    inner = grid[0].subgridspec(len(DISEASES), 1, hspace=0.12)
    image = None
    for row, (disease, panel) in enumerate(zip(DISEASES, panels)):
        axis = figure.add_subplot(inner[row])
        image = axis.imshow(
            panel,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-clip,
            vmax=clip,
            extent=[times[0], times[-1], len(BRANCHES) - 0.5, -0.5],
            interpolation="nearest",
        )
        axis.set_yticks(range(len(BRANCHES)))
        axis.set_yticklabels([BRANCH_LABELS[b] for b in BRANCHES])
        axis.set_ylabel(disease, rotation=0, labelpad=24, va="center", fontweight="bold")
        axis.axvline(0, color="black", linewidth=0.8, linestyle="--")
        if row == 0:
            axis.set_title("A  Contribution on R-peak-relative time", loc="left", fontweight="bold")
        if row < len(DISEASES) - 1:
            axis.set_xticklabels([])
        else:
            axis.set_xlabel("Time relative to R peak (ms)")
    figure.colorbar(
        image,
        ax=figure.axes[: len(DISEASES)],
        label=f"Signed contribution (clipped at the {int(CLIP_QUANTILE * 100)}th percentile)",
        pad=0.02,
    )

    colours = {"DEP": "#3a6ea5", "REP": "#c86a3a"}
    axis = figure.add_subplot(grid[2])
    width, gap = 0.36, 0.0
    ticks, labels = [], []
    for index, disease in enumerate(DISEASES):
        for slot, branch in enumerate(BRANCH_LABELS[b] for b in BRANCHES):
            centre = index * (len(BRANCHES) + 1) + slot
            ticks.append(centre)
            labels.append(branch)
            for side, phase in enumerate(PHASES):
                cell = summary[(disease, branch, phase, "absolute_density")]
                position = centre + (side - 0.5) * (width + gap)
                axis.bar(
                    position,
                    cell["mean"],
                    width=width,
                    color=colours[phase],
                    label=phase if index == 0 and slot == 0 else None,
                )
                axis.plot(
                    [position, position],
                    [cell["low"], cell["high"]],
                    color="black",
                    linewidth=1.0,
                )
        axis.text(
            index * (len(BRANCHES) + 1) + 1,
            -0.17,
            disease,
            transform=axis.get_xaxis_transform(),
            ha="center",
            fontweight="bold",
        )
    axis.set_xticks(ticks)
    axis.set_xticklabels(labels)
    axis.set_ylabel("Absolute contribution density")
    axis.set_title(
        "B  Contribution during depolarisation and repolarisation", loc="left", fontweight="bold"
    )
    axis.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))

    axis = figure.add_subplot(grid[4])
    branch_colour = {"R": "#7b9acc", "L": "#e0a458", "Q": "#5f9e6e"}
    ticks, labels = [], []
    for index, disease in enumerate(DISEASES):
        for slot, phase in enumerate(PHASES):
            centre = index * (len(PHASES) + 1) + slot
            ticks.append(centre)
            labels.append(phase)
            bottom = 0.0
            for branch in (BRANCH_LABELS[b] for b in BRANCHES):
                value = shares[(disease, branch, phase)]["share"]
                axis.bar(
                    centre,
                    value,
                    bottom=bottom,
                    width=0.6,
                    color=branch_colour[branch],
                    label=branch if index == 0 and slot == 0 else None,
                )
                bottom += 0.0 if not np.isfinite(value) else value
        axis.text(
            index * (len(PHASES) + 1) + 0.5,
            -0.20,
            disease,
            transform=axis.get_xaxis_transform(),
            ha="center",
            fontweight="bold",
        )
    axis.set_xticks(ticks)
    axis.set_xticklabels(labels)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Share of absolute mass")
    axis.set_title("C  R/L/Q composition within each phase", loc="left", fontweight="bold")
    axis.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    _save(figure, path)


def _save(figure, path):
    path = Path(path)
    for suffix in (".png", ".pdf"):
        figure.savefig(path.with_suffix(suffix), dpi=200, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(figure)


def qc_figure(report, beats, run, signals, rate, path):
    """Duration distributions, success rates per disease, and annotated examples."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.figure(figsize=(11, 8))
    grid = figure.add_gridspec(2, 3, hspace=0.45, wspace=0.28)
    qrs = np.concatenate([b["qrs_duration_ms"][b["valid_qrs"]] for b in beats])
    qt = np.concatenate([b["qt_duration_ms"][b["valid_twave"]] for b in beats])
    for column, (values, name) in enumerate(((qrs, "QRS duration"), (qt, "QT duration"))):
        axis = figure.add_subplot(grid[0, column])
        axis.hist(values, bins=40, color="#3a6ea5")
        axis.axvline(np.median(values), color="black", linestyle="--", linewidth=1)
        axis.set_xlabel(f"{name} (ms)")
        axis.set_ylabel("Beats")
        axis.set_title(f"{name}: median {np.median(values):.0f} ms", loc="left")
    axis = figure.add_subplot(grid[0, 2])
    groups = [row["group"] for row in report]
    positions = np.arange(len(groups))
    axis.bar(positions - 0.2, [row["valid_qrs_rate"] for row in report], 0.4, label="valid QRS")
    axis.bar(positions + 0.2, [row["valid_twave_rate"] for row in report], 0.4, label="valid T")
    axis.set_xticks(positions)
    axis.set_xticklabels(groups, rotation=45, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("Fraction of beats")
    axis.set_title("Delineation success", loc="left")
    axis.legend(frameon=False, fontsize=8)
    # Examples: the first record of each disease whose first beat is fully delineated.
    shown = 0
    for disease in DISEASES:
        positive = np.flatnonzero(run["labels"][:, CLASSES.index(disease)] == 1)
        chosen = next((r for r in positive if beats[r]["valid_full_beat"].any()), None)
        if chosen is None or shown >= 3:
            continue
        axis = figure.add_subplot(grid[1, shown])
        beat = int(np.flatnonzero(beats[chosen]["valid_full_beat"])[0])
        _draw_example(axis, signals, run, beats, chosen, beat, rate, disease)
        shown += 1
    _save(figure, path)


def _draw_example(axis, signals, run, beats, record, beat, rate, disease):
    current = beats[record]
    peak = int(current["r_peak"][beat])
    low = max(0, peak - int(0.4 * rate))
    high = min(signals.shape[-1], peak + int(0.6 * rate))
    vector = np.asarray(signals[run["cache_rows"][record]])
    from .handcrafted import cardiac_vector

    magnitude = np.linalg.norm(cardiac_vector(vector), axis=0)[low:high]
    time = (np.arange(low, high) - peak) * 1000.0 / rate
    axis.plot(time, magnitude, color="black", linewidth=0.9)
    for name, colour in zip(FIDUCIALS, ("#3a6ea5", "#000000", "#3a6ea5", "#c86a3a", "#c86a3a")):
        sample = int(current[name][beat])
        if sample >= 0:
            axis.axvline((sample - peak) * 1000.0 / rate, color=colour, linewidth=0.9, alpha=0.8)
            axis.text(
                (sample - peak) * 1000.0 / rate,
                magnitude.max(),
                name.replace("_", " "),
                rotation=90,
                fontsize=6,
                va="top",
                ha="right",
            )
    axis.set_xlabel("Time relative to R peak (ms)")
    axis.set_ylabel("||V|| (mV)")
    axis.set_title(f"{disease}: record {int(run['ecg_ids'][record])}", loc="left", fontsize=9)


def robustness_figure(summaries, labels, path, title):
    """Absolute density per disease, branch and phase under each condition."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        len(PHASES), len(DISEASES), figsize=(12, 6), sharey="row", squeeze=False
    )
    width = 0.8 / len(summaries)
    for row, phase in enumerate(PHASES):
        for column, disease in enumerate(DISEASES):
            axis = axes[row][column]
            for index, (summary, label) in enumerate(zip(summaries, labels)):
                positions = np.arange(len(BRANCHES)) + (index - (len(summaries) - 1) / 2) * width
                values = [
                    summary[(disease, BRANCH_LABELS[b], phase, "absolute_density")]["mean"]
                    for b in BRANCHES
                ]
                axis.bar(positions, values, width=width, label=label if not row + column else None)
            axis.set_xticks(range(len(BRANCHES)))
            axis.set_xticklabels([BRANCH_LABELS[b] for b in BRANCHES])
            if row == 0:
                axis.set_title(disease, fontweight="bold")
            if column == 0:
                axis.set_ylabel(f"{phase}\nabsolute density")
    axes[0][0].legend(frameon=False, fontsize=8)
    figure.suptitle(title)
    _save(figure, path)


def _correlation(first, second):
    first, second = np.ravel(first), np.ravel(second)
    keep = np.isfinite(first) & np.isfinite(second)
    if keep.sum() < 3:
        return float("nan")
    return float(np.corrcoef(first[keep], second[keep])[0, 1])


def _density_vector(summary):
    return [
        summary[(disease, BRANCH_LABELS[branch], phase, "absolute_density")]["mean"]
        for disease in DISEASES
        for branch in BRANCHES
        for phase in PHASES
    ]


def stability_rows(runs, summaries, panels):
    """Whether the DEP/REP reading survives what the fine time axis does not.

    The plan asks for this comparison and explicitly forbids claiming the phase level is
    steadier unless the numbers say so, so both correlations are reported side by side
    and neither is interpreted here.
    """
    rows = []
    for index, first in enumerate(runs):
        for second in runs[index + 1 :]:
            changed = "window" if first["window_ms"] != second["window_ms"] else "perturbation"
            if first["window_ms"] != second["window_ms"] and (
                first["quaternion_mode"] != second["quaternion_mode"]
            ):
                continue
            rows.append(
                {
                    "comparison": f"{_label(first)} vs {_label(second)}",
                    "varied": changed,
                    "r_time_resolved": round(
                        _correlation(panels[_label(first)], panels[_label(second)]), 4
                    ),
                    "r_phase_level": round(
                        _correlation(
                            _density_vector(summaries[_label(first)]),
                            _density_vector(summaries[_label(second)]),
                        ),
                        4,
                    ),
                }
            )
    return rows


def _label(run):
    return f"w{run['window_ms']}ms {run['quaternion_mode']}"


def run(config, root, output=None, bootstrap=BOOTSTRAP, seed=SEED):
    """Delineate, map every saved attribution onto the phases, write tables and figures.

    Delineation QC is computed and written before any phase-level interpretation, which
    is the order the plan requires: the phase numbers mean nothing until the boundaries
    behind them have been shown to be reliable.
    """
    global BOOTSTRAP
    BOOTSTRAP = bootstrap
    runs = load_runs(root)
    output = Path(output or Path(root) / "phases")
    output.mkdir(parents=True, exist_ok=True)
    primary = next(
        (r for r in runs if all(r[k] == v for k, v in PRIMARY.items())),
        None,
    )
    if primary is None:
        raise ValueError(
            f"No primary run (window {PRIMARY['window_ms']} ms, "
            f"{PRIMARY['quaternion_mode']} quaternion perturbation) under {root}"
        )
    cache = Path(config["data"]["cache"])
    signals = np.load(cache / "signals.npy", mmap_mode="r", allow_pickle=False)
    from .interpret import aggregate

    shared = {}
    summaries, tables, panels = {}, {}, {}
    for current in runs:
        key = current["cache_rows"].tobytes()
        if key not in shared:
            shared[key] = delineation(config, current)
        beats, rate = shared[key]
        table = patient_table(current, beats, rate, signals.shape[-1])
        tables[_label(current)] = table
        summaries[_label(current)] = disease_summary(table, seed)
        panels[_label(current)] = aggregate(current, bootstrap=1, seed=seed)["panels"]

    beats, rate = shared[primary["cache_rows"].tobytes()]
    report = delineation_report(beats, primary)
    write_csv(output / "delineation_qc.csv", report)
    write_csv(output / "beat_boundaries.csv", beat_rows(primary, beats, rate))
    qc_figure(report, beats, primary, signals, rate, output / "figure_delineation_qc")

    table, summary = tables[_label(primary)], summaries[_label(primary)]
    shares = branch_shares(table, seed)
    write_csv(
        output / "phase_contribution_patient.csv",
        [
            {
                "patient_id": patient,
                "disease": disease,
                "branch": branch,
                "phase": phase,
                "metric": metric,
                "value": round(value, 6),
            }
            for (disease, branch, phase, metric), values in sorted(table.items())
            for patient, value in sorted(values.items())
            if np.isfinite(value)
        ],
    )
    write_csv(output / "phase_contribution_disease_summary.csv", summary_rows(summary, primary))
    write_csv(
        output / "phase_branch_share.csv",
        [
            {
                "disease": disease,
                "phase": phase,
                "branch": branch,
                "share": round(values["share"], 6),
                "ci_low": round(values["low"], 6),
                "ci_high": round(values["high"], 6),
                "patients": values["patients"],
            }
            for (disease, branch, phase), values in sorted(shares.items())
        ],
    )

    primary_panels = panels[_label(primary)]
    clip = float(np.quantile(np.abs(primary_panels), CLIP_QUANTILE)) or 1.0
    main_figure(
        primary_panels,
        primary["times_ms"],
        summary,
        shares,
        output / "figure_interpretability_main",
        clip,
    )

    windows = sorted(
        (r for r in runs if r["quaternion_mode"] == PRIMARY["quaternion_mode"]),
        key=lambda r: r["window_ms"],
    )
    write_csv(
        output / "phase_robustness_window.csv",
        [row for r in windows for row in summary_rows(summaries[_label(r)], r)],
    )
    robustness_figure(
        [summaries[_label(r)] for r in windows],
        [f"{r['window_ms']} ms" for r in windows],
        output / "figure_window_robustness",
        "Phase-level contribution under 10 / 20 / 40 ms perturbation windows",
    )
    modes = [r for r in runs if r["window_ms"] == PRIMARY["window_ms"]]
    write_csv(
        output / "phase_robustness_q_perturbation.csv",
        [row for r in modes for row in summary_rows(summaries[_label(r)], r)],
    )
    robustness_figure(
        [summaries[_label(r)] for r in modes],
        [r["quaternion_mode"] for r in modes],
        output / "figure_identity_robustness",
        "Phase-level contribution under SLERP and identity quaternion perturbation",
    )
    stability = stability_rows(runs, summaries, panels)
    write_csv(output / "phase_stability.csv", stability)
    written = sorted(path.name for path in output.iterdir())
    print(
        json.dumps(
            {
                "output": str(output),
                "runs": [_label(r) for r in runs],
                "stability": stability,
                "files": written,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return {
        "summary": summary,
        "shares": shares,
        "delineation": report,
        "stability": stability,
    }
