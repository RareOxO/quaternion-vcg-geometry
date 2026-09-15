"""Disease x R/L/Q x R-peak-aligned-time contribution heatmap (plan section 9).

For each disease, each branch and each window of R-peak-relative time, the information
in that window is removed from that branch alone and the model is run again. The signed
change in the logit is the contribution:

    C = z - z_perturbed

positive where the removed information supported the prediction, negative where it
opposed it. Signs are kept; the main figure is not an absolute value.

Removal is by interpolation, not by zeroing, so no artificial edge is introduced. R and
L are interpolated linearly between the values just outside the window. Q is a unit
quaternion, so it is interpolated along the geodesic (SLERP) between the boundary
rotations -- a component-wise interpolation would leave the sphere and stop being a
rotation at all. Replacing the window with the identity rotation is available as the
robustness check the plan asks for.

Two engineering decisions the plan leaves open, recorded here:

* All beats of a record are perturbed at the same relative time in one pass, rather than
  one beat at a time. This measures the joint contribution of a cardiac phase across the
  record, which is what the figure is read as; the mean of per-beat contributions would
  be a different quantity and costs a factor of K more forward passes. The two coincide
  only if contributions add.
* A record whose R-peak detection fails the pre-specified quality rule is dropped from
  the analysis rather than aligned on unreliable peaks. The count of dropped records is
  reported alongside the figure.
"""

import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .data import CLASSES, PTBXLDataset, load_manifest, save_json
from .handcrafted import beat_quality, cardiac_vector, detect_r_peaks

DISEASES = ("MI", "STTC", "CD", "HYP")
BRANCHES = ("radial", "linear", "angular")
BRANCH_LABELS = {"radial": "R", "linear": "L", "angular": "Q"}
# Plan section 9.2 and 9.3. W is the perturbation window and is unrelated to the 20 ms
# quaternion lag, which stays what it is inside the frontend.
RELATIVE_MS = (-300, 600)
STRIDE_MS = 10
WINDOW_MS = 20
IDENTITY = (1.0, 0.0, 0.0, 0.0)


def relative_times(stride_ms=STRIDE_MS):
    return np.arange(RELATIVE_MS[0], RELATIVE_MS[1] + 1, stride_ms)


def record_peaks(cache, indices, sampling_rate):
    """R peaks per record, and which records pass the pre-specified quality rule."""
    signals = np.load(Path(cache) / "signals.npy", mmap_mode="r", allow_pickle=False)
    peaks, usable = [], []
    for index in tqdm(indices, desc="R peaks", leave=False):
        found = detect_r_peaks(cardiac_vector(np.asarray(signals[index])), sampling_rate)
        peaks.append(found)
        usable.append(beat_quality(found, sampling_rate, signals.shape[-1]))
    return peaks, np.array(usable)


def _slerp(start, end, weights, eps=1e-7):
    """Geodesic interpolation between unit quaternions.

    start/end are (..., 4); weights is (..., n). Returns (..., n, 4). The shorter arc is
    taken by flipping `end` where the pair points into opposite hemispheres, and the
    nearly-parallel case falls back to a normalised linear blend, which is what SLERP
    tends to there anyway.
    """
    end = torch.where((start * end).sum(-1, keepdim=True) < 0, -end, end)
    dot = (start * end).sum(-1, keepdim=True).clamp(-1 + eps, 1 - eps)
    omega = dot.arccos()
    start, end = start.unsqueeze(-2), end.unsqueeze(-2)
    weights = weights.unsqueeze(-1)
    omega = omega.unsqueeze(-2)
    blended = torch.where(
        omega.abs() < eps,
        start * (1 - weights) + end * weights,
        (((1 - weights) * omega).sin() * start + (weights * omega).sin() * end) / omega.sin(),
    )
    return blended / blended.norm(dim=-1, keepdim=True).clamp_min(eps)


def perturb(features, positions, valid, quaternion, mode="interpolate"):
    """Replace `positions` of `features` with information-free filler.

    features  (B, C, T)
    positions (B, K, W) sample indices, one window per beat
    valid     (B, K) which beats are real rather than padding
    """
    batch, channels, steps = features.shape
    width = positions.shape[-1]
    left = (positions[..., 0] - 1).clamp(0, steps - 1)
    right = (positions[..., -1] + 1).clamp(0, steps - 1)
    start = features.gather(2, left.unsqueeze(1).expand(batch, channels, -1))
    end = features.gather(2, right.unsqueeze(1).expand(batch, channels, -1))
    weights = (torch.arange(width, device=features.device) + 1) / (width + 1)
    if quaternion and mode == "identity":
        filler = (
            features.new_tensor(IDENTITY)
            .view(1, 4, 1, 1)
            .expand(batch, 4, positions.shape[1], width)
        )
    elif quaternion:
        # (B, 4, K) -> (B, K, 4) so the interpolation runs over whole quaternions.
        filler = _slerp(
            start.permute(0, 2, 1),
            end.permute(0, 2, 1),
            weights.expand(batch, positions.shape[1], width),
        ).permute(0, 3, 1, 2)
    else:
        filler = start.unsqueeze(-1) * (1 - weights) + end.unsqueeze(-1) * weights
    out = features.clone()
    index = positions.unsqueeze(1).expand(batch, channels, -1, -1)
    keep = valid.view(batch, 1, -1, 1).expand_as(filler)
    source = torch.where(
        keep, filler, out.gather(2, index.reshape(batch, channels, -1)).view_as(filler)
    )
    out.scatter_(2, index.reshape(batch, channels, -1), source.reshape(batch, channels, -1))
    return out


def _windows(peaks, offset, width, steps, device, max_beats):
    """Sample indices of the perturbation window for every beat, padded to max_beats."""
    positions = torch.zeros(len(peaks), max_beats, width, dtype=torch.long, device=device)
    valid = torch.zeros(len(peaks), max_beats, dtype=torch.bool, device=device)
    span = torch.arange(width, device=device) - width // 2
    for row, found in enumerate(peaks):
        if not len(found):
            continue
        centres = torch.as_tensor(found[:max_beats], device=device) + offset
        window = centres.unsqueeze(-1) + span
        inside = (window[:, 0] >= 1) & (window[:, -1] < steps - 1)
        positions[row, : len(centres)] = window.clamp(1, steps - 2)
        valid[row, : len(centres)] = inside
    return positions, valid


def contributions(
    config,
    checkpoint,
    device=None,
    stride_ms=STRIDE_MS,
    window_ms=WINDOW_MS,
    quaternion_mode="interpolate",
    batch_size=32,
    limit=None,
):
    """The A tensor of plan section 9.7, plus the bookkeeping to interpret it."""
    from .models import build_model

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    manifest = load_manifest(config["data"])
    rate = manifest["stats"]["sampling_rate"]
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(state["config"]["model"], state["stats"]).to(device).eval()
    model.load_state_dict(state["model"])
    if set(model.branches) != set(BRANCHES):
        raise ValueError(
            f"The heatmap needs all three branches; this checkpoint has {model.branches}"
        )
    dataset = PTBXLDataset(config["data"]["cache"], "test", limit, config["training"]["seed"])
    patients = np.load(Path(config["data"]["cache"]) / "patient_ids.npy", allow_pickle=False)
    peaks, usable = record_peaks(config["data"]["cache"], dataset.indices, rate)
    keep = np.flatnonzero(usable)
    times = relative_times(stride_ms)
    width = max(2, int(round(window_ms * rate / 1000)))
    max_beats = max((len(peaks[i]) for i in keep), default=0)
    scores = np.zeros((len(keep), len(times), len(BRANCHES), len(CLASSES)), dtype=np.float32)
    labels = dataset.labels[dataset.indices][keep]
    record_patients = patients[dataset.indices][keep]
    with torch.inference_mode():
        for start in tqdm(range(0, len(keep), batch_size), desc="Perturbation"):
            rows = keep[start : start + batch_size]
            ecg = torch.stack([dataset[int(r)]["ecg"] for r in rows]).to(device)
            features = {b: model.frontends[b](ecg) for b in BRANCHES}
            baseline = model.head(model.fuse(features)).float().cpu().numpy()
            batch_peaks = [peaks[r] for r in rows]
            for step, relative in enumerate(times):
                offset = int(round(relative * rate / 1000))
                positions, valid = _windows(
                    batch_peaks, offset, width, ecg.shape[-1], device, max_beats
                )
                for axis, branch in enumerate(BRANCHES):
                    altered = dict(features)
                    altered[branch] = perturb(
                        features[branch],
                        positions,
                        valid,
                        quaternion=branch == "angular",
                        mode=quaternion_mode,
                    )
                    logits = model.head(model.fuse(altered)).float().cpu().numpy()
                    scores[start : start + len(rows), step, axis] = baseline - logits
    return {
        "contributions": scores,
        "labels": labels,
        "patients": record_patients,
        "times_ms": times,
        "window_ms": window_ms,
        "stride_ms": stride_ms,
        "quaternion_mode": quaternion_mode,
        "records_analysed": int(len(keep)),
        "records_dropped": int(len(dataset.indices) - len(keep)),
        "checkpoint": str(checkpoint),
    }


def aggregate(result, bootstrap=1000, seed=0):
    """record -> patient -> disease population (plan section 9.6), with a patient-level
    bootstrap so one patient contributing several records cannot count several times."""
    scores, labels, patients = result["contributions"], result["labels"], result["patients"]
    times = result["times_ms"]
    panels = np.zeros((len(DISEASES), len(BRANCHES), len(times)), dtype=np.float32)
    lower = np.zeros_like(panels)
    upper = np.zeros_like(panels)
    rng = np.random.default_rng(seed)
    counts = {}
    for index, disease in enumerate(DISEASES):
        positive = np.flatnonzero(labels[:, CLASSES.index(disease)] == 1)
        counts[disease] = {"records": int(len(positive))}
        if not len(positive):
            continue
        # Records first collapse to their patient, so a patient with several recordings
        # carries the same weight as one with a single recording.
        unique, inverse = np.unique(patients[positive], return_inverse=True)
        counts[disease]["patients"] = int(len(unique))
        per_patient = np.zeros((len(unique), len(times), len(BRANCHES)), dtype=np.float32)
        for slot in range(len(unique)):
            per_patient[slot] = scores[
                positive[inverse == slot], :, :, CLASSES.index(disease)
            ].mean(0)
        panels[index] = per_patient.mean(0).T
        draws = rng.integers(0, len(unique), size=(bootstrap, len(unique)))
        samples = per_patient[draws].mean(axis=1)
        lower[index] = np.quantile(samples, 0.025, axis=0).T
        upper[index] = np.quantile(samples, 0.975, axis=0).T
    return {"panels": panels, "lower": lower, "upper": upper, "counts": counts}


def heatmap(panels, times, path, title=None):
    """The single main figure: one panel per disease, three rows each, shared axis and a
    symmetric diverging scale so positive and negative read alike."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    limit = float(np.abs(panels).max()) or 1.0
    figure, axes = plt.subplots(len(DISEASES), 1, figsize=(9, 7), sharex=True)
    image = None
    for axis, disease, panel in zip(axes, DISEASES, panels):
        image = axis.imshow(
            panel,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            extent=[times[0], times[-1], len(BRANCHES) - 0.5, -0.5],
            interpolation="nearest",
        )
        axis.set_yticks(range(len(BRANCHES)))
        axis.set_yticklabels([BRANCH_LABELS[b] for b in BRANCHES])
        axis.set_ylabel(disease, rotation=0, labelpad=26, va="center", fontweight="bold")
        axis.axvline(0, color="black", linewidth=0.8, linestyle="--")
    axes[-1].set_xlabel("Time relative to R peak (ms)")
    figure.colorbar(image, ax=axes, label="Contribution  (logit drop when removed)", pad=0.02)
    if title:
        figure.suptitle(title)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def run(config, checkpoint, output=None, **kwargs):
    """Compute, aggregate, save the tensor and draw the figure."""
    result = contributions(config, checkpoint, **kwargs)
    summary = aggregate(result)
    output = Path(output or Path(config["training"]["output"]) / "interpretability")
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / f"contributions_{result['quaternion_mode']}.npz",
        panels=summary["panels"],
        lower=summary["lower"],
        upper=summary["upper"],
        times_ms=result["times_ms"],
    )
    report = {
        "diseases": list(DISEASES),
        "branches": [BRANCH_LABELS[b] for b in BRANCHES],
        "times_ms": result["times_ms"].tolist(),
        "window_ms": result["window_ms"],
        "stride_ms": result["stride_ms"],
        "quaternion_mode": result["quaternion_mode"],
        "records_analysed": result["records_analysed"],
        "records_dropped": result["records_dropped"],
        "counts": summary["counts"],
        "checkpoint": result["checkpoint"],
    }
    save_json(output / f"interpretability_{result['quaternion_mode']}.json", report)
    try:
        heatmap(
            summary["panels"],
            result["times_ms"],
            output / f"heatmap_{result['quaternion_mode']}.png",
            title=f"Disease x R/L/Q contribution ({result['quaternion_mode']} perturbation)",
        )
        report["figure"] = str(output / f"heatmap_{result['quaternion_mode']}.png")
    except ImportError:
        report["figure"] = "matplotlib not installed; tensor saved without the figure"
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    return report
