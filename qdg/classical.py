"""The handcrafted and hybrid arms of Experiment 5.

A, B and C feed handcrafted features to a conventional classifier; E concatenates the
learned embedding of the proposed model with those features and feeds the same
classifier. D is the proposed model itself and is not run here -- it is the existing
R+L+Q run, reused.

Two engineering decisions the plan leaves open, recorded here rather than buried:

* Classifier. Plan section 6.1 allows Logistic Regression, XGBoost or a small MLP and
  asks that the choice be "pre-specified or validated consistently". XGBoost is not a
  dependency of this project, so the candidates are Logistic Regression and a small MLP,
  both are fitted for every arm, and the one with the better VALIDATION Macro AUROC is
  reported. The same procedure runs for A, B, C and E, so no arm gets a classifier the
  others did not have the chance to use.
* Scaling. Features are standardised, with the mean and variance fitted on the training
  folds only. Everything upstream of that -- the features themselves -- is a
  deterministic function of a single record and cannot leak across the split.

Results are written in the same layout a neural run produces, so the same table code
picks them up.
"""

import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from .data import CLASSES, PTBXLDataset, load_manifest, save_json
from .handcrafted import FEATURE_SETS, features_for
from .metrics import multilabel_metrics, validation_thresholds

CACHE_NAME = "handcrafted_features.npy"


def feature_cache(config):
    """All three feature sets for every eligible record, computed once and reused.

    Cached beside the signal cache because it is a pure function of the waveforms and
    the fixed definitions in `handcrafted`; nothing about the split enters it.
    """
    cache = Path(config["data"]["cache"])
    path = cache / CACHE_NAME
    manifest = load_manifest(config["data"])
    rate = manifest["stats"]["sampling_rate"]
    if path.is_file():
        return np.load(path, allow_pickle=False), rate
    signals = np.load(cache / "signals.npy", mmap_mode="r", allow_pickle=False)
    rows = [
        features_for(signals[index], rate, FEATURE_SETS)
        for index in tqdm(range(len(signals)), desc="Handcrafted features")
    ]
    features = np.stack(rows).astype(np.float32)
    np.save(path, features, allow_pickle=False)
    return features, rate


def feature_columns(config, kinds):
    """Column slice for the named feature sets, in the canonical order."""
    features, rate = feature_cache(config)
    signals = np.load(
        Path(config["data"]["cache"]) / "signals.npy", mmap_mode="r", allow_pickle=False
    )
    widths = {kind: len(features_for(signals[0], rate, [kind])) for kind in FEATURE_SETS}
    columns, start = [], 0
    for kind in FEATURE_SETS:
        stop = start + widths[kind]
        if kind in kinds:
            columns.extend(range(start, stop))
        start = stop
    return features[:, columns]


def embeddings(config, checkpoint, split, indices, device):
    """Pre-classifier embedding of the proposed model, for the hybrid arm."""
    from .models import build_model

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model(state["config"]["model"], state["stats"]).to(device).eval()
    model.load_state_dict(state["model"])
    dataset = PTBXLDataset(config["data"]["cache"], split)
    out = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(dataset), 64), desc=f"Embedding {split}", leave=False):
            batch = torch.stack(
                [dataset[i]["ecg"] for i in range(start, min(start + 64, len(dataset)))]
            )
            out.append(model.forward_features(batch.to(device)).float().cpu().numpy())
    assert np.concatenate(out).shape[0] == len(indices)
    return np.concatenate(out)


def _candidates(seed):
    return {
        "logistic": OneVsRestClassifier(
            LogisticRegression(max_iter=2000, random_state=seed), n_jobs=1
        ),
        "mlp": OneVsRestClassifier(
            MLPClassifier(
                hidden_layer_sizes=(64,), max_iter=500, random_state=seed, early_stopping=True
            ),
            n_jobs=1,
        ),
    }


def run_classical(config, name, kinds, checkpoint=None, device="cpu"):
    """Fit, select on validation, evaluate once on test, and write a run directory."""
    started = time.perf_counter()
    training = config["training"]
    splits = {
        split: PTBXLDataset(config["data"]["cache"], split) for split in ("train", "val", "test")
    }
    manifest = load_manifest(config["data"])
    matrix = feature_columns(config, kinds) if kinds else None
    frozen = 0
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        frozen = sum(value.numel() for value in state["model"].values())
    parts = {}
    for split, dataset in splits.items():
        pieces = []
        if matrix is not None:
            pieces.append(matrix[dataset.indices])
        if checkpoint is not None:
            pieces.append(embeddings(config, checkpoint, split, dataset.indices, device))
        parts[split] = np.concatenate(pieces, axis=1)
    labels = {split: dataset.labels[dataset.indices] for split, dataset in splits.items()}
    # Fitted on the training folds only; validation and test are transformed, never fitted.
    scaler = StandardScaler().fit(parts["train"])
    scaled = {split: scaler.transform(values) for split, values in parts.items()}
    scored = {}
    for label, model in _candidates(training["seed"]).items():
        model.fit(scaled["train"], labels["train"])
        probability = model.predict_proba(scaled["val"])
        scored[label] = (multilabel_metrics(labels["val"], probability), model, probability)
    best = max(scored, key=lambda label: scored[label][0]["macro_auroc"])
    validation, model, validation_probability = scored[best]
    thresholds = (
        validation_thresholds(labels["val"], validation_probability)
        if training["threshold"] == "validation_f1"
        else np.full(len(CLASSES), 0.5)
    )
    probability = model.predict_proba(scaled["test"])
    run_dir = Path(training["output"]) / f"{name}_seed{training['seed']}"
    run_dir.mkdir(parents=True, exist_ok=False)
    environment = {
        "run_dir": str(run_dir),
        "variant": "classical",
        "seed": training["seed"],
        "settings": {
            "feature_sets": list(kinds),
            "classifier_candidates": sorted(_candidates(training["seed"])),
            "selected_classifier": best,
            "embedding_checkpoint": str(checkpoint) if checkpoint else None,
        },
        # The classifier's own coefficients, and separately anything frozen upstream of
        # it. Reporting only the former would put 915 next to the proposed model's
        # 62,617 and read as though a far smaller model had matched it, when the hybrid
        # arm runs that same model first and then fits a head on top.
        "classifier_parameters": int(sum(values.size for values in _coefficients(model))),
        "frozen_parameters": int(frozen),
        "parameters": int(sum(values.size for values in _coefficients(model)) + frozen),
        "sampling_rate": manifest["stats"]["sampling_rate"],
        "input_channels": int(parts["train"].shape[1]),
        "receptive_field_samples": None,
        "smoke_training": False,
        "python": platform.python_version(),
        "seconds": time.perf_counter() - started,
    }
    save_json(run_dir / "config.json", config)
    save_json(run_dir / "environment.json", environment)
    result = {
        "variant": "classical",
        "seed": training["seed"],
        "split": "test",
        "best_epoch": 1,
        "validation_macro_auroc": validation["macro_auroc"],
        "validation_macro_auprc": validation["macro_auprc"],
        "parameters": environment["parameters"],
        "classifier_parameters": environment["classifier_parameters"],
        "frozen_parameters": environment["frozen_parameters"],
        "settings": environment["settings"],
        "limited_evaluation": False,
        "smoke_training": False,
        "fixed_0.5": multilabel_metrics(labels["test"], probability),
        "validation_selected_thresholds": multilabel_metrics(
            labels["test"], probability, thresholds
        ),
    }
    save_json(run_dir / "best_test_metrics.json", result)
    print(json.dumps({"run_dir": str(run_dir), **environment}, ensure_ascii=False), flush=True)
    return result


def _coefficients(model):
    for estimator in model.estimators_:
        if hasattr(estimator, "coef_"):
            yield estimator.coef_
        else:
            for weights in estimator.coefs_:
                yield weights
