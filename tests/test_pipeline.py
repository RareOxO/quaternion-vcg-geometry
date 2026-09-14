import json
from pathlib import Path

import numpy as np
import pytest
import torch

from qdg.config import validate_config
from qdg.data import CLASSES, PTBXLDataset, load_manifest, split_masks
from qdg.engine import evaluate, train
from qdg.experiments import experiment_config, tables
from qdg.metrics import multilabel_metrics, validation_thresholds


def test_official_split_and_cache_contents(synthetic_cache):
    config, manifest = synthetic_cache
    report = manifest["report"]
    assert (
        report["splits"]["train"]["records"]
        + report["splits"]["val"]["records"]
        + report["splits"]["test"]["records"]
        == report["records_eligible"]
    )
    folds = np.load(f"{config['data']['cache']}/folds.npy")
    masks = split_masks(folds)
    assert masks["train"].sum() and masks["val"].sum() and masks["test"].sum()
    assert not (masks["train"] & masks["val"]).any()
    assert len(manifest["stats"]["vcg_std"]) == 3
    assert len(manifest["stats"]["pos_weight"]) == len(CLASSES)


def test_manifest_rejects_a_changed_configuration(synthetic_cache):
    config, _ = synthetic_cache
    with pytest.raises(ValueError):
        load_manifest({**config["data"], "bandpass": [0.5, 40.0]})


def test_dataset_items(synthetic_cache):
    config, manifest = synthetic_cache
    dataset = PTBXLDataset(config["data"]["cache"], "train")
    item = dataset[0]
    assert item["ecg"].shape == (12, manifest["stats"]["signal_length"])
    assert item["target"].shape == (len(CLASSES),)
    assert torch.isfinite(item["ecg"]).all()


def test_metrics_and_thresholds():
    rng = np.random.default_rng(0)
    target = rng.integers(0, 2, (60, 5)).astype(float)
    probability = rng.random((60, 5))
    thresholds = validation_thresholds(target, probability)
    assert thresholds.shape == (5,)
    metrics = multilabel_metrics(target, probability, thresholds)
    assert metrics["valid_auc_classes"] == 5
    assert 0 <= metrics["macro_auroc"] <= 1
    assert set(metrics["per_class"]) == set(CLASSES)
    with pytest.raises(ValueError):
        multilabel_metrics(target, probability[:, :3])


@pytest.mark.parametrize("name", ["M0", "M2", "M4"])
def test_train_and_evaluate_end_to_end(synthetic_cache, name):
    config = experiment_config(synthetic_cache[0], name)
    validate_config(config)
    run_dir = train(config, run_name=f"{name}_seed42")
    history = [json.loads(line) for line in (run_dir / "history.jsonl").read_text().splitlines()]
    assert len(history) == config["training"]["epochs"]
    assert (run_dir / "best.pt").exists()
    result = evaluate(run_dir / "best.pt", "test")
    assert result["variant"] == config["model"]["variant"]
    assert result["fixed_0.5"]["records"] == 4
    assert (run_dir / "best_test_metrics.json").exists()


def test_tables_are_generated_from_runs(synthetic_cache, tmp_path):
    config = synthetic_cache[0]
    for name in ("M0", "M1"):
        current = experiment_config(config, name)
        evaluate(train(current, run_name=f"{name}_seed42") / "best.pt")
    root = config["training"]["output"]
    summary = tables(root)
    assert set(summary) == {"M0", "M1"}
    text = (tmp_path / "runs" / "tables.md").read_text(encoding="utf-8")
    assert "Table 1" in text and "Table 4" in text
    assert "Explicit Geometry" in text


def test_validate_config_rejects_bad_settings(tiny_config):
    with pytest.raises(ValueError):
        validate_config({**tiny_config, "model": {**tiny_config["model"], "variant": "M9"}})
    with pytest.raises(ValueError):
        validate_config({**tiny_config, "model": {**tiny_config["model"], "kernel": 4}})
    with pytest.raises(ValueError):
        validate_config({**tiny_config, "model": {**tiny_config["model"], "scales_ms": [20, 20]}})


def test_fusion_tables_are_generated(synthetic_cache):
    """双分支方案 §14-16, and tables.md must stay untouched by them."""
    config = synthetic_cache[0]
    for name in ("M0", "M0_wide", "F1"):
        evaluate(train(experiment_config(config, name), run_name=f"{name}_seed42") / "best.pt")
    root = Path(config["training"]["output"])
    summary = tables(root)
    assert {"M0", "M0_wide", "F1"} <= set(summary)
    text = (root / "fusion_tables.md").read_text(encoding="utf-8")
    assert "Table 5" in text and "Table 6" in text and "Table 7" in text
    assert "Real geometry complementarity" in text
    assert "Geometry vs extra capacity" in text
    # The original four tables still exist and still describe only M0-M4.
    original = (root / "tables.md").read_text(encoding="utf-8")
    assert "Table 1" in original and "F1 Raw + Real Geo" not in original
