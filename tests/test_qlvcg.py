"""B0, V0 and the shared supervised protocol."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from lvcg.models import LVCG
from qdg.data import CLASSES
from qlvcg.config import DEFAULT_CONFIG, load_config, validate_config
from qlvcg.engine import CSV_COLUMNS, train, warmup_cosine
from qlvcg.models import (
    LVCG_RECONSTRUCTION_ONLY,
    SupervisedLVCG,
    build_model,
    record_shapes,
    standardize,
)
from qlvcg.tables import write_history


@pytest.fixture
def config():
    return load_config(DEFAULT_CONFIG)


def _ecg(batch=3, length=1000):
    torch.manual_seed(1)
    ecg = torch.randn(batch, 12, length) * 0.2
    beat = torch.zeros(length)
    beat[torch.arange(60, length, 80)] = 4.0
    ecg[:, :6] += beat
    return standardize(ecg)


def _names(model):
    return [type(m).__name__.lower() for m in model.modules()] + [
        n.lower() for n, _ in model.named_parameters()
    ]


def test_config_matches_the_author_architecture(config):
    lvcg = config["model"]["lvcg"]
    assert (lvcg["beat_len"], lvcg["state_dim"], lvcg["max_beats"]) == (128, 256, 20)
    assert lvcg["beat_encoder_stages"] == [96, 192, 256, 256]
    assert config["training"]["batch_size"] == 64 and config["training"]["lr"] == 5e-4
    broken = {**config, "model": {**config["model"], "lvcg": {**lvcg, "fs": 500}}}
    with pytest.raises(ValueError):
        validate_config(broken)


def test_standardize_is_per_record_and_per_lead():
    out = standardize(torch.randn(2, 12, 400) * 7 + 3)
    torch.testing.assert_close(out.mean(-1), torch.zeros(2, 12), atol=1e-5, rtol=0)
    torch.testing.assert_close(out.square().mean(-1).sqrt(), torch.ones(2, 12), atol=1e-5, rtol=0)


def test_warmup_then_cosine_to_zero():
    factor = warmup_cosine(10, 110)
    assert factor(0) == pytest.approx(0.1) and factor(9) == pytest.approx(1.0)
    assert factor(10) == pytest.approx(1.0) and factor(60) == pytest.approx(0.5)
    assert factor(110) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("experiment", ["B0", "V0A", "V0B"])
def test_logits_are_multilabel_and_finite(config, experiment):
    model = build_model(config, experiment)
    logits = model(_ecg())
    assert logits.shape == (3, len(CLASSES)) and torch.isfinite(logits).all()


@pytest.mark.parametrize("experiment", ["B0", "V0A"])
def test_no_quaternion_component(config, experiment):
    assert not any("quat" in name for name in _names(build_model(config, experiment)))


def test_b0_has_no_lvcg_machinery(config):
    names = set(_names(build_model(config, "B0")))
    for forbidden in ("beatencoder", "stategru", "beatsegmenter", "globalrrembedding"):
        assert forbidden not in names


def test_b0_sees_exactly_the_vcg_v0_sees(config):
    """Same geometry, same pseudo-inverse, same eps: the B0/V0 gap is the architecture."""
    ecg = _ecg()
    b0, v0 = build_model(config, "B0"), build_model(config, "V0A")
    torch.testing.assert_close(
        b0.vcg(ecg), v0.backbone._recover_vcg(ecg, torch.arange(12).expand(3, -1))
    )


def test_v0_is_the_unmodified_author_class(config):
    model = build_model(config, "V0A")
    assert isinstance(model, SupervisedLVCG) and type(model.backbone) is LVCG


def test_classification_gradients_skip_exactly_the_reconstruction_modules(config):
    """What `parameter_counts()['classification_path']` claims, checked by autograd."""
    model = build_model(config, "V0A")
    model(_ecg()).sum().backward()
    for name, module in model.backbone.named_children():
        grads = [p.grad for p in module.parameters()]
        if not grads:
            continue
        if name in LVCG_RECONSTRUCTION_ONLY:
            assert all(g is None for g in grads), name
    reached = sum(p.numel() for p in model.parameters() if p.grad is not None)
    counts = model.parameter_counts()
    assert counts["classification_path"] >= reached
    assert counts["total"] > counts["classification_path"]


def test_auxiliary_objective_trains_the_decoders(config):
    torch.manual_seed(0)
    model = build_model(config, "V0B")
    terms = model.auxiliary_losses(_ecg(4), num_visible=3)
    assert set(terms) == {"ecg", "temporal", "beat", "base"}
    assert all(torch.isfinite(v) for v in terms.values())
    sum(terms.values()).backward()
    assert any(p.grad is not None for p in model.backbone.beat_decoder.parameters())
    assert any(p.grad is not None for p in model.backbone.ecg_decoder.parameters())


def test_record_shapes_reports_every_stage(config):
    shapes = record_shapes(build_model(config, "V0A"), _ecg())
    assert shapes["vcg (vcg_inverse)"] == [[3, 3, 1000]]
    beats = shapes["beats, rr, mask (beat_segmenter)"]
    assert beats[0][0] == 3 and beats[0][2:] == [3, 128]
    assert shapes["logits (head)"] == [[3, len(CLASSES)]]
    assert build_model(config, "B0").parameter_counts()["total"] < 500_000


def _smoke_config(synthetic_cache, tmp_path):
    base = load_config(DEFAULT_CONFIG)
    base["data"] = dict(synthetic_cache[0]["data"])
    base["training"].update(
        epochs=1,
        batch_size=4,
        num_workers=0,
        cpu_threads=2,
        warmup_steps=1,
        device="cpu",
        output=str(tmp_path / "runs"),
        results=str(tmp_path / "results"),
        reports=str(tmp_path / "reports"),
    )
    return validate_config(base)


def test_end_to_end_training_writes_results_and_history(synthetic_cache, tmp_path):
    config = _smoke_config(synthetic_cache, tmp_path)
    for experiment in ("B0", "V0A", "V0B"):
        run_dir = train(json.loads(json.dumps(config)), experiment)
        for name in ("best.pt", "history.jsonl", "environment.json", "best_test_metrics.json"):
            assert (run_dir / name).exists(), (experiment, name)
        result = json.loads((run_dir / "best_test_metrics.json").read_text())
        for key in ("macro_auroc", "micro_auroc", "macro_f1", "micro_f1"):
            assert np.isfinite(result["fixed_0.5"][key])
        history = json.loads((run_dir / "history.jsonl").read_text().splitlines()[0])
        if experiment == "V0B":
            assert {"train_ecg", "train_temporal", "train_beat", "train_base"} <= set(history)
    with (tmp_path / "results" / "quaternion_experiments.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [row["experiment"] for row in rows] == ["B0", "V0A", "V0B"]
    assert tuple(rows[0]) == CSV_COLUMNS
    text = write_history(tmp_path / "runs", tmp_path / "reports").read_text()
    assert "V0 protocol selection" in text and "V0B dAUROC vs B0" in text


def test_smoke_runs_never_reach_the_results_csv(synthetic_cache, tmp_path):
    config = _smoke_config(synthetic_cache, tmp_path)
    train(config, "B0", limit_train=4, limit_val=4)
    assert not Path(tmp_path / "results" / "quaternion_experiments.csv").exists()
