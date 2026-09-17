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
from qlvcg.engine import CSV_COLUMNS, resolve_objective, train, warmup_cosine
from qlvcg.models import (
    EXPERIMENTS,
    LVCG_RECONSTRUCTION_ONLY,
    QDFLVCG,
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


# --- V1 QDF-LVCG ---

V1_EXPERIMENTS = [name for name, spec in EXPERIMENTS.items() if spec["model"] == "qdf_lvcg"]


def test_v1_covers_the_section_g_ablation_and_the_control():
    features = {name: EXPERIMENTS[name]["features"] for name in V1_EXPERIMENTS}
    assert features["V1"] == ("q", "theta", "omega")
    assert {features["V1q"], features["V1theta"], features["V1omega"]} == {
        ("q",),
        ("theta",),
        ("omega",),
    }
    assert features["V1ctrl"] == ("position", "next_position", "delta")


@pytest.mark.parametrize("experiment", V1_EXPERIMENTS)
def test_v1_logits_and_gradients_are_finite(config, experiment):
    torch.manual_seed(0)
    model = build_model(config, experiment)
    logits = model(_ecg())
    assert logits.shape == (3, len(CLASSES)) and torch.isfinite(logits).all()
    logits.square().mean().backward()
    grads = [p.grad for p in model.dynamic_encoder.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


def test_v1_keeps_the_v0_model_intact(config):
    """Section G: the V0 branch is untouched; V1 only adds beside it."""
    torch.manual_seed(0)
    v0 = build_model(config, "V0A")
    torch.manual_seed(0)
    v1 = build_model(config, "V1")
    assert isinstance(v1, QDFLVCG) and type(v1.backbone) is LVCG
    assert v1.backbone.state_dict().keys() == v0.backbone.state_dict().keys()
    for (name, a), b in zip(v0.backbone.state_dict().items(), v1.backbone.state_dict().values()):
        torch.testing.assert_close(a, b, msg=name)
    ecg = _ecg()
    v0.eval()
    v1.eval()
    with torch.no_grad():
        torch.testing.assert_close(
            v0.backbone.forward_inference(ecg), v1.backbone.forward_inference(ecg)
        )
    assert v1.head.net.in_features == v0.head.net.in_features + 128


def test_v1_uses_the_vcg_the_backbone_segments(config):
    model, ecg = build_model(config, "V1"), _ecg()
    torch.testing.assert_close(model.vcg(ecg), build_model(config, "B0").vcg(ecg))


def test_v1_feature_channels_and_masked_identity(config):
    model = build_model(config, "V1")
    ecg = _ecg()
    ecg[:, :, 400:500] = 0.0  # a flat stretch: no direction to rotate
    features, mask = model.dynamics(model.vcg(ecg))
    assert features.shape == (3, 4 + 1 + 1 + 1, 999)
    assert not mask[:, 405:495].any()
    flat = features[:, :, 405:495]
    assert torch.allclose(flat[:, 1:4], torch.zeros_like(flat[:, 1:4]))  # q_xyz
    assert torch.allclose(flat[:, 0].abs(), torch.ones_like(flat[:, 0]))  # |q_w| = 1
    # theta and omega carry the safe-norm floor: 2e-6 rad and 2e-4 rad/s, against a
    # median omega of about 17.5 rad/s on real records.
    assert flat[:, 4].abs().max() < 1e-5 and flat[:, 5].abs().max() < 1e-3
    assert torch.equal(features[:, -1], mask.float())


def test_omega_is_theta_over_dt(config):
    model = build_model(config, "V1")
    features, _ = model.dynamics(model.vcg(_ecg()))
    torch.testing.assert_close(features[:, 5], features[:, 4] * config["model"]["lvcg"]["fs"])


def test_optional_features_extend_the_quaternion_sets_only(config):
    config["model"]["qdf_lvcg"].update(include_magnitude=True, include_linear_velocity=True)
    assert build_model(config, "V1").dynamics.features[-2:] == ("magnitude", "linear_velocity")
    control = build_model(config, "V1ctrl").dynamics.features
    assert control == ("position", "next_position", "delta")


def test_real_control_is_parameter_matched(config):
    quaternion = build_model(config, "V1").parameter_counts()
    control = build_model(config, "V1ctrl").parameter_counts()
    v0 = build_model(config, "V0A").parameter_counts()["total"]
    assert quaternion["total"] - v0 == quaternion["dynamic_branch"]
    gap = abs(control["dynamic_branch"] - quaternion["dynamic_branch"])
    assert gap / quaternion["dynamic_branch"] < 0.02


def test_v1_will_not_train_before_v0_selects_its_objective(config):
    assert config["protocol"]["objective"] is None
    with pytest.raises(ValueError, match="V0 selected"):
        resolve_objective(config, "V1")
    config["protocol"]["objective"] = "classification+auxiliary"
    assert resolve_objective(config, "V1") == "classification+auxiliary"
    assert resolve_objective(config, "V0A") == "classification"


def test_v1_auxiliary_objective_still_trains_the_backbone(config):
    torch.manual_seed(0)
    model = build_model(config, "V1")
    terms = model.auxiliary_losses(_ecg(4), num_visible=3)
    assert all(torch.isfinite(v) for v in terms.values())


def test_v1_shapes_are_recorded(config):
    shapes = record_shapes(build_model(config, "V1"), _ecg())
    assert shapes["features, mask (dynamics)"] == [[3, 7, 999], [3, 999]]
    assert shapes["e_Q (dynamic_encoder)"] == [[3, 128]]
    assert shapes["logits (head)"] == [[3, len(CLASSES)]]


def test_v1_end_to_end_and_history_against_v0(synthetic_cache, tmp_path):
    config = _smoke_config(synthetic_cache, tmp_path)
    for experiment in ("B0", "V0A", "V0B"):
        train(json.loads(json.dumps(config)), experiment)
    config["protocol"]["objective"] = "classification"
    run_dir = train(json.loads(json.dumps(config)), "V1")
    environment = json.loads((run_dir / "environment.json").read_text())
    assert environment["objective"] == "classification"
    text = write_history(tmp_path / "runs", tmp_path / "reports").read_text()
    assert "V1 dAUROC vs V0" in text and "## Against V0" in text
