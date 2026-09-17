"""B0, V0, the Quaternion-LVCG variants and the shared supervised protocol."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from lvcg.models import LVCG
from qdg.data import CLASSES
from qlvcg.config import DEFAULT_CONFIG, load_config, validate_config
from qlvcg.engine import CSV_COLUMNS, evaluate, train, warmup_cosine
from qlvcg.models import (
    EXPERIMENTS,
    LVCG_RECONSTRUCTION_ONLY,
    QDFLVCG,
    QDTLVCG,
    MRQLVCG,
    BeatQuaternionFeatures,
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


@pytest.mark.parametrize("experiment", ["B0", "V0"])
def test_logits_are_multilabel_and_finite(config, experiment):
    model = build_model(config, experiment)
    logits = model(_ecg())
    assert logits.shape == (3, len(CLASSES)) and torch.isfinite(logits).all()


@pytest.mark.parametrize("experiment", ["B0", "V0"])
def test_no_quaternion_component(config, experiment):
    assert not any("quat" in name for name in _names(build_model(config, experiment)))


def test_b0_has_no_lvcg_machinery(config):
    names = set(_names(build_model(config, "B0")))
    for forbidden in ("beatencoder", "stategru", "beatsegmenter", "globalrrembedding"):
        assert forbidden not in names


def test_b0_sees_exactly_the_vcg_v0_sees(config):
    """Same geometry, same pseudo-inverse, same eps: the B0/V0 gap is the architecture."""
    ecg = _ecg()
    b0, v0 = build_model(config, "B0"), build_model(config, "V0")
    torch.testing.assert_close(
        b0.vcg(ecg), v0.backbone._recover_vcg(ecg, torch.arange(12).expand(3, -1))
    )


def test_v0_is_the_unmodified_author_class(config):
    model = build_model(config, "V0")
    assert isinstance(model, SupervisedLVCG) and type(model.backbone) is LVCG


def test_classification_gradients_skip_exactly_the_reconstruction_modules(config):
    """What `parameter_counts()['classification_path']` claims, checked by autograd."""
    model = build_model(config, "V0")
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


def test_record_shapes_reports_every_stage(config):
    shapes = record_shapes(build_model(config, "V0"), _ecg())
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
    for experiment in ("B0", "V0"):
        run_dir = train(json.loads(json.dumps(config)), experiment)
        for name in ("best.pt", "history.jsonl", "environment.json", "best_test_metrics.json"):
            assert (run_dir / name).exists(), (experiment, name)
        result = json.loads((run_dir / "best_test_metrics.json").read_text())
        for key in ("macro_auroc", "micro_auroc", "macro_f1", "micro_f1"):
            assert np.isfinite(result["fixed_0.5"][key])
        history = json.loads((run_dir / "history.jsonl").read_text().splitlines()[0])
        assert "train_loss" in history
    with (tmp_path / "results" / "quaternion_experiments.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [row["experiment"] for row in rows] == ["B0", "V0"]
    assert tuple(rows[0]) == CSV_COLUMNS
    text = write_history(tmp_path / "runs", tmp_path / "reports").read_text()
    assert "V0 dAUROC vs B0" in text and "selection" not in text


def test_runs_saved_as_v0a_are_read_as_v0(synthetic_cache, tmp_path):
    """Runs from before V0 became a single baseline must keep counting, and keep loading."""
    config = _smoke_config(synthetic_cache, tmp_path)
    run_dir = train(json.loads(json.dumps(config)), "V0")
    for name in ("best_test_metrics.json",):
        path = run_dir / name
        path.write_text(path.read_text().replace('"experiment": "V0"', '"experiment": "V0A"'))
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    checkpoint["experiment"] = "V0A"
    torch.save(checkpoint, run_dir / "best.pt")
    run_dir.rename(run_dir.with_name("V0A_seed42"))
    legacy = run_dir.with_name("V0A_seed42")
    assert evaluate(legacy / "best.pt", device="cpu")["experiment"] == "V0"
    text = write_history(tmp_path / "runs", tmp_path / "reports").read_text()
    assert "| V0 | lvcg |" in text


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
    v0 = build_model(config, "V0")
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
    v0 = build_model(config, "V0").parameter_counts()["total"]
    assert quaternion["total"] - v0 == quaternion["dynamic_branch"]
    gap = abs(control["dynamic_branch"] - quaternion["dynamic_branch"])
    assert gap / quaternion["dynamic_branch"] < 0.02


def test_v1_shapes_are_recorded(config):
    shapes = record_shapes(build_model(config, "V1"), _ecg())
    assert shapes["features, mask (dynamics)"] == [[3, 7, 999], [3, 999]]
    assert shapes["e_Q (dynamic_encoder)"] == [[3, 128]]
    assert shapes["logits (head)"] == [[3, len(CLASSES)]]


def test_v1_end_to_end_and_history_against_v0(synthetic_cache, tmp_path):
    config = _smoke_config(synthetic_cache, tmp_path)
    for experiment in ("B0", "V0", "V1"):
        run_dir = train(json.loads(json.dumps(config)), experiment)
    environment = json.loads((run_dir / "environment.json").read_text())
    assert environment["objective"] == "classification"
    train(json.loads(json.dumps(config)), "V2")
    text = write_history(tmp_path / "runs", tmp_path / "reports").read_text()
    assert "V1 dAUROC vs V0" in text and "## Against V0" in text
    v2_row = next(line for line in text.splitlines() if line.startswith("| V2 | qdt_lvcg |"))
    assert v2_row.split("|")[7].strip() != "-", "V2 must be compared with V1"
    v1_row = next(line for line in text.splitlines() if line.startswith("| V1 | qdf_lvcg | 42"))
    reused = next(line for line in text.splitlines() if line.startswith("| V3vcgrot (= V1) |"))
    assert v1_row.split("|")[4:] == reused.split("|")[4:], "V3 VCG+Rotation is read from V1"


# --- V2 QDT-LVCG ---

V2_EXPERIMENTS = [name for name, spec in EXPERIMENTS.items() if spec["model"] == "qdt_lvcg"]


def _v2_matching_v0(config, experiment):
    torch.manual_seed(0)
    v0 = build_model(config, "V0").eval()
    v2 = build_model(config, experiment).eval()
    v2.backbone.load_state_dict(v0.backbone.state_dict())
    return v0, v2


def test_v2_covers_section_h_and_its_control():
    specs = {name: EXPERIMENTS[name] for name in V2_EXPERIMENTS}
    assert specs["V2"]["fusion"] == "concat" and specs["V2gated"]["fusion"] == "gated"
    assert specs["V2"]["features"] == EXPERIMENTS["V1"]["features"], "same features as V1"
    assert specs["V2ctrl"]["features"] == ("position", "next_position", "delta")


@pytest.mark.parametrize("experiment", V2_EXPERIMENTS)
def test_untrained_v2_reproduces_v0_exactly(config, experiment):
    """The embedding path is the author's forward_inference, and the fusion starts as identity."""
    v0, v2 = _v2_matching_v0(config, experiment)
    ecg = _ecg()
    with torch.no_grad():
        torch.testing.assert_close(v2.embed(ecg), v0.backbone.forward_inference(ecg))


@pytest.mark.parametrize("experiment", V2_EXPERIMENTS)
def test_v2_logits_and_gradients(config, experiment):
    torch.manual_seed(0)
    model = build_model(config, experiment)
    logits = model(_ecg())
    assert logits.shape == (3, len(CLASSES)) and torch.isfinite(logits).all()
    logits.square().mean().backward()
    for module in (model.quaternion_beat_encoder, model.fusion):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
    # The zero-initialised quaternion block of the fusion learns from the first step.
    block = (
        model.fusion.projection.weight.grad[:, model.backbone.state_dim :]
        if experiment != "V2gated"
        else model.fusion.value.weight.grad
    )
    assert block.abs().sum() > 0


def test_only_the_anchor_beat_token_reaches_the_embedding(config):
    """The released StateGRU rolls out from beat 1's token and reads no other token."""
    torch.manual_seed(0)
    model = build_model(config, "V2").eval()
    with torch.no_grad():
        model.fusion.projection.weight.normal_(0, 0.05)
    ecg = _ecg()
    original = model.encode_beats

    def embed_with_nudge(index):
        def nudged(features, beat_mask):
            tokens = original(features, beat_mask).clone()
            tokens[:, index] += 3.0
            return tokens

        model.encode_beats = nudged
        try:
            with torch.no_grad():
                return model.embed(ecg)
        finally:
            model.encode_beats = original

    with torch.no_grad():
        base = model.embed(ecg)
    torch.testing.assert_close(embed_with_nudge(3), base)
    assert not torch.allclose(embed_with_nudge(1), base)


def test_beat_omega_is_a_physical_speed_whatever_the_beat_length():
    fs, steps, omega = 100, 128, 12.0  # rad/s
    beats, rr = [], []
    for length in (50, 100):
        t = torch.arange(steps) * (length - 1) / (steps - 1) / fs
        beats.append(torch.stack((torch.cos(omega * t), torch.sin(omega * t), torch.zeros(steps))))
        rr.append(float(length))
    beats = torch.stack(beats)[None]  # [1, 2, 3, P]
    features, mask = BeatQuaternionFeatures(("q", "theta", "omega"), fs)(
        beats, torch.tensor([rr]), torch.ones(1, 2, dtype=torch.bool), torch.ones(1, 3, 500)
    )
    assert mask.all()
    measured = features[0, :, 5]  # omega channel
    torch.testing.assert_close(measured, torch.full_like(measured, omega), rtol=1e-3, atol=1e-3)
    theta = features[0, :, 4].mean(-1)
    assert theta[1] / theta[0] == pytest.approx(99 / 49, rel=1e-3), "theta per step is not"


def test_beat_mask_uses_the_record_reference(config):
    """A quiet boundary interval must not call its own noise reliable."""
    beats = torch.ones(1, 2, 3, 128)
    beats[:, 0] *= 1e-3  # a quiet interval in a loud record
    vcg = torch.ones(1, 3, 1000)
    features, mask = BeatQuaternionFeatures(("q",), 100)(
        beats, torch.tensor([[90.0, 90.0]]), torch.ones(1, 2, dtype=torch.bool), vcg
    )
    assert not mask[0, 0].any() and mask[0, 1].all()


def test_padding_beats_never_reach_the_quaternion_encoder(config):
    model = build_model(config, "V2").train()
    features = torch.randn(2, 4, model.beat_dynamics.channels, 127)
    beat_mask = torch.tensor([[True, True, False, False], [True, True, True, False]])
    seen = []
    handle = model.quaternion_beat_encoder.register_forward_hook(
        lambda module, inputs, output: seen.append(inputs[0].shape[0])
    )
    try:
        tokens = model.encode_beats(features, beat_mask)
    finally:
        handle.remove()
    assert seen == [5]
    assert tokens[~beat_mask].abs().max() == 0 and tokens[beat_mask].abs().sum() > 0


def test_v2_parameters_and_shapes(config):
    v2 = build_model(config, "V2")
    assert isinstance(v2, QDTLVCG)
    counts = v2.parameter_counts()
    v0 = build_model(config, "V0").parameter_counts()["total"]
    assert counts["total"] - v0 == counts["dynamic_branch"]
    control = build_model(config, "V2ctrl").parameter_counts()["dynamic_branch"]
    assert abs(control - counts["dynamic_branch"]) / counts["dynamic_branch"] < 0.02
    shapes = record_shapes(v2, _ecg())
    beats = shapes["beats, rr, mask (beat_segmenter)"][0]
    assert shapes["beat features, mask (beat_dynamics)"][0] == [3, beats[1], 7, 127]
    assert shapes["fused tokens (fusion)"] == [[3, beats[1], 256]]
    assert shapes["logits (head)"] == [[3, len(CLASSES)]]


# --- V3 MRQ-LVCG ---

V3_EXPERIMENTS = [name for name, spec in EXPERIMENTS.items() if spec["model"] == "mrq_lvcg"]
V3_TRAINED = [name for name in V3_EXPERIMENTS if not EXPERIMENTS[name].get("reuses")]


def test_v3_covers_the_six_section_i_ablations_and_the_control():
    branches = {EXPERIMENTS[name]["branches"] for name in V3_EXPERIMENTS}
    assert {
        ("magnitude",),
        ("rotation",),
        ("magnitude", "rotation"),
        ("vcg", "magnitude"),
        ("vcg", "rotation"),
        ("vcg", "magnitude", "rotation"),
    } <= branches
    assert EXPERIMENTS["V3"]["branches"] == ("vcg", "magnitude", "rotation")
    assert EXPERIMENTS["V3ctrl"]["branches"] == ("vcg", "position", "delta")


@pytest.mark.parametrize("experiment", V3_TRAINED)
def test_v3_logits_and_gradients_are_finite(config, experiment):
    torch.manual_seed(0)
    model = build_model(config, experiment)
    logits = model(_ecg())
    assert logits.shape == (3, len(CLASSES)) and torch.isfinite(logits).all()
    logits.square().mean().backward()
    for encoder in model.branch_encoders.values():
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all() for p in encoder.parameters()
        )


def test_vcg_plus_rotation_is_v1(config):
    """The reuse is only honest if the two are the same model."""
    v1, v3 = build_model(config, "V1"), build_model(config, "V3vcgrot")
    assert v1.parameter_counts()["total"] == v3.parameter_counts()["total"]
    assert (
        v1.dynamic_encoder.net[0].num_features == v3.branch_encoders["rotation"].net[0].num_features
    )
    assert v1.dynamics.features == v3.branch_features["rotation"].features
    assert v1.head.net.in_features == v3.head.net.in_features
    ecg = _ecg()
    features_v1, _ = v1.dynamics(v1.vcg(ecg))
    features_v3, _ = v3.branch_features["rotation"](v3.vcg(ecg))
    torch.testing.assert_close(features_v1, features_v3)


def test_reused_experiments_are_not_trained(config):
    with pytest.raises(ValueError, match="same configuration as V1"):
        train(config, "V3vcgrot")


def test_branches_without_vcg_carry_no_lvcg_backbone(config):
    for name in ("V3mag", "V3rot", "V3magrot"):
        model = build_model(config, name)
        assert model.backbone is None
        assert not any(type(m).__name__ == "LVCG" for m in model.modules())
        torch.testing.assert_close(model.vcg(_ecg()), build_model(config, "B0").vcg(_ecg()))


def test_magnitude_branch_encodes_the_norm_alone(config):
    model = build_model(config, "V3mag")
    vcg = model.vcg(_ecg())
    features, _ = model.branch_features["magnitude"](vcg)
    assert features.shape == (3, 1, 999)
    torch.testing.assert_close(features[:, 0], vcg[..., :-1].norm(dim=1), atol=1e-5, rtol=0)


def test_rotation_and_magnitude_factorise_the_vector(config):
    """P_t = r_t u_t: the magnitude and the direction the rotation is built from recover P."""
    from qlvcg.quaternion_utils import rotate_vector_by_quaternion

    model = build_model(config, "V3")
    p = model.vcg(_ecg()).transpose(1, 2)
    r = p.norm(dim=-1, keepdim=True)
    u = p / r
    q, _ = model.branch_features["rotation"](p.transpose(1, 2))
    q = q[:, :4].transpose(1, 2)
    valid = (r[:, :-1, 0] > 0.05 * r.amax(dim=1)) & (r[:, 1:, 0] > 0.05 * r.amax(dim=1))
    rotated = rotate_vector_by_quaternion(u[:, :-1], q)
    torch.testing.assert_close(rotated[valid], u[:, 1:][valid], atol=1e-4, rtol=0)
    torch.testing.assert_close(r * u, p, atol=1e-5, rtol=0)


def test_real_control_matches_v3_parameter_for_parameter(config):
    v3 = build_model(config, "V3").parameter_counts()
    control = build_model(config, "V3ctrl").parameter_counts()
    assert v3["total"] == control["total"]
    assert (
        v3["total"] - build_model(config, "V0").parameter_counts()["total"] == v3["dynamic_branch"]
    )


def test_v3_shapes_are_recorded(config):
    shapes = record_shapes(build_model(config, "V3"), _ecg())
    assert shapes["magnitude features, mask"][0] == [3, 1, 999]
    assert shapes["rotation features, mask"][0] == [3, 7, 999]
    assert shapes["e_magnitude"] == [[3, 128]] and shapes["e_rotation"] == [[3, 128]]
    assert shapes["logits (head)"] == [[3, len(CLASSES)]]
    only = record_shapes(build_model(config, "V3rot"), _ecg())
    assert "vcg (lift)" in only and "beat tokens (beat_encoder)" not in only
    assert isinstance(build_model(config, "V3"), MRQLVCG)
