"""Experiment 1 encoder benchmark: shape contract, pairing, and the real QLSTM.

The QLSTM here is the quaternion LSTM of Parcollet et al., not the quantum LSTM that
shares the abbreviation. Its gates must act through the Hamilton product, so the tests
check the algebra rather than only the tensor shapes.
"""

from pathlib import Path

import pytest
import torch

from qdg.config import load_config, validate_config
from qdg.encoders import (
    ENCODERS,
    GENERIC_ENCODERS,
    QUATERNION_ENCODERS,
    QUATERNION_PAIR,
    QuaternionLSTM,
    branch_encoder,
    build_encoder,
    encoder_widths,
)
from qdg.experiments import BENCHMARK, BENCHMARK_LABELS, experiment_config, select_encoder
from qdg.geometry import hamilton_product
from qdg.models import BRANCH_BLOCKS, build_model
from qdg.quaternion_nn import QuaternionConv1d, QuaternionLinear

STATS = {"sampling_rate": 500, "signal_length": 5000, "vcg_std": [0.19716, 0.12212, 0.17387]}


@pytest.fixture
def full_config():
    return load_config(Path(__file__).resolve().parents[1] / "configs/base.yaml")


@pytest.fixture
def ecg():
    return torch.randn(2, 12, 5000, generator=torch.manual_seed(31))


def _model(config, name):
    current = experiment_config(config, name)
    validate_config(current)
    return build_model(current["model"], STATS)


# --- QuaternionLSTM: the algebra, not just the shapes ---


def test_qlstm_gates_use_the_hamilton_product():
    """One quaternion in, one per gate out: each pre-activation must be W (x) x + b."""
    rnn = QuaternionLSTM(1, 1)
    x = torch.randn(1, 1, 4)
    with torch.no_grad():
        out = rnn(x)
        gate = rnn.split_gates(rnn.input_projection(x))
        # weight is (4 components, out_quaternions=4 gates, in_quaternions=1, ...)
        for index, name in enumerate(rnn.GATES):
            weight = rnn.input_projection.weight[:, index].reshape(4)
            bias = rnn.split_gates(rnn.input_projection.bias)[name]
            torch.testing.assert_close(
                gate[name][0, 0], hamilton_product(weight, x[0, 0]) + bias, atol=1e-6, rtol=1e-5
            )
    # At t=0 the hidden state is zero, so only the input path contributes.
    cell = gate["i"].sigmoid() * gate["g"].tanh()
    expected = gate["o"].sigmoid() * cell.tanh()
    torch.testing.assert_close(out[0, 0], expected[0, 0], atol=1e-6, rtol=1e-5)


def test_qlstm_gate_split_respects_the_component_major_layout():
    """A flat four-way slice would mix components; the split must go inside each block."""
    rnn = QuaternionLSTM(2, 3)
    projected = torch.randn(5, 16 * 3)
    gates = rnn.split_gates(projected)
    blocks = projected.reshape(5, 4, 4, 3)
    for index, name in enumerate(rnn.GATES):
        assert gates[name].shape == (5, 12)
        torch.testing.assert_close(gates[name], blocks[:, :, index, :].reshape(5, 12))
    # The naive contiguous split is a different tensor, which is the bug this guards.
    assert not torch.allclose(gates["f"], projected[:, :12])


def test_qlstm_uses_split_activations_and_componentwise_gating():
    """sigma/alpha act on each component; gating is a component-wise product."""
    rnn = QuaternionLSTM(2, 3)
    x = torch.randn(4, 7, 8)
    out = rnn(x)
    assert out.shape == (4, 7, 12)
    # tanh bounds the hidden state componentwise, which componentwise gating preserves.
    assert out.abs().max() <= 1.0


def test_qlstm_is_built_from_quaternion_layers_only():
    rnn = QuaternionLSTM(2, 2)
    linears = [m for m in rnn.modules() if isinstance(m, QuaternionLinear)]
    assert len(linears) == 2  # one fused input projection, one fused recurrent one
    assert not any(isinstance(m, torch.nn.Linear) for m in rnn.modules())
    # Four gates of 2 output quaternions each, from 2 input quaternions, four components.
    assert rnn.input_projection.weight.shape == (4, 8, 2)
    assert rnn.recurrent_projection.weight.shape == (4, 8, 2)
    # A real LSTM of the same width would hold four times these gate weights.
    assert sum(layer.weight.numel() for layer in linears) == 2 * 4 * 8 * 2


def test_qlstm_is_recurrent_and_trainable():
    rnn = QuaternionLSTM(2, 2)
    x = torch.randn(2, 12, 8)
    out = rnn(x)
    out.square().mean().backward()
    for layer in (rnn.input_projection, rnn.recurrent_projection):
        assert layer.weight.grad is not None and torch.isfinite(layer.weight.grad).all()
        # Every one of the four quaternion components must receive gradient.
        assert (layer.weight.grad.flatten(1).abs().sum(-1) > 0).all()
    # Later steps depend on earlier ones: perturbing step 0 must move the last output.
    perturbed = x.clone()
    perturbed[:, 0] += 5.0
    with torch.no_grad():
        assert not torch.allclose(rnn(x)[:, -1], rnn(perturbed)[:, -1], atol=1e-5)


# --- The benchmark contract ---


@pytest.mark.parametrize("name", ENCODERS)
def test_every_encoder_meets_the_shape_contract(name):
    width, _ = encoder_widths(name, 88)
    encoder = build_encoder(name, 8, width).eval()
    with torch.inference_mode():
        out = encoder(torch.randn(2, 8, 5000))
    assert out.shape == (2, width)
    assert torch.isfinite(out).all()
    assert encoder.width == width


@pytest.mark.parametrize("name", QUATERNION_ENCODERS)
def test_quaternion_encoders_contain_quaternion_layers(name):
    width, _ = encoder_widths(name, 88)
    encoder = build_encoder(name, 8, width)
    assert any(isinstance(m, (QuaternionLinear, QuaternionConv1d)) for m in encoder.modules())


@pytest.mark.parametrize("name", GENERIC_ENCODERS)
def test_generic_encoders_contain_no_quaternion_layers(name):
    width, _ = encoder_widths(name, 88)
    encoder = build_encoder(name, 8, width)
    assert not any(isinstance(m, (QuaternionLinear, QuaternionConv1d)) for m in encoder.modules())


def test_r_and_l_branches_stay_real(full_config):
    """Plan section 2.1: only the Q branch becomes quaternion-valued."""
    for quaternion, generic in QUATERNION_PAIR.items():
        model = _model(full_config, f"E1_{quaternion}")
        assert model.settings["branch_encoders"]["angular"] == quaternion
        for branch in ("radial", "linear"):
            assert model.settings["branch_encoders"][branch] == generic
            assert not model.encoders[branch].quaternion
        assert branch_encoder(quaternion, "angular") == quaternion
        assert branch_encoder(quaternion, "radial") == generic


def test_quaternion_pair_differs_only_in_the_q_branch(full_config, ecg):
    """lstm vs qlstm: identical R and L encoders, identical input, one operator apart."""
    for quaternion, generic in QUATERNION_PAIR.items():
        pair = (_model(full_config, f"E1_{generic}"), _model(full_config, f"E1_{quaternion}"))
        for branch in ("radial", "linear"):
            assert type(pair[0].encoders[branch]) is type(pair[1].encoders[branch])
            assert pair[0].encoders[branch].width == pair[1].encoders[branch].width
        for block in BRANCH_BLOCKS:
            torch.testing.assert_close(
                pair[0].frontends[block](ecg), pair[1].frontends[block](ecg), atol=0, rtol=0
            )


@pytest.mark.parametrize("name", BENCHMARK)
def test_benchmark_models_run_and_train(full_config, ecg, name):
    model = _model(full_config, name)
    logits = model(ecg)
    assert logits.shape == (2, 5) and torch.isfinite(logits).all()
    logits.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.settings["angular_blocks"] == ["rotation"]
    assert set(BENCHMARK_LABELS) == set(ENCODERS)


def test_recurrent_and_global_encoders_report_no_finite_field(full_config):
    for name in ("E1_lstm", "E1_gru", "E1_transformer", "E1_qlstm", "E1_qtransformer"):
        assert _model(full_config, name).describe()["receptive_field_samples"] is None
    # The convolutional and graph encoders do have one, and it is the same for both.
    assert _model(full_config, "E1_tcn").describe()["receptive_field_samples"] == 1320
    assert _model(full_config, "E1_qgnn").describe()["receptive_field_samples"] == 1320


def test_selection_rule_uses_validation_only():
    rows = {
        "E1_lstm": {
            "validation_macro_auroc_mean": 0.90,
            "validation_macro_auprc_mean": 0.7,
            "parameters": 10,
        },
        "E1_tcn": {
            "validation_macro_auroc_mean": 0.92,
            "validation_macro_auprc_mean": 0.7,
            "parameters": 99,
        },
    }
    assert "not selected" in select_encoder({})
    assert "2/10" in select_encoder(rows)
    full = {
        f"E1_{name}": {
            "validation_macro_auroc_mean": 0.90,
            "validation_macro_auprc_mean": 0.70,
            "parameters": 100 - index,  # tie on both metrics; fewest params wins
        }
        for index, name in enumerate(ENCODERS)
    }
    assert select_encoder(full) == BENCHMARK_LABELS[ENCODERS[-1]]
    full["E1_gru"]["validation_macro_auroc_mean"] = 0.95
    assert select_encoder(full) == "GRU"


def test_earlier_variants_are_untouched(full_config, ecg):
    """No `encoder` key means the previous TCN-based model, unchanged."""
    for name, channels in (("M0", 3), ("M1", 4), ("M4", 32), ("RLA", 8)):
        assert _model(full_config, name).frontend.out_channels == channels
    legacy = _model(full_config, "RLA_standard")
    assert legacy.settings["encoder"] is None
    assert legacy.describe()["receptive_field_samples"] == 640
