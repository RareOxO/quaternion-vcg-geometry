"""The §5 mathematical pre-checks of the Temporal evolution 方案.

This round introduces the first genuine rotation quaternion in the project, so the
tests are about the algebra being right, not just the tensors being the right shape.
The distinction that carries the whole experiment is A1 vs A2: same two operands,
subtraction in R^4 versus composition in the rotation group.
"""

import math
from pathlib import Path

import pytest
import torch

from qdg.config import load_config, validate_config
from qdg.experiments import EVOLUTION_NEW, NOISE_BAND, experiment_config, interpret_evolution
from qdg.geometry import (
    canonical_sign,
    hamilton_product,
    local_rotation,
    quaternion_conjugate,
    relation_descriptor,
    rotation_difference,
    rotation_evolution,
    rotation_quaternion,
    unit_direction,
)
from qdg.models import ANGULAR_BLOCKS, build_model

STATS = {"sampling_rate": 500, "signal_length": 5000, "vcg_std": [0.19716, 0.12212, 0.17387]}
X = torch.tensor([1.0, 0.0, 0.0])
Y = torch.tensor([0.0, 1.0, 0.0])
Z = torch.tensor([0.0, 0.0, 1.0])


@pytest.fixture
def full_config():
    return load_config(Path(__file__).resolve().parents[1] / "configs/base.yaml")


@pytest.fixture
def ecg():
    return torch.randn(2, 12, 5000, generator=torch.manual_seed(23))


def _model(config, name):
    current = experiment_config(config, name)
    validate_config(current)
    return build_model(current["model"], STATS)


def _rotate(q, v):
    """Apply the rotation q to a 3-vector: q (x) [0, v] (x) q*."""
    pure = torch.cat((torch.zeros_like(v[..., :1]), v), dim=-1)
    return hamilton_product(hamilton_product(q, pure), quaternion_conjugate(q))[..., 1:]


# 1. The minimal rotation from u to v is built correctly, in half-angle form.
def test_rotation_quaternion_is_half_angle():
    q = rotation_quaternion(X, Y)
    half = math.cos(math.pi / 4)
    torch.testing.assert_close(q, torch.tensor([half, 0.0, 0.0, half]), atol=1e-6, rtol=1e-5)
    # The full-angle descriptor of the same pair is a different object entirely.
    torch.testing.assert_close(relation_descriptor(X, Y), torch.tensor([0.0, 0.0, 0.0, 1.0]))
    assert not torch.allclose(q, relation_descriptor(X, Y))


def test_rotation_quaternion_actually_rotates():
    """The real test of a rotation quaternion: applying it must move u onto v."""
    u = unit_direction(torch.randn(64, 3, generator=torch.manual_seed(1)))
    v = unit_direction(torch.randn(64, 3, generator=torch.manual_seed(2)))
    torch.testing.assert_close(_rotate(rotation_quaternion(u, v), u), v, atol=1e-5, rtol=1e-4)


def test_rotation_quaternion_is_the_minimal_rotation():
    """Its axis is perpendicular to both endpoints, i.e. no spin about the path."""
    u = unit_direction(torch.randn(32, 3, generator=torch.manual_seed(4)))
    v = unit_direction(torch.randn(32, 3, generator=torch.manual_seed(5)))
    axis = rotation_quaternion(u, v)[..., 1:]
    assert (axis * u).sum(-1).abs().max() < 1e-5
    assert (axis * v).sum(-1).abs().max() < 1e-5


# 2. Parallel and antiparallel inputs stay finite and meaningful.
def test_degenerate_pairs_are_stable():
    torch.testing.assert_close(rotation_quaternion(X, X), torch.tensor([1.0, 0.0, 0.0, 0.0]))
    for u in (X, Y, Z, unit_direction(torch.tensor([0.3, -0.7, 0.2]))):
        q = rotation_quaternion(u, -u)
        assert torch.isfinite(q).all()
        torch.testing.assert_close(q.norm(), torch.tensor(1.0), atol=1e-5, rtol=0)
        # 180 degrees: zero scalar, and an axis perpendicular to u.
        assert q[0].abs() < 1e-5
        assert (q[1:] * u).sum().abs() < 1e-5
        torch.testing.assert_close(_rotate(q, u), -u, atol=1e-5, rtol=1e-4)
    # Nearly parallel and nearly antiparallel both stay finite.
    for scale in (1e-7, 1e-4):
        near = unit_direction(X + scale * Y)
        assert torch.isfinite(rotation_quaternion(X, near)).all()
        assert torch.isfinite(rotation_quaternion(X, -near)).all()


def test_gradients_survive_degenerate_pairs():
    u = torch.tensor([[1.0, 0.0, 0.0]], requires_grad=True)
    rotation_quaternion(u, -u.detach()).sum().backward()
    assert torch.isfinite(u.grad).all()


# 3. Sign convention: w >= 0, so q never flips against -q across time.
def test_sign_is_canonical():
    u = unit_direction(torch.randn(4, 300, 3, generator=torch.manual_seed(7)))
    q = local_rotation(u, 10)
    assert (q[..., 0] >= 0).all()
    assert (rotation_evolution(q, 10)[..., 0] >= 0).all()
    flipped = torch.tensor([-0.5, 0.5, -0.5, 0.5])
    torch.testing.assert_close(canonical_sign(flipped), -flipped)
    torch.testing.assert_close(canonical_sign(-flipped), -flipped)


# 4. Unit norm, and inverse/composition are numerically correct.
def test_unit_norm_and_composition():
    u = unit_direction(torch.randn(3, 200, 3, generator=torch.manual_seed(8)))
    q = local_rotation(u, 10)
    assert (q.norm(dim=-1) - 1).abs().max() < 1e-5
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand_as(q)
    torch.testing.assert_close(
        hamilton_product(q, quaternion_conjugate(q)), identity, atol=1e-5, rtol=1e-4
    )
    e = rotation_evolution(q, 10)
    assert (e.norm(dim=-1) - 1).abs().max() < 1e-5
    # e composes: q_t (x) e_t == q_{t+tau}, up to the canonical sign.
    composed = hamilton_product(q[:, :-10], e[:, :-10])
    torch.testing.assert_close(
        canonical_sign(composed), canonical_sign(q[:, 10:]), atol=1e-4, rtol=1e-3
    )


# 5. e_t is a rotation; the A1 difference is not. They must not be conflated.
def test_evolution_and_difference_are_different_objects():
    u = unit_direction(torch.randn(2, 120, 3, generator=torch.manual_seed(9)))
    q = local_rotation(u, 10)
    evolution, difference = rotation_evolution(q, 10), rotation_difference(q, 10)
    assert evolution.shape == difference.shape
    assert not torch.allclose(evolution, difference, atol=1e-2)
    # The composition is unit norm; the coordinate-wise difference is not.
    assert (evolution.norm(dim=-1) - 1).abs().max() < 1e-5
    assert (difference.norm(dim=-1) - 1).abs().max() > 0.1
    # A constant rotation sequence: evolution is the identity, difference is zero.
    constant = torch.tensor([0.8, 0.6, 0.0, 0.0]).expand(1, 60, 4).contiguous()
    torch.testing.assert_close(
        rotation_evolution(constant, 5)[:, :50],
        torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(1, 50, 4),
        atol=1e-6,
        rtol=1e-5,
    )
    torch.testing.assert_close(rotation_difference(constant, 5)[:, :50], torch.zeros(1, 50, 4))


def test_difference_is_plain_subtraction():
    q = torch.randn(1, 40, 4)
    torch.testing.assert_close(rotation_difference(q, 4)[:, :36], q[:, 4:] - q[:, :-4])


# Model-level checks: only the angular representation moves.
@pytest.mark.parametrize("name", EVOLUTION_NEW)
def test_variants_build_and_train(full_config, ecg, name):
    model = _model(full_config, name)
    logits = model(ecg)
    assert logits.shape == (2, 5) and torch.isfinite(logits).all()
    logits.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.settings["angular_blocks"] == ANGULAR_BLOCKS[name]
    assert model.settings["tau_ms"] == 20


def test_a1_and_a2_are_matched(full_config, ecg):
    """The comparison that matters: identical everything except the combination rule."""
    a1, a2 = _model(full_config, "A1"), _model(full_config, "A2")
    assert sum(p.numel() for p in a1.parameters()) == sum(p.numel() for p in a2.parameters())
    assert a1.angular_parameters() == a2.angular_parameters()
    assert a1.frontends["angular"].out_channels == a2.frontends["angular"].out_channels == 8
    assert a1.settings["tau_ms"] == a2.settings["tau_ms"]
    for block in ("radial", "linear"):
        torch.testing.assert_close(
            a1.frontends[block](ecg), a2.frontends[block](ecg), atol=0, rtol=0
        )
    # Their first four angular channels are the same q_t; only the second half differs.
    first, second = a1.frontends["angular"](ecg), a2.frontends["angular"](ecg)
    torch.testing.assert_close(first[:, :4], second[:, :4], atol=0, rtol=0)
    assert not torch.allclose(first[:, 4:], second[:, 4:], atol=1e-2)


def test_a0_is_the_local_only_floor(full_config, ecg):
    a0, a1 = _model(full_config, "A0"), _model(full_config, "A1")
    assert a0.frontends["angular"].out_channels == 4
    torch.testing.assert_close(
        a0.frontends["angular"](ecg), a1.frontends["angular"](ecg)[:, :4], atol=0, rtol=0
    )


def test_no_quaternion_conv_anywhere(full_config):
    """§4/§8: the backend is a plain real encoder in every variant this round."""
    from qdg.quaternion_nn import QuaternionConv1d

    for name in EVOLUTION_NEW:
        model = _model(full_config, name)
        assert not any(isinstance(m, QuaternionConv1d) for m in model.modules())
        assert model.settings["angular_algebra"] == "standard"
        assert all(not encoder.quaternion for encoder in model.encoders.values())
        assert len({e.receptive_field for e in model.encoders.values()}) == 1


def test_evolution_verdicts():
    big = 4 * NOISE_BAND
    assert "Incomplete" in interpret_evolution({})
    assert "positive signal" in interpret_evolution({"A2 - A1": big, "A1 - A0": big})
    assert "is behind" in interpret_evolution({"A2 - A1": -big, "A1 - A0": 0.0})
    flat = interpret_evolution({"A2 - A1": 0.001, "A1 - A0": 0.001})
    assert "indistinguishable" in flat and "not distinguishable from local state" in flat
    mixed = interpret_evolution({"A2 - A1": 0.001, "A1 - A0": big})
    assert "does add information over the local angular state" in mixed


def test_earlier_variants_are_untouched(full_config, ecg):
    for name, channels in (("M0", 3), ("M1", 4), ("M4", 32), ("RLA", 8)):
        assert _model(full_config, name).frontend.out_channels == channels
    # M1's full-angle descriptor must not have been replaced by the rotation quaternion.
    m1 = _model(full_config, "M1").frontend(ecg)
    assert (m1.square().sum(1) - 1).abs().max() < 1e-4
