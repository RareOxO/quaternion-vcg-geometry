"""Master prompt section U, items 1-12, for the shared quaternion utilities."""

import math

import pytest
import torch

from qlvcg.quaternion_utils import (
    angular_velocity_from_quaternion,
    enforce_sign_continuity,
    normalize_quaternion,
    quaternion_angle,
    quaternion_conjugate,
    quaternion_geodesic_distance,
    quaternion_inverse,
    quaternion_multiply,
    quaternion_sequence,
    quaternion_to_rotation_matrix,
    rotate_vector_by_quaternion,
    valid_rotation_mask,
    vectors_to_quaternion,
)

IDENTITY = torch.tensor([1.0, 0.0, 0.0, 0.0])


def _random_q(n=512, seed=0):
    return normalize_quaternion(torch.randn(n, 4, generator=torch.manual_seed(seed)))


def _random_v(n=512, seed=1):
    return torch.randn(n, 3, generator=torch.manual_seed(seed))


def _axis_angle(axis, angle):
    axis = torch.as_tensor(axis, dtype=torch.float32)
    axis = axis / axis.norm()
    return torch.cat((torch.tensor([math.cos(angle / 2)]), math.sin(angle / 2) * axis))


def test_1_identity_leaves_vectors_unchanged():
    v = _random_v()
    torch.testing.assert_close(rotate_vector_by_quaternion(v, IDENTITY.expand(len(v), 4)), v)


def test_2_rotation_preserves_norm():
    v, q = _random_v(), _random_q()
    torch.testing.assert_close(rotate_vector_by_quaternion(v, q).norm(dim=-1), v.norm(dim=-1))


def test_3_rotation_matrix_is_orthogonal():
    matrix = quaternion_to_rotation_matrix(_random_q())
    eye = torch.eye(3).expand_as(matrix)
    torch.testing.assert_close(matrix.transpose(-1, -2) @ matrix, eye, atol=1e-5, rtol=0)


def test_4_rotation_matrix_has_unit_determinant():
    det = torch.linalg.det(quaternion_to_rotation_matrix(_random_q()))
    torch.testing.assert_close(det, torch.ones_like(det), atol=1e-5, rtol=0)


def test_5_q_and_minus_q_are_the_same_rotation():
    q, v = _random_q(), _random_v()
    torch.testing.assert_close(quaternion_to_rotation_matrix(q), quaternion_to_rotation_matrix(-q))
    torch.testing.assert_close(
        rotate_vector_by_quaternion(v, q), rotate_vector_by_quaternion(v, -q)
    )
    distance = quaternion_geodesic_distance(q, -q)
    torch.testing.assert_close(distance, torch.zeros_like(distance), atol=1e-3, rtol=0)
    torch.testing.assert_close(quaternion_angle(q), quaternion_angle(-q))


def test_6_vectors_to_quaternion_aligns_v1_to_v2():
    v1, v2 = _random_v(seed=2), _random_v(seed=3)
    q = vectors_to_quaternion(v1, v2)
    rotated = rotate_vector_by_quaternion(v1 / v1.norm(dim=-1, keepdim=True), q)
    torch.testing.assert_close(rotated, v2 / v2.norm(dim=-1, keepdim=True), atol=1e-5, rtol=0)
    assert (q[:, 0] >= 0).all(), "shortest arc: the scalar part is never negative"


def test_7_parallel_vectors_give_the_identity():
    v = _random_v()
    q = vectors_to_quaternion(v, 3.0 * v)
    torch.testing.assert_close(q, IDENTITY.expand_as(q), atol=1e-5, rtol=0)


def test_8_antiparallel_vectors_give_a_finite_half_turn():
    v = _random_v()
    q = vectors_to_quaternion(v, -v)
    assert torch.isfinite(q).all()
    angle = quaternion_angle(q)
    torch.testing.assert_close(angle, torch.full_like(angle, math.pi), atol=1e-4, rtol=0)
    unit = v / v.norm(dim=-1, keepdim=True)
    torch.testing.assert_close(rotate_vector_by_quaternion(unit, q), -unit, atol=1e-5, rtol=0)


def test_8b_the_antiparallel_fallback_is_deterministic():
    v = torch.tensor([[0.2, 0.9, -0.4]])
    torch.testing.assert_close(vectors_to_quaternion(v, -v), vectors_to_quaternion(v, -v))


def test_9_zero_and_near_zero_vectors_are_finite():
    zero, tiny = torch.zeros(4, 3), torch.full((4, 3), 1e-20)
    for a, b in ((zero, _random_v(4)), (_random_v(4), zero), (zero, zero), (tiny, -tiny)):
        assert torch.isfinite(vectors_to_quaternion(a, b)).all()


def test_10_sequences_are_finite_and_sign_continuous():
    p = torch.randn(3, 400, 3, generator=torch.manual_seed(4))
    p[:, 100:120] = 0.0  # a flat stretch
    q = quaternion_sequence(p)
    assert q.shape == (3, 399, 4) and torch.isfinite(q).all()
    assert ((q[:, 1:] * q[:, :-1]).sum(-1) >= 0).all()


@pytest.mark.parametrize("zero_input", [False, True])
def test_11_12_backward_gives_finite_gradients(zero_input):
    p = torch.randn(2, 50, 3, generator=torch.manual_seed(5))
    if zero_input:
        p[:, 10:20] = 0.0
        p[:, 30] = -p[:, 29]  # an exactly antiparallel step
    p.requires_grad_(True)
    q = quaternion_sequence(p)
    loss = q.sum() + angular_velocity_from_quaternion(q, 0.01).sum()
    loss = loss + quaternion_geodesic_distance(q[:, 1:], q[:, :-1]).sum()
    loss.backward()
    assert p.grad is not None and torch.isfinite(p.grad).all()


def test_sign_continuity_flips_a_suffix_not_a_single_sample():
    q = _axis_angle([0, 0, 1], 0.3).expand(5, 4).clone()
    q[2:] = -q[2:]
    fixed = enforce_sign_continuity(q)
    torch.testing.assert_close(fixed, q[:1].expand(5, 4))


def test_angle_and_angular_speed():
    q = _axis_angle([1, 2, 3], 0.4)
    assert quaternion_angle(q).item() == pytest.approx(0.4, abs=1e-5)
    assert angular_velocity_from_quaternion(q, 0.01).item() == pytest.approx(40.0, rel=1e-4)
    with pytest.raises(ValueError):
        angular_velocity_from_quaternion(q, 0.0)


def test_hamilton_algebra():
    a, b, c = _random_q(64, 6), _random_q(64, 7), _random_q(64, 8)
    torch.testing.assert_close(
        quaternion_multiply(quaternion_multiply(a, b), c),
        quaternion_multiply(a, quaternion_multiply(b, c)),
        atol=1e-5,
        rtol=0,
    )
    product = quaternion_multiply(a, quaternion_inverse(a))
    torch.testing.assert_close(product, IDENTITY.expand_as(product), atol=1e-5, rtol=0)
    # Composition order: rotating by (a * b) is rotating by b, then by a.
    v = _random_v(64, 9)
    torch.testing.assert_close(
        rotate_vector_by_quaternion(v, quaternion_multiply(a, b)),
        rotate_vector_by_quaternion(rotate_vector_by_quaternion(v, b), a),
        atol=1e-5,
        rtol=0,
    )
    torch.testing.assert_close(quaternion_conjugate(quaternion_conjugate(a)), a)


def test_geodesic_distance_is_the_relative_angle():
    a = _axis_angle([0, 1, 0], 0.2)
    b = _axis_angle([0, 1, 0], 0.9)
    assert quaternion_geodesic_distance(a, b).item() == pytest.approx(0.7, abs=1e-4)


def test_valid_rotation_mask_is_relative_to_the_record():
    p = torch.ones(1, 100, 3)
    p[:, 40:50] = 1e-3
    mask = valid_rotation_mask(p, lag=1, min_fraction=0.02)
    assert mask.shape == (1, 99)
    assert not mask[0, 39:50].any() and mask[0, :39].all() and mask[0, 50:].all()
    torch.testing.assert_close(valid_rotation_mask(p * 1e4), mask)
