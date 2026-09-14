import math

import pytest
import torch

from qdg.geometry import (
    KORS,
    VCGGeometry,
    first_order,
    hamilton_product,
    kors_transform,
    lag_samples,
    pure_quaternion_product,
    quaternion_inverse,
    relation_descriptor,
    relation_norm_error,
    second_order,
    unit_direction,
)
from qdg.sanity import sanity_checks

X = torch.tensor([1.0, 0.0, 0.0])
Y = torch.tensor([0.0, 1.0, 0.0])
Z = torch.tensor([0.0, 0.0, 1.0])


def test_theoretical_sanity_checks_all_pass():
    """方案 §6 in full; the CLI prints the same expected-vs-actual table."""
    assert sanity_checks()["all_passed"]


def test_hamilton_basis_multiplication_table():
    one, i, j, k = torch.eye(4)
    torch.testing.assert_close(hamilton_product(i, j), k)
    torch.testing.assert_close(hamilton_product(j, k), i)
    torch.testing.assert_close(hamilton_product(k, i), j)
    torch.testing.assert_close(hamilton_product(j, i), -k)
    torch.testing.assert_close(hamilton_product(i, i), -one)
    torch.testing.assert_close(hamilton_product(one, k), k)


def test_inverse_returns_identity():
    q = torch.randn(5, 4)
    torch.testing.assert_close(
        hamilton_product(q, quaternion_inverse(q)),
        torch.tensor([1.0, 0, 0, 0]).expand(5, 4).contiguous(),
    )


def test_raw_product_and_descriptor_differ_only_in_the_scalar_sign():
    """方案 §4.1: the raw product is [-dot, cross], the descriptor is [+dot, cross]."""
    u = unit_direction(torch.randn(20, 3))
    v = unit_direction(torch.randn(20, 3))
    raw, descriptor = pure_quaternion_product(u, v), relation_descriptor(u, v)
    torch.testing.assert_close(raw[:, 0], -descriptor[:, 0])
    torch.testing.assert_close(raw[:, 1:], descriptor[:, 1:])


def test_descriptor_known_geometry():
    torch.testing.assert_close(relation_descriptor(X, Y), torch.tensor([0.0, 0, 0, 1]))
    torch.testing.assert_close(relation_descriptor(Y, X), torch.tensor([0.0, 0, 0, -1]))
    torch.testing.assert_close(relation_descriptor(X, -X), torch.tensor([-1.0, 0, 0, 0]))
    torch.testing.assert_close(relation_descriptor(X, X), torch.tensor([1.0, 0, 0, 0]))
    torch.testing.assert_close(relation_descriptor(Y, Z), torch.tensor([0.0, 1, 0, 0]))


def test_descriptor_is_cos_theta_and_n_sin_theta():
    """方案 §4.2: full angle, not the half angle of a rotation quaternion."""
    theta = torch.tensor(0.7)
    q = relation_descriptor(X, torch.stack((theta.cos(), theta.sin(), torch.zeros(()))))
    torch.testing.assert_close(q, torch.tensor([theta.cos(), 0.0, 0.0, theta.sin()]))
    assert not math.isclose(q[0].item(), math.cos(0.35), abs_tol=1e-3)


def test_descriptor_has_unit_norm():
    u = unit_direction(torch.randn(3, 100, 3))
    assert relation_norm_error(first_order(u, 4)) < 1e-5


def test_lag_samples_follows_the_sampling_rate():
    assert [lag_samples(ms, 500) for ms in (10, 20, 40, 80)] == [5, 10, 20, 40]
    assert [lag_samples(ms, 100) for ms in (10, 20, 40, 80)] == [1, 2, 4, 8]
    with pytest.raises(ValueError):
        lag_samples(5, 100)


def test_first_and_second_order_alignment_and_padding():
    u = unit_direction(torch.randn(2, 40, 3))
    q = first_order(u, 3)
    assert q.shape == (2, 40, 4)
    # The relation t -> t+lag sits at t; the tail replicates the last valid value.
    torch.testing.assert_close(q[:, 7], relation_descriptor(u[:, 7], u[:, 10]))
    torch.testing.assert_close(q[:, -3:], q[:, -4:-3].expand(2, 3, 4))
    s = second_order(u, 3)
    assert s.shape == (2, 40, 4)
    expected = relation_descriptor(u[:, 9], u[:, 12]) - relation_descriptor(u[:, 6], u[:, 9])
    torch.testing.assert_close(s[:, 9], expected)


def test_second_order_is_the_difference_of_first_order():
    u = unit_direction(torch.randn(2, 60, 3))
    q = first_order(u, 5)
    torch.testing.assert_close(second_order(u, 5)[:, 5:-5], q[:, 5:-5] - q[:, :-10])


def test_zero_vectors_stay_finite():
    v = torch.zeros(4, 3, requires_grad=True)
    unit_direction(v).sum().backward()
    assert torch.isfinite(v.grad).all()


def test_kors_uses_the_eight_independent_leads():
    ecg = torch.zeros(1, 12, 20)
    ecg[:, 0] = 1.0
    torch.testing.assert_close(
        kors_transform(ecg, torch.tensor(KORS)),
        torch.tensor(KORS)[:, 0].view(1, 3, 1).expand(1, 3, 20).contiguous(),
    )
    # Lead III is dependent and must not enter the transform.
    dependent = torch.zeros(1, 12, 20)
    dependent[:, 2] = 5.0
    assert not kors_transform(dependent, torch.tensor(KORS)).any()


@pytest.mark.parametrize(
    "feature,scales,channels",
    [
        ("raw", [], 3),
        ("first", [20], 4),
        ("first", [10, 20, 40, 80], 16),
        ("first_second", [10, 20, 40, 80], 32),
    ],
)
def test_frontend_channel_counts(feature, scales, channels):
    frontend = VCGGeometry(feature, scales, 500)
    assert frontend.out_channels == channels
    assert frontend(torch.randn(2, 12, 1000)).shape == (2, channels, 1000)


def test_frontend_layout_is_component_major():
    """[r for every channel, then i, then j, then k] is what the Hamilton kernels expect."""
    frontend = VCGGeometry("first", [10, 20], 500)
    ecg = torch.randn(1, 12, 500)
    out = frontend(ecg)
    u = unit_direction(kors_transform(ecg, frontend.kors).transpose(1, 2))
    for channel, lag in enumerate(frontend.lags):
        expected = first_order(u, lag)[0]
        for component in range(4):
            torch.testing.assert_close(out[0, component * 2 + channel], expected[:, component])


def test_renormalize_is_opt_in():
    assert not VCGGeometry("first", [20], 500).renormalize
    assert VCGGeometry("first", [20], 500, renormalize=True).renormalize


def test_invalid_frontend_configuration():
    with pytest.raises(ValueError):
        VCGGeometry("raw", [20], 500)
    with pytest.raises(ValueError):
        VCGGeometry("first", [20, 20], 500)
    with pytest.raises(ValueError):
        VCGGeometry("nonsense", [20], 500)
