import pytest
import torch

from qdg.geometry import hamilton_product
from qdg.quaternion_nn import (
    QuaternionConv1d,
    QuaternionLinear,
    TCNEncoder,
    hamilton_kernel,
)


def _to_quaternion(x):
    """Component-major (4 * Q,) -> (Q, 4)."""
    return x.reshape(4, -1).transpose(0, 1)


def test_linear_reproduces_the_hamilton_product():
    """A one-in one-out QuaternionLinear must equal W (x) Q for the stored weight."""
    layer = QuaternionLinear(1, 1, bias=False)
    q = torch.randn(4)
    weight = layer.weight.reshape(4)
    torch.testing.assert_close(layer(q), hamilton_product(weight, q), atol=1e-6, rtol=1e-5)


def test_conv_reproduces_the_hamilton_product():
    layer = QuaternionConv1d(1, 1, 1, bias=False)
    q = torch.randn(1, 4, 7)
    weight = layer.weight.reshape(4)
    expected = hamilton_product(weight.expand(7, 4), q[0].transpose(0, 1)).transpose(0, 1)
    torch.testing.assert_close(layer(q)[0], expected, atol=1e-6, rtol=1e-5)


def test_hamilton_kernel_block_signs():
    weight = torch.arange(4.0).reshape(4, 1, 1, 1)
    r, i, j, k = 0.0, 1.0, 2.0, 3.0
    expected = torch.tensor([[r, -i, -j, -k], [i, r, -k, j], [j, k, r, -i], [k, -j, i, r]]).reshape(
        4, 4, 1
    )
    torch.testing.assert_close(hamilton_kernel(weight), expected)


def test_weight_sharing_keeps_a_quarter_of_the_dense_parameters():
    quaternion = QuaternionConv1d(8, 8, 5)
    real = torch.nn.Conv1d(32, 32, 5, bias=False)
    assert quaternion.weight.numel() * 4 == real.weight.numel()


def test_real_control_matches_the_quaternion_weight_count():
    """real_width = 2 * quaternions equalizes the convolution weight counts (方案 §7.1)."""
    quaternions = 16
    quaternion = QuaternionConv1d(quaternions, quaternions, 5)
    real = torch.nn.Conv1d(2 * quaternions, 2 * quaternions, 5, bias=False)
    assert quaternion.weight.numel() == real.weight.numel()


def test_components_are_actually_coupled():
    """Perturbing only the scalar part must move the vector outputs, and vice versa."""
    layer = QuaternionConv1d(2, 2, 3, bias=False)
    x = torch.zeros(1, 8, 5)
    x[0, 0, 2] = 1.0  # a single scalar-component impulse
    out = _to_quaternion(layer(x)[0, :, 2])
    assert out[:, 1:].abs().max() > 1e-6
    x = torch.zeros(1, 8, 5)
    x[0, 4, 2] = 1.0  # a single i-component impulse
    assert _to_quaternion(layer(x)[0, :, 2])[:, 0].abs().max() > 1e-6


@pytest.mark.parametrize("quaternion,operator", [(False, "conv"), (True, "conv"), (False, "mlp")])
def test_encoder_shapes_and_gradients(quaternion, operator):
    encoder = TCNEncoder(16, 64, quaternion=quaternion, operator=operator, dropout=0.0)
    out = encoder(torch.randn(2, 16, 1000))
    assert out.shape == (2, 64)
    out.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in encoder.parameters())


def test_receptive_field_and_mlp_operator():
    # Measured layer by layer, so the three avg_pool(2,2) stages are counted too; the
    # closed form this replaced omitted them and reported 605 instead of 640.
    assert TCNEncoder(4, 32, kernel=5, depth=4, stem_stride=5).receptive_field == 640
    mlp = TCNEncoder(4, 32, operator="mlp", stem_stride=5)
    # kernel 1 removes every *learned* temporal mixing inside the blocks, but the fixed
    # pooling between them still aggregates across time, so the field is 40 samples
    # (80 ms at 500 Hz), not one stem stride.
    assert all(conv.weight.shape[-1] == 1 for block in mlp.blocks for conv in block.convs)
    assert mlp.receptive_field == 40


def test_quaternion_encoder_rejects_misaligned_widths():
    with pytest.raises(ValueError):
        TCNEncoder(16, 30, quaternion=True)
    with pytest.raises(ValueError):
        TCNEncoder(4, 32, operator="nonsense")
