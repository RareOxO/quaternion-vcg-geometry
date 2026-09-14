"""Hamilton weight-sharing layers and the matched real controls.

Channel layout is component-major throughout: a tensor of 4 * Q real channels is
read as [r_0..r_{Q-1}, i_0..i_{Q-1}, j_0..j_{Q-1}, k_0..k_{Q-1}].

A quaternion layer stores four real kernels W_r, W_i, W_j, W_k and expands them
into the block matrix implied by 方案 §7:

    Y_r = W_r Q_r - W_i Q_i - W_j Q_j - W_k Q_k
    Y_i = W_r Q_i + W_i Q_r + W_j Q_k - W_k Q_j
    Y_j = W_r Q_j - W_i Q_k + W_j Q_r + W_k Q_i
    Y_k = W_r Q_k + W_i Q_j - W_j Q_i + W_k Q_r

That is a constrained interaction rule, not an SO(3)-equivariant layer.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def hamilton_kernel(weight):
    """(4, out, in, ...) real kernels -> (4 * out, 4 * in, ...) Hamilton block kernel."""
    r, i, j, k = weight.unbind(0)
    return torch.cat(
        (
            torch.cat((r, -i, -j, -k), 1),
            torch.cat((i, r, -k, j), 1),
            torch.cat((j, k, r, -i), 1),
            torch.cat((k, -j, i, r), 1),
        ),
        0,
    )


class QuaternionConv1d(nn.Module):
    """Hamilton-constrained Conv1d over (B, 4 * in_quaternions, T)."""

    def __init__(
        self, in_quaternions, out_quaternions, kernel_size, dilation=1, stride=1, bias=True
    ):
        super().__init__()
        self.in_quaternions, self.out_quaternions = in_quaternions, out_quaternions
        self.stride, self.dilation = stride, dilation
        self.padding = dilation * (kernel_size - 1) // 2 if stride == 1 else 0
        self.weight = nn.Parameter(torch.empty(4, out_quaternions, in_quaternions, kernel_size))
        bound = 1 / math.sqrt(4 * in_quaternions * kernel_size)
        nn.init.uniform_(self.weight, -bound, bound)
        # One real bias per output channel, exactly as nn.Conv1d, so the real control
        # of the same width has an identical parameter count (方案 §7.1).
        self.bias = nn.Parameter(torch.zeros(4 * out_quaternions)) if bias else None

    def forward(self, x):
        return F.conv1d(
            x,
            hamilton_kernel(self.weight),
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )


class QuaternionLinear(nn.Module):
    """Hamilton-constrained Linear over a trailing dimension of 4 * in_quaternions."""

    def __init__(self, in_quaternions, out_quaternions, bias=True):
        super().__init__()
        self.in_quaternions, self.out_quaternions = in_quaternions, out_quaternions
        self.weight = nn.Parameter(torch.empty(4, out_quaternions, in_quaternions))
        bound = 1 / math.sqrt(4 * in_quaternions)
        nn.init.uniform_(self.weight, -bound, bound)
        self.bias = nn.Parameter(torch.zeros(4 * out_quaternions)) if bias else None

    def forward(self, x):
        return F.linear(x, hamilton_kernel(self.weight), self.bias)


class QuaternionRMSNorm(nn.Module):
    """RMS over all four components at each time step; one gain per quaternion channel."""

    def __init__(self, quaternions):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(1, 1, quaternions, 1))

    def forward(self, x):
        z = x.float().reshape(x.shape[0], 4, -1, x.shape[-1])
        z = z * torch.rsqrt(z.square().mean((1, 2), keepdim=True) + 1e-4)
        return (z * self.gain).flatten(1, 2).to(x.dtype)


class QuaternionDropout(nn.Module):
    """Drop a whole quaternion, never an individual component."""

    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        z = x.reshape(x.shape[0], 4, -1, x.shape[-1])
        mask = F.dropout(torch.ones_like(z[:, :1]), self.p, True)
        return (z * mask).flatten(1, 2)


class ResidualBlock(nn.Module):
    """Pre-norm residual block; `quaternion` swaps the interaction rule and nothing else."""

    def __init__(self, width, kernel, dropout, quaternion):
        super().__init__()
        if quaternion:
            self.norms = nn.ModuleList(QuaternionRMSNorm(width // 4) for _ in range(2))
            self.convs = nn.ModuleList(
                QuaternionConv1d(width // 4, width // 4, kernel) for _ in range(2)
            )
            self.dropout = QuaternionDropout(dropout)
        else:
            self.norms = nn.ModuleList(nn.GroupNorm(1, width) for _ in range(2))
            self.convs = nn.ModuleList(
                nn.Conv1d(width, width, kernel, padding=(kernel - 1) // 2) for _ in range(2)
            )
            self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        y = x
        for norm, conv in zip(self.norms, self.convs):
            y = self.dropout(conv(F.gelu(norm(y))))
        return x + y


class TCNEncoder(nn.Module):
    """Strided stem, residual blocks halving time between them, mean over time.

    `operator="mlp"` sets every kernel to 1, which removes temporal mixing inside the
    blocks and leaves a per-time-step MLP -- the Real MLP control of Table 3.
    """

    def __init__(
        self,
        in_channels,
        width,
        kernel=5,
        depth=4,
        stem_stride=5,
        dropout=0.1,
        quaternion=False,
        operator="conv",
    ):
        super().__init__()
        if operator not in ("conv", "mlp"):
            raise ValueError("operator must be conv or mlp")
        if quaternion and (width % 4 or in_channels % 4):
            raise ValueError("Quaternion encoders need channel counts divisible by four")
        kernel = 1 if operator == "mlp" else kernel
        self.quaternion, self.width = quaternion, width
        if quaternion:
            self.stem = QuaternionConv1d(
                in_channels // 4, width // 4, stem_stride, stride=stem_stride
            )
            self.norm = QuaternionRMSNorm(width // 4)
        else:
            self.stem = nn.Conv1d(in_channels, width, stem_stride, stride=stem_stride)
            self.norm = nn.GroupNorm(1, width)
        self.blocks = nn.ModuleList(
            ResidualBlock(width, kernel, dropout, quaternion) for _ in range(depth)
        )
        # Block s sees units of stem_stride * 2**s samples after the s poolings before it.
        self.receptive_field = stem_stride * (
            1 + 2 * (kernel - 1) * sum(2**s for s in range(depth))
        )

    def forward(self, x):
        x = self.stem(x)
        for index, block in enumerate(self.blocks):
            if index:
                x = F.avg_pool1d(x, 2, 2)
            x = block(x)
        return self.norm(x).mean(-1)
