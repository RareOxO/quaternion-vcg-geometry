"""Temporal encoders for the Experiment 1 benchmark of the R/L/Q Long-Context plan.

Every encoder maps (B, C, T) -> (B, width). They share one downsampling stem so the
benchmark varies the temporal model and not the input rate, and they share the same
fusion and head downstream.

Quaternion pairing (plan section 2.1): R and L are always real. A quaternion-specific
encoder replaces the operator of the Q branch only, and its R/L branches use the
matched real encoder, so `lstm` vs `qlstm` differ in exactly one thing.

Width convention: a quaternion layer of Q channels holds a quarter of the weights of a
real layer of 4Q channels, so the real counterpart is 2Q wide. That is the same
convention the earlier rounds used, and it keeps the parameter counts comparable.

The three quaternion papers in `references/` all come from other domains -- 3-D sound
field SELD (QTCN), image denoising (QFormer), spatio-temporal graph forecasting
(QSTGNN). What is reused here is their operator structure, adapted to a 1-D
(B, 4Q, T) sequence; none is a reimplementation of the original pipeline.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .quaternion_nn import (
    QuaternionConv1d,
    QuaternionDropout,
    QuaternionLinear,
    QuaternionRMSNorm,
    ResidualBlock,
    receptive_field_samples,
)

GENERIC_ENCODERS = ("lstm", "gru", "lstm_attention", "tcn", "tcn_attention", "transformer")
QUATERNION_ENCODERS = ("qlstm", "qtcn", "qtransformer", "qgnn")
ENCODERS = (*GENERIC_ENCODERS, *QUATERNION_ENCODERS)
# A quaternion encoder's R/L branches use its matched real encoder, so the pair differs
# only in the Q-branch operator. QGNN has no listed real counterpart; the project's
# established real encoder is used so it is still a controlled comparison.
QUATERNION_PAIR = {
    "qlstm": "lstm",
    "qtcn": "tcn",
    "qtransformer": "transformer",
    "qgnn": "tcn",
}


class QuaternionLSTM(nn.Module):
    """Quaternion LSTM of Parcollet et al. (QRNN, ICLR 2019; QLSTM, arXiv:1811.02566).

        f_t = sigma(W_f (x) x_t + R_f (x) h_{t-1} + b_f)
        i_t = sigma(W_i (x) x_t + R_i (x) h_{t-1} + b_i)
        o_t = sigma(W_o (x) x_t + R_o (x) h_{t-1} + b_o)
        c_t = f_t * c_{t-1} + i_t * alpha(W_c (x) x_t + R_c (x) h_{t-1} + b_c)
        h_t = o_t * alpha(c_t)

    `(x)` is the Hamilton product -- every weight matrix is quaternion-valued, which is
    where the parameter saving comes from. `*` is the component-wise product, and sigma
    and alpha are SPLIT activations: ordinary sigmoid/tanh applied to each of the four
    components independently. That is the paper's formulation, not a componentwise
    rename of a real LSTM.

    This is a quaternion-valued network, not the quantum LSTM of Chen et al.; the two
    are unrelated despite both being abbreviated QLSTM.

    Gates are kept as four separate QuaternionLinear layers rather than one stacked
    projection: the component-major layout means a stacked output would have to be
    split inside each component block, which is easy to get silently wrong.
    """

    GATES = "figo"

    def __init__(self, in_quaternions, hidden_quaternions):
        super().__init__()
        self.in_quaternions, self.hidden_quaternions = in_quaternions, hidden_quaternions
        self.width = 4 * hidden_quaternions
        # All four gates in one Hamilton projection each. The input path does not depend
        # on h, so it is computed once for the whole sequence; only the recurrent path
        # has to run step by step.
        self.input_projection = QuaternionLinear(in_quaternions, 4 * hidden_quaternions)
        self.recurrent_projection = QuaternionLinear(
            hidden_quaternions, 4 * hidden_quaternions, bias=False
        )

    def split_gates(self, projected):
        """(..., 16 * hidden) -> dict of four (..., 4 * hidden) gate pre-activations.

        The quaternion layout is component-major, so a projection of 4 * hidden
        quaternions is laid out as [r for all, then i, then j, then k]. Splitting the
        gates therefore has to happen *inside* each component block, not by slicing the
        flat vector into four contiguous pieces.
        """
        hidden = self.hidden_quaternions
        blocks = projected.reshape(*projected.shape[:-1], 4, 4, hidden)
        return {
            name: blocks[..., index, :].reshape(*projected.shape[:-1], 4 * hidden)
            for index, name in enumerate(self.GATES)
        }

    def forward(self, x):
        """(B, T, 4 * in_quaternions) -> (B, T, 4 * hidden_quaternions), as nn.LSTM."""
        batch, steps, _ = x.shape
        projected = self.split_gates(self.input_projection(x))
        hidden = x.new_zeros(batch, self.width)
        cell = x.new_zeros(batch, self.width)
        outputs = []
        for step in range(steps):
            recurrent = self.split_gates(self.recurrent_projection(hidden))
            gate = {name: projected[name][:, step] + recurrent[name] for name in self.GATES}
            cell = gate["f"].sigmoid() * cell + gate["i"].sigmoid() * gate["g"].tanh()
            hidden = gate["o"].sigmoid() * cell.tanh()
            outputs.append(hidden)
        return torch.stack(outputs, dim=1)


class AttentionPool(nn.Module):
    """Additive attention over time: one learned query scores each step."""

    def __init__(self, width):
        super().__init__()
        self.project = nn.Linear(width, width)
        self.score = nn.Linear(width, 1, bias=False)

    def forward(self, x):
        """(B, T, width) -> (B, width)."""
        weights = self.score(self.project(x).tanh()).softmax(dim=1)
        return (x * weights).sum(dim=1)


class Downsample(nn.Module):
    """Shared stem: (B, C, T) -> (B, width, T / 40), so every encoder sees one rate.

    Stride-5 convolution then three poolings, i.e. 5000 samples become 125 steps of
    40 samples (80 ms) each. Recurrent and attention encoders are only tractable at
    this rate, and holding it fixed is what makes the benchmark about the temporal
    model rather than about the sampling.
    """

    stride, poolings = 5, 3

    def __init__(self, in_channels, width, quaternion=False, stride=None, poolings=None):
        super().__init__()
        self.stride = self.stride if stride is None else stride
        self.poolings = self.poolings if poolings is None else poolings
        self.quaternion, self.width = quaternion, width
        if quaternion:
            self.stem = QuaternionConv1d(
                in_channels // 4, width // 4, self.stride, stride=self.stride
            )
            self.norm = QuaternionRMSNorm(width // 4)
        else:
            self.stem = nn.Conv1d(in_channels, width, self.stride, stride=self.stride)
            self.norm = nn.GroupNorm(1, width)
        self.layer_spec = ((self.stride, self.stride, 1), *(((2, 2, 1),) * self.poolings))
        self.factor = self.stride * 2**self.poolings

    def forward(self, x):
        x = self.norm(self.stem(x))
        for _ in range(self.poolings):
            x = F.avg_pool1d(x, 2, 2)
        return x


class _Encoder(nn.Module):
    """Common shape contract: (B, C, T) -> (B, width), with a measured field."""

    def __init__(self, in_channels, width, quaternion=False, stem=None):
        super().__init__()
        self.down = Downsample(in_channels, width, quaternion, **(stem or {}))
        self.quaternion, self.width = quaternion, width

    def receptive_field_of(self, body_layers=()):
        return receptive_field_samples([*self.down.layer_spec, *body_layers])


class RecurrentEncoder(_Encoder):
    """LSTM / GRU / QLSTM, optionally with attention pooling instead of a mean."""

    def __init__(
        self,
        in_channels,
        width,
        kind="lstm",
        attention=False,
        dropout=0.1,
        stem=None,
        context_steps=None,
    ):
        super().__init__(in_channels, width, quaternion=kind == "qlstm", stem=stem)
        if kind == "qlstm":
            self.rnn = QuaternionLSTM(width // 4, width // 4)
        else:
            cell = nn.LSTM if kind == "lstm" else nn.GRU
            self.rnn = cell(width, width, batch_first=True)
        self.kind = kind
        self.pool = AttentionPool(width) if attention else None
        self.dropout = nn.Dropout(dropout)
        # Accessible temporal context, in downsampled steps. None leaves the recurrence
        # unrestricted, which is what every experiment other than Experiment 4 uses.
        self.context_steps = context_steps
        # A recurrent pass sees every earlier step, so the field is the whole input.
        self.receptive_field = None

    def _recur(self, x):
        return self.rnn(x) if self.kind == "qlstm" else self.rnn(x)[0]

    def _chunked(self, x):
        """Reset the recurrent state every `context_steps`, so nothing is integrated
        across a chunk boundary.

        Implemented by folding the chunk axis into the batch: each row then starts from
        the zero state, which IS the reset, and the whole sweep runs at one matmul per
        step exactly as the unrestricted model does. The tail is zero-padded to a whole
        number of chunks and cropped afterwards; because the recurrence is causal,
        padding placed AFTER a real step cannot reach it, so the padding is inert rather
        than merely small.
        """
        batch, steps, channels = x.shape
        window = self.context_steps
        padding = (-steps) % window
        if padding:
            x = F.pad(x, (0, 0, 0, padding))
        folded = x.reshape(batch * (x.shape[1] // window), window, channels)
        out = self._recur(folded)
        return out.reshape(batch, -1, out.shape[-1])[:, :steps]

    def forward(self, x):
        x = self.down(x).transpose(1, 2)
        restricted = self.context_steps and self.context_steps < x.shape[1]
        x = self._chunked(x) if restricted else self._recur(x)
        x = self.dropout(x)
        return self.pool(x) if self.pool is not None else x.mean(dim=1)


class ConvEncoder(_Encoder):
    """TCN / QTCN, optionally with attention pooling instead of a mean."""

    def __init__(
        self, in_channels, width, kernel=5, depth=4, dropout=0.1, quaternion=False, attention=False
    ):
        super().__init__(in_channels, width, quaternion)
        self.blocks = nn.ModuleList(
            ResidualBlock(width, kernel, dropout, quaternion) for _ in range(depth)
        )
        self.pool = AttentionPool(width) if attention else None
        self.receptive_field = self.receptive_field_of([(kernel, 1, 1), (kernel, 1, 1)] * depth)

    def forward(self, x):
        x = self.down(x)
        for block in self.blocks:
            x = block(x)
        x = x.transpose(1, 2)
        return self.pool(x) if self.pool is not None else x.mean(dim=1)


class TransformerEncoder(_Encoder):
    """Real Transformer: learned positions, pre-norm blocks, mean over time."""

    def __init__(self, in_channels, width, depth=4, heads=4, dropout=0.1, steps=125):
        super().__init__(in_channels, width)
        self.position = nn.Parameter(torch.zeros(1, steps, width))
        nn.init.trunc_normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            width,
            heads,
            dim_feedforward=2 * width,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.body = nn.TransformerEncoder(layer, depth)
        self.receptive_field = None  # global attention sees the whole sequence

    def forward(self, x):
        x = self.down(x).transpose(1, 2)
        return self.body(x + self.position[:, : x.shape[1]]).mean(dim=1)


class QuaternionTransformerEncoder(_Encoder):
    """QFormer-style attention: quaternion projections, real attention weights.

    Adapted from the image-denoising QFormer to a 1-D sequence. Q, K, V and the output
    projection are Hamilton-coupled QuaternionLinear layers; the attention weights
    themselves are ordinary scaled dot products over the 4Q-dimensional real vectors,
    since a quaternion-valued softmax has no standard definition. This is an adaptation,
    not a reimplementation of the paper's pipeline.
    """

    def __init__(self, in_channels, width, depth=4, heads=4, dropout=0.1, steps=125):
        super().__init__(in_channels, width, quaternion=True)
        self.position = nn.Parameter(torch.zeros(1, steps, width))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.heads, self.depth = heads, depth
        quaternions = width // 4
        self.norms = nn.ModuleList(QuaternionRMSNorm(quaternions) for _ in range(2 * depth))
        self.qkv = nn.ModuleList(
            QuaternionLinear(quaternions, 3 * quaternions) for _ in range(depth)
        )
        self.projections = nn.ModuleList(
            QuaternionLinear(quaternions, quaternions) for _ in range(depth)
        )
        self.feedforward = nn.ModuleList(
            nn.ModuleList(
                [
                    QuaternionLinear(quaternions, 2 * quaternions),
                    QuaternionLinear(2 * quaternions, quaternions),
                ]
            )
            for _ in range(depth)
        )
        self.dropout = nn.Dropout(dropout)
        self.receptive_field = None

    def _norm(self, index, x):
        return self.norms[index](x.transpose(1, 2)).transpose(1, 2)

    def forward(self, x):
        x = self.down(x).transpose(1, 2)
        x = x + self.position[:, : x.shape[1]]
        batch, steps, width = x.shape
        for layer in range(self.depth):
            y = self._norm(2 * layer, x)
            qkv = self.qkv[layer](y).reshape(batch, steps, 3, self.heads, -1)
            query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
            attended = F.scaled_dot_product_attention(query, key, value)
            attended = attended.transpose(1, 2).reshape(batch, steps, width)
            x = x + self.dropout(self.projections[layer](attended))
            y = self._norm(2 * layer + 1, x)
            first, second = self.feedforward[layer]
            x = x + self.dropout(second(F.gelu(first(y))))
        return x.mean(dim=1)


class QuaternionGraphEncoder(_Encoder):
    """QSTGNN-style message passing over a temporal graph, minimal mapping.

    Plan section 2.1 asks for the minimal sequence-to-graph mapping and no extra ECG or
    VCG information: node t is q_t, and the adjacency joins each node to its `span`
    neighbours on either side. Each layer is a Hamilton-coupled self transform plus a
    Hamilton-coupled transform of the mean neighbour message, which is the quaternion
    analogue of a GCN layer on a path graph.
    """

    def __init__(self, in_channels, width, depth=4, span=4, dropout=0.1):
        super().__init__(in_channels, width, quaternion=True)
        quaternions = width // 4
        self.span, self.depth = span, depth
        self.self_transform = nn.ModuleList(
            QuaternionLinear(quaternions, quaternions) for _ in range(depth)
        )
        self.neighbour_transform = nn.ModuleList(
            QuaternionLinear(quaternions, quaternions, bias=False) for _ in range(depth)
        )
        self.norms = nn.ModuleList(QuaternionRMSNorm(quaternions) for _ in range(depth))
        self.dropout = QuaternionDropout(dropout)
        # Each layer reaches `span` steps on both sides: a (2 span + 1) kernel. span=4
        # makes the field equal to the paired TCN, so the pair differs in operator only.
        self.receptive_field = self.receptive_field_of([(2 * span + 1, 1, 1)] * depth)

    def forward(self, x):
        x = self.down(x)
        window = 2 * self.span + 1
        for layer in range(self.depth):
            y = self.norms[layer](x)
            # Mean over the temporal neighbourhood, self included, as the message.
            message = F.avg_pool1d(y, window, 1, padding=self.span, count_include_pad=False)
            updated = self.self_transform[layer](y.transpose(1, 2)) + self.neighbour_transform[
                layer
            ](message.transpose(1, 2))
            x = x + self.dropout(F.gelu(updated).transpose(1, 2))
        return x.transpose(1, 2).mean(dim=1)


def build_encoder(name, in_channels, width, **shared):
    """Construct one benchmark encoder. Width is the real channel count it outputs."""
    if name not in ENCODERS:
        raise ValueError(f"encoder must be one of {ENCODERS}")
    dropout = shared.get("dropout", 0.1)
    recurrent = {"stem": shared.get("stem"), "context_steps": shared.get("context_steps")}
    if name in ("lstm", "gru", "qlstm"):
        return RecurrentEncoder(in_channels, width, kind=name, dropout=dropout, **recurrent)
    if name == "lstm_attention":
        return RecurrentEncoder(
            in_channels, width, kind="lstm", attention=True, dropout=dropout, **recurrent
        )
    if name in ("tcn", "qtcn", "tcn_attention"):
        return ConvEncoder(
            in_channels,
            width,
            kernel=shared.get("kernel", 5),
            depth=shared.get("depth", 4),
            dropout=dropout,
            quaternion=name == "qtcn",
            attention=name == "tcn_attention",
        )
    if name == "transformer":
        return TransformerEncoder(in_channels, width, depth=shared.get("depth", 4), dropout=dropout)
    if name == "qtransformer":
        return QuaternionTransformerEncoder(
            in_channels, width, depth=shared.get("depth", 4), dropout=dropout
        )
    return QuaternionGraphEncoder(in_channels, width, depth=shared.get("depth", 4), dropout=dropout)


def encoder_widths(name, quaternion_width):
    """(Q-branch width, R/L-branch width) for an encoder name.

    A quaternion layer of Q channels holds a quarter of the weights of a real layer of
    4Q channels, so the real counterpart is half as wide. Generic encoders use the real
    width everywhere; quaternion encoders keep 4Q on the Q branch only.
    """
    if name in QUATERNION_ENCODERS:
        return quaternion_width, quaternion_width // 2
    return quaternion_width // 2, quaternion_width // 2


def branch_encoder(name, branch):
    """Which encoder a branch actually uses (plan section 2.1: R and L stay real)."""
    if branch == "angular":
        return name
    return QUATERNION_PAIR.get(name, name)


def positional_steps(signal_length):
    return signal_length // (Downsample.stride * 2**Downsample.poolings)
