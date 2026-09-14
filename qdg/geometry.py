"""VCG direction dynamics expressed as quaternion relation descriptors.

Two quaternions are deliberately kept apart (方案 §4.1); never silently swap them.

    raw pure-quaternion Hamilton product   [0, u] (x) [0, v] = [-u.v,  u x v]
    relation descriptor used in this study q(u, v)          = [+u.v,  u x v]

The descriptor is [cos(theta), n sin(theta)], a full-angle object. A rigid-body
rotation quaternion is [cos(theta/2), n sin(theta/2)], so the descriptor is NOT a
standard physical rotation quaternion (方案 §4.2). Call it a relation descriptor.

Tensor conventions inside this module: directions are (..., T, 3) and quaternions
are (..., T, 4). The VCGGeometry frontend transposes to channel-major (B, C, T)
for the convolutional encoders.
"""

import torch
from torch import nn

LEADS = ("I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6")
INDEPENDENT_LEADS = (0, 1, 6, 7, 8, 9, 10, 11)
KORS = (
    (0.38, -0.07, -0.13, 0.05, -0.01, 0.14, 0.06, 0.54),
    (-0.07, 0.93, 0.06, -0.02, -0.05, 0.06, -0.17, 0.13),
    (0.11, -0.23, -0.43, -0.06, -0.14, -0.20, -0.11, 0.31),
)
FEATURES = ("raw", "first", "first_second")


def kors_transform(ecg, matrix):
    """(B, 12, T) physical-mV ECG -> (B, 3, T) VCG via the eight independent leads."""
    if ecg.shape[-2] != 12:
        raise ValueError("Kors transform expects all twelve leads")
    return torch.einsum("ci,bit->bct", matrix, ecg[..., INDEPENDENT_LEADS, :])


def unit_direction(v, eps=1e-8):
    """(..., 3) -> unit direction u = v / (||v|| + eps); eps keeps zeros differentiable."""
    return v / (torch.linalg.vector_norm(v, dim=-1, keepdim=True) + eps)


def hamilton_product(q1, q2):
    """Full Hamilton product of (..., 4) quaternions [a, b, c, d] (x) [e, f, g, h]."""
    a, b, c, d = q1.unbind(-1)
    e, f, g, h = q2.unbind(-1)
    return torch.stack(
        (
            a * e - b * f - c * g - d * h,
            a * f + b * e + c * h - d * g,
            a * g - b * h + c * e + d * f,
            a * h + b * g - c * f + d * e,
        ),
        dim=-1,
    )


def quaternion_conjugate(q):
    return q * q.new_tensor((1.0, -1.0, -1.0, -1.0))


def quaternion_inverse(q, eps=1e-12):
    return quaternion_conjugate(q) / q.square().sum(-1, keepdim=True).clamp_min(eps)


def pure_quaternion(v):
    """(..., 3) -> pure quaternion [0, v]."""
    return torch.cat((torch.zeros_like(v[..., :1]), v), dim=-1)


def pure_quaternion_product(u, v):
    """Raw algebraic result [0, u] (x) [0, v] = [-u.v, u x v]. Sign check only."""
    return hamilton_product(pure_quaternion(u), pure_quaternion(v))


def relation_descriptor(u, v):
    """The study's descriptor q = [+u.v, u x v] (方案 §4.1), on broadcastable (..., 3)."""
    u, v = torch.broadcast_tensors(u, v)
    return torch.cat(((u * v).sum(-1, keepdim=True), torch.linalg.cross(u, v)), dim=-1)


def lag_samples(lag_ms, sampling_rate):
    """Convert a millisecond scale to a sample offset at the actual sampling rate."""
    lag = int(round(lag_ms * sampling_rate / 1000))
    if lag < 1:
        raise ValueError(f"{lag_ms} ms is below one sample at {sampling_rate} Hz")
    return lag


def _pad_edges(q, left, right):
    """Replicate the first/last valid relation so every scale keeps the full length T."""
    parts = [q[..., :1, :].expand(*q.shape[:-2], left, 4)] if left else []
    parts.append(q)
    if right:
        parts.append(q[..., -1:, :].expand(*q.shape[:-2], right, 4))
    return torch.cat(parts, dim=-2)


def first_order(u, lag):
    """q_t^lag = [u_t . u_{t+lag}, u_t x u_{t+lag}], placed at t, (..., T, 3) -> (..., T, 4)."""
    if lag >= u.shape[-2]:
        raise ValueError("Lag is not shorter than the signal")
    return _pad_edges(relation_descriptor(u[..., :-lag, :], u[..., lag:, :]), 0, lag)


def second_order(u, lag):
    """s_t^lag = q(u_t, u_{t+lag}) - q(u_{t-lag}, u_t), placed at t (方案 §5.2)."""
    if 2 * lag >= u.shape[-2]:
        raise ValueError("Twice the lag is not shorter than the signal")
    previous = relation_descriptor(u[..., : -2 * lag, :], u[..., lag:-lag, :])
    following = relation_descriptor(u[..., lag:-lag, :], u[..., 2 * lag :, :])
    return _pad_edges(following - previous, lag, lag)


def relation_norm_error(q):
    """max | ||q|| - 1 |. First-order descriptors satisfy dot^2 + ||cross||^2 = 1 (方案 §4.3)."""
    return (torch.linalg.vector_norm(q.float(), dim=-1) - 1).abs().max()


class VCGGeometry(nn.Module):
    """ECG -> VCG -> unit direction -> the feature block a model actually consumes.

    Output is channel-major (B, C, T). For quaternion features the layout is
    component-major, [r for every channel, then i, then j, then k], which is the
    layout the Hamilton block kernels in quaternion_nn expect. M1 (real) and M2
    (quaternion) therefore receive bit-identical numbers (方案 §7.1).
    """

    def __init__(
        self,
        feature,
        scales_ms,
        sampling_rate,
        vcg_scale=None,
        eps=1e-8,
        renormalize=False,
    ):
        super().__init__()
        if feature not in FEATURES:
            raise ValueError(f"feature must be one of {FEATURES}")
        scales_ms = tuple(scales_ms)
        if feature == "raw":
            if scales_ms:
                raise ValueError("Raw VCG input takes no temporal scales")
        elif not scales_ms or len(set(scales_ms)) != len(scales_ms):
            raise ValueError("Geometry features need unique temporal scales")
        self.feature, self.scales_ms, self.eps = feature, scales_ms, eps
        self.sampling_rate = sampling_rate
        # Explicit configuration: re-normalization is never applied silently (方案 §4.3).
        self.renormalize = renormalize
        self.lags = tuple(lag_samples(ms, sampling_rate) for ms in scales_ms)
        self.register_buffer("kors", torch.tensor(KORS, dtype=torch.float32))
        scale = torch.ones(3) if vcg_scale is None else torch.tensor(vcg_scale, dtype=torch.float32)
        self.register_buffer("vcg_scale", scale.view(1, 3, 1))

    @property
    def quaternion_channels(self):
        if self.feature == "raw":
            return 0
        return len(self.lags) * (2 if self.feature == "first_second" else 1)

    @property
    def out_channels(self):
        return 3 if self.feature == "raw" else 4 * self.quaternion_channels

    def forward(self, ecg):
        # Geometry is always float32: cosines near +-1 are the point of the representation.
        with torch.autocast(device_type=ecg.device.type, enabled=False):
            vcg = kors_transform(ecg.float(), self.kors)
            if self.feature == "raw":
                return vcg / self.vcg_scale
            u = unit_direction(vcg.transpose(1, 2), self.eps)
            channels = [first_order(u, lag) for lag in self.lags]
            if self.renormalize:
                channels = [
                    q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(self.eps)
                    for q in channels
                ]
            if self.feature == "first_second":
                # Second-order differences are not unit norm and are never renormalized.
                channels += [second_order(u, lag) for lag in self.lags]
            # (B, Q, T, 4) -> component-major (B, 4 * Q, T)
            stacked = torch.stack(channels, dim=1)
            return stacked.permute(0, 3, 1, 2).flatten(1, 2)
