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
FEATURES = ("raw", "first", "first_second", "composite")
# Real-valued blocks of the representation diagnostic (v2 指导书 §2). Concatenated in
# this canonical order, so a block always occupies the same channel slice.
BLOCKS = (
    "radial",
    "direction",
    "linear",
    "angular",
    # Rotation-quaternion blocks (Temporal evolution 方案 §2). Unlike `angular`, these
    # are genuine half-angle rotation quaternions, so composition is meaningful.
    "rotation",
    "rotation_delta",
    "rotation_evolution",
)


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


def radial_magnitude(v):
    """(..., 3) -> (..., 1) radial magnitude r = ||V||_2. Carries the amplitude that
    `unit_direction` divides away; never normalized per record (v2 指导书 §6)."""
    return torch.linalg.vector_norm(v, dim=-1, keepdim=True)


def linear_velocity(v):
    """(..., T, 3) -> (..., T, 3) forward difference of the RAW vector, edge-replicated.

    v2 指导书 §2 defines l_t = (V_{t+1} - V_t) / dt with dt = 1 / Fs. This returns the
    bare difference V_{t+1} - V_t; the caller multiplies by the constant it needs. It
    must be built on raw XYZ, never on the normalized direction u.
    """
    return _pad_edges3(v[..., 1:, :] - v[..., :-1, :], 0, 1)


def _pad_edges3(x, left, right):
    """Edge replication for (..., T, 3), matching the convention `_pad_edges` uses."""
    parts = [x[..., :1, :].expand(*x.shape[:-2], left, 3)] if left else []
    parts.append(x)
    if right:
        parts.append(x[..., -1:, :].expand(*x.shape[:-2], right, 3))
    return torch.cat(parts, dim=-2)


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


def canonical_sign(q):
    """Flip q so its scalar part is non-negative. q and -q are the same rotation, so
    this removes a sign ambiguity that would otherwise look like a jump in time (§5)."""
    return torch.where(q[..., :1] < 0, -q, q)


def rotation_quaternion(u, v, degenerate_threshold=1e-3, eps=1e-12):
    """Minimal rotation taking unit u to unit v, as a HALF-angle quaternion (§2).

    This is the standard physical rotation quaternion [cos(theta/2), n sin(theta/2)] --
    a different object from `relation_descriptor`, which is the full-angle
    [cos theta, n sin theta] used by M1-M4 and never composable as a rotation.

    Built without trigonometry from the identity

        [1 + u.v, u x v] / || [1 + u.v, u x v] ||  ==  [cos(theta/2), n sin(theta/2)]

    since the norm is sqrt(2(1 + u.v)) = 2 cos(theta/2). The scalar part 1 + u.v is
    never negative, so the result already satisfies the w >= 0 sign convention.

    Antiparallel inputs (theta -> pi) collapse both parts to zero; there the rotation is
    pi about any axis perpendicular to u, so one is chosen deterministically from the
    coordinate axis least aligned with u. Parallel inputs need no special case: the
    scalar dominates and the result tends to the identity [1, 0, 0, 0].
    """
    u, v = torch.broadcast_tensors(u, v)
    dot = (u * v).sum(-1, keepdim=True)
    raw = torch.cat((1 + dot, torch.linalg.cross(u, v)), dim=-1)
    norm = torch.linalg.vector_norm(raw, dim=-1, keepdim=True)
    # Deterministic perpendicular axis for the antiparallel case.
    index = u.abs().argmin(-1, keepdim=True)
    basis = torch.zeros_like(u).scatter(-1, index, 1.0)
    perpendicular = torch.linalg.cross(u, basis)
    perpendicular = perpendicular / (
        torch.linalg.vector_norm(perpendicular, dim=-1, keepdim=True) + eps
    )
    fallback = torch.cat((torch.zeros_like(dot), perpendicular), dim=-1)
    return torch.where(norm < degenerate_threshold, fallback, raw / norm.clamp_min(eps))


def local_rotation(u, lag):
    """q_t = Rotation(u_t -> u_{t+lag}), placed at t, (..., T, 3) -> (..., T, 4)."""
    if lag >= u.shape[-2]:
        raise ValueError("Lag is not shorter than the signal")
    return _pad_edges(rotation_quaternion(u[..., :-lag, :], u[..., lag:, :]), 0, lag)


def rotation_difference(q, tau):
    """q_{t+tau} - q_t: a coordinate-wise difference in R^4.

    Deliberately NOT a rotation. It is the plain temporal difference control of §3, and
    must never be described as a relative rotation (§5).
    """
    if tau >= q.shape[-2]:
        raise ValueError("Tau is not shorter than the signal")
    return _pad_edges(q[..., tau:, :] - q[..., :-tau, :], 0, tau)


def rotation_evolution(q, tau):
    """e_t = q_t^-1 (x) q_{t+tau}: the relative rotation from one local rotation to the next.

    This one IS a rotation: it is the group element carrying the frame of q_t onto that
    of q_{t+tau}, expressed in q_t's own frame. That is the whole contrast with
    `rotation_difference` -- same two operands, composition instead of subtraction.
    """
    if tau >= q.shape[-2]:
        raise ValueError("Tau is not shorter than the signal")
    previous, following = q[..., :-tau, :], q[..., tau:, :]
    # q is unit by construction, so the inverse is the conjugate.
    evolution = hamilton_product(quaternion_conjugate(previous), following)
    return _pad_edges(canonical_sign(evolution), 0, tau)


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
        blocks=(),
        tau_ms=None,
    ):
        super().__init__()
        if feature not in FEATURES:
            raise ValueError(f"feature must be one of {FEATURES}")
        scales_ms = tuple(scales_ms)
        blocks = tuple(blocks)
        if feature == "raw":
            if scales_ms:
                raise ValueError("Raw VCG input takes no temporal scales")
        elif not scales_ms or len(set(scales_ms)) != len(scales_ms):
            raise ValueError("Geometry features need unique temporal scales")
        if feature == "composite":
            if not blocks or len(set(blocks)) != len(blocks):
                raise ValueError("A composite feature needs unique blocks")
            if any(block not in BLOCKS for block in blocks):
                raise ValueError(f"blocks must be drawn from {BLOCKS}")
            if len(scales_ms) != 1:
                raise ValueError("The composite diagnostic fixes a single 20 ms scale")
        elif blocks:
            raise ValueError("blocks only apply to a composite feature")
        # Canonical order, so a block always occupies the same channel slice.
        self.blocks = tuple(block for block in BLOCKS if block in blocks)
        self.feature, self.scales_ms, self.eps = feature, scales_ms, eps
        self.sampling_rate = sampling_rate
        # Explicit configuration: re-normalization is never applied silently (方案 §4.3).
        self.renormalize = renormalize
        self.lags = tuple(lag_samples(ms, sampling_rate) for ms in scales_ms)
        # tau is the evolution step of the rotation blocks. Fixed to the angular lag by
        # default; the plan forbids searching it alongside delta and the receptive field.
        self.tau_ms = tau_ms if tau_ms is not None else (scales_ms[0] if scales_ms else None)
        self.tau = lag_samples(self.tau_ms, sampling_rate) if self.tau_ms else None
        self.register_buffer("kors", torch.tensor(KORS, dtype=torch.float32))
        scale = torch.ones(3) if vcg_scale is None else torch.tensor(vcg_scale, dtype=torch.float32)
        self.register_buffer("vcg_scale", scale.view(1, 3, 1))
        # RMS of r over the training folds, exactly sqrt(sum of the per-axis RMS squares)
        # since vcg_std is itself an uncentred training-fold RMS. Derived from the cache
        # that already exists, so no re-prepare and no new normalization recipe (§6).
        # persistent=False: derived entirely from vcg_scale, so it must never enter
        # state_dict -- an earlier checkpoint has to keep loading with strict=True.
        self.register_buffer(
            "radial_scale", scale.square().sum().sqrt().view(1, 1, 1), persistent=False
        )

    BLOCK_CHANNELS = {
        "radial": 1,
        "direction": 3,
        "linear": 3,
        "angular": 4,
        "rotation": 4,
        "rotation_delta": 4,
        "rotation_evolution": 4,
    }

    @property
    def quaternion_channels(self):
        if self.feature in ("raw", "composite"):
            return 0
        return len(self.lags) * (2 if self.feature == "first_second" else 1)

    @property
    def block_slices(self):
        """block name -> (start, stop) channel indices, for the per-block unit tests."""
        slices, start = {}, 0
        for block in self.blocks:
            stop = start + self.BLOCK_CHANNELS[block]
            slices[block] = (start, stop)
            start = stop
        return slices

    @property
    def out_channels(self):
        if self.feature == "raw":
            return 3
        if self.feature == "composite":
            return sum(self.BLOCK_CHANNELS[block] for block in self.blocks)
        return 4 * self.quaternion_channels

    def forward(self, ecg):
        # Geometry is always float32: cosines near +-1 are the point of the representation.
        with torch.autocast(device_type=ecg.device.type, enabled=False):
            vcg = kors_transform(ecg.float(), self.kors)
            if self.feature == "raw":
                return vcg / self.vcg_scale
            u = unit_direction(vcg.transpose(1, 2), self.eps)
            if self.feature == "composite":
                return self._composite(vcg, u)
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

    def _composite(self, vcg, u):
        """Real concatenation of the diagnostic blocks, (B, C, T) in canonical order.

        Scaling is by training-fold constants only, and is a pure per-axis rescale, so
        no information is added or removed:
          radial  r / RMS_train(r)
          linear  (V_{t+1} - V_t) / vcg_std, i.e. the §2 velocity (V_{t+1}-V_t)/dt
                  divided by the constant Fs * vcg_std
          direction, angular  already bounded, left alone
        """
        parts = []
        for block in self.blocks:
            if block == "radial":
                parts.append(
                    radial_magnitude(vcg.transpose(1, 2)).transpose(1, 2) / self.radial_scale
                )
            elif block == "direction":
                parts.append(u.transpose(1, 2))
            elif block == "linear":
                velocity = linear_velocity(vcg.transpose(1, 2)).transpose(1, 2)
                parts.append(velocity / self.vcg_scale)
            elif block == "rotation":
                parts.append(local_rotation(u, self.lags[0]).transpose(1, 2))
            elif block == "rotation_delta":
                q = local_rotation(u, self.lags[0])
                parts.append(rotation_difference(q, self.tau).transpose(1, 2))
            elif block == "rotation_evolution":
                q = local_rotation(u, self.lags[0])
                parts.append(rotation_evolution(q, self.tau).transpose(1, 2))
            else:
                # Bit-identical to the M1 input: one quaternion channel is already
                # [dot, cross_x, cross_y, cross_z] in component-major order.
                parts.append(first_order(u, self.lags[0]).transpose(1, 2))
        return torch.cat(parts, dim=1)
