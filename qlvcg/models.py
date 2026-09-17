"""B0 and V0, the two baselines every Quaternion-LVCG variant is compared against.

B0  Traditional VCG. ECG -> fixed lift -> VCG [B, 3, T] -> a small real-valued 1D CNN
    -> global average pooling -> linear multi-label head. No beat bottleneck, no
    temporal module, nothing Quaternion.
V0  Supervised original LVCG. The author's ``LVCG`` class, unmodified, trained end to
    end from scratch with a linear head on its 640-d ``ecg_emb``. No Quaternion.

B0's lift is V0's lift -- the same Table 7 lead geometry and the same Tikhonov
pseudo-inverse with the same eps -- so B0 and V0 receive an identical VCG and the gap
between them is the latent beat architecture, not a different ECG-to-VCG transform.

V0 is run under two objectives (master prompt section E):

    V0-A  L = L_cls
    V0-B  L = lambda_cls L_cls + lambda_ecg L_ECG + lambda_beat L_beat
              + lambda_base L_base + lambda_temporal L_temp

The auxiliary terms are the author's own functions and weights (``scripts/train.py``
and ``configs/train/lvcg_v5_gru.yaml``); ``lambda_cls`` is not in that config and is 1.
They need the author's masked training pass, which lifts only three random visible
leads, whereas classification uses all twelve. Classifying from the masked pass would
train the head on a view the test set never shows, so V0-B runs both passes over shared
weights: the classifier input is then identical in V0-A and V0-B, and the comparison
isolates the effect of the auxiliary objective.

One property of the author's reconstruction path carries into L_ECG unchanged: its
BeatStitcher cross-fades with windows that reach exactly zero at the first and last
sample of every inner beat and places beats without overlap, so the reconstruction is
zero at roughly two samples per beat boundary (about 2% of a record).
"""

import torch
from torch import nn

from lvcg.data import get_lead_directions
from lvcg.models.heads import ClassificationHead
from lvcg.models.lvcg import LVCG, base_beat_loss, beat_level_loss, temporal_loss
from lvcg.models.utils.loss import masked_reconstruction_loss, random_lead_mask
from lvcg.models.vcg import VCGPseudoInverse

from qdg.data import CLASSES

from .quaternion_utils import (
    _safe_norm,
    enforce_sign_continuity,
    quaternion_angle,
    valid_rotation_mask,
    vectors_to_quaternion,
)

# The eps LVCG hard-codes for its lift; B0 must use the same one.
LIFT_EPS = 0.1

OBJECTIVES = ("classification", "classification+auxiliary")
# V1-V8 use whichever objective won V0A vs V0B. "protocol" defers to the config, which
# refuses to train until that choice has been recorded.
FROM_PROTOCOL = "protocol"

EXPERIMENTS = {
    "B0": {
        "model": "traditional_vcg",
        "objective": "classification",
        "variant": "traditional_vcg",
        "report": "reports/B0_TRADITIONAL_VCG_report.md",
    },
    "V0A": {
        "model": "lvcg",
        "objective": "classification",
        "variant": "lvcg_cls",
        "report": "reports/V0_SUPERVISED_LVCG_report.md",
    },
    "V0B": {
        "model": "lvcg",
        "objective": "classification+auxiliary",
        "variant": "lvcg_cls_aux",
        "report": "reports/V0_SUPERVISED_LVCG_report.md",
    },
}
_V1 = {"model": "qdf_lvcg", "objective": FROM_PROTOCOL, "report": "reports/V1_QDF_report.md"}
EXPERIMENTS.update(
    {
        # The full feature set, then the section G ablation, then the section O control.
        "V1": {**_V1, "variant": "qdf_lvcg", "features": ("q", "theta", "omega")},
        "V1q": {**_V1, "variant": "qdf_lvcg_q", "features": ("q",)},
        "V1theta": {**_V1, "variant": "qdf_lvcg_theta", "features": ("theta",)},
        "V1omega": {**_V1, "variant": "qdf_lvcg_omega", "features": ("omega",)},
        "V1ctrl": {
            **_V1,
            "variant": "qdf_lvcg_real_control",
            "features": ("position", "next_position", "delta"),
        },
    }
)
QUATERNION_FEATURES = ("q", "theta", "omega", "magnitude", "linear_velocity")
CONTROL_FEATURES = ("position", "next_position", "delta")
FEATURE_CHANNELS = {
    "q": 4,
    "theta": 1,
    "omega": 1,
    "magnitude": 1,
    "linear_velocity": 1,
    "position": 3,
    "next_position": 3,
    "delta": 3,
}
# Modules the classification path of LVCG never touches. They are counted separately
# so a parameter count does not credit V0-A with decoders it does not use.
LVCG_RECONSTRUCTION_ONLY = ("beat_decoder", "ecg_decoder", "struct_proj", "dynamic_proj")


def standardize(ecg, eps=1e-8):
    """Per-record, per-lead z-score, the paper's normalisation (Appendix B.1).

    Population standard deviation, matching the release's probing loader, which uses
    NumPy's default.
    """
    centred = ecg - ecg.mean(dim=-1, keepdim=True)
    return centred / (centred.square().mean(dim=-1, keepdim=True).sqrt() + eps)


class TraditionalVCG(nn.Module):
    """B0: fixed ECG-to-VCG lift and a small convolutional encoder."""

    def __init__(
        self, num_classes, lead_order="ptbxl", channels=(32, 64, 128, 128), kernel=7, dropout=0.1
    ):
        super().__init__()
        if kernel % 2 == 0:
            raise ValueError("kernel must be odd")
        self.lift = VCGPseudoInverse(eps=LIFT_EPS)
        self.register_buffer("directions", get_lead_directions(lead_order, as_tensor=True))
        layers, width = [], 3
        for out in channels:
            layers += [
                nn.Conv1d(width, out, kernel, stride=2, padding=kernel // 2),
                nn.BatchNorm1d(out),
                nn.GELU(),
            ]
            width = out
        self.encoder = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(width, num_classes)
        self.embedding_dim = width

    def vcg(self, ecg):
        return self.lift(ecg, self.directions.expand(ecg.shape[0], -1, -1))

    def embed(self, ecg):
        return self.encoder(self.vcg(ecg)).mean(dim=-1)

    def forward(self, ecg):
        return self.head(self.dropout(self.embed(ecg)))

    def parameter_counts(self):
        total = sum(p.numel() for p in self.parameters())
        return {"total": total, "classification_path": total}


class SupervisedLVCG(nn.Module):
    """V0: the author's LVCG backbone with a linear multi-label head."""

    def __init__(self, num_classes, **lvcg):
        super().__init__()
        self.backbone = LVCG(**lvcg)
        self.head = ClassificationHead("linear", self.backbone.out_features, num_classes)

    def forward(self, ecg):
        return self.head(self.backbone.forward_inference(ecg, use_all_leads=True))

    def auxiliary_losses(self, ecg, num_visible):
        """The author's pretraining losses, computed exactly as ``scripts/train.py`` does."""
        visible, mask = random_lead_mask(
            ecg.shape[0], num_leads=ecg.shape[1], num_visible=num_visible, device=ecg.device
        )
        out = self.backbone.forward_train(ecg, visible)
        predicted, target = out["V_hat_beats"], out["V_beats"]
        if predicted.shape[1] == target.shape[1]:
            beat = beat_level_loss(predicted, target, out["beat_mask_full"])
        else:  # the TTT path decodes N beats but encodes only the N-2 core ones
            beat = beat_level_loss(predicted[:, 1:-1], target, out["beat_mask_full"][:, 1:-1])
        return {
            "ecg": masked_reconstruction_loss(out["recon"], ecg, mask),
            "temporal": temporal_loss(out["states_pred"], out["states_real"], out["beat_mask"]),
            "beat": beat,
            "base": base_beat_loss(out["V_base_hat"], out["V_base"]),
        }

    def parameter_counts(self):
        total = sum(p.numel() for p in self.parameters())
        unused = sum(
            p.numel()
            for name in LVCG_RECONSTRUCTION_ONLY
            for p in getattr(self.backbone, name).parameters()
        )
        return {"total": total, "classification_path": total - unused}


class QuaternionDynamicFeatures(nn.Module):
    """VCG [B, 3, T] -> per-transition features [B, C, T - 1] and a validity mask.

    Parameter-free. For adjacent cardiac vectors P_t, P_{t+1} (one sample apart, so
    dt = 1 / fs):

        q_t      shortest-arc rotation u_t -> u_{t+1}                   4 channels
        theta_t  2 atan2(||q_xyz||, |q_w| + eps)                        1
        omega_t  theta_t / dt, rad/s                                    1
        magnitude         ||P_t||              (optional, section G)    1
        linear_velocity   ||P_{t+1} - P_t|| / dt  (optional)            1

    The real-valued control of section O replaces all of these with the raw pair and
    its difference: P_t, P_{t+1}, P_{t+1} - P_t (3 channels each).

    A transition whose endpoints are too small to carry a direction gets the identity
    rotation -- no rotation, theta = omega = 0 -- instead of the arbitrary one noise
    would produce; this happens before sign continuity is enforced, so the filled
    stretch joins the sequence without a sign jump. The mask is appended as a final
    channel for every feature set, control included, so the encoder can tell "no
    rotation" from "rotation undefined" and both branches see the same information
    about where the signal was too small.
    """

    def __init__(self, features, dt, min_fraction=0.02, sign_continuity=True):
        super().__init__()
        unknown = [f for f in features if f not in FEATURE_CHANNELS]
        if unknown or not features:
            raise ValueError(f"Unknown or empty feature set {features!r}")
        if any(f in CONTROL_FEATURES for f in features) and any(
            f in QUATERNION_FEATURES for f in features
        ):
            raise ValueError("A feature set is either quaternion or the real control, not both")
        self.features = tuple(features)
        self.dt = float(dt)
        self.min_fraction = float(min_fraction)
        self.sign_continuity = bool(sign_continuity)
        self.channels = sum(FEATURE_CHANNELS[f] for f in self.features) + 1
        self.register_buffer("identity", torch.tensor([1.0, 0.0, 0.0, 0.0]), persistent=False)

    def forward(self, vcg):
        p = vcg.transpose(1, 2)  # [B, T, 3]
        current, following = p[:, :-1], p[:, 1:]
        mask = valid_rotation_mask(p, lag=1, min_fraction=self.min_fraction)
        parts = {}
        if any(f in ("q", "theta", "omega") for f in self.features):
            q = torch.where(
                mask.unsqueeze(-1),
                vectors_to_quaternion(current, following),
                self.identity.to(p.dtype),
            )
            if self.sign_continuity:
                q = enforce_sign_continuity(q)
            theta = quaternion_angle(q).unsqueeze(-1)
            parts.update(q=q, theta=theta, omega=theta / self.dt)
        parts.update(
            magnitude=_safe_norm(current, keepdim=True),
            linear_velocity=_safe_norm(following - current, keepdim=True) / self.dt,
            position=current,
            next_position=following,
            delta=following - current,
        )
        x = torch.cat([parts[f] for f in self.features] + [mask.unsqueeze(-1).to(p.dtype)], -1)
        return x.transpose(1, 2), mask


class DynamicEncoder(nn.Module):
    """Lightweight temporal encoder, identical for the quaternion branch and its control.

    Input batch normalisation puts theta (radians per step) and omega (radians per
    second) on the same footing. It is scale-invariant, so at a fixed dt the theta-only
    and omega-only ablations see the same input up to its epsilon: they can differ only
    through noise, which is itself a check on the run-to-run variation.
    """

    def __init__(self, in_channels, embedding_dim=128, hidden=64, kernel=7, dropout=0.1):
        super().__init__()
        if kernel % 2 == 0:
            raise ValueError("kernel must be odd")
        layers, width = [nn.BatchNorm1d(in_channels)], in_channels
        for out in (hidden, embedding_dim, embedding_dim):
            layers += [
                nn.Conv1d(width, out, kernel, stride=2, padding=kernel // 2),
                nn.BatchNorm1d(out),
                nn.GELU(),
            ]
            width = out
        self.net = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        # LVCG layer-normalises each embedding part before concatenation; e_Q follows suit.
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        return self.norm(self.dropout(self.net(x).mean(dim=-1)))


class QDFLVCG(SupervisedLVCG):
    """V1: the V0 model untouched, plus e_Q from explicit cardiac-vector rotation.

        e_base = LVCG(ECG)                        640-d, exactly V0's embedding
        e_Q    = DynamicEncoder(features(VCG))    128-d
        logits = Linear([e_base ; e_Q])

    The VCG is lifted with the backbone's own direction buffer and pseudo-inverse, so
    e_Q is computed from the same VCG the backbone segments. The auxiliary losses, if
    the V0 protocol selected them, are inherited unchanged and act on the backbone only.
    """

    def __init__(
        self,
        num_classes,
        lvcg,
        features,
        embedding_dim=128,
        hidden=64,
        kernel=7,
        dropout=0.1,
        min_magnitude_fraction=0.02,
        sign_continuity=True,
    ):
        super().__init__(num_classes, **lvcg)
        self.dynamics = QuaternionDynamicFeatures(
            features, 1.0 / lvcg["fs"], min_magnitude_fraction, sign_continuity
        )
        self.dynamic_encoder = DynamicEncoder(
            self.dynamics.channels, embedding_dim, hidden, kernel, dropout
        )
        self.head = ClassificationHead(
            "linear", self.backbone.out_features + embedding_dim, num_classes
        )

    def vcg(self, ecg):
        directions = self.backbone.all_lead_directions.expand(ecg.shape[0], -1, -1)
        return self.backbone.vcg_inverse(ecg, directions)

    def forward(self, ecg):
        e_base = self.backbone.forward_inference(ecg, use_all_leads=True)
        features, _ = self.dynamics(self.vcg(ecg))
        return self.head(torch.cat((e_base, self.dynamic_encoder(features)), dim=-1))

    def parameter_counts(self):
        counts = super().parameter_counts()
        extra_head = self.dynamic_encoder.norm.normalized_shape[0] * self.head.net.out_features
        counts["dynamic_branch"] = (
            sum(p.numel() for p in self.dynamic_encoder.parameters()) + extra_head
        )
        return counts


def build_model(config, experiment):
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment {experiment!r}; expected one of {list(EXPERIMENTS)}")
    kind = EXPERIMENTS[experiment]["model"]
    settings = config["model"][kind]
    if kind == "qdf_lvcg":
        features = tuple(EXPERIMENTS[experiment]["features"])
        if features[0] in QUATERNION_FEATURES:
            features += tuple(
                name
                for name, flag in (
                    ("magnitude", settings["include_magnitude"]),
                    ("linear_velocity", settings["include_linear_velocity"]),
                )
                if flag
            )
        return QDFLVCG(
            len(CLASSES),
            config["model"]["lvcg"],
            features,
            embedding_dim=settings["embedding_dim"],
            hidden=settings["hidden"],
            kernel=settings["kernel"],
            dropout=settings["dropout"],
            min_magnitude_fraction=settings["min_magnitude_fraction"],
            sign_continuity=settings["sign_continuity"],
        )
    if kind == "traditional_vcg":
        return TraditionalVCG(
            len(CLASSES),
            lead_order=config["model"]["lvcg"]["lead_order"],
            channels=tuple(settings["channels"]),
            kernel=settings["kernel"],
            dropout=settings["dropout"],
        )
    return SupervisedLVCG(len(CLASSES), **settings)


def record_shapes(model, ecg):
    """Actual tensor shapes at every stage, from a real forward pass (report section 6)."""
    shapes, handles = {"ecg": list(ecg.shape)}, []

    def hook(name):
        def save(_module, _inputs, output):
            items = output if isinstance(output, tuple) else (output,)
            shapes[name] = [list(item.shape) for item in items if torch.is_tensor(item)]

        return save

    if isinstance(model, SupervisedLVCG):
        stages = {
            "vcg (vcg_inverse)": model.backbone.vcg_inverse,
            "beats, rr, mask (beat_segmenter)": model.backbone.beat_segmenter,
            "beat tokens (beat_encoder)": model.backbone.beat_encoder,
            "states_pred, h_last (state_generator)": model.backbone.state_generator,
            "emb_rhythm (global_rr_embedding)": model.backbone.global_rr_embedding,
            "logits (head)": model.head,
        }
        if isinstance(model, QDFLVCG):
            stages["features, mask (dynamics)"] = model.dynamics
            stages["e_Q (dynamic_encoder)"] = model.dynamic_encoder
    else:
        stages = {
            "vcg (lift)": model.lift,
            "features (encoder)": model.encoder,
            "logits (head)": model.head,
        }
    handles = [module.register_forward_hook(hook(name)) for name, module in stages.items()]
    try:
        with torch.no_grad():
            was_training = model.training
            model.eval()
            model(ecg)
            model.train(was_training)
    finally:
        for handle in handles:
            handle.remove()
    return shapes
