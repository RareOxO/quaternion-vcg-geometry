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

# The eps LVCG hard-codes for its lift; B0 must use the same one.
LIFT_EPS = 0.1

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


def build_model(config, experiment):
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment {experiment!r}; expected one of {list(EXPERIMENTS)}")
    kind = EXPERIMENTS[experiment]["model"]
    settings = config["model"][kind]
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
