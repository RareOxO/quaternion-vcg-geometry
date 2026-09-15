"""The five main models of 方案 §8, built from one shared backbone.

    M0  raw VCG [X, Y, Z]                        real encoder
    M1  first-order [dot, cross] at 20 ms        real encoder
    M2  the same numbers as M1                   quaternion encoder
    M3  first-order at 10/20/40/80 ms            quaternion encoder
    M4  M3 plus the second-order difference      quaternion encoder

The fusion ladder of the 双分支 addendum keeps M0 intact as a raw branch and adds
one of M1-M4 as a second branch, joined by late concatenation:

    F1 = M0 + M1   F2 = M0 + M2   F3 = M0 + M3   F4 = M0 + M4

M0-M4 themselves are untouched, so their published numbers stay reproducible; the
fusion models answer a different question ("does geometry add anything on top of a
strong raw baseline?") than M1-M0 did ("is geometry a better replacement?").

Only the feature block and the interaction rule change between them. Stem depth,
kernel, pooling, dropout, pooling-to-logits head and training recipe are shared, so
a difference in the result table is attributable to the ablated component.

Width fairness (方案 §7.1): a quaternion conv with Q channels in and out holds
4 * Q^2 * K weights; a real conv of width C holds C^2 * K. Setting C = 2 * Q makes
those exactly equal, which is what `real_width = 2 * quaternions` does by default.
Per-channel terms (bias, norm gain) cannot match at the same time -- the quaternion
encoder has 4Q channels and the real control 2Q -- so totals stay close, not equal.
Table 3 prints the parameter count of every row for exactly this reason.
"""

import torch
from torch import nn

from .data import CLASSES
from .geometry import VCGGeometry
from .quaternion_nn import TCNEncoder

# Representation diagnostic of the v2 指导书 §3. All Real, single 20 ms scale.
DIAGNOSTIC_VARIANTS = ("R", "RA", "RU", "RLA")
SINGLE_VARIANTS = ("M0", "M1", "M2", "M3", "M4", *DIAGNOSTIC_VARIANTS)
# F_n keeps M0 as an intact raw branch and adds M_n as a second branch (双分支方案 §2).
FUSION_PAIRS = {"F1": "M1", "F2": "M2", "F3": "M3", "F4": "M4"}
FUSION_VARIANTS = tuple(FUSION_PAIRS)
VARIANTS = (*SINGLE_VARIANTS, *FUSION_VARIANTS)

VARIANT_DEFAULTS = {
    "M0": {"feature": "raw", "scales_ms": [], "algebra": "real"},
    "M1": {"feature": "first", "scales_ms": [20], "algebra": "real"},
    "M2": {"feature": "first", "scales_ms": [20], "algebra": "quaternion"},
    "M3": {"feature": "first", "scales_ms": [10, 20, 40, 80], "algebra": "quaternion"},
    "M4": {"feature": "first_second", "scales_ms": [10, 20, 40, 80], "algebra": "quaternion"},
    # r alone: how much diagnostic information does magnitude itself carry?
    "R": {"feature": "composite", "blocks": ["radial"], "scales_ms": [20], "algebra": "real"},
    # r + M1's angular relation: does magnitude restore what angular-only lost?
    "RA": {
        "feature": "composite",
        "blocks": ["radial", "angular"],
        "scales_ms": [20],
        "algebra": "real",
    },
    # r + absolute direction: V = r * u, so this is a reparameterization of raw XYZ.
    # Not a candidate model -- a representation audit (§4).
    "RU": {
        "feature": "composite",
        "blocks": ["radial", "direction"],
        "scales_ms": [20],
        "algebra": "real",
    },
    # radial + linear + angular, the decomposition the cited work suggests.
    "RLA": {
        "feature": "composite",
        "blocks": ["radial", "linear", "angular"],
        "scales_ms": [20],
        "algebra": "real",
    },
}


def branch_configs(config):
    """Split a fusion config into its raw (M0) and geometry (M1-M4) branch configs.

    The geometry branch inherits every model key the user set, so `scales_ms`,
    `operator` and the width settings stay under one knob; the raw branch is pinned
    to M0's defaults so it can never be perturbed by a geometry-side override
    (双分支方案 §4: "不要因为做 fusion 而修改 M0 branch").
    """
    if config["variant"] not in FUSION_VARIANTS:
        raise ValueError(f"variant must be one of {FUSION_VARIANTS}")
    geometry = {**config, "variant": FUSION_PAIRS[config["variant"]]}
    raw = {
        **config,
        "variant": "M0",
        **VARIANT_DEFAULTS["M0"],
        "operator": "conv",
        # M0 is a real encoder, so only real_width applies; keep the config's own value.
        "real_width": config.get("raw_width") or config.get("real_width"),
    }
    return raw, geometry


def model_settings(config):
    """Resolve variant defaults, letting explicit keys override them for the ablations."""
    if config["variant"] not in SINGLE_VARIANTS:
        raise ValueError(f"variant must be one of {SINGLE_VARIANTS}")
    settings = dict(VARIANT_DEFAULTS[config["variant"]])
    for key in ("feature", "scales_ms", "algebra"):
        if config.get(key) is not None:
            settings[key] = config[key]
    settings.setdefault("blocks", [])
    if settings["feature"] == "composite" and settings["algebra"] != "real":
        raise ValueError("The representation diagnostic is Real only (v2 指导书 §1)")
    if settings["algebra"] not in ("real", "quaternion"):
        raise ValueError("algebra must be real or quaternion")
    if settings["algebra"] == "quaternion" and settings["feature"] == "raw":
        raise ValueError("Raw XYZ VCG is not a quaternion feature")
    settings["operator"] = config.get("operator", "conv")
    settings["quaternions"] = config["quaternions"]
    settings["real_width"] = config.get("real_width") or 2 * config["quaternions"]
    for key in ("kernel", "depth", "stem_stride"):
        settings[key] = config[key]
    settings["dropout"] = config["dropout"]
    settings["renormalize"] = config.get("renormalize", False)
    return settings


class GeometryNet(nn.Module):
    def __init__(self, config, stats, classifier=True):
        super().__init__()
        settings = model_settings(config)
        self.variant, self.settings = config["variant"], settings
        quaternion = settings["algebra"] == "quaternion"
        self.frontend = VCGGeometry(
            settings["feature"],
            settings["scales_ms"],
            stats["sampling_rate"],
            vcg_scale=stats["vcg_std"],
            renormalize=settings["renormalize"],
            blocks=settings["blocks"],
        )
        width = 4 * settings["quaternions"] if quaternion else settings["real_width"]
        self.encoder = TCNEncoder(
            self.frontend.out_channels,
            width,
            kernel=settings["kernel"],
            depth=settings["depth"],
            stem_stride=settings["stem_stride"],
            dropout=settings["dropout"],
            quaternion=quaternion,
            operator=settings["operator"],
        )
        self.features = width
        # The head is real and identical everywhere; only the encoder is ablated.
        # A fusion branch drops it so its parameters do not inflate the fusion count.
        self.head = nn.Linear(width, len(CLASSES)) if classifier else None

    def describe(self):
        """Reporting surface shared with FusionNet, so the engine logs both alike."""
        return {
            "encoder_parameters": sum(p.numel() for p in self.encoder.parameters()),
            "receptive_field_samples": self.encoder.receptive_field,
            "input_channels": self.frontend.out_channels,
        }

    def forward_features(self, ecg):
        """Pre-classifier embedding, (B, features). Used as-is by the fusion branches."""
        return self.encoder(self.frontend(ecg)).float()

    def forward(self, ecg):
        if self.head is None:
            raise RuntimeError("This model was built without a classifier head")
        return self.head(self.forward_features(ecg))


class FusionNet(nn.Module):
    """M0 raw branch + an M1-M4 geometry branch, joined by late concat (双分支方案 §3).

    Deliberately late fusion: each branch keeps its own stem, so an effect can be
    attributed to the geometry features rather than to a wider first layer or a
    larger input channel count. The join is the same for F1-F4 -- project each
    branch to `fusion_dim`, concatenate, normalize, one linear to the classes
    (§9) -- with no attention, gating or hidden MLP.
    """

    def __init__(self, config, stats):
        super().__init__()
        raw_config, geometry_config = branch_configs(config)
        self.variant = config["variant"]
        self.raw = GeometryNet(raw_config, stats, classifier=False)
        self.geometry = GeometryNet(geometry_config, stats, classifier=False)
        dim = config.get("fusion_dim") or self.raw.features
        self.raw_project = nn.Linear(self.raw.features, dim)
        self.geometry_project = nn.Linear(self.geometry.features, dim)
        self.norm = nn.LayerNorm(2 * dim)
        self.head = nn.Linear(2 * dim, len(CLASSES))
        self.settings = {
            "fusion_dim": dim,
            "raw": self.raw.settings,
            "geometry": self.geometry.settings,
        }

    def describe(self):
        raw, geometry = self.raw.describe(), self.geometry.describe()
        return {
            "encoder_parameters": raw["encoder_parameters"] + geometry["encoder_parameters"],
            "receptive_field_samples": max(
                raw["receptive_field_samples"], geometry["receptive_field_samples"]
            ),
            "input_channels": {
                "raw": raw["input_channels"],
                "geometry": geometry["input_channels"],
            },
            "branch_parameters": {
                "raw": sum(p.numel() for p in self.raw.parameters()),
                "geometry": sum(p.numel() for p in self.geometry.parameters()),
            },
        }

    def forward_features(self, ecg):
        raw = self.raw_project(self.raw.forward_features(ecg))
        geometry = self.geometry_project(self.geometry.forward_features(ecg))
        return self.norm(torch.cat((raw, geometry), dim=-1))

    def forward(self, ecg):
        return self.head(self.forward_features(ecg))


def build_model(config, stats):
    if config["variant"] in FUSION_VARIANTS:
        return FusionNet(config, stats)
    return GeometryNet(config, stats)
