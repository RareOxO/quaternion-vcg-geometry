"""The five main models of 方案 §8, built from one shared backbone.

    M0  raw VCG [X, Y, Z]                        real encoder
    M1  first-order [dot, cross] at 20 ms        real encoder
    M2  the same numbers as M1                   quaternion encoder
    M3  first-order at 10/20/40/80 ms            quaternion encoder
    M4  M3 plus the second-order difference      quaternion encoder

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

from torch import nn

from .data import CLASSES
from .geometry import VCGGeometry
from .quaternion_nn import TCNEncoder

VARIANTS = ("M0", "M1", "M2", "M3", "M4")

VARIANT_DEFAULTS = {
    "M0": {"feature": "raw", "scales_ms": [], "algebra": "real"},
    "M1": {"feature": "first", "scales_ms": [20], "algebra": "real"},
    "M2": {"feature": "first", "scales_ms": [20], "algebra": "quaternion"},
    "M3": {"feature": "first", "scales_ms": [10, 20, 40, 80], "algebra": "quaternion"},
    "M4": {"feature": "first_second", "scales_ms": [10, 20, 40, 80], "algebra": "quaternion"},
}


def model_settings(config):
    """Resolve variant defaults, letting explicit keys override them for the ablations."""
    if config["variant"] not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    settings = dict(VARIANT_DEFAULTS[config["variant"]])
    for key in ("feature", "scales_ms", "algebra"):
        if config.get(key) is not None:
            settings[key] = config[key]
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
    def __init__(self, config, stats):
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
        # The head is real and identical everywhere; only the encoder is ablated.
        self.head = nn.Linear(width, len(CLASSES))

    def forward(self, ecg):
        return self.head(self.encoder(self.frontend(ecg)).float())


def build_model(config, stats):
    return GeometryNet(config, stats)
