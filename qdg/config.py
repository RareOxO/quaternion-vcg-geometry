from copy import deepcopy
from pathlib import Path

import yaml

from .geometry import FEATURES, lag_samples
from .models import VARIANTS


def merge(base, overrides):
    result = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path):
    """Paths inside a YAML resolve against that YAML, not the working directory."""
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    parent = config.pop("extends", None)
    for section, key in (("data", "root"), ("data", "cache"), ("training", "output")):
        if key in config.get(section, {}):
            config[section][key] = str((path.parent / config[section][key]).resolve())
    if parent:
        config = merge(load_config(path.parent / parent), config)
    validate_config(config)
    return config


def validate_config(config):
    data, model, train = (config[key] for key in ("data", "model", "training"))
    if data["sampling_rate"] not in (100, 500):
        raise ValueError("PTB-XL sampling_rate must be 100 or 500")
    if data["bandpass"] is not None:
        low, high = data["bandpass"]
        if not 0 < low < high < data["sampling_rate"] / 2:
            raise ValueError("Bandpass frequencies must lie below Nyquist")
    if model["variant"] not in VARIANTS:
        raise ValueError(f"model.variant must be one of {VARIANTS}")
    if model.get("feature") is not None and model["feature"] not in FEATURES:
        raise ValueError(f"model.feature must be one of {FEATURES}")
    if model.get("operator", "conv") not in ("conv", "mlp"):
        raise ValueError("model.operator must be conv or mlp")
    if model.get("algebra") is not None and model["algebra"] not in ("real", "quaternion"):
        raise ValueError("model.algebra must be real or quaternion")
    for key in ("quaternions", "kernel", "depth", "stem_stride"):
        if not isinstance(model[key], int) or model[key] < 1:
            raise ValueError(f"model.{key} must be a positive integer")
    if model["kernel"] % 2 == 0:
        raise ValueError("model.kernel must be odd so padding keeps the length")
    if (10 * data["sampling_rate"]) % (model["stem_stride"] * 2 ** (model["depth"] - 1)):
        raise ValueError("stem_stride and depth must divide the signal length evenly")
    if not 0 <= model["dropout"] < 1:
        raise ValueError("model.dropout must lie in [0, 1)")
    scales = model.get("scales_ms")
    if scales is not None:
        if len(set(scales)) != len(scales):
            raise ValueError("model.scales_ms must be unique")
        for lag_ms in scales:
            # Second order needs two lags of history on each side of the signal.
            if 2 * lag_samples(lag_ms, data["sampling_rate"]) >= 10 * data["sampling_rate"]:
                raise ValueError(f"Scale {lag_ms} ms is too long for a 10 s record")
    for key in ("epochs", "patience", "batch_size", "cpu_threads"):
        if not isinstance(train[key], int) or train[key] < 1:
            raise ValueError(f"training.{key} must be positive")
    if train["threshold"] not in ("validation_f1", "fixed_0.5"):
        raise ValueError("training.threshold must be validation_f1 or fixed_0.5")
    return config
