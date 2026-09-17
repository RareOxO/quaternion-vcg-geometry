"""Configuration for the LVCG-family experiments."""

from pathlib import Path

import yaml

from qdg.config import merge
from qdg.data import CLASSES

PATH_KEYS = (
    ("data", "root"),
    ("data", "cache"),
    ("training", "output"),
    ("training", "results"),
    ("training", "reports"),
)
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "lvcg.yaml"


def load_config(path=DEFAULT_CONFIG):
    """Paths inside a YAML resolve against that YAML, not the working directory."""
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    parent = config.pop("extends", None)
    for section, key in PATH_KEYS:
        if key in config.get(section, {}):
            config[section][key] = str((path.parent / config[section][key]).resolve())
    if parent:
        config = merge(load_config(path.parent / parent), config)
    return validate_config(config)


def validate_config(config):
    data, lvcg, train = config["data"], config["model"]["lvcg"], config["training"]
    rate = data["sampling_rate"]
    if rate != lvcg["fs"]:
        raise ValueError("data.sampling_rate must equal model.lvcg.fs")
    if 10 * rate != lvcg["time_len"]:
        raise ValueError("model.lvcg.time_len must equal ten seconds of samples")
    if data["bandpass"] is not None:
        low, high = data["bandpass"]
        if not 0 < low < high < rate / 2:
            raise ValueError("Band-pass frequencies must lie below Nyquist")
    if lvcg["lead_order"] != "ptbxl":
        raise ValueError("Records are fed in PTB-XL's native order; lead_order must be ptbxl")
    for key in ("epochs", "patience", "batch_size", "cpu_threads"):
        if not isinstance(train[key], int) or train[key] < 1:
            raise ValueError(f"training.{key} must be a positive integer")
    if not isinstance(train["warmup_steps"], int) or train["warmup_steps"] < 0:
        raise ValueError("training.warmup_steps must be a non-negative integer")
    if len(CLASSES) != 5:
        raise ValueError("The PTB-XL superclass task has five labels")
    return config
