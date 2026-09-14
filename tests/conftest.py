from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import wfdb

from qdg.config import load_config
from qdg.data import CLASSES, prepare
from qdg.geometry import LEADS

torch.set_num_threads(2)


@pytest.fixture
def tiny_config():
    config = load_config(Path(__file__).resolve().parents[1] / "configs/base.yaml")
    config["model"].update(quaternions=4, depth=2, dropout=0.0)
    config["training"].update(
        device="cpu", num_workers=0, batch_size=3, epochs=2, cpu_threads=2, amp=False
    )
    return config


@pytest.fixture
def synthetic_cache(tmp_path, tiny_config):
    """A miniature PTB-XL at 100 Hz so the whole pipeline runs in seconds."""
    root = tmp_path / "source"
    root.mkdir()
    pd.DataFrame(
        {"diagnostic": [1] * len(CLASSES), "diagnostic_class": CLASSES}, index=list(CLASSES)
    ).to_csv(root / "scp_statements.csv")
    rows = []
    rng = np.random.default_rng(123)
    for i in range(18):
        fold = i % 8 + 1 if i < 10 else 9 if i < 14 else 10
        name = f"record{i:03d}"
        waveform = rng.normal(0, 0.1, (1000, 12))
        wfdb.wrsamp(
            name,
            fs=100,
            units=["mV"] * 12,
            sig_name=list(LEADS),
            p_signal=waveform,
            fmt=["16"] * 12,
            write_dir=str(root),
        )
        rows.append(
            {
                "ecg_id": i + 1,
                "patient_id": i + 100,
                "strat_fold": fold,
                "scp_codes": repr({CLASSES[i % 5]: 0, CLASSES[(i + 1) % 5]: 100}),
                "filename_lr": name,
                "filename_hr": name,
            }
        )
    pd.DataFrame(rows).set_index("ecg_id").to_csv(root / "ptbxl_database.csv")
    config = tiny_config
    config["data"].update(
        root=str(root), cache=str(tmp_path / "cache"), sampling_rate=100, bandpass=[0.5, 45.0]
    )
    config["training"]["output"] = str(tmp_path / "runs")
    return config, prepare(config["data"])
