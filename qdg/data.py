"""PTB-XL only, the official split, five diagnostic superclasses (方案 §9).

prepare() writes a memory-mapped cache of band-passed 12-lead waveforms plus the
training-fold statistics the models need. Folds 1-8 train, fold 9 validates,
fold 10 tests; the official strat_fold is never re-derived or re-shuffled.
"""

import ast
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wfdb
from scipy.signal import butter, sosfiltfilt
from torch.utils.data import Dataset
from tqdm import tqdm

from .geometry import INDEPENDENT_LEADS, KORS, LEADS

CLASSES = ("NORM", "MI", "STTC", "CD", "HYP")
CACHE_VERSION = 1


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(temporary, path)


def split_masks(folds):
    return {"train": folds <= 8, "val": folds == 9, "test": folds == 10}


def metadata(root):
    root = Path(root)
    frame = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    statements = pd.read_csv(root / "scp_statements.csv", index_col=0)
    if not frame.index.is_unique or frame[["patient_id", "strat_fold"]].isna().any().any():
        raise ValueError("Missing patient/fold information or duplicate ECG IDs")
    if not frame.strat_fold.isin(range(1, 11)).all():
        raise ValueError("Invalid official strat_fold")
    if frame.groupby("patient_id").strat_fold.nunique().max() != 1:
        raise ValueError("Patient leakage: a patient appears in multiple official folds")
    mapping = statements.loc[statements.diagnostic == 1, "diagnostic_class"].to_dict()
    labels = np.zeros((len(frame), len(CLASSES)), dtype=np.float32)
    for i, codes in enumerate(frame.scp_codes):
        # The benchmark uses the presence of a diagnostic code, including likelihood 0.
        for code in ast.literal_eval(codes):
            superclass = mapping.get(code)
            if superclass in CLASSES:
                labels[i, CLASSES.index(superclass)] = 1
    keep = labels.any(axis=1)
    frame, labels = frame.loc[keep].copy(), labels[keep]
    report = {
        "records_eligible": len(frame),
        "records_without_superclass": int((~keep).sum()),
        "patients": int(frame.patient_id.nunique()),
        "class_counts": dict(zip(CLASSES, labels.sum(0).astype(int).tolist())),
        "splits": {
            name: {
                "records": int(mask.sum()),
                "patients": int(frame.loc[mask].patient_id.nunique()),
                "class_counts": dict(zip(CLASSES, labels[mask].sum(0).astype(int).tolist())),
            }
            for name, mask in split_masks(frame.strat_fold.to_numpy()).items()
        },
    }
    return frame, labels, report


def cache_signature(config):
    root = Path(config["root"])
    return {
        "version": CACHE_VERSION,
        "database_sha256": sha256(root / "ptbxl_database.csv"),
        "statements_sha256": sha256(root / "scp_statements.csv"),
        "sampling_rate": config["sampling_rate"],
        "bandpass": config["bandpass"],
        "classes": list(CLASSES),
    }


def audit(config):
    frame, _, report = metadata(config["root"])
    filename = "filename_hr" if config["sampling_rate"] == 500 else "filename_lr"
    missing = [
        str(Path(config["root"]) / (relative + suffix))
        for relative in frame[filename]
        for suffix in (".hea", ".dat")
        if not (Path(config["root"]) / (relative + suffix)).is_file()
    ]
    report["sampling_rate"] = config["sampling_rate"]
    report["missing_file_count"] = len(missing)
    report["missing_file_examples"] = missing[:10]
    return report


def read_waveform(record_path, sampling_rate, bandpass=None):
    signal, fields = wfdb.rdsamp(str(record_path))
    names = [name.upper() for name in fields["sig_name"]]
    if len(names) != 12 or set(names) != set(LEADS):
        raise ValueError(f"Unexpected lead names in {record_path}: {names}")
    order = [names.index(name) for name in LEADS]
    if any(fields["units"][i].lower() != "mv" for i in order):
        raise ValueError(f"Expected physical mV units: {record_path}")
    if fields["fs"] != sampling_rate or signal.shape != (sampling_rate * 10, 12):
        raise ValueError(f"Unexpected sampling rate or duration: {record_path}")
    signal = signal[:, order].T
    if not np.isfinite(signal).all():
        raise ValueError(f"Nonfinite waveform values: {record_path}")
    if bandpass is not None:
        sos = butter(4, bandpass, btype="bandpass", fs=sampling_rate, output="sos")
        signal = sosfiltfilt(sos, signal, axis=-1)
    return np.ascontiguousarray(signal, dtype=np.float32)


def load_manifest(config):
    cache = Path(config["cache"])
    path = cache / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"No completed cache at {cache}; run `qdg prepare` first")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["signature"] != cache_signature(config):
        raise ValueError("Cache/config mismatch; choose a new data.cache path and prepare it")
    for name, size in manifest["file_sizes"].items():
        entry = cache / name
        if not entry.is_file() or entry.stat().st_size != size:
            raise ValueError(f"Missing or truncated cache file: {entry}")
    return manifest


def prepare(config):
    cache = Path(config["cache"])
    if (cache / "manifest.json").exists():
        return load_manifest(config)
    if cache.exists() and any(cache.iterdir()):
        raise FileExistsError(f"Incomplete/nonempty cache {cache}; use a new cache path")
    report = audit(config)
    if report["missing_file_count"]:
        raise FileNotFoundError(json.dumps(report, ensure_ascii=False))
    frame, labels, _ = metadata(config["root"])
    cache.mkdir(parents=True, exist_ok=True)
    length, count = config["sampling_rate"] * 10, len(frame)
    signals = np.lib.format.open_memmap(
        cache / "signals.npy", mode="w+", dtype="float32", shape=(count, 12, length)
    )
    kors = np.asarray(KORS, dtype=np.float32)
    filename = "filename_hr" if config["sampling_rate"] == 500 else "filename_lr"
    # VCG scale is estimated on the training folds only and used to normalize M0 input.
    vcg_squares, train_samples = np.zeros(3, np.float64), 0
    for i, (_, row) in enumerate(tqdm(frame.iterrows(), total=count, desc="Prepare ECG")):
        signal = read_waveform(
            Path(config["root"]) / row[filename], config["sampling_rate"], config["bandpass"]
        )
        signals[i] = signal
        if row.strat_fold <= 8:
            vcg = kors @ signal[list(INDEPENDENT_LEADS)].astype(np.float64)
            vcg_squares += np.square(vcg).sum(axis=1)
            train_samples += length
    signals.flush()
    del signals
    folds = frame.strat_fold.to_numpy(dtype=np.int64)
    positives = labels[folds <= 8].sum(0)
    if np.any(positives == 0):
        raise ValueError("Training folds have a class with no positive labels")
    stats = {
        "sampling_rate": config["sampling_rate"],
        "signal_length": length,
        "vcg_std": np.sqrt(np.maximum(vcg_squares / train_samples, 1e-12)).tolist(),
        "pos_weight": (((folds <= 8).sum() - positives) / positives).tolist(),
        "training_records": int((folds <= 8).sum()),
    }
    arrays = {
        "labels": labels,
        "folds": folds,
        "ecg_ids": frame.index.to_numpy(dtype=np.int64),
        "patient_ids": frame.patient_id.to_numpy(dtype=np.int64),
    }
    for name, array in arrays.items():
        np.save(cache / f"{name}.npy", array, allow_pickle=False)
    names = ["signals.npy"] + [f"{name}.npy" for name in arrays]
    manifest = {
        "signature": cache_signature(config),
        "report": report,
        "stats": stats,
        "source_root": str(Path(config["root"]).resolve()),
        "file_sizes": {name: (cache / name).stat().st_size for name in names},
    }
    save_json(cache / "manifest.json", manifest)
    return manifest


class PTBXLDataset(Dataset):
    def __init__(self, cache, split, limit=None, seed=0):
        self.cache = Path(cache)
        folds = np.load(self.cache / "folds.npy", allow_pickle=False)
        self.indices = np.flatnonzero(split_masks(folds)[split])
        if limit is not None:
            if limit < 1:
                raise ValueError("Subset limit must be positive")
            rng = np.random.default_rng(seed)
            self.indices = np.sort(rng.choice(self.indices, min(limit, len(self.indices)), False))
        self.labels = np.load(self.cache / "labels.npy", allow_pickle=False)
        self.ecg_ids = np.load(self.cache / "ecg_ids.npy", allow_pickle=False)
        self._signals = None

    def __getstate__(self):
        return {**self.__dict__, "_signals": None}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        if self._signals is None:
            self._signals = np.load(self.cache / "signals.npy", mmap_mode="r", allow_pickle=False)
        i = self.indices[index]
        return {
            "ecg": torch.from_numpy(self._signals[i].copy()),
            "target": torch.from_numpy(self.labels[i].copy()),
            "ecg_id": int(self.ecg_ids[i]),
        }
