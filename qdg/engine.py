"""Supervised training and evaluation. Identical recipe for every variant.

Model selection uses validation macro AUROC (fold 9) only; fold 10 is touched once,
by `evaluate`, with the checkpoint and thresholds already fixed (方案 §12).
"""

import json
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import CLASSES, PTBXLDataset, load_manifest, save_json
from .metrics import multilabel_metrics, validation_thresholds
from .models import build_model


def setup(config):
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(config["cpu_threads"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = config["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable in this process")
    return torch.device(device)


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def make_loader(dataset, config, shuffle, generator=None):
    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=shuffle,
        num_workers=config["num_workers"],
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=False,
    )


def train_epoch(model, loader, criterion, optimizer, scaler, device, config):
    model.train()
    amp = config["amp"] and device.type == "cuda"
    total_loss, count = 0.0, 0
    for batch in tqdm(loader, desc="Train", leave=False):
        ecg = batch["ecg"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            loss = criterion(model(ecg), target)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * len(ecg)
        count += len(ecg)
    return total_loss / count


@torch.inference_mode()
def predict(model, loader, device, amp=False):
    model.eval()
    targets, probabilities, ecg_ids = [], [], []
    for batch in tqdm(loader, desc="Predict", leave=False):
        with torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"
        ):
            logits = model(batch["ecg"].to(device, non_blocking=True))
        probabilities.append(logits.float().sigmoid().cpu().numpy())
        targets.append(batch["target"].numpy())
        ecg_ids.append(batch["ecg_id"].numpy())
    return {
        "targets": np.concatenate(targets),
        "probabilities": np.concatenate(probabilities),
        "ecg_ids": np.concatenate(ecg_ids),
    }


def train(config, run_name=None, limit_train=None, limit_val=None):
    manifest = load_manifest(config["data"])
    tc = config["training"]
    device = setup(tc)
    model = build_model(config["model"], manifest["stats"]).to(device)
    weight = (
        torch.tensor(manifest["stats"]["pos_weight"], device=device) if tc["weighted_bce"] else None
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, tc["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=tc["amp"] and device.type == "cuda")
    generator = torch.Generator().manual_seed(tc["seed"])
    train_loader = make_loader(
        PTBXLDataset(config["data"]["cache"], "train", limit_train, tc["seed"]), tc, True, generator
    )
    val_loader = make_loader(
        PTBXLDataset(config["data"]["cache"], "val", limit_val, tc["seed"]), tc, False
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = run_name or f"{config['model']['variant']}_seed{tc['seed']}_{stamp}"
    if Path(run_name).name != run_name:
        raise ValueError("run_name must be a single directory name")
    run_dir = Path(tc["output"]) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    save_json(run_dir / "config.json", config)
    environment = {
        "run_dir": str(run_dir),
        "variant": config["model"]["variant"],
        "seed": tc["seed"],
        "settings": model.settings,
        "parameters": sum(p.numel() for p in model.parameters()),
        **model.describe(),
        "smoke_training": bool(limit_train or limit_val),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    save_json(run_dir / "environment.json", environment)
    print(json.dumps(environment, ensure_ascii=False), flush=True)
    best_score, stale = -1.0, 0
    for epoch in range(tc["epochs"]):
        if stale >= tc["patience"]:
            break
        started = time.perf_counter()
        loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device, tc)
        scheduler.step()
        predictions = predict(model, val_loader, device, tc["amp"])
        metrics = multilabel_metrics(predictions["targets"], predictions["probabilities"])
        score = metrics["macro_auroc"]
        if score is None:
            raise ValueError("Validation has no evaluable class; increase limit_val")
        improved = score > best_score + tc["min_delta"]
        best_score, stale = (score, 0) if improved else (best_score, stale + 1)
        if improved:
            thresholds = (
                validation_thresholds(predictions["targets"], predictions["probabilities"])
                if tc["threshold"] == "validation_f1"
                else np.full(len(CLASSES), 0.5)
            )
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": config,
                    "stats": manifest["stats"],
                    "signature": manifest["signature"],
                    "epoch": epoch,
                    "best_score": best_score,
                    "thresholds": thresholds.tolist(),
                    "environment": environment,
                },
                run_dir / "best.pt",
            )
            save_json(
                run_dir / "validation_metrics.json",
                {"epoch": epoch + 1, "metrics": metrics, "thresholds": thresholds.tolist()},
            )
        log = {
            "epoch": epoch + 1,
            "train_loss": loss,
            "lr": optimizer.param_groups[0]["lr"],
            "val_macro_auroc": score,
            "val_macro_auprc": metrics["macro_auprc"],
            "seconds": time.perf_counter() - started,
            "best": improved,
            "stale_epochs": stale,
        }
        with (run_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(log, allow_nan=False) + "\n")
        print(json.dumps(log), flush=True)
    return run_dir


def evaluate(checkpoint_path, split="test", device=None, limit=None):
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    tc = dict(config["training"], device=device or config["training"]["device"])
    torch_device = setup(tc)
    manifest = load_manifest(config["data"])
    if manifest["signature"] != checkpoint["signature"]:
        raise ValueError("Checkpoint was trained on a different cache")
    model = build_model(config["model"], checkpoint["stats"]).to(torch_device)
    model.load_state_dict(checkpoint["model"])
    loader = make_loader(PTBXLDataset(config["data"]["cache"], split, limit, tc["seed"]), tc, False)
    predictions = predict(model, loader, torch_device, tc["amp"])
    thresholds = np.asarray(checkpoint["thresholds"])
    result = {
        "variant": config["model"]["variant"],
        "seed": tc["seed"],
        "split": split,
        "best_epoch": checkpoint["epoch"] + 1,
        "validation_macro_auroc": checkpoint["best_score"],
        "parameters": checkpoint["environment"]["parameters"],
        "settings": checkpoint["environment"]["settings"],
        "limited_evaluation": limit is not None,
        "smoke_training": checkpoint["environment"]["smoke_training"],
        "fixed_0.5": multilabel_metrics(predictions["targets"], predictions["probabilities"]),
        "validation_selected_thresholds": multilabel_metrics(
            predictions["targets"], predictions["probabilities"], thresholds
        ),
    }
    run_dir = checkpoint_path.parent
    save_json(run_dir / f"best_{split}_metrics.json", result)
    np.savez_compressed(run_dir / f"{split}_predictions.npz", **predictions)
    return result
