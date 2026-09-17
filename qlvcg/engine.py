"""Training and evaluation, one recipe for B0, V0 and every later variant.

Checkpoint selection uses validation macro AUROC only; the test fold is read once, by
``evaluate``, after the checkpoint and the F1 thresholds are fixed on validation.

Two properties of the released LVCG code that matter for reading these numbers:

* ``forward_inference`` unrolls its GRU for N - 1 steps, where N is the largest beat
  count in the *batch*, so a record's dynamic embedding depends on which records share
  its batch. Evaluation therefore never shuffles and always uses the training batch
  size, which keeps it reproducible; it does not make it batch-independent.
* The release takes ``emb_struct`` from the first complete beat, where the paper
  describes mean-pooled beat tokens (``docs/ARCHITECTURE.md`` notes this). With the
  classification loss alone, only that beat's morphology reaches the logits; later
  beats contribute their R-R intervals and, through the rollout length, their count.
"""

import csv
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from qdg.data import PTBXLDataset, load_manifest, save_json
from qdg.engine import make_loader, setup
from qdg.metrics import validation_thresholds

from .metrics import full_metrics
from .models import EXPERIMENTS, LEGACY_NAMES, build_model, record_shapes, standardize

CSV_COLUMNS = (
    "experiment",
    "variant",
    "seed",
    "parameter_count",
    "best_epoch",
    "macro_auroc",
    "micro_auroc",
    "macro_f1",
    "micro_f1",
    "training_time",
    "checkpoint",
    "config",
    "report",
)


def warmup_cosine(warmup_steps, total_steps):
    """Linear warmup to the base rate, then cosine decay to zero at ``total_steps``."""

    def factor(step):
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        span = max(1, total_steps - warmup_steps)
        progress = min(1.0, (step - warmup_steps) / span)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return factor


def check_cache(manifest, config):
    stats, lvcg = manifest["stats"], config["model"]["lvcg"]
    if stats["sampling_rate"] != lvcg["fs"] or stats["signal_length"] != lvcg["time_len"]:
        raise ValueError(
            f"Cache is {stats['sampling_rate']} Hz x {stats['signal_length']} samples; "
            f"the model expects {lvcg['fs']} Hz x {lvcg['time_len']}. Run `qlvcg prepare`."
        )


def train_epoch(model, loader, optimizer, scheduler, device, config):
    model.train()
    tc = config["training"]
    total, count = 0.0, 0
    for batch in tqdm(loader, desc="Train", leave=False):
        ecg = standardize(batch["ecg"].to(device, non_blocking=True))
        target = batch["target"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(model(ecg), target)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), tc["gradient_clip"])
        optimizer.step()
        scheduler.step()
        total += loss.item() * len(ecg)
        count += len(ecg)
    return total / count


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    targets, probabilities, ecg_ids, loss, cells = [], [], [], 0.0, 0
    for batch in tqdm(loader, desc="Predict", leave=False):
        logits = model(standardize(batch["ecg"].to(device, non_blocking=True))).float()
        target = batch["target"].to(device)
        loss += F.binary_cross_entropy_with_logits(logits, target, reduction="sum").item()
        cells += target.numel()
        probabilities.append(logits.sigmoid().cpu().numpy())
        targets.append(batch["target"].numpy())
        ecg_ids.append(batch["ecg_id"].numpy())
    return {
        "targets": np.concatenate(targets),
        "probabilities": np.concatenate(probabilities),
        "ecg_ids": np.concatenate(ecg_ids),
        "loss": loss / cells,
    }


def train(config, experiment, run_name=None, limit_train=None, limit_val=None, epochs=None):
    spec = EXPERIMENTS[experiment]
    if spec.get("reuses"):
        raise ValueError(
            f"{experiment} is the same configuration as {spec['reuses']}; its results are read "
            f"from the {spec['reuses']} run, so it is not trained again."
        )
    tc = config["training"]
    if epochs is not None:
        tc["epochs"] = epochs
    manifest = load_manifest(config["data"])
    check_cache(manifest, config)
    device = setup(tc)
    model = build_model(config, experiment).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc["weight_decay"])
    generator = torch.Generator().manual_seed(tc["seed"])
    train_set = PTBXLDataset(config["data"]["cache"], "train", limit_train, tc["seed"])
    train_loader = make_loader(train_set, tc, True, generator)
    val_loader = make_loader(
        PTBXLDataset(config["data"]["cache"], "val", limit_val, tc["seed"]), tc, False
    )
    total_steps = tc["epochs"] * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, warmup_cosine(tc["warmup_steps"], total_steps)
    )
    run_name = run_name or f"{experiment}_seed{tc['seed']}"
    if Path(run_name).name != run_name:
        raise ValueError("run_name must be a single directory name")
    run_dir = Path(tc["output"]) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    config = {**config, "experiment": experiment}
    save_json(run_dir / "config.json", config)
    sample = standardize(torch.stack([train_set[i]["ecg"] for i in range(min(4, len(train_set)))]))
    environment = {
        "run_dir": str(run_dir),
        "experiment": experiment,
        **spec,
        "objective": "classification",
        "seed": tc["seed"],
        "parameters": model.parameter_counts(),
        "tensor_shapes": record_shapes(model, sample.to(device)),
        "train_records": len(train_set),
        "val_records": len(val_loader.dataset),
        "steps_per_epoch": len(train_loader),
        "smoke_training": bool(limit_train or limit_val),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }
    save_json(run_dir / "environment.json", environment)
    print(json.dumps(environment, ensure_ascii=False), flush=True)
    started_all = time.perf_counter()
    best_score, stale = -1.0, 0
    for epoch in range(tc["epochs"]):
        if stale >= tc["patience"]:
            break
        started = time.perf_counter()
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, config)
        predictions = predict(model, val_loader, device)
        metrics = full_metrics(predictions["targets"], predictions["probabilities"])
        score = metrics["macro_auroc"]
        if score is None:
            raise ValueError("Validation has no evaluable class; increase limit_val")
        improved = score > best_score + tc["min_delta"]
        best_score, stale = (score, 0) if improved else (best_score, stale + 1)
        if improved:
            thresholds = validation_thresholds(predictions["targets"], predictions["probabilities"])
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": config,
                    "experiment": experiment,
                    "signature": manifest["signature"],
                    "epoch": epoch,
                    "best_score": best_score,
                    "validation_loss": predictions["loss"],
                    "thresholds": thresholds.tolist(),
                    "environment": environment,
                },
                run_dir / "best.pt",
            )
            save_json(
                run_dir / "validation_metrics.json",
                {
                    "epoch": epoch + 1,
                    "loss": predictions["loss"],
                    "metrics": metrics,
                    "thresholds": thresholds.tolist(),
                },
            )
        log = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": predictions["loss"],
            "val_macro_auroc": score,
            "val_micro_auroc": metrics["micro_auroc"],
            "val_macro_f1": metrics["macro_f1"],
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.perf_counter() - started,
            "best": improved,
            "stale_epochs": stale,
        }
        with (run_dir / "history.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(log, allow_nan=False) + "\n")
        print(json.dumps(log), flush=True)
    training_time = time.perf_counter() - started_all
    save_json(run_dir / "training_time.json", {"seconds": training_time})
    result = evaluate(run_dir / "best.pt", device=str(device))
    if not environment["smoke_training"]:
        append_result(config, experiment, run_dir, result, training_time)
    return run_dir


def evaluate(checkpoint_path, split="test", device=None, limit=None):
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    experiment = LEGACY_NAMES.get(checkpoint["experiment"], checkpoint["experiment"])
    tc = dict(config["training"], device=device or config["training"]["device"])
    torch_device = setup(tc)
    manifest = load_manifest(config["data"])
    if manifest["signature"] != checkpoint["signature"]:
        raise ValueError("Checkpoint was trained on a different cache")
    model = build_model(config, experiment).to(torch_device)
    model.load_state_dict(checkpoint["model"])
    loader = make_loader(PTBXLDataset(config["data"]["cache"], split, limit, tc["seed"]), tc, False)
    predictions = predict(model, loader, torch_device)
    thresholds = np.asarray(checkpoint["thresholds"])
    result = {
        "experiment": experiment,
        "seed": tc["seed"],
        "split": split,
        "best_epoch": checkpoint["epoch"] + 1,
        "validation_macro_auroc": checkpoint["best_score"],
        "validation_loss": checkpoint["validation_loss"],
        f"{split}_loss": predictions["loss"],
        "parameters": checkpoint["environment"]["parameters"],
        "limited_evaluation": limit is not None,
        "smoke_training": checkpoint["environment"]["smoke_training"],
        "fixed_0.5": full_metrics(predictions["targets"], predictions["probabilities"]),
        "validation_selected_thresholds": full_metrics(
            predictions["targets"], predictions["probabilities"], thresholds
        ),
    }
    run_dir = checkpoint_path.parent
    save_json(run_dir / f"best_{split}_metrics.json", result)
    np.savez_compressed(
        run_dir / f"{split}_predictions.npz",
        **{k: v for k, v in predictions.items() if k != "loss"},
    )
    return result


def append_result(config, experiment, run_dir, result, training_time):
    """One row per model x seed, appended; an existing row is never rewritten."""
    path = Path(config["training"]["results"]) / "quaternion_experiments.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = result["fixed_0.5"]
    row = {
        "experiment": experiment,
        "variant": EXPERIMENTS[experiment]["variant"],
        "seed": result["seed"],
        "parameter_count": result["parameters"]["total"],
        "best_epoch": result["best_epoch"],
        "macro_auroc": round(metrics["macro_auroc"], 6),
        "micro_auroc": round(metrics["micro_auroc"], 6),
        "macro_f1": round(metrics["macro_f1"], 6),
        "micro_f1": round(metrics["micro_f1"], 6),
        "training_time": round(training_time, 1),
        "checkpoint": str(Path(run_dir) / "best.pt"),
        "config": str(Path(run_dir) / "config.json"),
        "report": EXPERIMENTS[experiment]["report"],
    }
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow(row)
