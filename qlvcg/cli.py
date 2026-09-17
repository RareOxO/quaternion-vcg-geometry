import argparse
import json
from pathlib import Path

import torch

from qdg.data import PTBXLDataset, load_manifest, prepare

from .config import DEFAULT_CONFIG, load_config
from .engine import check_cache, evaluate, train
from .models import EXPERIMENTS, build_model, record_shapes, standardize
from .tables import write_history


def profile(config, experiment):
    """What master prompt section W asks to state before training, from real objects."""
    manifest = load_manifest(config["data"])
    check_cache(manifest, config)
    tc, report = config["training"], manifest["report"]
    model = build_model(config, experiment)
    dataset = PTBXLDataset(config["data"]["cache"], "train")
    ecg = standardize(torch.stack([dataset[i]["ecg"] for i in range(4)]))
    spec = EXPERIMENTS[experiment]
    return {
        "experiment": experiment,
        "task": "PTB-XL diagnostic superclass, multi-label",
        "labels": list(report["class_counts"]),
        "split_sizes": {name: s["records"] for name, s in report["splits"].items()},
        "preprocessing": {
            "sampling_rate": config["data"]["sampling_rate"],
            "bandpass": config["data"]["bandpass"],
            "normalisation": "per-record per-lead z-score",
        },
        "ecg_to_vcg": "Tikhonov pseudo-inverse of the paper's Table 7 lead directions, eps 0.1",
        "model": spec["model"],
        "objective": spec["objective"],
        "tensor_shapes": record_shapes(model, ecg),
        "parameters": model.parameter_counts(),
        "loss": "BCEWithLogitsLoss (no positive weighting)"
        + (" + author auxiliary losses" if spec["objective"] != "classification" else ""),
        "optimizer": "AdamW",
        "lr": tc["lr"],
        "weight_decay": tc["weight_decay"],
        "schedule": f"linear warmup {tc['warmup_steps']} steps, cosine to zero",
        "batch_size": tc["batch_size"],
        "epochs": tc["epochs"],
        "early_stopping": f"validation macro AUROC, patience {tc['patience']}",
        "seed": tc["seed"],
        "command": f"python -m qlvcg train --experiment {experiment}",
    }


def main():
    parser = argparse.ArgumentParser(prog="qlvcg")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "profile", "train", "evaluate", "tables"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        if name in ("profile", "train"):
            sub.add_argument("--experiment", choices=list(EXPERIMENTS), required=True)
        if name == "train":
            sub.add_argument("--seed", type=int)
            sub.add_argument("--run-name")
            sub.add_argument("--epochs", type=int)
            sub.add_argument("--device")
            sub.add_argument("--limit-train", type=int)
            sub.add_argument("--limit-val", type=int)
        if name == "evaluate":
            sub.add_argument("--checkpoint", type=Path, required=True)
            sub.add_argument("--split", choices=("val", "test"), default="test")
            sub.add_argument("--device")
        if name == "tables":
            sub.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "prepare":
        manifest = prepare(config["data"])
        print(json.dumps(manifest["report"], ensure_ascii=False, indent=2))
    elif args.command == "profile":
        print(json.dumps(profile(config, args.experiment), ensure_ascii=False, indent=2))
    elif args.command == "train":
        if args.seed is not None:
            config["training"]["seed"] = args.seed
        if args.device:
            config["training"]["device"] = args.device
        run_dir = train(
            config,
            args.experiment,
            run_name=args.run_name,
            limit_train=args.limit_train,
            limit_val=args.limit_val,
            epochs=args.epochs,
        )
        result = json.loads((run_dir / "best_test_metrics.json").read_text())
        test = result["fixed_0.5"]
        print(
            json.dumps(
                {
                    "run_dir": str(run_dir),
                    "best_epoch": result["best_epoch"],
                    "val_macro_auroc": result["validation_macro_auroc"],
                    "test_macro_auroc": test["macro_auroc"],
                    "test_micro_auroc": test["micro_auroc"],
                    "test_macro_f1": test["macro_f1"],
                    "test_micro_f1": test["micro_f1"],
                },
                indent=2,
            )
        )
    elif args.command == "evaluate":
        print(json.dumps(evaluate(args.checkpoint, args.split, args.device), indent=2))
    elif args.command == "tables":
        path = write_history(args.root, config["training"]["reports"])
        print(path.read_text())
