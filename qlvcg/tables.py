"""reports/EXPERIMENT_HISTORY.md, regenerated from the run directories.

Numbers only, and the V0 protocol-selection rule applied mechanically. Interpretation
belongs in the per-stage reports, written after the numbers exist.
"""

import json
from pathlib import Path

from qdg.data import CLASSES

from .models import EXPERIMENTS

ORDER = tuple(EXPERIMENTS)


def collect(root):
    """experiment -> list of (run_dir, test result), full-data runs only."""
    runs = {}
    for path in sorted(Path(root).rglob("best_test_metrics.json")):
        result = json.loads(path.read_text())
        if result.get("experiment") not in EXPERIMENTS:
            continue
        if result["smoke_training"] or result["limited_evaluation"]:
            continue
        runs.setdefault(result["experiment"], []).append((path.parent, result))
    return runs


def _f(value, digits=4):
    return "-" if value is None else f"{value:.{digits}f}"


def _table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def select_v0(results):
    """Master prompt section E: V0-A or V0-B, by validation macro AUROC, fixed thereafter."""
    if "V0A" not in results or "V0B" not in results:
        return None
    score = {name: results[name]["validation_macro_auroc"] for name in ("V0A", "V0B")}
    return max(score, key=score.get), score


def write_history(root, reports):
    runs = collect(root)
    # One seed per experiment at this stage; the latest run of each is reported.
    latest = {name: runs[name][-1] for name in ORDER if name in runs}
    results = {name: result for name, (_, result) in latest.items()}
    overall = []
    for name, (run_dir, result) in latest.items():
        test = result["fixed_0.5"]
        seconds = json.loads((run_dir / "training_time.json").read_text())["seconds"]
        overall.append(
            [
                name,
                EXPERIMENTS[name]["variant"],
                result["seed"],
                f"{result['parameters']['total']:,}",
                f"{result['parameters']['classification_path']:,}",
                result["best_epoch"],
                _f(result["validation_macro_auroc"]),
                _f(result["validation_loss"]),
                _f(result["test_loss"]),
                _f(test["macro_auroc"]),
                _f(test["micro_auroc"]),
                _f(test["macro_f1"]),
                _f(test["micro_f1"]),
                f"{seconds / 60:.1f}",
            ]
        )
    text = [
        "# Experiment history",
        "",
        "PTB-XL superclass task (5 labels, multi-label), official folds 1-8 / 9 / 10,",
        "trained from scratch end to end. Test metrics at a fixed 0.5 threshold; F1 at",
        "validation-selected thresholds is in each run's best_test_metrics.json.",
        "",
        "## Overall",
        "",
        _table(
            [
                "Exp",
                "Variant",
                "Seed",
                "Params (total)",
                "Params (cls path)",
                "Best epoch",
                "Val macro AUROC",
                "Val loss",
                "Test loss",
                "Macro AUROC",
                "Micro AUROC",
                "Macro F1",
                "Micro F1",
                "Train min",
            ],
            overall,
        ),
    ]
    base = results.get("B0")
    if results:
        rows = []
        for label in CLASSES:
            row = [label]
            for name in results:
                cls = results[name]["fixed_0.5"]["per_class"][label]
                row += [_f(cls["auroc"]), _f(cls["f1"])]
                if name != "B0" and base is not None:
                    ref = base["fixed_0.5"]["per_class"][label]
                    row += [f"{cls['auroc'] - ref['auroc']:+.4f}"]
            rows.append(row)
        header = ["Label"]
        for name in results:
            header += [f"{name} AUROC", f"{name} F1"]
            if name != "B0" and base is not None:
                header += [f"{name} dAUROC vs B0"]
        text += ["", "## Per label", "", _table(header, rows)]
    chosen = select_v0(results)
    if chosen:
        protocol, score = chosen
        text += [
            "",
            "## V0 protocol selection",
            "",
            f"Validation macro AUROC: V0A {score['V0A']:.4f}, V0B {score['V0B']:.4f}. "
            f"Rule: the higher validation score is fixed as the V1-V8 protocol -> **{protocol}**. "
            "Training stability is read from each run's history.jsonl before confirming.",
        ]
    path = Path(reports) / "EXPERIMENT_HISTORY.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(text) + "\n", encoding="utf-8")
    return path
