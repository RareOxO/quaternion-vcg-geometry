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
    chosen = select_v0(results)
    reference = results[chosen[0]] if chosen else None
    if results:
        # V0A and V0B are measured against B0; everything after V0 against the V0 run
        # whose objective was selected.
        def baseline(name):
            if name == "B0":
                return None, None
            if name in ("V0A", "V0B"):
                return "B0", results.get("B0")
            return "V0", reference

        header, rows = ["Label"], []
        for name in results:
            header += [f"{name} AUROC", f"{name} F1"]
            label, ref = baseline(name)
            if ref is not None:
                header += [f"{name} dAUROC vs {label}", f"{name} dF1 vs {label}"]
        for cls_name in CLASSES:
            row = [cls_name]
            for name in results:
                cls = results[name]["fixed_0.5"]["per_class"][cls_name]
                row += [_f(cls["auroc"]), _f(cls["f1"])]
                _, ref = baseline(name)
                if ref is not None:
                    other = ref["fixed_0.5"]["per_class"][cls_name]
                    row += [
                        f"{cls['auroc'] - other['auroc']:+.4f}",
                        f"{cls['f1'] - other['f1']:+.4f}",
                    ]
            rows.append(row)
        text += ["", "## Per label", "", _table(header, rows)]
    later = [name for name in results if name not in ("B0", "V0A", "V0B")]
    if reference is not None and later:
        v0_params = reference["parameters"]["total"]
        v1 = results.get("V1")
        rows = []
        for name in later:
            test, ref = results[name]["fixed_0.5"], reference["fixed_0.5"]
            # Section H's key comparison is V2 against V1 as well as against V0.
            against_v1 = (
                f"{test['macro_auroc'] - v1['fixed_0.5']['macro_auroc']:+.4f}"
                if v1 is not None and not name.startswith("V1")
                else "-"
            )
            rows.append(
                [name, EXPERIMENTS[name]["variant"]]
                + [
                    f"{test[key] - ref[key]:+.4f}"
                    for key in ("macro_auroc", "micro_auroc", "macro_f1", "micro_f1")
                ]
                + [against_v1, f"{results[name]['parameters']['total'] - v0_params:+,}"]
            )
        text += [
            "",
            f"## Against V0 ({chosen[0]})",
            "",
            _table(
                [
                    "Exp",
                    "Variant",
                    "dMacro AUROC",
                    "dMicro AUROC",
                    "dMacro F1",
                    "dMicro F1",
                    "dMacro AUROC vs V1",
                    "dParams",
                ],
                rows,
            ),
        ]
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
