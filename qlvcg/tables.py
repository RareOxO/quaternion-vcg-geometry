"""reports/EXPERIMENT_HISTORY.md, regenerated from the run directories.

Numbers only. Interpretation belongs in the per-stage reports, written after the
numbers exist.

Who is compared with whom: V0 against B0; every Quaternion variant against V0; and from
V2 on, also against V1, which section H names as the key comparison. An experiment
whose configuration is identical to an earlier one declares ``reuses`` in the registry
and is read from that run rather than trained twice.
"""

import json
from pathlib import Path

from qdg.data import CLASSES

from .models import EXPERIMENTS, LEGACY_NAMES

ORDER = tuple(EXPERIMENTS)
METRICS = ("macro_auroc", "micro_auroc", "macro_f1", "micro_f1")


def collect(root):
    """experiment -> list of (run_dir, test result), full-data runs only."""
    runs = {}
    for path in sorted(Path(root).rglob("best_test_metrics.json")):
        result = json.loads(path.read_text())
        name = LEGACY_NAMES.get(result.get("experiment"), result.get("experiment"))
        if name not in EXPERIMENTS or EXPERIMENTS[name].get("reuses"):
            continue
        if result["smoke_training"] or result["limited_evaluation"]:
            continue
        runs.setdefault(name, []).append((path.parent, result))
    for name, spec in EXPERIMENTS.items():
        if spec.get("reuses") in runs:
            runs[name] = runs[spec["reuses"]]
    return runs


def _f(value, digits=4):
    return "-" if value is None else f"{value:.{digits}f}"


def _table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def _baseline(name, results):
    """(label, result) this experiment's per-label deltas are taken against."""
    if name == "B0":
        return None, None
    if name == "V0":
        return "B0", results.get("B0")
    return "V0", results.get("V0")


def write_history(root, reports):
    runs = collect(root)
    # One seed per experiment at this stage; the latest run of each is reported.
    latest = {name: runs[name][-1] for name in ORDER if name in runs}
    results = {name: result for name, (_, result) in latest.items()}

    overall = []
    for name, (run_dir, result) in latest.items():
        test = result["fixed_0.5"]
        seconds = json.loads((run_dir / "training_time.json").read_text())["seconds"]
        reused = EXPERIMENTS[name].get("reuses")
        overall.append(
            [
                name if not reused else f"{name} (= {reused})",
                EXPERIMENTS[name]["variant"],
                result["seed"],
                f"{result['parameters']['total']:,}",
                f"{result['parameters']['classification_path']:,}",
                result["best_epoch"],
                _f(result["validation_macro_auroc"]),
                _f(result["validation_loss"]),
                _f(result["test_loss"]),
                *[_f(test[key]) for key in METRICS],
                f"{seconds / 60:.1f}",
            ]
        )
    text = [
        "# Experiment history",
        "",
        "PTB-XL superclass task (5 labels, multi-label), official folds 1-8 / 9 / 10,",
        "trained from scratch end to end with the classification loss only. Test metrics at a",
        "fixed 0.5 threshold; F1 at validation-selected thresholds is in each run's",
        "best_test_metrics.json.",
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

    if results:
        header, rows = ["Label"], []
        for name in results:
            header += [f"{name} AUROC", f"{name} F1"]
            label, reference = _baseline(name, results)
            if reference is not None:
                header += [f"{name} dAUROC vs {label}", f"{name} dF1 vs {label}"]
        for cls_name in CLASSES:
            row = [cls_name]
            for name in results:
                cls = results[name]["fixed_0.5"]["per_class"][cls_name]
                row += [_f(cls["auroc"]), _f(cls["f1"])]
                _, reference = _baseline(name, results)
                if reference is not None:
                    other = reference["fixed_0.5"]["per_class"][cls_name]
                    row += [
                        f"{cls['auroc'] - other['auroc']:+.4f}",
                        f"{cls['f1'] - other['f1']:+.4f}",
                    ]
            rows.append(row)
        text += ["", "## Per label", "", _table(header, rows)]

    v0, v1 = results.get("V0"), results.get("V1")
    later = [name for name in results if name not in ("B0", "V0")]
    if v0 is not None and later:
        rows = []
        for name in later:
            test, reference = results[name]["fixed_0.5"], v0["fixed_0.5"]
            against_v1 = (
                f"{test['macro_auroc'] - v1['fixed_0.5']['macro_auroc']:+.4f}"
                if v1 is not None and not name.startswith("V1")
                else "-"
            )
            rows.append(
                [name, EXPERIMENTS[name]["variant"]]
                + [f"{test[key] - reference[key]:+.4f}" for key in METRICS]
                + [
                    against_v1,
                    f"{results[name]['parameters']['total'] - v0['parameters']['total']:+,}",
                ]
            )
        text += [
            "",
            "## Against V0",
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

    path = Path(reports) / "EXPERIMENT_HISTORY.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(text) + "\n", encoding="utf-8")
    return path
