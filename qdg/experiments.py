"""The experiment registry and the four result tables of 方案 §10.

Every row of every table comes from a run in this registry, so the tables are
generated, never typed by hand. M2 is also the 20 ms row of Table 2 and the
Quaternion Conv row of Table 3; it is trained once and reused.
"""

import csv
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from .data import CLASSES, load_manifest, save_json
from .engine import evaluate, setup, train
from .models import FUSION_PAIRS, build_model

# name -> model-config overrides applied on top of the base YAML.
EXPERIMENTS = {
    "M0": {"variant": "M0"},
    "M1": {"variant": "M1"},
    "M2": {"variant": "M2"},
    "M3": {"variant": "M3"},
    "M4": {"variant": "M4"},
    "M2_s10": {"variant": "M2", "scales_ms": [10]},
    "M2_s40": {"variant": "M2", "scales_ms": [40]},
    "M2_s80": {"variant": "M2", "scales_ms": [80]},
    "M1_mlp": {"variant": "M1", "operator": "mlp"},
    # Fusion ladder (双分支方案). M0_wide is the capacity control: raw XYZ only, width
    # solved so its parameter count lands within 0.1% of F1 and 2.1% of F4 (§10).
    "M0_wide": {"variant": "M0", "real_width": 92},
    "F1": {"variant": "F1"},
    "F2": {"variant": "F2"},
    "F3": {"variant": "F3"},
    "F4": {"variant": "F4"},
}
MAIN = ("M0", "M1", "M2", "M3", "M4")
FUSION = ("M0", "M0_wide", "F1", "F2", "F3", "F4")
# 双分支方案 §12: train M0 / M0_wide / F1 first and stop for a decision on F1.
FUSION_STAGE_A = ("M0", "M0_wide", "F1")
FUSION_LABELS = {
    "M0": "M0 Raw XYZ",
    "M0_wide": "M0-Wide (capacity control)",
    "F1": "F1 Raw + Real Geo",
    "F2": "F2 Raw + Quat Geo",
    "F3": "F3 Raw + Quat MS",
    "F4": "F4 Raw + Full Geo",
}
# Raw / Geometry / Quaternion / Multi-scale / 2nd-order
FUSION_MARKS = {
    "M0": ("v", "x", "x", "x", "x"),
    "M0_wide": ("v", "x", "x", "x", "x"),
    "F1": ("v", "v", "x", "x", "x"),
    "F2": ("v", "v", "v", "x", "x"),
    "F3": ("v", "v", "v", "v", "x"),
    "F4": ("v", "v", "v", "v", "v"),
}
# 双分支方案 §16. The old M1-M0 is explicitly NOT an "added geometry" gain.
FUSION_GAINS = (
    ("F1 - M0", "Real geometry complementarity", "F1", "M0"),
    ("F1 - M0-Wide", "Geometry vs extra capacity", "F1", "M0_wide"),
    ("F2 - F1", "Quaternion-specific increment", "F2", "F1"),
    ("F3 - F2", "Multi-scale increment", "F3", "F2"),
    ("F4 - F3", "Second-order increment", "F4", "F3"),
)
SCALE_ROWS = (
    ("10 ms", "M2_s10"),
    ("20 ms", "M2"),
    ("40 ms", "M2_s40"),
    ("80 ms", "M2_s80"),
    ("10+20+40+80 ms", "M3"),
)
OPERATOR_ROWS = (
    ("dot+cross", "Real Conv", "M1"),
    ("dot+cross", "Real MLP", "M1_mlp"),
    ("dot+cross", "Quaternion Conv", "M2"),
)
COMPONENT_ROWS = (
    ("Explicit Geometry", "M1", "M0"),
    ("Quaternion Interaction", "M2", "M1"),
    ("Multi-scale", "M3", "M2"),
    ("Second-order", "M4", "M3"),
)
MARKS = {
    "M0": ("x", "x", "x", "x"),
    "M1": ("v", "x", "x", "x"),
    "M2": ("v", "v", "x", "x"),
    "M3": ("v", "v", "v", "x"),
    "M4": ("v", "v", "v", "v"),
}


def experiment_config(config, name):
    if name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment {name}; choose from {sorted(EXPERIMENTS)}")
    current = deepcopy(config)
    current["model"].update(EXPERIMENTS[name])
    return current


def profile(config):
    """Parameter counts and output shapes for every registered experiment."""
    manifest = load_manifest(config["data"])
    device = setup(config["training"])
    rows = []
    for name in EXPERIMENTS:
        current = experiment_config(config, name)
        model = build_model(current["model"], manifest["stats"]).to(device).eval()
        with torch.inference_mode():
            logits = model(torch.zeros(1, 12, manifest["stats"]["signal_length"], device=device))
        if not torch.isfinite(logits).all():
            raise FloatingPointError(f"Nonfinite {name} logits")
        rows.append(
            {
                "experiment": name,
                "parameters": sum(p.numel() for p in model.parameters()),
                **model.describe(),
                "output_shape": list(logits.shape),
            }
        )
        del model
    return rows


def suite(config, names, seeds):
    if len(set(seeds)) != len(seeds) or len(set(names)) != len(names):
        raise ValueError("Suite experiments and seeds must be unique")
    for name in names:
        for seed in seeds:
            current = experiment_config(config, name)
            current["training"]["seed"] = seed
            run_dir = Path(current["training"]["output"]) / f"{name}_seed{seed}"
            if (run_dir / "best_test_metrics.json").exists():
                print(f"skip completed {run_dir.name}", flush=True)
                continue
            if run_dir.exists():
                raise FileExistsError(f"Incomplete run {run_dir}; remove it before rerunning")
            evaluate(train(current, run_name=run_dir.name) / "best.pt")
    return tables(config["training"]["output"])


def collect(root):
    """experiment name -> per-seed test results, one entry per completed run."""
    groups = {}
    for path in sorted(Path(root).rglob("best_test_metrics.json")):
        result = json.loads(path.read_text())
        if result["limited_evaluation"] or result["smoke_training"]:
            continue
        name = path.parent.name.rsplit("_seed", 1)[0]
        groups.setdefault(name, []).append(result)
    for name, results in groups.items():
        seeds = [result["seed"] for result in results]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"Duplicate seeds for {name}: {seeds}")
    return groups


def aggregate(results):
    """Mean and std over seeds of macro and per-class AUROC at fixed thresholds."""
    row = {"n_seeds": len(results), "seeds": sorted(r["seed"] for r in results)}
    row["parameters"] = results[0]["parameters"]
    values = {"macro_auroc": [], "macro_auprc": []}
    for name in CLASSES:
        values[name] = []
    for result in results:
        metrics = result["fixed_0.5"]
        values["macro_auroc"].append(metrics["macro_auroc"])
        values["macro_auprc"].append(metrics["macro_auprc"])
        for name in CLASSES:
            values[name].append(metrics["per_class"][name]["auroc"])
    for key, samples in values.items():
        if any(sample is None for sample in samples):
            raise ValueError(f"Undefined AUROC for {key}")
        row[f"{key}_mean"] = float(np.mean(samples))
        row[f"{key}_std"] = float(np.std(samples, ddof=1)) if len(samples) > 1 else None
    return row


def _cell(row, key):
    mean, std = row[f"{key}_mean"], row[f"{key}_std"]
    return f"{mean:.4f}" if std is None else f"{mean:.4f}±{std:.4f}"


def _table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def tables(root):
    """Write tables.md / results.csv from every completed full-data run under root."""
    root = Path(root)
    groups = collect(root)
    if not groups:
        raise ValueError(f"No completed full-data test evaluations under {root}")
    summary = {name: aggregate(results) for name, results in groups.items()}
    missing = lambda names: [name for name in names if name not in summary]  # noqa: E731
    sections = []

    present = [name for name in MAIN if name in summary]
    sections.append(
        "## Table 1: PTB-XL main results\n\n"
        + _table(
            [
                "Model",
                "Geometry",
                "Quaternion",
                "Multi-scale",
                "2nd-order",
                "Macro AUROC",
                *CLASSES,
            ],
            [
                [name, *MARKS[name], _cell(summary[name], "macro_auroc")]
                + [_cell(summary[name], cls) for cls in CLASSES]
                for name in present
            ],
        )
        + (f"\n\nNot yet trained: {', '.join(missing(MAIN))}" if missing(MAIN) else "")
    )

    scale_names = [name for _, name in SCALE_ROWS]
    sections.append(
        "## Table 2: Temporal-scale analysis\n\n"
        + _table(
            ["Scale", "Macro AUROC", *CLASSES],
            [
                [label, _cell(summary[name], "macro_auroc")]
                + [_cell(summary[name], cls) for cls in CLASSES]
                for label, name in SCALE_ROWS
                if name in summary
            ],
        )
        + (
            f"\n\nNot yet trained: {', '.join(missing(scale_names))}"
            if missing(scale_names)
            else ""
        )
    )

    sections.append(
        "## Table 3: Real vs Quaternion on identical input\n\n"
        + _table(
            ["Representation", "Operator", "Params", "Macro AUROC"],
            [
                [
                    representation,
                    operator,
                    f"{summary[name]['parameters']:,}",
                    _cell(summary[name], "macro_auroc"),
                ]
                for representation, operator, name in OPERATOR_ROWS
                if name in summary
            ],
        )
    )

    deltas = [
        [label, f"{summary[new]['macro_auroc_mean'] - summary[old]['macro_auroc_mean']:+.4f}"]
        for label, new, old in COMPONENT_ROWS
        if new in summary and old in summary
    ]
    sections.append(
        "## Table 4: Component increments\n\n"
        + _table(["Component Added", "Delta Macro AUROC"], deltas)
    )

    text = (
        "# Quaternion-VCG dynamic geometry: PTB-XL results\n\n"
        "Test fold 10, thresholds fixed at 0.5, mean+-std over seeds.\n"
        "Geometry/Quaternion/Multi-scale/2nd-order: v = present, x = absent.\n\n"
        + "\n\n".join(sections)
        + "\n"
    )
    (root / "tables.md").write_text(text, encoding="utf-8")
    save_json(root / "results.json", summary)
    rows = [{"experiment": name, **row} for name, row in sorted(summary.items())]
    with (root / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(text, flush=True)
    fusion_tables(root, summary)
    return summary


def fusion_tables(root, summary):
    """Tables 5-7 of the 双分支 addendum (§14-16), written to fusion_tables.md.

    Kept in their own file so tables.md -- the M0-M4 result of the first report --
    stays byte-stable and reproducible.
    """
    root = Path(root)
    present = [name for name in FUSION if name in summary]
    if not any(name in summary for name in ("M0_wide", *FUSION_PAIRS)):
        return None
    baseline = summary.get("M0")

    def delta(name):
        if baseline is None or name == "M0":
            return "--"
        return f"{summary[name]['macro_auroc_mean'] - baseline['macro_auroc_mean']:+.4f}"

    main = _table(
        [
            "Model",
            "Raw XYZ",
            "Geometry",
            "Quaternion",
            "Multi-scale",
            "2nd-order",
            "Params",
            "Macro AUROC",
            "Delta vs M0",
        ],
        [
            [
                FUSION_LABELS[name],
                *FUSION_MARKS[name],
                f"{summary[name]['parameters']:,}",
                _cell(summary[name], "macro_auroc"),
                delta(name),
            ]
            for name in present
        ],
    )
    per_class = _table(
        ["Model", "Macro AUROC", *CLASSES],
        [
            [FUSION_LABELS[name], _cell(summary[name], "macro_auroc")]
            + [_cell(summary[name], cls) for cls in CLASSES]
            for name in present
        ],
    )
    gains = _table(
        ["Comparison", "Meaning", "Delta Macro AUROC"],
        [
            [
                label,
                meaning,
                f"{summary[new]['macro_auroc_mean'] - summary[old]['macro_auroc_mean']:+.4f}",
            ]
            for label, meaning, new, old in FUSION_GAINS
            if new in summary and old in summary
        ],
    )
    missing = [name for name in FUSION if name not in summary]
    text = (
        "# Raw VCG + Geometry fusion: PTB-XL results\n\n"
        "Test fold 10, thresholds fixed at 0.5, mean+-std over seeds.\n"
        "F_n = M0 raw branch + M_n geometry branch, joined by late concat.\n"
        "M0-Wide is raw XYZ only, widened to match the fusion parameter count.\n\n"
        "## Table 5: Fusion main results\n\n" + main + "\n\n"
        "## Table 6: Per-class AUROC\n\n" + per_class + "\n\n"
        "## Table 7: Component gains\n\n" + gains + "\n\n"
        "Note: the earlier M1 - M0 is a geometry-only replacement gap, not an\n"
        "added-geometry gain. The added-geometry effect is F1 - M0, and F1 - M0-Wide\n"
        "is the part that survives the capacity control.\n"
    )
    if missing:
        text += f"\nNot yet trained: {', '.join(missing)}\n"
    (root / "fusion_tables.md").write_text(text, encoding="utf-8")
    print(text, flush=True)
    return text
