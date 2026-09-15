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
    # Representation diagnostic (v2 指导书 §3). Real only, single 20 ms scale, seed 42.
    "R": {"variant": "R"},
    "RA": {"variant": "RA"},
    "RU": {"variant": "RU"},
    "RLA": {"variant": "RLA"},
    # Temporal context ablation (RLA temporal 指导书 §3). The RLA input is identical in
    # all three; the only systematic variable is the receptive field. Widths are solved
    # so the parameter counts stay within 1.3% of RLA-Long, which IS the existing RLA.
    "RLA_short": {"variant": "RLA", "depth": 1, "kernel": 5, "real_width": 127},
    "RLA_medium": {"variant": "RLA", "depth": 2, "kernel": 7, "real_width": 76},
}
MAIN = ("M0", "M1", "M2", "M3", "M4")
# v2 指导书 §8: M0 and M1 are existing anchors, never retrained for this stage.
DIAGNOSTIC = ("M0", "M1", "R", "RA", "RU", "RLA")
DIAGNOSTIC_NEW = ("R", "RA", "RU", "RLA")
DIAGNOSTIC_LABELS = {
    "M0": "XYZ",
    "M1": "Angular [dot,cross]",
    "R": "Radial",
    "RA": "Radial + Angular",
    "RU": "Radial + Absolute Direction",
    "RLA": "Radial + Linear + Angular",
}
# v2 指导书 §8. Each entry is (label, new, old, what the difference asks).
DIAGNOSTIC_DIFFS = (
    ("RA - M1", "RA", "M1", "How much does adding magnitude restore?"),
    ("RU - M0", "RU", "M0", "Cost of the (r, u) reparameterization of raw XYZ"),
    ("RA - RU", "RA", "RU", "Cost of relative angular compression vs absolute direction"),
    ("RLA - RA", "RLA", "RA", "Does linear dynamics recover anything further?"),
    ("RLA - M0", "RLA", "M0", "Gap from the full decomposition to the raw baseline"),
)
# Seed spread of the completed three-seed runs is 0.0002-0.0031 Macro AUROC, so an
# effect under this band is not distinguishable at seed=42 only (v2 指导书 §9).
NOISE_BAND = 0.005

# RLA temporal 指导书 §3/§8. Long is the existing RLA run, never retrained (§7 step 5).
TEMPORAL = (("RLA-Short", "RLA_short"), ("RLA-Medium", "RLA_medium"), ("RLA-Long", "RLA"))
TEMPORAL_NEW = ("RLA_short", "RLA_medium")
TEMPORAL_DIFFS = (
    ("Medium - Short", "RLA_medium", "RLA_short"),
    ("Long - Medium", "RLA", "RLA_medium"),
    ("Long - Short", "RLA", "RLA_short"),
)
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
    """experiment name -> per-seed test results, one entry per completed run.

    A run directory is named `<experiment>_seed<n>`; `qdg train` without --run-name
    appends a timestamp, so a manual run can collide with a suite run of the same
    seed. That is ambiguous rather than wrong, so it is an error -- but the message
    has to name the directories, or there is no way to act on it.
    """
    groups, sources = {}, {}
    for path in sorted(Path(root).rglob("best_test_metrics.json")):
        result = json.loads(path.read_text())
        if result["limited_evaluation"] or result["smoke_training"]:
            continue
        name = path.parent.name.rsplit("_seed", 1)[0]
        groups.setdefault(name, []).append(result)
        sources.setdefault(name, {}).setdefault(result["seed"], []).append(path.parent)
    for name, by_seed in sources.items():
        clashes = {seed: dirs for seed, dirs in by_seed.items() if len(dirs) > 1}
        if clashes:
            detail = "; ".join(
                f"seed {seed}: " + ", ".join(str(d) for d in dirs)
                for seed, dirs in sorted(clashes.items())
            )
            raise ValueError(
                f"{name} has more than one completed run for the same seed ({detail}). "
                "Move or delete the run you do not want summarized, then rerun."
            )
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
    diagnostic_tables(root, summary)
    temporal_tables(root, summary, encoder_stats(root))
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


def interpret(gap):
    """Map the observed differences onto the v2 指导书 §9 decision table.

    Returns (verdict, pattern, next step). The thresholds are NOISE_BAND, which is set
    from the seed spread already measured on the completed three-seed runs; at seed=42
    only there is no per-variant standard deviation to test against, so anything inside
    the band is reported as indistinguishable rather than as an effect.
    """
    band = NOISE_BAND
    ra_m1, ru_m0 = gap.get("RA - M1"), gap.get("RU - M0")
    ra_ru, rla_ra = gap.get("RA - RU"), gap.get("RLA - RA")
    if any(value is None for value in (ra_m1, ru_m0, ra_ru, rla_ra)):
        return "incomplete", "Not every diagnostic variant has a result yet.", "Finish stage one."
    if ru_m0 < -band and ra_m1 < band and rla_ra < band:
        return (
            "not supported",
            "Even RU, an information-preserving reparameterization of raw XYZ, is well "
            "below M0, so the shortfall is not explained by what the representation drops.",
            "Audit normalization/scaling before any new model; do not add structure.",
        )
    if ra_m1 > band and abs(ru_m0) <= band and ra_ru < -band:
        return (
            "partially supported",
            "Magnitude restores part of the angular-only loss and RU matches M0, but RA "
            "stays below RU, so the remaining loss is in the u -> [dot,cross] compression.",
            "Stop elaborating [dot,cross]; study representations that keep absolute direction.",
        )
    if ra_m1 > band and rla_ra <= band:
        return (
            "supported",
            "Adding magnitude alone recovers a large part of the angular-only gap.",
            "Add seeds 43/44 for R, RA, RU; then consider magnitude-aware angular modelling.",
        )
    if ra_m1 <= band and rla_ra > band:
        return (
            "partially supported",
            "Magnitude alone changes little, but linear displacement dynamics recover more, "
            "so the missing information is in the linear rather than the radial term.",
            "Add seeds 43/44 for RLA and a parameter-matched raw control.",
        )
    if ra_m1 <= band and rla_ra <= band and ru_m0 >= -band:
        return (
            "not supported",
            "No diagnostic variant improves beyond the seed noise band, yet RU reproduces "
            "M0, so raw XYZ is not being beaten by anything these blocks expose.",
            "Drop this line; look at raw morphology, phase-specific or robustness questions.",
        )
    return (
        "partially supported",
        "The differences do not match a single pattern in the decision table cleanly.",
        "Inspect the per-class table before committing to a second stage.",
    )


def diagnostic_tables(root, summary):
    """Tables of the v2 指导书 §8/§15, written to diagnostic_tables.md."""
    root = Path(root)
    if not any(name in summary for name in DIAGNOSTIC_NEW):
        return None
    present = [name for name in DIAGNOSTIC if name in summary]
    main = _table(
        ["Model", "Representation", "Params", "Macro AUROC", *CLASSES],
        [
            [
                name,
                DIAGNOSTIC_LABELS[name],
                f"{summary[name]['parameters']:,}",
                _cell(summary[name], "macro_auroc"),
                *[_cell(summary[name], cls) for cls in CLASSES],
            ]
            for name in present
        ],
    )
    gap = {
        label: summary[new]["macro_auroc_mean"] - summary[old]["macro_auroc_mean"]
        for label, new, old, _ in DIAGNOSTIC_DIFFS
        if new in summary and old in summary
    }
    diffs = _table(
        ["Comparison", "Question", "Delta Macro AUROC"],
        [
            [label, question, f"{gap[label]:+.4f}"]
            for label, new, old, question in DIAGNOSTIC_DIFFS
            if label in gap
        ],
    )
    verdict, pattern, nxt = interpret(gap)
    missing = [name for name in DIAGNOSTIC if name not in summary]
    text = (
        "# Representation diagnostic: radial / direction / linear / angular\n\n"
        "Test fold 10, thresholds fixed at 0.5. Stage one is seed 42 only, so no\n"
        "standard deviation is available and differences are read against a\n"
        f"{NOISE_BAND:.4f} noise band taken from the completed three-seed runs.\n"
        "All variants are Real; no QuaternionConv, multi-scale or second-order.\n\n"
        "## Table 8: Diagnostic results\n\n" + main + "\n\n"
        "## Table 9: Component differences\n\n" + diffs + "\n\n"
        "## Verdict\n\n"
        f"**Assumption: {verdict}.** {pattern}\n\n"
        f"Suggested next step: {nxt}\n"
    )
    if missing:
        text += f"\nNot yet trained: {', '.join(missing)}\n"
    (root / "diagnostic_tables.md").write_text(text, encoding="utf-8")
    print(text, flush=True)
    return text


def encoder_stats(root):
    """experiment name -> the receptive field each run actually recorded at train time.

    Read back from environment.json rather than recomputed, so the table reports the
    encoder that produced the number, not whatever the current config would build.
    """
    stats = {}
    for path in sorted(Path(root).rglob("environment.json")):
        environment = json.loads(path.read_text())
        name = path.parent.name.rsplit("_seed", 1)[0]
        samples = environment.get("receptive_field_samples")
        if samples is None:
            continue
        rate = environment.get("sampling_rate", 500)
        stats[name] = {
            "receptive_field_samples": samples,
            "receptive_field_ms": round(1000 * samples / rate),
        }
    return stats


def temporal_tables(root, summary, stats=None):
    """Tables of the RLA temporal 指导书 §8, written to temporal_tables.md.

    The receptive field printed here is measured from the layers the encoder actually
    applies, never asserted from the variant name (§5/§6).
    """
    root = Path(root)
    if not any(name in summary for name in TEMPORAL_NEW):
        return None
    stats = stats or {}
    rows, gap = [], {}
    for label, name in TEMPORAL:
        if name not in summary:
            continue
        field = stats.get(name, {})
        rows.append(
            [
                label,
                str(field.get("receptive_field_samples", "?")),
                f"{field.get('receptive_field_ms', '?')}",
                f"{summary[name]['parameters']:,}",
                _cell(summary[name], "macro_auroc"),
                *[_cell(summary[name], cls) for cls in CLASSES],
            ]
        )
    main = _table(["Model", "RF (samples)", "RF (ms)", "Params", "Macro AUROC", *CLASSES], rows)
    for label, new, old in TEMPORAL_DIFFS:
        if new in summary and old in summary:
            gap[label] = summary[new]["macro_auroc_mean"] - summary[old]["macro_auroc_mean"]
    diffs = _table(
        ["Comparison", "Delta Macro AUROC"],
        [[label, f"{value:+.4f}"] for label, value in gap.items()],
    )
    per_class = ""
    if "RLA" in summary and "RLA_short" in summary:
        per_class = _table(
            ["Class", "Long - Short"],
            [
                [cls, f"{summary['RLA'][f'{cls}_mean'] - summary['RLA_short'][f'{cls}_mean']:+.4f}"]
                for cls in CLASSES
            ],
        )
    verdict = interpret_temporal(gap)
    text = (
        "# RLA temporal context ablation\n\n"
        "The RLA input is fixed and identical in all three models; the only systematic\n"
        "variable is the receptive field, measured from the encoder's actual layers.\n"
        f"Seed 42 only, so differences are read against a {NOISE_BAND:.4f} noise band.\n\n"
        "## Table 10: Temporal context results\n\n" + main + "\n\n"
        "## Table 11: Context differences\n\n"
        + diffs
        + "\n\n"
        + ("## Table 12: Per-class Long - Short\n\n" + per_class + "\n\n" if per_class else "")
        + "## Verdict\n\n"
        + verdict
        + "\n"
    )
    missing = [name for _, name in TEMPORAL if name not in summary]
    if missing:
        text += f"\nNot yet trained: {', '.join(missing)}\n"
    (root / "temporal_tables.md").write_text(text, encoding="utf-8")
    print(text, flush=True)
    return text


def interpret_temporal(gap):
    """The §9 decision table for the temporal ablation."""
    medium_short, long_medium = gap.get("Medium - Short"), gap.get("Long - Medium")
    long_short = gap.get("Long - Short")
    if long_short is None or medium_short is None or long_medium is None:
        return "**Incomplete.** Train RLA-Short and RLA-Medium before reading a verdict."
    band = NOISE_BAND
    if long_short > band and medium_short > 0 and long_medium > 0:
        return (
            "**Longer temporal context helps.** Short < Medium < Long and the total gain "
            f"({long_short:+.4f}) clears the {band:.3f} noise band, so temporal evolution "
            "carries diagnostic information beyond the instantaneous R/L/A dynamics.\n\n"
            "Next: add seeds for the Short/Long contrast, then design temporal architecture."
        )
    if medium_short > band and long_medium <= 0:
        return (
            f"**A bounded useful range.** Medium gains {medium_short:+.4f} over Short but "
            f"Long adds {long_medium:+.4f} on top, so the value saturates rather than "
            "growing with context.\n\nNext: design around the Medium context, not longer."
        )
    if abs(long_short) <= band:
        return (
            f"**Not supported.** Long - Short is {long_short:+.4f}, inside the "
            f"{band:.3f} noise band, so at seed 42 the three contexts are "
            "indistinguishable.\n\nNext: stop elaborating temporal architecture; "
            "re-examine the representation or the task."
        )
    return (
        f"**Inconclusive.** Long - Short is {long_short:+.4f} but the ordering is not "
        "monotone, so seed 42 alone does not support a temporal claim.\n\n"
        "Next: add seeds only if the gap sits near the noise boundary."
    )
