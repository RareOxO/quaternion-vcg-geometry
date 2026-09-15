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
from .encoders import ENCODERS, QUATERNION_ENCODERS, QUATERNION_PAIR
from .engine import evaluate, setup, train
from .models import ANGULAR_BLOCKS, FUSION_PAIRS, SINGLE_VARIANTS, build_model
from .quaternion_nn import receptive_field_samples

# E* frozen for every later experiment (plan section 2.2). This constant is what the
# later registries build against; `select_encoder` independently recomputes the
# pre-specified rule from the runs, so the two can be compared.
SELECTED_ENCODER = "lstm_attention"
# The pre-specified rule (validation Macro AUROC, then AUPRC, then fewer parameters)
# ranked QGNN first at 0.9146 against LSTM + Attention at 0.9117. E* was set by hand
# instead, and that deviation has to travel with the results rather than sit in a commit
# message: the benchmark table prints this note whenever E* is not the rule's top pick.
SELECTION_NOTE = (
    "E* was set by hand rather than taken from the pre-specified rule. The rule's top "
    "pick led the chosen encoder by 0.0029 validation Macro AUROC, inside the 0.005 "
    "screening band and below the spread of a benchmark run across three different "
    "GPUs, so the ranking at the top is not established; the chosen encoder is generic "
    "and holds a third of the parameters. This is a deviation and must be reported as "
    "one: a clean selection needs Experiment 1 rerun on one machine with several seeds."
)
# Experiment 3 keeps R, L and the operator fixed and varies only the angular
# representation. A quaternion E* cannot take the 3-vector U at all, so it would force a
# generic operator here; a generic E* needs no such deviation and lets the Q row reuse
# the factorial's R+L+Q run.
REPRESENTATION_ENCODER = QUATERNION_PAIR.get(SELECTED_ENCODER, SELECTED_ENCODER)
REPRESENTATION_REUSES_FACTORIAL = REPRESENTATION_ENCODER == SELECTED_ENCODER

# Experiment 4: accessible temporal-context ablation (plan section 5, as redefined).
# A stem of stride 5 with one pooling gives 20 ms per step, which is what puts the
# shortest level within reach; every level shares it, so the sweep is internally
# controlled even though this stem is finer than the one Experiments 1-3 use.
CONTEXT_STEM = {"stride": 5, "poolings": 1}
CONTEXT_STEP_MS = 20
# (label, accessible context in ms). None is the unrestricted upper anchor: the same
# model with no state reset, which says whether 1280 ms has already saturated.
CONTEXT_LEVELS = (
    ("20ms", 20),
    ("40ms", 40),
    ("80ms", 80),
    ("160ms", 160),
    ("320ms", 320),
    ("640ms", 640),
    ("1280ms", 1280),
    ("full", None),
)
CONTEXT_SCALE = {
    20: "local",
    40: "local",
    80: "local",
    160: "phase-scale",
    320: "phase-scale",
    640: "cycle-scale",
    1280: "cycle-scale",
    None: "unrestricted",
}
CONTEXT = tuple(f"E4_{label}" for label, _ in CONTEXT_LEVELS)


# Experiment 5 (plan section 6): handcrafted summaries against learned dynamics.
# A, B and C are classical arms run by qdg.classical; D is the proposed model itself,
# reused; E concatenates D's embedding with all three handcrafted sets.
HANDCRAFTED_ARMS = {
    "E5_stat": ("A  statistical aggregation", ("stat",)),
    "E5_velocity": ("B  Cruces-2016-inspired velocity summary", ("velocity",)),
    "E5_biomarker": ("C  Cruces-2020-inspired dynamic biomarkers", ("biomarker",)),
    "E5_hybrid": ("E  Proposed + handcrafted", ("stat", "velocity", "biomarker")),
}
HANDCRAFTED_NEW = tuple(HANDCRAFTED_ARMS)
HANDCRAFTED_PROPOSED = "E2_RLQ"
HANDCRAFTED = (
    ("A  statistical aggregation", "E5_stat"),
    ("B  Cruces-2016-inspired velocity summary", "E5_velocity"),
    ("C  Cruces-2020-inspired dynamic biomarkers", "E5_biomarker"),
    ("D  Proposed full-sequence learning", HANDCRAFTED_PROPOSED),
    ("E  Proposed + handcrafted", "E5_hybrid"),
)

# Experiment 6 (plan section 7): local and long context together. The two scales come
# from the Experiment 4 sweep under its pre-specified reading -- local at the end of the
# local group, long at the point where the gain saturates -- and NOT from inspecting the
# test set. Local-only and long-only are the corresponding Experiment 4 runs, reused.
LOCAL_CONTEXT_MS, LONG_CONTEXT_MS = 80, 640
COMBINED = "E6_local_long"
LOCAL_LONG = (
    (f"Local only ({LOCAL_CONTEXT_MS} ms)", f"E4_{LOCAL_CONTEXT_MS}ms"),
    (f"Long only ({LONG_CONTEXT_MS} ms)", f"E4_{LONG_CONTEXT_MS}ms"),
    ("Local + Long", COMBINED),
)

# Experiment 7: temporal ordering (plan section 8). Original is E2_RLQ, reused. Block
# shuffles keep local dynamics and break longer-range organisation; the point-wise
# shuffle breaks both. Each is run twice: permuting the Q branch alone, and permuting
# all three with ONE shared permutation so cross-component alignment survives.
SHUFFLE_LEVELS = (("block40", 40), ("block80", 80), ("block160", 160), ("full", None))
SHUFFLE_SCOPES = (("q", "angular"), ("joint", "all"))
ORDERING = tuple(
    f"E7_{scope}_{label}" for scope, _ in SHUFFLE_SCOPES for label, _ in SHUFFLE_LEVELS
)
ORDERING_ORIGINAL = "E2_RLQ"
SHUFFLE_LABELS = {
    "block40": "Block shuffle 40 ms",
    "block80": "Block shuffle 80 ms",
    "block160": "Block shuffle 160 ms",
    "full": "Point-wise shuffle",
}
SCOPE_LABELS = {"q": "Q only", "joint": "Joint R/L/Q"}

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
    # Angular temporal operator (Angular temporal 指导书 §6). Branched RLA so the angular
    # encoder can be swapped alone; widths solved so the two hold matched weight counts.
    "RLA_standard": {
        "variant": "RLAB",
        "angular_algebra": "standard",
        "angular_quaternions": 22,
        "branch_width": 32,
    },
    "RLA_quaternion": {
        "variant": "RLAB",
        "angular_algebra": "quaternion",
        "angular_quaternions": 22,
        "branch_width": 32,
    },
    # Experiment 1: temporal encoder benchmark (R/L/Q Long-Context plan section 2).
    # R + L + Q with Q frozen as the rotation quaternion; only the encoder changes.
    **{
        f"E1_{name}": {
            "variant": "RLAB",
            "encoder": name,
            "angular_algebra": "standard",
            "angular_blocks": ["rotation"],
            "angular_quaternions": 22,
            "branch_width": 32,
        }
        for name in ENCODERS
    },
    # Experiment 2: the 2^3-1 R/L/Q factorial plus a raw XYZ reference, on the frozen
    # E*. Experiment 3: the same R + L + <angular> model with the angular representation
    # swapped for absolute direction (U) or the full-angle descriptor (D); R+L+Q is
    # E2_RLQ and is never retrained.
    **{
        f"E2_{name}": {
            "variant": "RLAB",
            "encoder": SELECTED_ENCODER,
            "angular_algebra": "standard",
            "angular_blocks": ["rotation"],
            "angular_quaternions": 22,
            "branch_width": 32,
            "branches": branches,
        }
        for name, branches in {
            "R": ["radial"],
            "L": ["linear"],
            "Q": ["angular"],
            "RL": ["radial", "linear"],
            "RQ": ["radial", "angular"],
            "LQ": ["linear", "angular"],
            "RLQ": ["radial", "linear", "angular"],
            "raw": ["raw"],
        }.items()
    },
    # Experiment 3 uses the GENERIC counterpart of E* on the angular branch, for all
    # three representations. The absolute direction U is a 3-vector and cannot enter a
    # quaternion operator at all, and the plan forbids padding a branch merely to fit
    # one (section 2.1). Holding the operator generic across U, D and Q is what makes
    # "only the angular representation changes" literally true.
    **{
        f"E3_{name}": {
            "variant": "RLAB",
            "encoder": REPRESENTATION_ENCODER,
            "angular_algebra": "standard",
            "angular_blocks": [block],
            "angular_quaternions": 22,
            "branch_width": 32,
            "branches": ["radial", "linear", "angular"],
        }
        for name, block in (("U", "direction"), ("D", "angular"), ("Q", "rotation"))
        if not (name == "Q" and REPRESENTATION_REUSES_FACTORIAL)
    },
    # Experiment 4: accessible temporal-context ablation. E* stays LSTM + Attention and
    # its architecture is untouched; what changes is how much time the recurrence may
    # integrate, set by resetting the recurrent state every `context_ms`. A finer stem
    # than the other experiments use puts the 20 ms floor within reach.
    **{
        f"E4_{label}": {
            "variant": "RLAB",
            "encoder": SELECTED_ENCODER,
            "angular_algebra": "standard",
            "angular_blocks": ["rotation"],
            "angular_quaternions": 22,
            "branch_width": 32,
            "branches": ["radial", "linear", "angular"],
            "stem": CONTEXT_STEM,
            "context_ms": context,
        }
        for label, context in CONTEXT_LEVELS
    },
    # Experiment 6: both scales at once. Two encoders per branch, one restricted to the
    # local context and one to the long context, concatenated. The width is solved so the
    # total parameter count matches the single-scale runs it is compared against.
    COMBINED: {
        "variant": "RLAB",
        "encoder": SELECTED_ENCODER,
        "angular_algebra": "standard",
        "angular_blocks": ["rotation"],
        "angular_quaternions": 22,
        "branch_width": 32,
        "branches": ["radial", "linear", "angular"],
        "stem": CONTEXT_STEM,
        "context_ms": [LOCAL_CONTEXT_MS, LONG_CONTEXT_MS],
        # Two encoders per branch, so the width is solved to land within 1.1% of the
        # single-scale runs rather than 7% above them.
        "width_override": 31,
    },
    # Experiment 7: temporal ordering. Identical to E2_RLQ except that the feature
    # sequence is permuted before the encoder.
    **{
        f"E7_{scope}_{label}": {
            "variant": "RLAB",
            "encoder": SELECTED_ENCODER,
            "angular_algebra": "standard",
            "angular_blocks": ["rotation"],
            "angular_quaternions": 22,
            "branch_width": 32,
            "branches": ["radial", "linear", "angular"],
            "shuffle": {"scope": target, "block_ms": block},
        }
        for scope, target in SHUFFLE_SCOPES
        for label, block in SHUFFLE_LEVELS
    },
    # Temporal evolution of the angular representation (Temporal evolution 方案 §3).
    # All three use the SAME plain real temporal encoder; only the angular input changes.
    **{
        name: {
            "variant": "RLAB",
            "angular_algebra": "standard",
            "angular_blocks": blocks,
            "angular_quaternions": 22,
            "branch_width": 32,
        }
        for name, blocks in ANGULAR_BLOCKS.items()
    },
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
# Angular temporal 指导书 §6/§7. Both are trained: the single-encoder RLA-Long has no
# separable angular encoder, so it does not meet the §6 reuse condition.
ANGULAR = (("RLA-Standard", "RLA_standard"), ("RLA-Quaternion", "RLA_quaternion"))
ANGULAR_NEW = ("RLA_standard", "RLA_quaternion")

# Experiment 1 of the R/L/Q Long-Context plan. E* is chosen on validation Macro AUROC,
# then Macro AUPRC, then the smaller parameter count (section 2.2).
BENCHMARK = tuple(f"E1_{name}" for name in ENCODERS)
BENCHMARK_LABELS = {
    "lstm": "LSTM",
    "gru": "GRU",
    "lstm_attention": "LSTM + Attention",
    "tcn": "TCN",
    "tcn_attention": "TCN + Attention",
    "transformer": "Transformer",
    "qlstm": "QLSTM",
    "qtcn": "QTCN",
    "qtransformer": "Q-Transformer",
    "qgnn": "QGNN",
}

# Experiment 2 (plan section 3): conditional contributions come from the differences
# RLQ - RL, RLQ - RQ and RLQ - LQ, which is why the full factorial is run.
FACTORIAL = ("E2_R", "E2_L", "E2_Q", "E2_RL", "E2_RQ", "E2_LQ", "E2_RLQ")
FACTORIAL_REFERENCE = "E2_raw"
FACTORIAL_LABELS = {
    "E2_R": "R",
    "E2_L": "L",
    "E2_Q": "Q",
    "E2_RL": "R+L",
    "E2_RQ": "R+Q",
    "E2_LQ": "L+Q",
    "E2_RLQ": "R+L+Q",
    "E2_raw": "Raw XYZ (reference)",
}
FACTORIAL_CONTRIBUTIONS = (
    ("Q", "E2_RLQ", "E2_RL", "rotational dynamics on top of R+L"),
    ("L", "E2_RLQ", "E2_RQ", "linear motion on top of R+Q"),
    ("R", "E2_RLQ", "E2_LQ", "magnitude on top of L+Q"),
)

# Experiment 3 (plan section 4): R and L, E*, budget and protocol frozen; only the
# angular representation changes. R+L+Q is E2_RLQ, reused rather than retrained.
# Names Experiment 3 has to train. The Q row is the factorial's R+L+Q whenever E* is
# generic, so it is reused rather than retrained.
REPRESENTATION_NEW = (
    ("E3_U", "E3_D")
    if REPRESENTATION_REUSES_FACTORIAL
    else (
        "E3_U",
        "E3_D",
        "E3_Q",
    )
)
REPRESENTATIONS = (
    ("R+L+U  absolute unit direction", "E3_U", "u_t"),
    ("R+L+D  full-angle dot/cross", "E3_D", "[cos t, n sin t]"),
    (
        "R+L+Q  rotation quaternion",
        "E2_RLQ" if REPRESENTATION_REUSES_FACTORIAL else "E3_Q",
        "[cos(t/2), n sin(t/2)]",
    ),
)

# Angular representation 指导书 §2. Both runs already exist: the baseline is the
# full-angle control of the operator round, and the new variant is the registry's A0,
# whose angular branch is the 4-channel half-angle rotation quaternion. Nothing is
# retrained -- this round only formalises the comparison (§1, §9).
REPRESENTATION = (
    ("Baseline: full-angle [dot, cross]", "RLA_standard", "[cos t, n sin t]"),
    ("New: half-angle rotation quaternion", "A0", "[cos(t/2), n sin(t/2)]"),
)
# The recorded matched seed-42 baseline, used only when its run directory is absent.
BASELINE_MACRO_AUROC = 0.9130

# Temporal evolution 方案 §3. A1 vs A2 is the question; A0 is the local-only floor.
EVOLUTION = (
    ("A0 Local only", "A0", "q_t"),
    ("A1 Real evolution", "A1", "q_t + (q_{t+tau} - q_t)"),
    ("A2 Quaternion evolution", "A2", "q_t + (q_t^-1 (x) q_{t+tau})"),
)
EVOLUTION_NEW = tuple(name for _, name, _ in EVOLUTION)
EVOLUTION_DIFFS = (
    ("A2 - A1", "A2", "A1", "Quaternion-specific rotation composition vs plain difference"),
    ("A1 - A0", "A1", "A0", "Does temporal evolution add anything over local state?"),
    ("A2 - A0", "A2", "A0", "Total gain of quaternion evolution over local state"),
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
    for key in ("classifier_parameters", "frozen_parameters", "settings"):
        if key in results[0]:
            row[key] = results[0][key]
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
    # Union of every row's keys, in first-seen order: the classical arms of Experiment 5
    # carry fields a neural run does not, and taking the header from the first row alone
    # fails as soon as a later row has more.
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (root / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)
    print(text, flush=True)
    fusion_tables(root, summary)
    diagnostic_tables(root, summary)
    temporal_tables(root, summary, encoder_stats(root))
    angular_tables(root, summary, angular_stats(root))
    evolution_tables(root, summary, angular_stats(root))
    representation_tables(root, summary, angular_stats(root))
    benchmark_tables(root, summary, angular_stats(root))
    factorial_tables(root, summary)
    representation_ablation_tables(root, summary)
    context_tables(root, summary)
    ordering_tables(root, summary)
    handcrafted_tables(root, summary)
    local_long_tables(root, summary)
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
        "The differences do not match any single row of the decision table cleanly.",
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
    """experiment name -> receptive field, recomputed from each run's saved config.

    Deliberately NOT read back from environment.json: a run trained before the field
    calculator was corrected recorded the old closed-form value, which would put a
    stale number next to freshly computed ones in the same table. The architecture is
    fully determined by the config the run stored, so recomputing is both authoritative
    and consistent across runs of different vintages (§5).
    """
    stats = {}
    for path in sorted(Path(root).rglob("config.json")):
        config = json.loads(path.read_text())
        model, data = config.get("model"), config.get("data", {})
        if not model:
            continue
        if model["variant"] not in SINGLE_VARIANTS:
            continue  # a fusion model has two encoders; no single field applies
        layers = [(model["stem_stride"],) * 2 + (1,)]
        kernel = 1 if model.get("operator", "conv") == "mlp" else model["kernel"]
        for index in range(model["depth"]):
            if index:
                layers.append((2, 2, 1))
            layers += [(kernel, 1, 1), (kernel, 1, 1)]
        samples = receptive_field_samples(layers)
        rate = data.get("sampling_rate", 500)
        stats[path.parent.name.rsplit("_seed", 1)[0]] = {
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


def angular_tables(root, summary, angular=None):
    """Tables of the Angular temporal 指导书 §7, written to angular_tables.md."""
    root = Path(root)
    if not all(name in summary for name in ANGULAR_NEW):
        if not any(name in summary for name in ANGULAR_NEW):
            return None
    angular = angular or {}
    present = [(label, name) for label, name in ANGULAR if name in summary]
    main = _table(
        ["Model", "Total Params", "Angular Params", "RF (ms)", "Macro AUROC", *CLASSES],
        [
            [
                label,
                f"{summary[name]['parameters']:,}",
                f"{angular.get(name, {}).get('angular_parameters', '?'):,}"
                if angular.get(name, {}).get("angular_parameters")
                else "?",
                str(angular.get(name, {}).get("receptive_field_ms", "?")),
                _cell(summary[name], "macro_auroc"),
                *[_cell(summary[name], cls) for cls in CLASSES],
            ]
            for label, name in present
        ],
    )
    standard, quaternion = "RLA_standard", "RLA_quaternion"
    diffs, verdict = "", "**Incomplete.** Train both angular encoders before reading a verdict."
    if standard in summary and quaternion in summary:
        gap = summary[quaternion]["macro_auroc_mean"] - summary[standard]["macro_auroc_mean"]
        rows = [["Macro AUROC", f"{gap:+.4f}"]]
        rows += [
            [cls, f"{summary[quaternion][f'{cls}_mean'] - summary[standard][f'{cls}_mean']:+.4f}"]
            for cls in CLASSES
        ]
        total = summary[quaternion]["parameters"] - summary[standard]["parameters"]
        rows.append(
            [
                "Total params",
                f"{total:+,} ({total / summary[standard]['parameters']:+.2%})",
            ]
        )
        if angular.get(standard, {}).get("angular_parameters"):
            delta = (
                angular[quaternion]["angular_parameters"] - angular[standard]["angular_parameters"]
            )
            rows.append(
                [
                    "Angular params",
                    f"{delta:+,} ({delta / angular[standard]['angular_parameters']:+.2%})",
                ]
            )
        diffs = _table(["Quantity", "Quaternion - Standard"], rows)
        verdict = interpret_angular(gap)
    text = (
        "# Angular temporal modelling: Standard vs Quaternion\n\n"
        "One angular sequence, two temporal operators. The R and L branches, the input\n"
        "A tensor, the receptive field, the fusion, the head and the recipe are shared;\n"
        "only the angular encoder changes. Seed 42 only, noise band "
        f"{NOISE_BAND:.4f}.\n\n"
        "The descriptor q = dot + cross_x i + cross_y j + cross_z k is a full-angle\n"
        "relation, not a physical rotation quaternion, and a vanilla QuaternionConv\n"
        "carries no SO(3) guarantee. The claim under test is only whether Hamilton\n"
        "scalar-vector coupling is a better inductive bias for this sequence.\n\n"
        "## Table 13: Angular operator results\n\n"
        + main
        + "\n\n"
        + ("## Table 14: Quaternion - Standard\n\n" + diffs + "\n\n" if diffs else "")
        + "## Verdict\n\n"
        + verdict
        + "\n"
    )
    missing = [name for _, name in ANGULAR if name not in summary]
    if missing:
        text += f"\nNot yet trained: {', '.join(missing)}\n"
    (root / "angular_tables.md").write_text(text, encoding="utf-8")
    print(text, flush=True)
    return text


def interpret_angular(gap):
    """The §8 decision table. 0.005 is this project's screening band, not a p-value."""
    band = NOISE_BAND
    if gap > band:
        return (
            f"**A signal worth more seeds.** Quaternion - Standard is {gap:+.4f}, above "
            f"the {band:.3f} screening band, so Hamilton-structured angular temporal "
            "modelling shows a seed-42 advantage worth confirming.\n\n"
            "Next: train both with seeds 43/44. No formal claim before that."
        )
    if gap < -band:
        return (
            f"**Standard is ahead.** Quaternion - Standard is {gap:+.4f}, below "
            f"-{band:.3f}, so Hamilton-structured angular temporal modelling does not "
            "hold an advantage here.\n\nNext: stop the Quaternion line; do not stack "
            "further Quaternion layers in the hope of recovering it."
        )
    return (
        f"**Indistinguishable.** Quaternion - Standard is {gap:+.4f}, inside the "
        f"{band:.3f} band, so the two inductive biases cannot be told apart at seed 42."
        "\n\nNext: do not extend the Quaternion architecture. Stop here."
    )


def angular_stats(root):
    """experiment name -> angular encoder parameter count and receptive field."""
    stats = {}
    for path in sorted(Path(root).rglob("environment.json")):
        environment = json.loads(path.read_text())
        if "angular_parameters" not in environment:
            continue
        samples = environment["receptive_field_samples"]
        rate = environment.get("sampling_rate", 500)
        stats[path.parent.name.rsplit("_seed", 1)[0]] = {
            "angular_parameters": environment["angular_parameters"],
            "receptive_field_samples": samples,
            "receptive_field_ms": round(1000 * samples / rate) if samples else None,
        }
    return stats


def evolution_tables(root, summary, angular=None):
    """Tables of the Temporal evolution 方案 §3/§6, written to evolution_tables.md."""
    root = Path(root)
    if not any(name in summary for name in EVOLUTION_NEW):
        return None
    angular = angular or {}
    main = _table(
        ["Model", "Angular input", "Angular Params", "Total Params", "Macro AUROC", *CLASSES],
        [
            [
                label,
                formula,
                f"{angular.get(name, {}).get('angular_parameters', 0):,}"
                if angular.get(name, {}).get("angular_parameters")
                else "?",
                f"{summary[name]['parameters']:,}",
                _cell(summary[name], "macro_auroc"),
                *[_cell(summary[name], cls) for cls in CLASSES],
            ]
            for label, name, formula in EVOLUTION
            if name in summary
        ],
    )
    gap = {
        label: summary[new]["macro_auroc_mean"] - summary[old]["macro_auroc_mean"]
        for label, new, old, _ in EVOLUTION_DIFFS
        if new in summary and old in summary
    }
    diffs = _table(
        ["Comparison", "Question", "Delta Macro AUROC"],
        [
            [label, question, f"{gap[label]:+.4f}"]
            for label, new, old, question in EVOLUTION_DIFFS
            if label in gap
        ],
    )
    per_class = ""
    if "A2" in summary and "A1" in summary:
        per_class = _table(
            ["Class", "A2 - A1"],
            [
                [cls, f"{summary['A2'][f'{cls}_mean'] - summary['A1'][f'{cls}_mean']:+.4f}"]
                for cls in CLASSES
            ],
        )
    verdict = interpret_evolution(gap)
    text = (
        "# Quaternion temporal evolution of the angular representation\n\n"
        "q_t = Rotation(u_t -> u_{t+delta}) is a standard HALF-angle rotation quaternion,\n"
        "not the full-angle [dot, cross] descriptor used by M1-M4. A1 and A2 take the\n"
        "same two operands and differ only in how they are combined: subtraction in R^4\n"
        "versus composition in the rotation group. e_t = q_t^-1 (x) q_{t+tau} is a real\n"
        "relative rotation expressed in q_t's own frame; q_{t+tau} - q_t is a\n"
        "coordinate-wise difference with no rotation-group meaning.\n\n"
        "Every variant uses the same plain real temporal encoder, the same R and L\n"
        "branches, fusion, head, receptive field and recipe. No QuaternionConv is used\n"
        f"anywhere in this round. Seed 42 only, noise band {NOISE_BAND:.4f}.\n\n"
        "## Table 15: Angular temporal evolution results\n\n" + main + "\n\n"
        "## Table 16: Component differences\n\n"
        + diffs
        + "\n\n"
        + ("## Table 17: Per-class A2 - A1\n\n" + per_class + "\n\n" if per_class else "")
        + "## Verdict\n\n"
        + verdict
        + "\n"
    )
    missing = [name for _, name, _ in EVOLUTION if name not in summary]
    if missing:
        text += f"\nNot yet trained: {', '.join(missing)}\n"
    (root / "evolution_tables.md").write_text(text, encoding="utf-8")
    print(text, flush=True)
    return text


def interpret_evolution(gap):
    """The §6 decision table, plus what A0 says about evolution itself."""
    quaternion, evolution = gap.get("A2 - A1"), gap.get("A1 - A0")
    if quaternion is None:
        return "**Incomplete.** Train A0, A1 and A2 before reading a verdict."
    band = NOISE_BAND
    if quaternion > band:
        verdict = (
            f"**Quaternion evolution shows a positive signal.** A2 - A1 is {quaternion:+.4f}, "
            f"above the {band:.3f} band, so the gain is not merely from adding a temporal "
            "difference: rotation-group composition carries something the plain difference "
            "does not.\n\nNext: add seeds 43/44, then run the cardiac-cycle interpretability "
            "analysis of the plan's section 7. No method claim before both."
        )
    elif quaternion < -band:
        verdict = (
            f"**Quaternion evolution is behind.** A2 - A1 is {quaternion:+.4f}, below "
            f"-{band:.3f}.\n\nNext: stop this quaternion representation line."
        )
    else:
        verdict = (
            f"**Quaternion-specific value is indistinguishable.** A2 - A1 is "
            f"{quaternion:+.4f}, inside the {band:.3f} band, so composition in the "
            "rotation group cannot be told apart from a plain temporal difference at "
            "seed 42.\n\nNext: do not package quaternion representation as a necessary "
            "component; do not stack further quaternion architecture."
        )
    if evolution is not None:
        if evolution > band:
            verdict += (
                f"\n\nSeparately, A1 - A0 is {evolution:+.4f}: angular temporal evolution "
                "itself does add information over the local angular state, independent of "
                "how that evolution is encoded."
            )
        else:
            verdict += (
                f"\n\nSeparately, A1 - A0 is {evolution:+.4f}, inside the band, so at seed "
                "42 even plain angular evolution is not distinguishable from local state."
            )
    return verdict


def representation_tables(root, summary, angular=None):
    """Tables of the Angular representation 指导书 §8, written to representation_tables.md.

    Neither model is trained here: the baseline is the operator round's full-angle
    control and the new variant is the registry's A0. The two differ only in the four
    angular channels, and their parameter counts are identical, so the comparison is
    already matched (§4).
    """
    root = Path(root)
    baseline_name, new_name = REPRESENTATION[0][1], REPRESENTATION[1][1]
    if new_name not in summary:
        return None
    angular = angular or {}
    rows = []
    for label, name, formula in REPRESENTATION:
        if name not in summary:
            rows.append([label, formula, "reused, run not present", "--", "--", *["--"] * 5])
            continue
        rows.append(
            [
                label,
                formula,
                f"{summary[name]['parameters']:,}",
                f"{angular.get(name, {}).get('angular_parameters', 0):,}"
                if angular.get(name, {}).get("angular_parameters")
                else "?",
                _cell(summary[name], "macro_auroc"),
                *[_cell(summary[name], cls) for cls in CLASSES],
            ]
        )
    main = _table(
        [
            "Variant",
            "Angular representation",
            "Total Params",
            "Angular Params",
            "Macro AUROC",
            *CLASSES,
        ],
        rows,
    )
    if baseline_name in summary:
        baseline = summary[baseline_name]["macro_auroc_mean"]
        source = "both runs present"
    else:
        baseline = BASELINE_MACRO_AUROC
        source = f"baseline run absent; using the recorded {BASELINE_MACRO_AUROC:.4f}"
    gap = summary[new_name]["macro_auroc_mean"] - baseline
    per_class = ""
    if baseline_name in summary:
        per_class = _table(
            ["Class", "New - Baseline"],
            [
                [
                    cls,
                    f"{summary[new_name][f'{cls}_mean'] - summary[baseline_name][f'{cls}_mean']:+.4f}",
                ]
                for cls in CLASSES
            ],
        )
    text = (
        "# Angular representation: full-angle descriptor vs rotation quaternion\n\n"
        "Same 1280 ms long-context temporal modelling, same R and L branches, same\n"
        "fusion, LayerNorm and head, same 4 angular channels, same angular encoder and\n"
        "parameter count, delta fixed at 20 ms. The only variable is whether the four\n"
        "angular numbers are the full-angle descriptor [cos t, n sin t] -- which is NOT a\n"
        "physical rotation quaternion -- or the standard half-angle rotation quaternion\n"
        f"[cos(t/2), n sin(t/2)]. Seed 42 only, screening band {NOISE_BAND:.4f}.\n\n"
        "## Table 18: Angular representation results\n\n"
        + main
        + "\n\n"
        + ("## Table 19: Per-class New - Baseline\n\n" + per_class + "\n\n" if per_class else "")
        + f"## Verdict\n\nDelta Macro AUROC = {gap:+.4f} ({source}).\n\n"
        + interpret_representation(gap)
        + "\n"
    )
    (root / "representation_tables.md").write_text(text, encoding="utf-8")
    print(text, flush=True)
    return text


def interpret_representation(gap):
    """The §6 decision rule: a single screening call on one number."""
    band = NOISE_BAND
    if gap > band:
        return (
            f"**Clear positive signal.** {gap:+.4f} exceeds +{band:.3f}, so replacing the "
            "full-angle descriptor with a standard rotation quaternion is worth "
            "confirming.\n\nNext: run seeds 43/44 for both variants."
        )
    if gap < -band:
        return (
            f"**Negative signal.** {gap:+.4f} is below -{band:.3f}.\n\n"
            "Next: stop the quaternion representation line."
        )
    return (
        f"**Weak / indistinguishable signal.** {gap:+.4f} lies inside the "
        f"+-{band:.3f} screening band, so this must be recorded as indistinguishable "
        "and must NOT be written up as a quaternion advantage.\n\n"
        "Next: do not extend the architecture. Because the control is unusually clean "
        "-- identical parameter counts, identical everything but the four angular "
        "channels -- a low-cost seeds 43/44 confirmation is a defensible option, but "
        "that is a separate decision, not an automatic follow-on."
    )


def benchmark_tables(root, summary, encoders=None):
    """Experiment 1 table and the E* selection rule (plan section 2.2).

    Selection is validation Macro AUROC, then validation Macro AUPRC, then the smaller
    parameter count. Test metrics are reported but never used to select.
    """
    root = Path(root)
    present = [name for name in BENCHMARK if name in summary]
    if not present:
        return None
    encoders = encoders or {}

    def field(name):
        milliseconds = encoders.get(name, {}).get("receptive_field_ms")
        return f"{milliseconds} ms" if milliseconds else "global"

    rows = []
    for name in present:
        short = name[len("E1_") :]
        rows.append(
            [
                BENCHMARK_LABELS[short],
                "quaternion" if short in QUATERNION_ENCODERS else "generic",
                f"{summary[name]['parameters']:,}",
                field(name),
                _cell(summary[name], "macro_auroc"),
                _cell(summary[name], "macro_auprc"),
                *[_cell(summary[name], cls) for cls in CLASSES],
            ]
        )
    main = _table(
        ["Encoder", "Family", "Params", "RF", "Macro AUROC", "Macro AUPRC", *CLASSES], rows
    )
    best = select_encoder(summary)
    pairs = _table(
        ["Quaternion encoder", "Matched generic", "Delta Macro AUROC"],
        [
            [
                BENCHMARK_LABELS[q],
                BENCHMARK_LABELS[QUATERNION_PAIR[q]],
                f"{summary[f'E1_{q}']['macro_auroc_mean'] - summary[f'E1_{QUATERNION_PAIR[q]}']['macro_auroc_mean']:+.4f}",
            ]
            for q in QUATERNION_ENCODERS
            if f"E1_{q}" in summary and f"E1_{QUATERNION_PAIR[q]}" in summary
        ],
    )
    missing = [name for name in BENCHMARK if name not in summary]
    text = (
        "# Experiment 1: temporal encoder benchmark\n\n"
        "R + L + Q with Q frozen as the rotation quaternion. Preprocessing, split,\n"
        "fusion, head and optimisation are identical; only the temporal encoder changes.\n"
        "R and L are always real: a quaternion encoder replaces the Q-branch operator\n"
        "only, and its R/L branches use the matched generic encoder, so each\n"
        "quaternion/generic pair differs in exactly one thing (plan section 2.1).\n\n"
        "Selection rule: validation Macro AUROC, then validation Macro AUPRC, then the\n"
        "smaller parameter count. Test metrics below are reported, never used to select.\n\n"
        "## Table E1.1: Encoder benchmark\n\n" + main + "\n\n"
        "## Table E1.2: Quaternion vs its matched generic encoder\n\n" + pairs + "\n\n"
        f"## Selected encoder\n\n"
        f"**E\\* = {BENCHMARK_LABELS[SELECTED_ENCODER]}** (frozen for Experiments 2-7)\n\n"
        f"Pre-specified rule would select: **{best}**\n"
    )
    if best != BENCHMARK_LABELS.get(SELECTED_ENCODER) and "not selected" not in best:
        text += f"\n> {SELECTION_NOTE}\n"
    if missing:
        text += f"\nNot yet trained: {', '.join(missing)}\n"
    (root / "benchmark_tables.md").write_text(text, encoding="utf-8")
    print(text, flush=True)
    return text


def select_encoder(summary):
    """E* under the pre-specified rule; validation metrics only (plan section 2.2)."""
    present = [name for name in BENCHMARK if name in summary]
    if not present:
        return "not selected: no benchmark runs"
    if len(present) < len(BENCHMARK):
        return f"not selected: {len(present)}/{len(BENCHMARK)} encoders trained"

    def key(name):
        row = summary[name]
        return (
            -row["validation_macro_auroc_mean"],
            -row["validation_macro_auprc_mean"],
            row["parameters"],
        )

    if any("validation_macro_auroc_mean" not in summary[name] for name in present):
        return "not selected: validation metrics missing from the run summaries"
    return BENCHMARK_LABELS[min(present, key=key)[len("E1_") :]]


def factorial_tables(root, summary):
    """Experiment 2: the R/L/Q factorial and the conditional contributions (plan §3)."""
    root = Path(root)
    present = [name for name in (*FACTORIAL, FACTORIAL_REFERENCE) if name in summary]
    if not present:
        return None
    marks = {"E2_raw": ("-", "-", "-")}
    for name in FACTORIAL:
        short = name[len("E2_") :]
        marks[name] = tuple("v" if letter in short else "x" for letter in "RLQ")
    main = _table(
        ["Variant", "R", "L", "Q", "Params", "Macro AUROC", "Macro AUPRC", *CLASSES],
        [
            [
                FACTORIAL_LABELS[name],
                *marks[name],
                f"{summary[name]['parameters']:,}",
                _cell(summary[name], "macro_auroc"),
                _cell(summary[name], "macro_auprc"),
                *[_cell(summary[name], cls) for cls in CLASSES],
            ]
            for name in present
        ],
    )
    contributions = _table(
        ["Component", "Difference", "Meaning", "Delta Macro AUROC"],
        [
            [
                letter,
                f"{FACTORIAL_LABELS[full]} - {FACTORIAL_LABELS[without]}",
                meaning,
                f"{summary[full]['macro_auroc_mean'] - summary[without]['macro_auroc_mean']:+.4f}",
            ]
            for letter, full, without, meaning in FACTORIAL_CONTRIBUTIONS
            if full in summary and without in summary
        ],
    )
    missing = [name for name in (*FACTORIAL, FACTORIAL_REFERENCE) if name not in summary]
    text = (
        "# Experiment 2: R/L/Q component ablation\n\n"
        f"Full 2^3-1 factorial on the frozen E* = {BENCHMARK_LABELS[SELECTED_ENCODER]}, plus a\n"
        "raw XYZ reference that is NOT part of the factorial design. A conditional\n"
        "contribution is the difference between the full model and the model without that\n"
        f"component. Seed 42, screening band {NOISE_BAND:.4f}.\n\n"
        "## Table E2.1: Factorial results\n\n" + main + "\n\n"
        "## Table E2.2: Conditional contributions\n\n" + contributions + "\n"
    )
    if missing:
        text += f"\nNot yet trained: {', '.join(missing)}\n"
    (root / "factorial_tables.md").write_text(text, encoding="utf-8")
    print(text, flush=True)
    return text


def representation_ablation_tables(root, summary):
    """Experiment 3: R and L fixed, only the angular representation changes (plan §4)."""
    root = Path(root)
    present = [
        (label, name, formula) for label, name, formula in REPRESENTATIONS if name in summary
    ]
    if not present:
        return None
    main = _table(
        ["Variant", "Angular representation", "Params", "Macro AUROC", "Macro AUPRC", *CLASSES],
        [
            [
                label,
                formula,
                f"{summary[name]['parameters']:,}",
                _cell(summary[name], "macro_auroc"),
                _cell(summary[name], "macro_auprc"),
                *[_cell(summary[name], cls) for cls in CLASSES],
            ]
            for label, name, formula in present
        ],
    )
    pairs = _table(
        ["Comparison", "Question", "Delta Macro AUROC"],
        [
            [
                label,
                question,
                f"{summary[new]['macro_auroc_mean'] - summary[old]['macro_auroc_mean']:+.4f}",
            ]
            for label, new, old, question in (
                ("D - U", "E3_D", "E3_U", "Local angular dynamics vs absolute direction"),
                # The Q row is E2_RLQ whenever E* is generic, so it is looked up rather
                # than named: a hardcoded E3_Q would silently drop these two rows.
                (
                    "Q - D",
                    REPRESENTATIONS[-1][1],
                    "E3_D",
                    "Half-angle rotation vs full-angle descriptor",
                ),
                (
                    "Q - U",
                    REPRESENTATIONS[-1][1],
                    "E3_U",
                    "Rotation parameterization vs absolute direction",
                ),
            )
            if new in summary and old in summary
        ],
    )
    missing = [name for _, name, _ in REPRESENTATIONS if name not in summary]
    text = (
        "# Experiment 3: angular representation ablation\n\n"
        "R, L, E*, receptive field, channel and parameter budget, fusion, head and\n"
        "training protocol are frozen; only the angular representation changes.\n\n"
        + (
            f"E* ({BENCHMARK_LABELS[SELECTED_ENCODER]}) is generic, so it takes all three\n"
            "representations directly and the R+L+Q row is the factorial's E2_RLQ, reused\n"
            "rather than retrained.\n\n"
            if REPRESENTATION_REUSES_FACTORIAL
            else f"All three use the generic counterpart of E* "
            f"({BENCHMARK_LABELS[REPRESENTATION_ENCODER]}) on the angular branch: U is a\n"
            "3-vector and cannot enter a quaternion operator, and padding a branch to fit\n"
            "one is forbidden, so holding the operator generic is what makes 'only the\n"
            "representation changes' literally true.\n\n"
        )
        + "Q is a quaternion REPRESENTATION of the angular branch; its temporal\n"
        "dependencies are learned by the same generic real-valued encoder as U and D.\n"
        "This experiment compares representations, not quaternion-specific temporal\n"
        "computation, which is what Experiment 1 tested.\n\n"
        "D and Q derive from the same (theta, n); neither carries more raw information\n"
        "than the other, and the comparison is about parameterization and empirical\n"
        f"utility only. Seed 42, screening band {NOISE_BAND:.4f}.\n\n"
        "## Table E3.1: Representation results\n\n" + main + "\n\n"
        "## Table E3.2: Pairwise differences\n\n" + pairs + "\n"
    )
    if missing:
        text += f"\nNot yet trained: {', '.join(missing)}\n"
    (root / "representation_ablation_tables.md").write_text(text, encoding="utf-8")
    print(text, flush=True)
    return text


def context_tables(root, summary):
    """Experiment 4: accessible temporal-context ablation (plan section 5, redefined).

    The pre-specified reading is fixed here so it cannot be chosen after seeing the
    numbers: the trend is read over the ordered levels, and the local / phase-scale /
    cycle-scale grouping is the one the plan defines, not one fitted to the result.
    """
    root = Path(root)
    present = [(label, context) for label, context in CONTEXT_LEVELS if f"E4_{label}" in summary]
    if not present:
        return None
    main = _table(
        ["Accessible context", "Scale", "Params", "Macro AUROC", "Macro AUPRC", *CLASSES],
        [
            [
                f"{context} ms" if context else "unrestricted",
                CONTEXT_SCALE[context],
                f"{summary[f'E4_{label}']['parameters']:,}",
                _cell(summary[f"E4_{label}"], "macro_auroc"),
                _cell(summary[f"E4_{label}"], "macro_auprc"),
                *[_cell(summary[f"E4_{label}"], cls) for cls in CLASSES],
            ]
            for label, context in present
        ],
    )
    scores = {context: summary[f"E4_{label}"]["macro_auroc_mean"] for label, context in present}
    steps = _table(
        ["Step", "Delta Macro AUROC"],
        [
            [
                f"{b} ms - {a} ms" if b else f"unrestricted - {a} ms",
                f"{scores[b] - scores[a]:+.4f}",
            ]
            for a, b in zip([c for _, c in present], [c for _, c in present][1:])
        ],
    )
    text = (
        "# Experiment 4: accessible temporal-context ablation\n\n"
        f"E* ({BENCHMARK_LABELS[SELECTED_ENCODER]}) is unchanged: same architecture, same\n"
        "width, same parameter count at every level. What is varied is how much time the\n"
        "recurrence may integrate, by resetting the recurrent state every N steps so that\n"
        "nothing crosses a boundary. All three branches carry the same restriction, so\n"
        "they stay temporally aligned.\n\n"
        "Assumptions this rests on, stated rather than buried:\n\n"
        f"1. One step is {CONTEXT_STEP_MS} ms. The stem is stride {CONTEXT_STEM['stride']} with\n"
        f"   {CONTEXT_STEM['poolings']} pooling, finer than the stem Experiments 1-3 use, which\n"
        "   is what puts the 20 ms level within reach. Every level shares it, so the sweep\n"
        "   is internally controlled; its absolute numbers are not directly comparable to\n"
        "   Experiment 2's.\n"
        "2. The restriction bounds temporal INTEGRATION, not visibility. A step near the\n"
        "   end of a chunk sees the full window, one at the start sees less, so N is an\n"
        "   upper bound rather than a uniform context. The mechanism is identical at every\n"
        "   level, so the comparison is still controlled.\n"
        "3. Attention still pools over the whole record. The model may therefore weigh\n"
        "   local summaries from anywhere in the 10 s, but cannot represent a dependency\n"
        "   longer than the window. That is the intended meaning of accessible context.\n"
        "4. The tail is zero-padded to a whole number of chunks and cropped afterwards.\n"
        "   The recurrence is causal, so padding placed after a real step cannot reach it.\n\n"
        "## Table E4.1: Context levels\n\n" + main + "\n\n"
        "## Table E4.2: Step-to-step differences\n\n" + steps + "\n\n"
        "## Verdict\n\n" + interpret_context(scores) + "\n"
    )
    missing = [f"E4_{label}" for label, _ in CONTEXT_LEVELS if f"E4_{label}" not in summary]
    if missing:
        text += f"\nNot yet trained: {', '.join(missing)}\n"
    (root / "context_tables.md").write_text(text, encoding="utf-8")
    print(text, flush=True)
    return text


def interpret_context(scores):
    """Read the sweep against the plan's pre-specified scale grouping."""
    levels = [context for context in scores if context is not None]
    if len(levels) < len(CONTEXT_LEVELS) - 1:
        return "**Incomplete.** Train every context level before reading a trend."
    band = NOISE_BAND
    ordered = sorted(levels)
    best = max(ordered, key=lambda c: scores[c])
    grouped = {}
    for context in ordered:
        grouped.setdefault(CONTEXT_SCALE[context], []).append(scores[context])
    means = {scale: sum(values) / len(values) for scale, values in grouped.items()}
    summary = ", ".join(f"{scale} {value:.4f}" for scale, value in means.items())
    # The trend is read between the plan's scale GROUPS, so the number quoted has to be
    # the difference of their means -- not of the single shortest and longest levels,
    # which is a different and larger quantity.
    total = means["cycle-scale"] - means["local"]
    extremes = scores[ordered[-1]] - scores[ordered[0]]
    lines = [
        f"Scale means: {summary}.",
        f"Cycle-scale minus local: {total:+.4f}. Longest minus shortest level: {extremes:+.4f}.",
    ]
    if total > band:
        lines.append(
            f"**Longer accessible context helps.** Cycle-scale minus local is "
            f"{total:+.4f}, clear of the {band:.3f} screening band, so the model uses "
            "information it can only obtain by integrating over a longer window."
        )
    elif total < -band:
        lines.append(
            f"**Longer accessible context hurts.** The total change is {total:+.4f}; "
            "restricting integration is better than allowing it here."
        )
    else:
        lines.append(
            f"**Not supported.** Cycle-scale minus local is {total:+.4f}, inside the "
            f"{band:.3f} band, so at seed 42 the context levels are indistinguishable."
        )
    if best != ordered[-1] and scores[best] - scores[ordered[-1]] > band:
        lines.append(
            f"The best level is {best} ms rather than the longest, by "
            f"{scores[best] - scores[ordered[-1]]:+.4f}, which points to a bounded useful "
            "range rather than 'longer is better'. Experiment 6 should take its local and "
            "long settings from this."
        )
    if None in scores:
        lines.append(
            f"Unrestricted minus 1280 ms is {scores[None] - scores[ordered[-1]]:+.4f}: "
            "whether the longest tested window already saturates."
        )
    return "\n\n".join(lines)


def ordering_tables(root, summary):
    """Experiment 7: temporal ordering (plan section 8)."""
    root = Path(root)
    if not any(name in summary for name in ORDERING):
        return None
    rows = []
    if ORDERING_ORIGINAL in summary:
        rows.append(
            [
                "Original",
                "--",
                f"{summary[ORDERING_ORIGINAL]['parameters']:,}",
                _cell(summary[ORDERING_ORIGINAL], "macro_auroc"),
                _cell(summary[ORDERING_ORIGINAL], "macro_auprc"),
                *[_cell(summary[ORDERING_ORIGINAL], cls) for cls in CLASSES],
            ]
        )
    for scope, _ in SHUFFLE_SCOPES:
        for label, _ in SHUFFLE_LEVELS:
            name = f"E7_{scope}_{label}"
            if name not in summary:
                continue
            rows.append(
                [
                    SHUFFLE_LABELS[label],
                    SCOPE_LABELS[scope],
                    f"{summary[name]['parameters']:,}",
                    _cell(summary[name], "macro_auroc"),
                    _cell(summary[name], "macro_auprc"),
                    *[_cell(summary[name], cls) for cls in CLASSES],
                ]
            )
    main = _table(["Condition", "Scope", "Params", "Macro AUROC", "Macro AUPRC", *CLASSES], rows)
    losses = ""
    if ORDERING_ORIGINAL in summary:
        original = summary[ORDERING_ORIGINAL]["macro_auroc_mean"]
        losses = _table(
            ["Condition", "Scope", "Delta vs Original"],
            [
                [
                    SHUFFLE_LABELS[label],
                    SCOPE_LABELS[scope],
                    f"{summary[f'E7_{scope}_{label}']['macro_auroc_mean'] - original:+.4f}",
                ]
                for scope, _ in SHUFFLE_SCOPES
                for label, _ in SHUFFLE_LEVELS
                if f"E7_{scope}_{label}" in summary
            ],
        )
    text = (
        "# Experiment 7: temporal ordering\n\n"
        "Identical to R+L+Q in every respect except that the feature sequence is\n"
        "permuted before the encoder. R, L and Q are built from the intact recording\n"
        "first, so what a permutation destroys is the order of the dynamic states, not\n"
        "the states themselves; shuffling the signal instead would corrupt the lagged\n"
        "quantities as they are computed and would be testing something else.\n\n"
        "A block shuffle permutes whole blocks and keeps the order inside each, so local\n"
        "dynamics survive and longer-range organisation does not. The point-wise shuffle\n"
        "breaks both. The joint scope applies ONE permutation to all three branches, so\n"
        "cross-component alignment survives; permuting them independently would destroy\n"
        "that as well and confound the result.\n\n"
        "A fresh permutation is drawn per record so the model cannot learn to invert a\n"
        "fixed one; in evaluation the draw is re-seeded so the reported metric is\n"
        "reproducible. Where the length is not a whole number of blocks the remainder\n"
        "stays at the end -- 40 of 5000 samples at 160 ms, the same tail in every\n"
        f"condition. Seed 42, screening band {NOISE_BAND:.4f}.\n\n"
        "## Table E7.1: Ordering conditions\n\n"
        + main
        + "\n\n"
        + ("## Table E7.2: Cost of destroying order\n\n" + losses + "\n\n" if losses else "")
        + "## Verdict\n\n"
        + interpret_ordering(summary)
        + "\n"
    )
    missing = [name for name in ORDERING if name not in summary]
    if missing:
        text += f"\nNot yet trained: {', '.join(missing)}\n"
    (root / "ordering_tables.md").write_text(text, encoding="utf-8")
    print(text, flush=True)
    return text


def interpret_ordering(summary):
    """Original > block > point-wise would support temporal organisation carrying
    information beyond a distribution of local states. The hierarchy is stated in
    advance and not fitted to the result (plan section 8)."""
    if ORDERING_ORIGINAL not in summary:
        return "**Incomplete.** The Original condition (R+L+Q) has not been trained."
    missing = [name for name in ORDERING if name not in summary]
    if missing:
        return f"**Incomplete.** Still to train: {', '.join(missing)}."
    original = summary[ORDERING_ORIGINAL]["macro_auroc_mean"]
    band = NOISE_BAND
    lines = []
    for scope, _ in SHUFFLE_SCOPES:
        blocks = [summary[f"E7_{scope}_{label}"]["macro_auroc_mean"] for label, _ in SHUFFLE_LEVELS]
        *block_scores, point = blocks
        best_block = max(block_scores)
        if original - point > band and best_block - point > band and original - best_block > band:
            verdict = (
                "Original > block > point-wise, each step clear of the band: temporal "
                "organisation carries information beyond a distribution of local states."
            )
        elif original - point > band and original - best_block <= band:
            verdict = (
                "Block shuffles cost little but the point-wise shuffle costs "
                f"{original - point:+.4f}: local dynamics matter, longer-range order "
                "does not show a separable effect."
            )
        elif original - point <= band:
            verdict = (
                f"Even the point-wise shuffle costs only {original - point:+.4f}, inside "
                "the band: at seed 42 ordering is not distinguishable from a bag of "
                "local states."
            )
        else:
            verdict = "The ordering is not monotone; seed 42 alone supports no claim."
        lines.append(f"**{SCOPE_LABELS[scope]}.** {verdict}")
    return "\n\n".join(lines)


CLASSIFIER_LABELS = {"logistic": "Logistic Regression", "mlp": "MLP (64 hidden)"}


def _classifier_cell(row):
    """Which classifier the row actually used.

    Only D is the sequence model; A, B and C are a scikit-learn classifier on fixed
    features and must not be printed as though they ran E*.
    """
    settings = row.get("settings") or {}
    chosen = settings.get("selected_classifier")
    if chosen is None:
        return f"E* ({BENCHMARK_LABELS[SELECTED_ENCODER]})"
    label = CLASSIFIER_LABELS.get(chosen, chosen)
    return f"{label} on E*" if settings.get("embedding_checkpoint") else label


def _parameter_cell(row):
    """Fitted parameters, plus whatever is frozen upstream, kept apart."""
    frozen = row.get("frozen_parameters")
    fitted = row.get("classifier_parameters")
    if fitted is None:
        return f"{row['parameters']:,}"
    if not frozen:
        return f"{fitted:,}"
    return f"{fitted:,} + {frozen:,}"


def handcrafted_tables(root, summary):
    """Experiment 5: handcrafted summaries against learned dynamics (plan section 6)."""
    root = Path(root)
    if not any(name in summary for _, name in HANDCRAFTED):
        return None
    rows = []
    for label, name in HANDCRAFTED:
        if name not in summary:
            continue
        rows.append(
            [
                label,
                _classifier_cell(summary[name]),
                _parameter_cell(summary[name]),
                _cell(summary[name], "macro_auroc"),
                _cell(summary[name], "macro_auprc"),
                *[_cell(summary[name], cls) for cls in CLASSES],
            ]
        )
    main = _table(
        ["Arm", "Classifier", "Params (fitted + frozen)", "Macro AUROC", "Macro AUPRC", *CLASSES],
        rows,
    )
    diffs = ""
    proposed = HANDCRAFTED_PROPOSED
    if proposed in summary:
        comparisons = [
            (f"D - {label.split()[0]}", proposed, name)
            for label, name in HANDCRAFTED
            if name in summary and name not in (proposed, "E5_hybrid")
        ]
        if "E5_hybrid" in summary:
            comparisons.append(("E - D", "E5_hybrid", proposed))
        diffs = _table(
            ["Comparison", "Delta Macro AUROC"],
            [
                [
                    label,
                    f"{summary[new]['macro_auroc_mean'] - summary[old]['macro_auroc_mean']:+.4f}",
                ]
                for label, new, old in comparisons
            ],
        )
    text = (
        "# Experiment 5: handcrafted summaries vs learned temporal dynamics\n\n"
        "A, B and C compress the same R/L/Q signals into fixed summaries and hand them to\n"
        "a conventional classifier; D is the proposed model, reused; E concatenates D's\n"
        "pre-classifier embedding with all three handcrafted sets and uses the same\n"
        "classifier procedure as A/B/C.\n\n"
        "B and C are reimplementations on PTB-XL, not replications: the task, the dataset\n"
        "and the available annotation differ from the original papers, and every\n"
        "definition used is stated in qdg/handcrafted.py.\n\n"
        "Params are split because the arms are not commensurable: A, B and C fit only a\n"
        "classifier on fixed features, while D fits a whole sequence model and E fits a\n"
        "classifier on top of D frozen. The second number is what is frozen upstream.\n\n"
        "Classifier: Logistic Regression and a small MLP are both fitted for every arm and\n"
        "the one with the better VALIDATION Macro AUROC is reported, so no arm gets a\n"
        "classifier the others could not have had. Standardisation is fitted on the\n"
        "training folds only; the features themselves are a deterministic function of a\n"
        "single record and cannot leak across the split.\n\n"
        f"Seed 42, screening band {NOISE_BAND:.4f}.\n\n"
        "## Table E5.1: Arms\n\n"
        + main
        + "\n\n"
        + ("## Table E5.2: Against the proposed model\n\n" + diffs + "\n\n" if diffs else "")
        + "## Verdict\n\n"
        + interpret_handcrafted(summary)
        + "\n"
    )
    missing = [name for _, name in HANDCRAFTED if name not in summary]
    if missing:
        text += f"\nNot yet trained: {', '.join(missing)}\n"
    (root / "handcrafted_tables.md").write_text(text, encoding="utf-8")
    print(text, flush=True)
    return text


def interpret_handcrafted(summary):
    """Plan section 6.5, stated in advance: the reading depends on where E lands."""
    proposed, hybrid = HANDCRAFTED_PROPOSED, "E5_hybrid"
    arms = [name for _, name in HANDCRAFTED if name not in (proposed, hybrid)]
    if proposed not in summary or any(name not in summary for name in arms):
        return "**Incomplete.** Train every handcrafted arm and the proposed model first."
    band = NOISE_BAND
    best_handcrafted = max(arms, key=lambda name: summary[name]["macro_auroc_mean"])
    gap = summary[proposed]["macro_auroc_mean"] - summary[best_handcrafted]["macro_auroc_mean"]
    lines = [
        f"Best handcrafted arm: {best_handcrafted} at "
        f"{summary[best_handcrafted]['macro_auroc_mean']:.4f}; proposed "
        f"{summary[proposed]['macro_auroc_mean']:.4f}, a gap of {gap:+.4f}."
    ]
    if hybrid not in summary:
        lines.append("The hybrid arm is not trained, so no conclusion about complementarity.")
        return "\n\n".join(lines)
    addition = summary[hybrid]["macro_auroc_mean"] - summary[proposed]["macro_auroc_mean"]
    if gap > band and abs(addition) <= band:
        lines.append(
            f"**Learned dynamics win and the summaries add nothing on top.** E - D is "
            f"{addition:+.4f}, inside the band, so the predefined summaries carry little "
            "beyond what the learned sequence representation already holds."
        )
    elif addition > band:
        lines.append(
            f"**Complementary.** E - D is {addition:+.4f}, clear of the band: the "
            "clinically motivated features and the learned representation hold different "
            "information."
        )
    elif gap <= band:
        lines.append(
            f"**Not separated.** The proposed model leads the best handcrafted arm by "
            f"{gap:+.4f}, inside the band, so at seed 42 learning the full sequence is not "
            "distinguishable from compressing it first."
        )
    else:
        lines.append(f"**E is below D** by {addition:+.4f}; adding the summaries hurts.")
    return "\n\n".join(lines)


def local_long_tables(root, summary):
    """Experiment 6: local and long context together (plan section 7)."""
    root = Path(root)
    if COMBINED not in summary:
        return None
    present = [(label, name) for label, name in LOCAL_LONG if name in summary]
    main = _table(
        ["Condition", "Params", "Macro AUROC", "Macro AUPRC", *CLASSES],
        [
            [
                label,
                f"{summary[name]['parameters']:,}",
                _cell(summary[name], "macro_auroc"),
                _cell(summary[name], "macro_auprc"),
                *[_cell(summary[name], cls) for cls in CLASSES],
            ]
            for label, name in present
        ],
    )
    text = (
        "# Experiment 6: local and long-context information\n\n"
        f"Local is {LOCAL_CONTEXT_MS} ms and long is {LONG_CONTEXT_MS} ms. Both come from the\n"
        "Experiment 4 sweep under its pre-specified reading -- local at the end of the\n"
        "local group, long at the point where the gain saturates -- and not from\n"
        "inspecting the test set. The two single-scale conditions ARE the corresponding\n"
        "Experiment 4 runs, reused. Local + Long runs one encoder per scale on every\n"
        "branch and concatenates them; its width is solved so the parameter count lands\n"
        "within about 1% of the single-scale runs.\n\n"
        "This experiment gates a claim: only if Local + Long beats both single scales may\n"
        "the manuscript say the method integrates local and long-range dynamics.\n"
        "Otherwise the narrower 'long-context temporal modeling' is what is supported.\n\n"
        f"Seed 42, screening band {NOISE_BAND:.4f}.\n\n"
        "## Table E6.1: Local, long and both\n\n" + main + "\n\n"
        "## Verdict\n\n" + interpret_local_long(summary) + "\n"
    )
    missing = [name for _, name in LOCAL_LONG if name not in summary]
    if missing:
        text += f"\nNot yet trained: {', '.join(missing)}\n"
    (root / "local_long_tables.md").write_text(text, encoding="utf-8")
    print(text, flush=True)
    return text


def interpret_local_long(summary):
    """The claim this gates is stated in advance (plan section 7)."""
    names = [name for _, name in LOCAL_LONG]
    if any(name not in summary for name in names):
        return "**Incomplete.** Train all three conditions before reading a verdict."
    local, long_, both = (summary[name]["macro_auroc_mean"] for name in names)
    band = NOISE_BAND
    best_single = max(local, long_)
    gain = both - best_single
    if gain > band:
        return (
            f"**Integration supported.** Local + Long is {gain:+.4f} above the better "
            f"single scale ({max(('local', local), ('long', long_), key=lambda p: p[1])[0]}), "
            f"clear of the {band:.3f} band. The manuscript may claim the method integrates "
            "local and long-range dynamics."
        )
    return (
        f"**Integration not supported.** Local + Long is {gain:+.4f} against the better "
        f"single scale, inside the {band:.3f} band. Use the narrower claim, "
        "'long-context temporal modeling', not 'integrates local and long-range dynamics'."
    )
