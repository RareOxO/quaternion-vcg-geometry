"""Append one finished experiment to a shared Markdown results table.

    python scripts/append_result.py E1_qlstm --out /shared/exp1_results.md

Written for one experiment per machine with a cloud-shared results file: each host runs
the command for its own experiment and the row lands in the same table. Re-running an
experiment replaces its row rather than adding a duplicate, and rows are always written
back in the canonical experiment order, so the file stays a valid table whatever order
the machines finish in.

Only results go in the file -- no training log.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qdg.data import CLASSES  # noqa: E402
from qdg.encoders import QUATERNION_ENCODERS  # noqa: E402
from qdg.experiments import BENCHMARK, BENCHMARK_LABELS  # noqa: E402

COLUMNS = (
    "Experiment",
    "Encoder",
    "Family",
    "Params",
    "RF",
    "Val AUROC",
    "Val AUPRC",
    "Test AUROC",
    "Test AUPRC",
    *CLASSES,
    "GPU",
)


def row_for(name, run):
    result = json.loads((run / "best_test_metrics.json").read_text())
    environment = json.loads((run / "environment.json").read_text())
    test = result["fixed_0.5"]
    field = environment.get("receptive_field_samples")
    auprc = result.get("validation_macro_auprc")
    short = name[len("E1_") :] if name.startswith("E1_") else name
    return [
        name,
        BENCHMARK_LABELS.get(short, short),
        "quaternion" if short in QUATERNION_ENCODERS else "generic",
        f"{result['parameters']:,}",
        "global" if not field else f"{round(field * 2)} ms",
        f"{result['validation_macro_auroc']:.4f}",
        f"{auprc:.4f}" if auprc is not None else "--",
        f"{test['macro_auroc']:.4f}",
        f"{test['macro_auprc']:.4f}",
        *[f"{test['per_class'][cls]['auroc']:.4f}" for cls in CLASSES],
        (environment.get("gpu") or "cpu").replace("NVIDIA GeForce ", ""),
    ]


def read_rows(path):
    """Existing data rows, keyed by experiment, tolerating an absent or empty file."""
    if not path.is_file():
        return {}
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("|") or set(line) <= set("|- "):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if cells[0] in ("Experiment", ""):
            continue
        rows[cells[0]] = cells
    return rows


def write_table(path, rows):
    order = {name: index for index, name in enumerate(BENCHMARK)}
    ordered = sorted(rows.values(), key=lambda cells: (order.get(cells[0], len(order)), cells[0]))
    lines = [
        "# Experiment 1: temporal encoder benchmark (seed 42)",
        "",
        "One row per experiment, appended by `scripts/append_result.py` as each machine",
        "finishes. E* is selected on validation Macro AUROC, then validation Macro AUPRC,",
        "then fewer parameters; test columns are reported but never used to select.",
        "",
        "| " + " | ".join(COLUMNS) + " |",
        "| " + " | ".join(["---"] * len(COLUMNS)) + " |",
        *["| " + " | ".join(cells) + " |" for cells in ordered],
        "",
        f"{len(ordered)}/{len(BENCHMARK)} encoders recorded.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Append one result row to a shared table")
    parser.add_argument("experiment")
    parser.add_argument("--root", type=Path, default=Path("runs"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out", type=Path, default=Path("runs/exp1_results.md"), help="the shared table"
    )
    args = parser.parse_args()
    run = args.root / f"{args.experiment}_seed{args.seed}"
    if not (run / "best_test_metrics.json").is_file():
        raise SystemExit(f"No completed run at {run}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.out)
    replaced = args.experiment in rows
    rows[args.experiment] = row_for(args.experiment, run)
    write_table(args.out, rows)
    action = "replaced" if replaced else "appended"
    print(f"{action} {args.experiment} in {args.out} ({len(rows)}/{len(BENCHMARK)} recorded)")


if __name__ == "__main__":
    main()
