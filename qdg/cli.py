import argparse
import json
from pathlib import Path

from .config import load_config, validate_config
from .data import audit, prepare
from .engine import evaluate, train
from .experiments import (
    ANGULAR_NEW,
    BENCHMARK,
    CONTEXT,
    ORDERING,
    COMBINED,
    HANDCRAFTED_ARMS,
    HANDCRAFTED_PROPOSED,
    FACTORIAL,
    FACTORIAL_REFERENCE,
    REPRESENTATION_NEW,
    DIAGNOSTIC_NEW,
    EVOLUTION_NEW,
    EXPERIMENTS,
    FUSION,
    FUSION_STAGE_A,
    TEMPORAL_NEW,
    experiment_config,
    profile,
    suite,
    tables,
)
from .sanity import check_geometry, feature_stats, sanity_checks

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"


def main():
    parser = argparse.ArgumentParser(description="Quaternion-VCG dynamic geometry on PTB-XL")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in (
        "audit",
        "prepare",
        "sanity",
        "check-geometry",
        "feature-stats",
        "profile",
        "train",
        "suite",
    ):
        sub = subparsers.add_parser(command)
        sub.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
        if command in ("check-geometry", "feature-stats", "profile", "train", "suite"):
            sub.add_argument("--device")
        if command in ("check-geometry", "feature-stats"):
            sub.add_argument("--records", type=int, default=256)
        if command in ("train", "suite"):
            sub.add_argument("--epochs", type=int)
            sub.add_argument("--output", type=Path)
        if command == "train":
            sub.add_argument("--experiment", choices=sorted(EXPERIMENTS), default="M0")
            sub.add_argument("--seed", type=int)
            sub.add_argument("--run-name")
            sub.add_argument("--limit-train", type=int)
            sub.add_argument("--limit-val", type=int)
        if command == "suite":
            group = sub.add_mutually_exclusive_group()
            group.add_argument("--experiments", nargs="+", choices=sorted(EXPERIMENTS))
            # 双分支方案 §12: stage-a stops after F1 so its result can be judged
            # before F2-F4 are spent.
            group.add_argument(
                "--stage",
                choices=(
                    "stage-a",
                    "fusion",
                    "diagnostic",
                    "temporal",
                    "angular",
                    "evolution",
                    "benchmark",
                    "factorial",
                    "representation",
                    "context",
                    "ordering",
                    "local-long",
                ),
                help="benchmark: the ten Exp 1 encoders; factorial: the Exp 2 R/L/Q "
                "factorial plus the raw XYZ reference; representation: Exp 3 U/D/Q. "
                "Earlier stages: stage-a, fusion, diagnostic, temporal, angular, "
                "evolution. Anchors already trained are reused, never retrained.",
            )
            sub.add_argument("--seeds", nargs="+", type=int, default=[42])
    handcrafted = subparsers.add_parser("handcrafted")
    handcrafted.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    handcrafted.add_argument("--arm", choices=sorted(HANDCRAFTED_ARMS), required=True)
    handcrafted.add_argument("--seed", type=int)
    handcrafted.add_argument("--device", default="cpu")
    handcrafted.add_argument(
        "--checkpoint",
        type=Path,
        help=f"best.pt of the proposed model; required for the hybrid arm "
        f"(normally runs/{HANDCRAFTED_PROPOSED}_seed42/best.pt)",
    )
    evaluation = subparsers.add_parser("evaluate")
    evaluation.add_argument("--checkpoint", required=True, type=Path)
    evaluation.add_argument("--split", choices=("val", "test"), default="test")
    evaluation.add_argument("--device")
    evaluation.add_argument("--limit", type=int)
    report = subparsers.add_parser("tables")
    report.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()

    if args.command == "handcrafted":
        from .classical import run_classical

        config = load_config(args.config)
        if args.seed is not None:
            config["training"]["seed"] = args.seed
        label, kinds = HANDCRAFTED_ARMS[args.arm]
        hybrid = args.arm == "E5_hybrid"
        if hybrid and not args.checkpoint:
            parser.error("The hybrid arm needs --checkpoint of the proposed model")
        result = run_classical(
            config, args.arm, kinds, args.checkpoint if hybrid else None, args.device
        )
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return
    if args.command == "evaluate":
        result = evaluate(args.checkpoint, args.split, args.device, args.limit)
    elif args.command == "tables":
        result = tables(args.root)
    elif args.command == "sanity":
        result = sanity_checks()
        raise SystemExit(0 if result["all_passed"] else 1)
    else:
        config = load_config(args.config)
        for key in ("device", "seed", "epochs", "output"):
            value = getattr(args, key, None)
            if value is not None:
                config["training"][key] = str(value.resolve()) if key == "output" else value
        if getattr(args, "experiment", None):
            config = experiment_config(config, args.experiment)
        validate_config(config)
        if args.command == "audit":
            result = audit(config["data"])
            print(json.dumps(result, ensure_ascii=False, indent=2))
            raise SystemExit(1 if result["missing_file_count"] else 0)
        elif args.command == "prepare":
            result = prepare(config["data"])
        elif args.command == "check-geometry":
            result = check_geometry(config, args.records)
        elif args.command == "feature-stats":
            result = feature_stats(config, args.records)
        elif args.command == "profile":
            result = profile(config)
        elif args.command == "suite":
            stage = {
                "stage-a": list(FUSION_STAGE_A),
                "fusion": list(FUSION),
                "diagnostic": list(DIAGNOSTIC_NEW),
                "temporal": list(TEMPORAL_NEW),
                "angular": list(ANGULAR_NEW),
                "evolution": list(EVOLUTION_NEW),
                "benchmark": list(BENCHMARK),
                "factorial": [*FACTORIAL, FACTORIAL_REFERENCE],
                "representation": list(REPRESENTATION_NEW),
                "context": list(CONTEXT),
                "ordering": list(ORDERING),
                "local-long": [COMBINED],
            }
            names = args.experiments or stage.get(args.stage) or list(EXPERIMENTS)
            result = suite(config, names, args.seeds)
        else:
            # Default the run directory to the experiment name. Without this the
            # name comes from model.variant, and every benchmark experiment shares
            # the variant RLAB, so the runs would be indistinguishable on disk.
            run_name = args.run_name
            if run_name is None and args.experiment:
                run_name = f"{args.experiment}_seed{config['training']['seed']}"
            run_dir = train(config, run_name, args.limit_train, args.limit_val)
            result = evaluate(run_dir / "best.pt", "test", config["training"]["device"])
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
