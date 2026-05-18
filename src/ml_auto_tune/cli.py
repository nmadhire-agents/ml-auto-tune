from __future__ import annotations

import argparse
from pathlib import Path

from ml_auto_tune.config import load_config
from ml_auto_tune.sample_data import make_sample_data
from ml_auto_tune.tuning import run_tuning


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="ml-auto-tune")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run model tuning from a YAML config.")
    run_parser.add_argument("--config", required=True, help="Path to a tuning YAML config.")

    sample_parser = subparsers.add_parser("make-sample-data", help="Generate the bundled sample CSV.")
    sample_parser.add_argument("--output", default="data/sample_regression.csv", help="CSV output path.")
    sample_parser.add_argument("--rows", type=int, default=180, help="Number of rows to include.")
    sample_parser.add_argument("--random-state", type=int, default=42, help="Sampling seed.")

    args = parser.parse_args(argv)
    if args.command == "run":
        config = load_config(args.config)
        result = run_tuning(config)
        print(f"Best score: {result.best_score:.6f}")
        print(f"Artifacts: {result.output_directory}")
        return

    if args.command == "make-sample-data":
        output = make_sample_data(Path(args.output), rows=args.rows, random_state=args.random_state)
        print(f"Wrote sample data: {output}")
        return

    parser.error(f"Unknown command: {args.command}")

