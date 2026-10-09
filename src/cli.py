"""
src/cli.py — Universal CLI for Gaman AI
=======================================
Exposes single-shot commands (choice, noul, score) and high-throughput
streaming batch processing (gaman batch).

Universal CLI Invariants:
- Single-shot commands provide human-readable output by default.
- --json outputs exact schema from docs/api_contract.md.
- Streaming generator ingestion for batch processing (O(1) RAM usage).
- Complete preservation of input metadata in batch mode.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Generator, Sequence

from src.engine import GamanEngine


def parse_state_input(state_input: str) -> dict[str, Any]:
    """Parse state from JSON string or wrap plain text as dict."""
    state_input = state_input.strip()
    if (state_input.startswith("{") and state_input.endswith("}")) or (
        state_input.startswith("[") and state_input.endswith("]")
    ):
        try:
            parsed = json.loads(state_input)
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list):
                return {"items": parsed}
        except Exception:
            pass
    return {"text": state_input}


def _parse_options_list(options_args: list[str]) -> list[str]:
    """Parse options list from either whitespace-separated or comma-separated tokens."""
    if len(options_args) == 1 and "," in options_args[0]:
        return [opt.strip() for opt in options_args[0].split(",") if opt.strip()]
    return [opt.strip() for opt in options_args if opt.strip()]


# ─── Single-Shot Handlers ────────────────────────────────────────────────────

def handle_choice(args: argparse.Namespace, engine: GamanEngine) -> int:
    state = parse_state_input(args.state)
    options = _parse_options_list(args.options)
    if not options:
        print("Error: At least one option must be provided.", file=sys.stderr)
        return 1

    result = engine.choice(state, options)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Selection:  {result['selection']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Latency:    {result['latency_ms']:.2f}ms")
    return 0


def handle_noul(args: argparse.Namespace, engine: GamanEngine) -> int:
    state = parse_state_input(args.state)
    result = engine.noul(state, args.predicate)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        status_str = "PASSED" if result["passed"] else "FAILED"
        print(f"Predicate:   {status_str}")
        print(f"Probability: {result['probability']:.4f}")
        print(f"Latency:     {result['latency_ms']:.2f}ms")
    return 0


def handle_score(args: argparse.Namespace, engine: GamanEngine) -> int:
    state = parse_state_input(args.state)
    result = engine.score(state, args.criterion)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Score:   {result['value']:.4f}")
        print(f"Latency: {result['latency_ms']:.2f}ms")
    return 0


# ─── Batch Handlers ──────────────────────────────────────────────────────────

def _stream_csv_rows(file_path: Path) -> Generator[dict[str, Any], None, None]:
    """Stream CSV rows one by one to avoid loading entire file into memory."""
    with open(file_path, mode="r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def _stream_jsonl_rows(file_path: Path) -> Generator[dict[str, Any], None, None]:
    """Stream JSONL rows one by one to avoid loading entire file into memory."""
    with open(file_path, mode="r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _extract_state(row: dict[str, Any], state_column: str | None) -> dict[str, Any]:
    """Extract target state dict based on --state-column argument."""
    if state_column is not None:
        if state_column not in row:
            raise KeyError(f"Specified --state-column '{state_column}' not found in row keys.")
        val = row[state_column]
        if isinstance(val, dict):
            return val
        if isinstance(val, str):
            return parse_state_input(val)
        return {"value": val}
    return row


def handle_batch(args: argparse.Namespace, engine: GamanEngine) -> int:
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found at '{input_path}'", file=sys.stderr)
        return 1

    # Infer file formats
    input_suffix = input_path.suffix.lower()
    is_csv_input = input_suffix == ".csv"
    is_jsonl_input = input_suffix in (".jsonl", ".json")

    if not is_csv_input and not is_jsonl_input:
        print(
            f"Error: Unsupported file format '{input_suffix}'. Please provide .csv or .jsonl",
            file=sys.stderr,
        )
        return 1

    output_path = Path(args.output) if args.output else None
    if output_path is not None:
        output_is_csv = output_path.suffix.lower() == ".csv"
    else:
        output_is_csv = is_csv_input

    # Validate primitive requirements
    primitive = args.primitive
    if primitive == "choice":
        if not args.options:
            print("Error: --options is required for choice primitive in batch mode.", file=sys.stderr)
            return 1
        options = _parse_options_list(args.options)
    elif primitive == "noul":
        if not args.predicate:
            print("Error: --predicate is required for noul primitive in batch mode.", file=sys.stderr)
            return 1
    elif primitive == "score":
        if not args.criterion:
            print("Error: --criterion is required for score primitive in batch mode.", file=sys.stderr)
            return 1
    else:
        print(f"Error: Unknown primitive '{primitive}'", file=sys.stderr)
        return 1

    row_stream = _stream_csv_rows(input_path) if is_csv_input else _stream_jsonl_rows(input_path)

    # Process and write streaming
    out_file = open(output_path, mode="w", encoding="utf-8", newline="") if output_path else sys.stdout

    csv_writer: csv.DictWriter | None = None
    processed_count = 0

    try:
        for row in row_stream:
            state = _extract_state(row, args.state_column)

            if primitive == "choice":
                eval_res = engine.choice(state, options)
            elif primitive == "noul":
                eval_res = engine.noul(state, args.predicate)
            else:
                eval_res = engine.score(state, args.criterion)

            if output_is_csv:
                # Augment row with gaman_* columns
                augmented_row = dict(row)
                if primitive == "choice":
                    augmented_row["gaman_selection"] = eval_res["selection"]
                    augmented_row["gaman_confidence"] = eval_res["confidence"]
                elif primitive == "noul":
                    augmented_row["gaman_passed"] = eval_res["passed"]
                    augmented_row["gaman_probability"] = eval_res["probability"]
                elif primitive == "score":
                    augmented_row["gaman_score"] = eval_res["value"]
                augmented_row["gaman_latency_ms"] = eval_res["latency_ms"]

                if csv_writer is None:
                    fieldnames = list(augmented_row.keys())
                    csv_writer = csv.DictWriter(out_file, fieldnames=fieldnames)
                    csv_writer.writeheader()
                csv_writer.writerow(augmented_row)
            else:
                augmented_row = dict(row)
                augmented_row["gaman"] = eval_res
                out_file.write(json.dumps(augmented_row) + "\n")

            processed_count += 1
    finally:
        if output_path:
            out_file.close()

    if output_path:
        print(f"[OK] Processed {processed_count} rows -> {output_path}")

    return 0


# ─── Argument Parser Construction ────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gaman",
        description="Gaman AI -- High-performance, local System One decision engine.",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Path to models artifact directory (default: 'models').",
    )

    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # 1. choice
    p_choice = subparsers.add_parser("choice", help="Semantic routing over K options")
    p_choice.add_argument("--state", type=str, required=True, help="Input state (JSON or text)")
    p_choice.add_argument(
        "--options",
        type=str,
        nargs="+",
        required=True,
        help="List of options (space or comma separated)",
    )
    p_choice.add_argument("--json", action="store_true", help="Output raw schema JSON")

    # 2. noul
    p_noul = subparsers.add_parser("noul", help="Guardrail / predicate compliance check")
    p_noul.add_argument("--state", type=str, required=True, help="Input state (JSON or text)")
    p_noul.add_argument("--predicate", type=str, required=True, help="Predicate compliance condition")
    p_noul.add_argument("--json", action="store_true", help="Output raw schema JSON")

    # 3. score
    p_score = subparsers.add_parser("score", help="Continuous context evaluation [0.0, 1.0]")
    p_score.add_argument("--state", type=str, required=True, help="Input state (JSON or text)")
    p_score.add_argument("--criterion", type=str, required=True, help="Continuous evaluation criterion")
    p_score.add_argument("--json", action="store_true", help="Output raw schema JSON")

    # 4. batch
    p_batch = subparsers.add_parser("batch", help="Stream evaluation over CSV/JSONL datasets")
    p_batch.add_argument("--input", type=str, required=True, help="Path to input .csv or .jsonl file")
    p_batch.add_argument("--output", type=str, default=None, help="Path to write output file (default: stdout)")
    p_batch.add_argument(
        "--primitive",
        type=str,
        choices=["choice", "noul", "score"],
        required=True,
        help="Evaluation primitive to execute",
    )
    p_batch.add_argument("--state-column", type=str, default=None, help="Column to use as state (default: all columns)")
    p_batch.add_argument("--options", type=str, nargs="+", default=None, help="Options for choice primitive")
    p_batch.add_argument("--predicate", type=str, default=None, help="Predicate for noul primitive")
    p_batch.add_argument("--criterion", type=str, default=None, help="Criterion for score primitive")
    p_batch.add_argument("--batch-size", type=int, default=16, help="Evaluation batch size (default: 16)")

    return parser


def main(
    argv: Sequence[str] | None = None,
    engine: GamanEngine | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.subcommand:
        parser.print_help()
        return 0

    if engine is None:
        try:
            engine = GamanEngine(models_dir=args.models_dir)
        except Exception as exc:
            print(f"Error initializing GamanEngine: {exc}", file=sys.stderr)
            return 1

    if args.subcommand == "choice":
        return handle_choice(args, engine)
    elif args.subcommand == "noul":
        return handle_noul(args, engine)
    elif args.subcommand == "score":
        return handle_score(args, engine)
    elif args.subcommand == "batch":
        return handle_batch(args, engine)

    return 0


if __name__ == "__main__":
    sys.exit(main())
