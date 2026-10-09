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
import time
from pathlib import Path
from typing import Any, Generator, Sequence

import numpy as np

from src.calibration import (
    compute_ece,
    compute_nll,
    fit_temperature,
    load_calibration,
    save_calibration,
)
from src.engine import GamanEngine
from src.resolver import detect_hardware, get_default_cache_dir, resolve_model_path, resolve_slab
from src.serializer import build_nli_pair, serialize_state


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


def handle_predict(args: argparse.Namespace, engine: GamanEngine) -> int:
    state = parse_state_input(args.state)
    try:
        result = engine.predict(state, head_name=args.head)
    except Exception as exc:
        print(f"Error during prediction: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Head:        {result['head']}")
        print(f"Selection:   {result['selection']}")
        print(f"Confidence:  {result['confidence']:.4f}")
        print(f"Latency:     {result['latency_ms']:.2f}ms")
    return 0


def handle_fit(args: argparse.Namespace, engine: GamanEngine) -> int:
    from src.heads import fit_adapter

    t0 = time.perf_counter()
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Dataset not found at: {data_path}", file=sys.stderr)
        return 1

    states: list[dict[str, Any]] = []
    labels: list[str] = []

    with open(data_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if args.target_column not in row:
                print(f"Error: Target column '{args.target_column}' not found in CSV", file=sys.stderr)
                return 1
            lbl = row[args.target_column].strip()
            labels.append(lbl)

            if args.state_column:
                if args.state_column not in row:
                    print(f"Error: State column '{args.state_column}' not found in CSV", file=sys.stderr)
                    return 1
                raw_st = row[args.state_column]
                states.append(parse_state_input(raw_st))
            else:
                row_copy = {k: v for k, v in row.items() if k != args.target_column}
                states.append(row_copy)

    if not states:
        print("Error: Dataset is empty.", file=sys.stderr)
        return 1

    embeddings = engine.embed_batch(states)

    head = fit_adapter(
        embeddings=embeddings,
        labels=labels,
        name=args.name,
        l2_reg=args.l2_reg,
    )

    if args.output:
        save_path = Path(args.output)
    else:
        save_path = engine.models_dir / "heads" / f"{args.name}.json"

    head.save(save_path)
    elapsed = time.perf_counter() - t0

    preds = head.predict_proba(embeddings)
    top_indices = np.argmax(preds, axis=1)
    train_correct = sum(1 for i, idx in enumerate(top_indices) if head.classes[idx] == labels[i])
    train_acc = train_correct / len(labels)

    print("=" * 60)
    print(f"Gaman AI -- Linear Adapter Head Fitted: '{head.name}'")
    print("=" * 60)
    print(f"  Samples (N):        {len(labels)}")
    print(f"  Classes (K):        {len(head.classes)} ({', '.join(head.classes)})")
    print(f"  Hidden Dim (d):     {head.hidden_dim}")
    print(f"  L2 Regularization:  {args.l2_reg}")
    print(f"  Training Accuracy:  {train_acc * 100:.2f}%")
    print(f"  Fitting Time:       {elapsed:.2f}s")
    print(f"  Saved Head:         {save_path}")
    print("=" * 60)
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


# ─── Calibration Handler ─────────────────────────────────────────────────────

def _chunked_generator(
    gen: Generator[dict[str, Any], None, None], chunk_size: int = 32
) -> Generator[list[dict[str, Any]], None, None]:
    """Yield chunks of rows to prevent high memory usage on large datasets."""
    chunk: list[dict[str, Any]] = []
    for item in gen:
        chunk.append(item)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def handle_calibrate(args: argparse.Namespace, engine: GamanEngine) -> int:
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Dataset not found at '{data_path}'", file=sys.stderr)
        return 1

    suffix = data_path.suffix.lower()
    if suffix == ".csv":
        row_gen = _stream_csv_rows(data_path)
    elif suffix in (".jsonl", ".json"):
        row_gen = _stream_jsonl_rows(data_path)
    else:
        print(f"Error: Unsupported format '{suffix}'. Use .csv or .jsonl", file=sys.stderr)
        return 1

    primitive = args.primitive
    target_col = args.target_column
    state_col = args.state_column

    chunk_logits: list[np.ndarray] = []
    chunk_targets: list[np.ndarray] = []
    total_samples = 0

    if primitive == "choice":
        if not args.options:
            print("Error: --options is required for choice calibration.", file=sys.stderr)
            return 1
        options = _parse_options_list(args.options)
        clean_options = [opt.replace("_", " ") for opt in options]
        k_options = len(clean_options)

        for chunk in _chunked_generator(row_gen, chunk_size=16):
            pairs_to_encode: list[tuple[str, str]] = []
            targets_chunk: list[int] = []

            for row in chunk:
                if target_col not in row:
                    print(f"Error: Target column '{target_col}' not found in row.", file=sys.stderr)
                    return 1
                target_raw = str(row[target_col]).strip()
                if target_raw in options:
                    target_idx = options.index(target_raw)
                else:
                    try:
                        target_idx = int(target_raw)
                    except ValueError:
                        print(
                            f"Error: Target value '{target_raw}' does not match options: {options}",
                            file=sys.stderr,
                        )
                        return 1

                state = _extract_state(row, state_col)
                premise = serialize_state(state)
                for opt in clean_options:
                    pairs_to_encode.append((premise, f"The decision is to {opt}."))
                targets_chunk.append(target_idx)

            batch_enc = engine.tokenizer.encode_batch(pairs_to_encode)
            logits = engine._forward(batch_enc)
            entailment_logits = logits[:, engine.entailment_idx].reshape(len(chunk), k_options)
            chunk_logits.append(entailment_logits.astype(np.float32))
            chunk_targets.append(np.array(targets_chunk, dtype=np.int64))
            total_samples += len(chunk)

    elif primitive == "noul":
        if not args.predicate:
            print("Error: --predicate is required for noul calibration.", file=sys.stderr)
            return 1
        hypothesis = engine._normalize_predicate(args.predicate)

        for chunk in _chunked_generator(row_gen, chunk_size=32):
            pairs_to_encode = []
            targets_chunk = []

            for row in chunk:
                if target_col not in row:
                    print(f"Error: Target column '{target_col}' not found in row.", file=sys.stderr)
                    return 1
                target_raw = str(row[target_col]).strip().lower()
                target_val = 1 if target_raw in ("1", "true", "yes", "pass", "passed") else 0

                state = _extract_state(row, state_col)
                premise, hyp = build_nli_pair(state, hypothesis)
                pairs_to_encode.append((premise, hyp))
                targets_chunk.append(target_val)

            batch_enc = engine.tokenizer.encode_batch(pairs_to_encode)
            logits = engine._forward(batch_enc)
            chunk_logits.append(logits.astype(np.float32))
            chunk_targets.append(np.array(targets_chunk, dtype=np.int64))
            total_samples += len(chunk)

    elif primitive == "score":
        if not args.criterion:
            print("Error: --criterion is required for score calibration.", file=sys.stderr)
            return 1

        for chunk in _chunked_generator(row_gen, chunk_size=32):
            pairs_to_encode = []
            targets_chunk = []

            for row in chunk:
                if target_col not in row:
                    print(f"Error: Target column '{target_col}' not found in row.", file=sys.stderr)
                    return 1
                try:
                    target_num = float(row[target_col])
                    target_val = 1 if target_num >= 0.5 else 0
                except ValueError:
                    target_raw = str(row[target_col]).strip().lower()
                    target_val = 1 if target_raw in ("1", "true", "yes", "high") else 0

                state = _extract_state(row, state_col)
                premise, hyp = build_nli_pair(state, args.criterion)
                pairs_to_encode.append((premise, hyp))
                targets_chunk.append(target_val)

            batch_enc = engine.tokenizer.encode_batch(pairs_to_encode)
            logits = engine._forward(batch_enc)
            chunk_logits.append(logits.astype(np.float32))
            chunk_targets.append(np.array(targets_chunk, dtype=np.int64))
            total_samples += len(chunk)

    if total_samples == 0:
        print("Error: Dataset is empty.", file=sys.stderr)
        return 1

    Z = np.concatenate(chunk_logits, axis=0)
    Y = np.concatenate(chunk_targets, axis=0)

    # Initial uncalibrated metrics (T=1.0)
    init_nll = compute_nll(Z, Y, temperature=1.0)
    if primitive == "choice":
        init_exp = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        init_probs = init_exp / np.sum(init_exp, axis=1, keepdims=True)
        init_preds = np.argmax(init_probs, axis=1)
        init_confs = np.max(init_probs, axis=1)
    else:
        init_exp = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        init_p_all = init_exp / np.sum(init_exp, axis=1, keepdims=True)
        init_p = init_p_all[:, engine.entailment_idx]
        init_preds = (init_p > 0.5).astype(int)
        init_confs = np.where(init_preds == 1, init_p, 1.0 - init_p)

    ece_before = compute_ece(init_confs, init_preds, Y, strategy="quantile")

    # Optimize Temperature (with inverse beta convexity)
    T_opt = fit_temperature(Z, Y, bounds=(0.1, 10.0), enforce_gating=False if total_samples < 30 else True)

    # Calibrated metrics (T=T_opt)
    cal_nll = compute_nll(Z, Y, temperature=T_opt)
    if primitive == "choice":
        scaled_Z = Z / T_opt
        cal_exp = np.exp(scaled_Z - np.max(scaled_Z, axis=1, keepdims=True))
        cal_probs = cal_exp / np.sum(cal_exp, axis=1, keepdims=True)
        cal_preds = np.argmax(cal_probs, axis=1)
        cal_confs = np.max(cal_probs, axis=1)
    else:
        scaled_Z = Z / T_opt
        cal_exp = np.exp(scaled_Z - np.max(scaled_Z, axis=1, keepdims=True))
        cal_p_all = cal_exp / np.sum(cal_exp, axis=1, keepdims=True)
        cal_p = cal_p_all[:, engine.entailment_idx]
        cal_preds = (cal_p > 0.5).astype(int)
        cal_confs = np.where(cal_preds == 1, cal_p, 1.0 - cal_p)

    ece_after = compute_ece(cal_confs, cal_preds, Y, strategy="quantile")

    # Output file destination
    cal_file = Path(args.output) if args.output else (engine.models_dir / "calibration.json")
    cal_config = load_calibration(cal_file)
    cal_config["temperatures"][primitive] = T_opt
    cal_config["ece_before"] = round(ece_before, 4)
    cal_config["ece_after"] = round(ece_after, 4)
    save_calibration(cal_config, cal_file)

    # Reload into active engine
    engine.load_calibration(cal_file)

    print("==================================================")
    print(f"Calibration Complete for [{primitive}]")
    print("==================================================")
    print(f"Dataset:            {data_path} ({total_samples} samples)")
    print(f"Optimal Temp (T*):  {T_opt:.4f}")
    print(f"NLL Loss:           {init_nll:.4f} -> {cal_nll:.4f}")
    print(f"ECE (Quantile):     {ece_before * 100:.2f}% -> {ece_after * 100:.2f}% (Delta: {(ece_before - ece_after) * 100:+.2f}%)")
    print(f"Saved:              {cal_file}")
    return 0


# ─── System Diagnostic Handler ───────────────────────────────────────────────

def handle_info(args: argparse.Namespace) -> int:
    profile = detect_hardware()
    slab = resolve_slab(requested_slab=args.slab, profile=profile)
    model_dir = resolve_model_path(tier=slab.tier, models_dir=args.models_dir, auto_download=False)
    cache_dir = get_default_cache_dir()
    models_cache = cache_dir / "models"

    vram_gb = (
        round(profile.vram_bytes / (1024**3), 2)
        if profile.vram_bytes is not None
        else None
    )
    ram_gb = round(profile.system_ram_bytes / (1024**3), 2)

    if args.json:
        payload = {
            "hardware": {
                "providers": profile.providers,
                "system_ram_gb": ram_gb,
                "vram_gb": vram_gb,
                "cpu_cores": profile.cpu_cores,
                "has_avx512_or_arm_neon": profile.has_avx512_or_arm_neon,
            },
            "slab": {
                "requested": args.slab,
                "resolved_tier": slab.tier,
                "model_id": slab.model_id,
                "quantization": slab.quantization,
                "hidden_dim": slab.hidden_dim,
                "selected_provider": slab.provider,
                "model_dir": str(model_dir),
                "model_dir_exists": (model_dir / "backbone.onnx").exists(),
            },
            "cache": {
                "cache_dir": str(cache_dir),
                "models_cache_exists": models_cache.exists(),
            },
        }
        print(json.dumps(payload, indent=2))
    else:
        print("============================================================")
        print("Gaman AI -- Hardware Profile & Spec Slab Resolution")
        print("============================================================")
        print("Hardware Profile:")
        print(f"  System RAM:        {ram_gb:.1f} GB")
        print(f"  Dedicated VRAM:    {vram_gb if vram_gb is not None else 'None'}")
        print(f"  Logical CPU Cores: {profile.cpu_cores}")
        print(f"  SIMD Acceleration: {profile.has_avx512_or_arm_neon}")
        print(f"  Providers:         {', '.join(profile.providers)}")
        print("\nSpec Slab Configuration:")
        print(f"  Requested Slab:    {args.slab}")
        print(f"  Resolved Tier:     {slab.tier.upper()}")
        print(f"  Model ID:          {slab.model_id}")
        print(f"  Quantization:      {slab.quantization}")
        print(f"  Hidden Dimension:  {slab.hidden_dim}")
        print(f"  Selected Provider: {slab.provider}")
        print(f"  Model Path:        {model_dir} (exists: {(model_dir / 'backbone.onnx').exists()})")
        print(f"  Global Cache:      {cache_dir} (exists: {models_cache.exists()})")
        print("============================================================")
    return 0


def handle_cache(args: argparse.Namespace) -> int:
    import shutil

    cache_dir = get_default_cache_dir()
    models_cache = cache_dir / "models"

    if args.dir:
        print(str(cache_dir))
        return 0

    if args.clean:
        if models_cache.exists():
            shutil.rmtree(models_cache)
            print(f"Cache cleared: {models_cache}")
        else:
            print(f"Cache is already clean: {models_cache}")
        return 0

    print(f"Gaman AI Cache Directory: {cache_dir}")
    print(f"Models Cache Exists:      {models_cache.exists()}")
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
        default=None,
        help="Path to models artifact directory (default: None, auto-resolved).",
    )
    parser.add_argument(
        "--slab",
        type=str,
        default="auto",
        choices=["auto", "small", "base", "large"],
        help="Target Spec Slab tier (default: 'auto').",
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

    # 5. calibrate
    p_calibrate = subparsers.add_parser("calibrate", help="Optimize temperature scaling on validation dataset")
    p_calibrate.add_argument("--data", type=str, required=True, help="Path to validation CSV or JSONL dataset")
    p_calibrate.add_argument(
        "--primitive",
        type=str,
        choices=["choice", "noul", "score"],
        required=True,
        help="Primitive to calibrate",
    )
    p_calibrate.add_argument("--target-column", type=str, required=True, help="Column containing ground truth labels")
    p_calibrate.add_argument("--state-column", type=str, default=None, help="Column containing input state (default: all columns)")
    p_calibrate.add_argument("--options", type=str, nargs="+", default=None, help="Options for choice primitive")
    p_calibrate.add_argument("--predicate", type=str, default=None, help="Predicate for noul primitive")
    p_calibrate.add_argument("--criterion", type=str, default=None, help="Criterion for score primitive")
    p_calibrate.add_argument("--output", type=str, default=None, help="Path to write calibration.json (default: models/calibration.json)")

    # 6. info
    p_info = subparsers.add_parser("info", help="Print detected hardware topology and active spec slab")
    p_info.add_argument("--json", action="store_true", help="Output details as JSON")

    # 7. fit
    p_fit = subparsers.add_parser("fit", help="Fit a lightweight linear adapter head on labeled data")
    p_fit.add_argument("--data", type=str, required=True, help="Path to input dataset (CSV)")
    p_fit.add_argument(
        "--state-column",
        type=str,
        default=None,
        help="Column to extract state from (default: all row columns except target)",
    )
    p_fit.add_argument(
        "--target-column",
        type=str,
        required=True,
        help="Column containing target category label",
    )
    p_fit.add_argument("--name", type=str, required=True, help="Identifier name for the adapter head")
    p_fit.add_argument(
        "--l2-reg",
        type=float,
        default=1.0,
        help="L2 regularization strength (default: 1.0)",
    )
    p_fit.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom output path for head json (default: models/heads/<name>.json)",
    )

    # 8. predict
    p_pred = subparsers.add_parser("predict", help="Predict using a custom linear adapter head")
    p_pred.add_argument("--state", type=str, required=True, help="Input state (JSON or text)")
    p_pred.add_argument("--head", type=str, required=True, help="Adapter head name")
    p_pred.add_argument("--json", action="store_true", help="Output raw schema JSON")

    # 9. cache
    p_cache = subparsers.add_parser("cache", help="Manage Gaman AI downloaded models cache")
    p_cache.add_argument("--dir", action="store_true", help="Print active cache directory path")
    p_cache.add_argument("--clean", action="store_true", help="Wipe downloaded model weights cache")

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

    if args.subcommand == "info":
        return handle_info(args)
    elif args.subcommand == "cache":
        return handle_cache(args)

    silent = getattr(args, "json", False)
    if engine is None:
        try:
            engine = GamanEngine(slab=args.slab, models_dir=args.models_dir, silent=silent)
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
    elif args.subcommand == "calibrate":
        return handle_calibrate(args, engine)
    elif args.subcommand == "fit":
        return handle_fit(args, engine)
    elif args.subcommand == "predict":
        return handle_predict(args, engine)

    return 0


if __name__ == "__main__":
    sys.exit(main())
