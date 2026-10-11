#!/usr/bin/env python3
"""
scripts/optimize_model.py — Fused ONNX Graph Optimization for Gaman AI
=======================================================================
Generates a hardware-fused, statically optimized ONNX graph for CPU execution.
Applies constant folding, node eliminations, redundant reshape/transpose fusions,
and Nchwc layout transformations via ONNX Runtime graph optimization.

Zero runtime dependency on PyTorch.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import onnxruntime as ort


def optimize_onnx_model(input_path: Path, output_path: Path) -> bool:
    """
    Optimize an ONNX model graph and serialize the optimized artifact to output_path.

    Attempts transformers fused optimizer if available; seamlessly falls back
    to native ONNX Runtime C++ graph serialization with ORT_ENABLE_ALL.
    """
    if not input_path.exists():
        print(f"[OPTIMIZER] Error: Input model '{input_path}' does not exist.", file=sys.stderr)
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    input_size_mb = input_path.stat().st_size / (1024 * 1024)
    print(f"[OPTIMIZER] Optimizing '{input_path}' ({input_size_mb:.2f} MB)...")

    # Strategy 1: Attempt onnxruntime.transformers optimizer if available
    optimized = False
    try:
        from onnxruntime.transformers.optimizer import optimize_model  # type: ignore[import]

        opt_model = optimize_model(
            str(input_path),
            model_type="bert",
            num_heads=12,
            hidden_size=768,
        )
        opt_model.save_model_to_file(str(output_path))
        optimized = True
        print("[OPTIMIZER] Fused graph generated via onnxruntime.transformers.optimizer.")
    except Exception as exc:
        print(f"[OPTIMIZER] Note: transformers optimizer skipped ({exc}). Using native ORT C++ engine...")

    # Strategy 2: Native ONNX Runtime C++ graph serialization (constant folding, fusions)
    if not optimized:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.optimized_model_filepath = str(output_path)

        # Creating an InferenceSession with optimized_model_filepath compiles and serializes the graph
        _ = ort.InferenceSession(
            str(input_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

    if not output_path.exists() or output_path.stat().st_size == 0:
        print(f"[OPTIMIZER] Error: Failed to write optimized graph to '{output_path}'.", file=sys.stderr)
        return False

    output_size_mb = output_path.stat().st_size / (1024 * 1024)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    print(
        f"[OPTIMIZER] Successfully generated '{output_path}' "
        f"({output_size_mb:.2f} MB) in {elapsed_ms:.2f} ms."
    )

    # Verification: Validate that the optimized model loads cleanly in ORT
    try:
        test_opts = ort.SessionOptions()
        test_session = ort.InferenceSession(
            str(output_path),
            sess_options=test_opts,
            providers=["CPUExecutionProvider"],
        )
        assert len(test_session.get_inputs()) > 0
        print("[OPTIMIZER] Verification passed: Optimized model loads cleanly.")
        return True
    except Exception as exc:
        print(f"[OPTIMIZER] Verification failed: {exc}", file=sys.stderr)
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gaman AI — Fused ONNX Graph Optimization Utility"
    )
    parser.add_argument(
        "--model-path",
        "--model_path",
        type=Path,
        default=None,
        help="Path to unoptimized input ONNX model (default: models/backbone.onnx)",
    )
    parser.add_argument(
        "--output-path",
        "--output_path",
        type=Path,
        default=None,
        help="Path to write optimized ONNX model (default: models/backbone_optimized.onnx)",
    )
    parser.add_argument(
        "--tier",
        type=str,
        default=None,
        choices=["all", "small", "base", "large"],
        help="Model tier to optimize, or 'all' to optimize root and all tiered directories.",
    )

    args = parser.parse_args()

    # If tier is specified
    if args.tier:
        models_dir = Path("models")
        targets: list[tuple[Path, Path]] = []
        if args.tier == "all":
            root_input = models_dir / "backbone.onnx"
            if root_input.exists():
                targets.append((root_input, models_dir / "backbone_optimized.onnx"))
            for tier_name in ("small", "base", "large"):
                t_input = models_dir / tier_name / "backbone.onnx"
                if t_input.exists():
                    targets.append((t_input, models_dir / tier_name / "backbone_optimized.onnx"))
        else:
            t_input = models_dir / args.tier / "backbone.onnx"
            targets.append((t_input, models_dir / args.tier / "backbone_optimized.onnx"))

        success = True
        for inp, outp in targets:
            if not optimize_onnx_model(inp, outp):
                success = False
        return 0 if success else 1

    input_path = args.model_path or Path("models/backbone.onnx")
    output_path = args.output_path or (input_path.parent / f"{input_path.stem}_optimized.onnx")

    ok = optimize_onnx_model(input_path, output_path)

    # If optimizing root models/backbone.onnx, also mirror to models/small/ if it exists
    small_dir = Path("models/small")
    if ok and input_path == Path("models/backbone.onnx") and small_dir.exists():
        optimize_onnx_model(small_dir / "backbone.onnx", small_dir / "backbone_optimized.onnx")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
