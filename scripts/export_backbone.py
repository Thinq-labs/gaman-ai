#!/usr/bin/env python3
"""
Gaman AI -- Backbone Export Utility
====================================
Downloads a pre-trained Hugging Face NLI encoder, exports it to ONNX (FP32),
applies dynamic INT8 quantization, and saves all inference artifacts to models/.

This script is the ONLY place where torch / transformers / optimum are used.
These packages are hard-banned from src/ (the inference runtime).

Usage
-----
    # Default: DeBERTa-v3-small NLI -> models/
    python scripts/export_backbone.py

    # Spec Slab tier export
    python scripts/export_backbone.py --tier small
    python scripts/export_backbone.py --tier base
    python scripts/export_backbone.py --tier large

    # Custom model
    python scripts/export_backbone.py --model_id answerdotai/ModernBERT-small

    # Keep intermediate FP32 model for inspection
    python scripts/export_backbone.py --keep_fp32

    # Export FP32 only (skip quantization)
    python scripts/export_backbone.py --skip_quantization

    # Custom output directory
    python scripts/export_backbone.py --output_dir /tmp/gaman_models

Output Artifacts (in --output_dir)
-----------------------------------
    backbone.onnx           - INT8 quantized model (primary runtime artifact)
    backbone_fp32.onnx      - FP32 model (only with --keep_fp32)
    tokenizer.json          - HF fast tokenizer
    tokenizer_config.json
    special_tokens_map.json
    config.json             - Model config (num_labels, label2id, id2label)
    spm.model               - SentencePiece vocab (DeBERTa)
    manifest.json           - Export provenance record
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

# ─── Constants ──────────────────────────────────────────────────────────────
DEFAULT_MODEL_ID = "cross-encoder/nli-deberta-v3-small"
DEFAULT_OUTPUT_DIR = "models"

TIER_CONFIGS = {
    "small": {
        "model_id": "cross-encoder/nli-deberta-v3-small",
        "output_dir": "models/small",
        "quantize": True,
    },
    "base": {
        "model_id": "cross-encoder/nli-deberta-v3-base",
        "output_dir": "models/base",
        "quantize": True,
    },
    "large": {
        "model_id": "cross-encoder/nli-deberta-v3-large",
        "output_dir": "models/large",
        "quantize": True,
    },
}

# Artifacts to copy from the optimum export directory to the final output dir.
# Listed in priority order; missing files are silently skipped.
TOKENIZER_ARTIFACTS = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "config.json",
    "spm.model",       # SentencePiece (DeBERTa)
    "vocab.json",      # BPE models (BERT, RoBERTa)
    "merges.txt",      # BPE merge rules
    "vocab.txt",       # WordPiece (classic BERT)
]


# ─── CLI ────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a Hugging Face NLI encoder to ONNX and apply dynamic INT8 quantization."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--tier",
        type=str,
        choices=["small", "base", "large"],
        default=None,
        help="Spec Slab tier to export ('small', 'base', 'large'). Sets default model_id and output_dir.",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help=f"Hugging Face model ID to export. Default: {DEFAULT_MODEL_ID}",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to write ONNX artifacts. Default: '{DEFAULT_OUTPUT_DIR}'",
    )
    parser.add_argument(
        "--skip_quantization",
        action="store_true",
        help="Skip INT8 quantization and export FP32 model only.",
    )
    parser.add_argument(
        "--keep_fp32",
        action="store_true",
        help="Keep the intermediate FP32 ONNX file as backbone_fp32.onnx after quantization.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version for export. Default: 17",
    )
    args = parser.parse_args()
    if args.tier:
        tier_cfg = TIER_CONFIGS[args.tier]
        if args.model_id == DEFAULT_MODEL_ID:
            args.model_id = tier_cfg["model_id"]
        if args.output_dir == DEFAULT_OUTPUT_DIR:
            args.output_dir = tier_cfg["output_dir"]
    return args


# ─── Step 1: Export to ONNX ─────────────────────────────────────────────────
def export_to_onnx(model_id: str, tmp_dir: Path, opset: int) -> Path:
    """
    Download model from HF Hub and export to ONNX.
    If the repository already hosts pre-exported ONNX weights (e.g. cross-encoder/nli-deberta-v3-small),
    downloads them and tokenizer files directly via huggingface_hub.
    Otherwise, falls back to optimum.onnxruntime.

    Returns the path to the exported FP32 .onnx file inside tmp_dir.
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    # Fast path: check if pre-exported ONNX graph is available on HF Hub
    try:
        from huggingface_hub import HfApi, hf_hub_download
        api = HfApi()
        files = api.list_repo_files(model_id)
        onnx_candidates = ["onnx/model.onnx", "model.onnx"]
        found_onnx = next((c for c in onnx_candidates if c in files), None)
        if found_onnx:
            print(f"  Downloading pre-exported ONNX artifact '{found_onnx}' for '{model_id}'...")
            downloaded = hf_hub_download(model_id, found_onnx)
            target = tmp_dir / "model.onnx"
            shutil.copy2(downloaded, target)

            # Download tokenizer and config artifacts
            for fname in TOKENIZER_ARTIFACTS:
                if fname in files:
                    art = hf_hub_download(model_id, fname)
                    shutil.copy2(art, tmp_dir / fname)

            elapsed = time.perf_counter() - t0
            print(f"  [OK] Direct ONNX download done ({elapsed:.1f}s)")
            size_mb = target.stat().st_size / (1024 * 1024)
            print(f"  [INFO] FP32 model size: {size_mb:.1f} MB")
            return target
    except Exception as exc:
        print(f"  [INFO] Direct download check skipped ({exc}), falling back to optimum export...")

    # Full export path via optimum
    _check_export_deps()

    from optimum.onnxruntime import ORTModelForSequenceClassification  # type: ignore[import]
    from transformers import AutoTokenizer  # type: ignore[import]

    print(f"  Downloading and exporting '{model_id}' to ONNX via optimum (opset={opset})...")

    # Export model + bake NLI classification head into ONNX graph
    model = ORTModelForSequenceClassification.from_pretrained(
        model_id,
        export=True,
        provider="CPUExecutionProvider",
    )
    model.save_pretrained(tmp_dir)

    # Save tokenizer alongside
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(tmp_dir)

    elapsed = time.perf_counter() - t0
    print(f"  [OK] ONNX export done ({elapsed:.1f}s)")

    # optimum saves the ONNX as model.onnx; guard against version differences
    fp32_onnx = tmp_dir / "model.onnx"
    if not fp32_onnx.exists():
        candidates = sorted(tmp_dir.glob("*.onnx"))
        if not candidates:
            raise FileNotFoundError(
                f"optimum export produced no .onnx file in {tmp_dir}. "
                "Check optimum version compatibility."
            )
        fp32_onnx = candidates[0]
        print(f"  [INFO] Found ONNX at non-standard path: {fp32_onnx.name}")

    size_mb = fp32_onnx.stat().st_size / (1024 * 1024)
    print(f"  [INFO] FP32 model size: {size_mb:.1f} MB")
    return fp32_onnx


# ─── Step 2: Quantize to INT8 ───────────────────────────────────────────────
def quantize_int8(fp32_path: Path, output_path: Path) -> None:
    """
    Apply dynamic INT8 quantization to the FP32 ONNX model.

    Targets MatMul and Gemm operators (the dominant cost in transformer FFN
    and attention projection layers). Uses QInt8 weight type for maximum
    compatibility with AVX-512/VNNI/NEON CPU extensions.
    """
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic  # type: ignore[import]
    except ImportError as exc:
        _die(f"onnxruntime.quantization not available: {exc}")

    print(f"  Applying dynamic INT8 quantization...")
    t0 = time.perf_counter()

    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8,
    )

    elapsed = time.perf_counter() - t0
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  [OK] INT8 quantization done ({elapsed:.1f}s) -> {size_mb:.1f} MB")


# ─── Step 3: Copy tokenizer artifacts ───────────────────────────────────────
def copy_artifacts(src_dir: Path, dst_dir: Path) -> list[str]:
    """
    Copy tokenizer and config artifacts from the optimum export dir to the
    final output directory. Returns the list of files successfully copied.

    Raises SystemExit if tokenizer.json is absent — the runtime requires it
    for fully offline operation (zero network calls at inference time).
    """
    copied: list[str] = []
    for filename in TOKENIZER_ARTIFACTS:
        src = src_dir / filename
        if src.exists():
            shutil.copy2(src, dst_dir / filename)
            copied.append(filename)

    # Hard-fail guard: tokenizer.json is non-negotiable for offline runtime
    if "tokenizer.json" not in copied:
        _die(
            "tokenizer.json was NOT produced by the optimum export.\n"
            "  This means the model does not have a HuggingFace Fast Tokenizer.\n"
            "  The Gaman AI runtime requires tokenizer.json for fully offline operation.\n"
            "  Solution: choose a model with a fast tokenizer (most modern HF models do)."
        )

    return copied


# ─── Step 4: Write manifest ──────────────────────────────────────────────────
def write_manifest(
    output_dir: Path,
    model_id: str,
    quantized: bool,
    artifacts: list[str],
) -> None:
    """Write a JSON provenance record so the engine can validate its artifacts."""
    manifest = {
        "model_id": model_id,
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "backbone_onnx": "backbone.onnx",
        "quantization": "dynamic_int8" if quantized else "fp32",
        "artifacts": ["backbone.onnx"] + artifacts,
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"  [OK] Manifest written -> {manifest_path.name}")


# ─── Helpers ─────────────────────────────────────────────────────────────────
def _check_export_deps() -> None:
    """Fail fast with an actionable error if export deps are missing."""
    missing = []
    for pkg in ("optimum", "transformers", "torch"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        _die(
            f"Missing export dependencies: {', '.join(missing)}\n"
            "  Install them with:\n"
            "    pip install -r requirements-dev.txt"
        )


def _die(msg: str) -> None:
    print(f"\n  ERROR: {msg}\n", file=sys.stderr)
    sys.exit(1)


def _print_banner(args: argparse.Namespace) -> None:
    quant_label = "disabled (FP32 only)" if args.skip_quantization else "dynamic INT8"
    print()
    print("=" * 62)
    print("  Gaman AI -- Backbone Export Utility")
    print(f"  Model  : {args.model_id}")
    print(f"  Output : {Path(args.output_dir).resolve()}")
    print(f"  INT8   : {quant_label}")
    print(f"  Opset  : {args.opset}")
    print("=" * 62)
    print()


# ─── Entrypoint ──────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    _print_banner(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = output_dir / "_export_tmp"
    final_backbone = output_dir / "backbone.onnx"

    total_start = time.perf_counter()

    try:
        # ── Step 1: Export FP32 ONNX ──────────────────────────────
        print("[1/4] Exporting model to ONNX...")
        fp32_onnx = export_to_onnx(args.model_id, tmp_dir, args.opset)
        print()

        # ── Step 2: Quantize (or copy FP32 directly) ──────────────
        print("[2/4] Quantizing...")
        if args.skip_quantization:
            shutil.copy2(fp32_onnx, final_backbone)
            quantized = False
            print(f"  [WARN] Quantization skipped. FP32 model saved as backbone.onnx")
        else:
            if args.keep_fp32:
                fp32_copy = output_dir / "backbone_fp32.onnx"
                shutil.copy2(fp32_onnx, fp32_copy)
                print(f"  [INFO] FP32 copy saved -> {fp32_copy.name}")
            quantize_int8(fp32_onnx, final_backbone)
            quantized = True
        print()

        # ── Step 3: Copy tokenizer artifacts ──────────────────────
        print("[3/4] Copying tokenizer artifacts...")
        copied = copy_artifacts(tmp_dir, output_dir)
        if copied:
            print(f"  [OK] Copied: {', '.join(copied)}")
        else:
            print("  [WARN] No tokenizer artifacts found -- tokenizer.json may be missing.")
        print()

        # ── Step 4: Write manifest ─────────────────────────────────
        print("[4/4] Writing manifest...")
        write_manifest(output_dir, args.model_id, quantized, copied)
        print()

    finally:
        # Always clean up temp dir, even on failure
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)

    total_elapsed = time.perf_counter() - total_start
    print("=" * 62)
    print(f"  [OK] Export complete in {total_elapsed:.1f}s")
    print(f"  Artifacts: {output_dir.resolve()}")
    print("=" * 62)
    print()
    print("  Next step -- run inference:")
    print("    python -c \"from src.engine import GamanEngine; e = GamanEngine()\"")
    print()


if __name__ == "__main__":
    main()
