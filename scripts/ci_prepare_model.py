#!/usr/bin/env python3
"""
Gaman AI -- CI Model Pre-flight Utility
======================================
Ensures that model weights exist before running test suites in CI matrix jobs.
Checks for `models/backbone.onnx` or `models/small/backbone.onnx`.
If neither exists, triggers `scripts/export_backbone.py --tier small`.
Cross-platform compatible across Linux, macOS, and Windows runners without shell-specific branching.
"""

import subprocess
import sys
from pathlib import Path


def prepare_model() -> int:
    candidate_paths = [
        Path("models/backbone.onnx"),
        Path("models/small/backbone.onnx"),
    ]

    for path in candidate_paths:
        if path.exists() and path.stat().st_size > 0:
            print(f"[CI PRE-FLIGHT] Model weights verified at '{path}'. Skipping export.")
            return 0

    print("[CI PRE-FLIGHT] No existing model weights found. Exporting 'small' tier...")
    export_script = Path("scripts/export_backbone.py")
    if not export_script.exists():
        print(f"[CI PRE-FLIGHT] Error: '{export_script}' not found.", file=sys.stderr)
        return 1

    cmd = [sys.executable, str(export_script), "--tier", "small"]
    print(f"[CI PRE-FLIGHT] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    sys.exit(prepare_model())
