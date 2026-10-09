#!/usr/bin/env python3
"""
Gaman AI -- CI Model Pre-flight Utility
======================================
Ensures that model weights exist before running test suites in CI matrix jobs.
Prepares default weights at `models/` and links/copies to `models/small/` and `models/base/`
so automated Spec Slab hardware resolution (e.g. CoreML on macOS arm64 runners) succeeds.
Cross-platform compatible across Linux, macOS, and Windows runners.
"""

import shutil
import subprocess
import sys
from pathlib import Path


def prepare_model() -> int:
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    root_backbone = models_dir / "backbone.onnx"
    small_backbone = models_dir / "small" / "backbone.onnx"

    # If small tier was cached but root wasn't, restore root from small
    if not root_backbone.exists() and small_backbone.exists() and small_backbone.stat().st_size > 0:
        print("[CI PRE-FLIGHT] Restoring root models/ from cached models/small/...")
        for f in (models_dir / "small").iterdir():
            if f.is_file():
                shutil.copy2(f, models_dir / f.name)

    # If neither exists, trigger backbone export
    if not (root_backbone.exists() and root_backbone.stat().st_size > 0):
        print("[CI PRE-FLIGHT] Exporting model weights to models/...")
        export_script = Path("scripts/export_backbone.py")
        if not export_script.exists():
            print(f"[CI PRE-FLIGHT] Error: '{export_script}' not found.", file=sys.stderr)
            return 1

        cmd = [sys.executable, str(export_script)]
        print(f"[CI PRE-FLIGHT] Running: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            return result.returncode

    # Ensure tiered subdirectories exist so all auto-resolved tiers (small & base) succeed
    for tier in ("small", "base"):
        tier_dir = models_dir / tier
        tier_dir.mkdir(parents=True, exist_ok=True)
        tier_backbone = tier_dir / "backbone.onnx"
        if not (tier_backbone.exists() and tier_backbone.stat().st_size > 0):
            print(f"[CI PRE-FLIGHT] Populating tiered directory '{tier_dir}' from '{models_dir}'...")
            for f in models_dir.iterdir():
                if f.is_file():
                    shutil.copy2(f, tier_dir / f.name)

    print("[CI PRE-FLIGHT] All model weights verified across root, small, and base tiers.")
    return 0


if __name__ == "__main__":
    sys.exit(prepare_model())
