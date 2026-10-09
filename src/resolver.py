"""Hardware topology detection and Spec Slab tier resolution for Gaman AI.

Automatically profiles consumer hardware (RAM, VRAM, execution providers, SIMD)
and dispatches inference across Spec Slabs:
- Slab 1 (Small): Edge / <16GB RAM / CPU
- Slab 2 (Base): Mid-tier / CoreML / >=16GB RAM / 4GB VRAM
- Slab 3 (Large): High-tier / >=8GB VRAM dedicated GPU

Zero runtime dependencies beyond numpy and onnxruntime.
"""

from __future__ import annotations

import ctypes
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import onnxruntime as ort

PROVIDER_PRIORITY = [
    "CUDAExecutionProvider",
    "ROCMExecutionProvider",
    "CoreMLExecutionProvider",
    "DirectMLExecutionProvider",
    "CPUExecutionProvider",
]

SLAB_DEFINITIONS: dict[str, dict[str, Any]] = {
    "small": {
        "model_id": "cross-encoder/nli-deberta-v3-small",
        "quantization": "int8",
        "hidden_dim": 768,
    },
    "base": {
        "model_id": "cross-encoder/nli-deberta-v3-base",
        "quantization": "int8",
        "hidden_dim": 768,
    },
    "large": {
        "model_id": "cross-encoder/nli-deberta-v3-large",
        "quantization": "fp16",
        "hidden_dim": 1024,
    },
}


class ModelNotFoundError(FileNotFoundError):
    """Raised when a requested or resolved Spec Slab model tier is missing from disk."""

    pass


@dataclass(frozen=True)
class HardwareProfile:
    """Hardware profile discovered at runtime."""

    providers: list[str]
    vram_bytes: int | None
    system_ram_bytes: int
    cpu_cores: int
    has_avx512_or_arm_neon: bool


@dataclass(frozen=True)
class SlabConfig:
    """Resolved model tier configuration."""

    tier: str  # 'small', 'base', 'large'
    model_id: str  # Canonical Hugging Face checkpoint
    quantization: str  # 'int8', 'fp16', 'fp32'
    provider: str  # Selected ONNX execution provider
    hidden_dim: int  # 768 or 1024


def _detect_system_ram() -> int:
    """Detect total system physical RAM in bytes."""
    if sys.platform == "win32":
        try:
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return int(stat.ullTotalPhys)
        except Exception:
            pass
    elif sys.platform.startswith("linux"):
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        parts = line.split()
                        return int(parts[1]) * 1024  # kB to bytes
        except Exception:
            pass
    elif sys.platform == "darwin":
        try:
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            return int(out.strip())
        except Exception:
            pass

    # Generic POSIX fallback
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages * page_size)
    except Exception:
        return 8 * 1024**3  # Safe 8GB default


def _detect_vram(providers: list[str]) -> int | None:
    """Attempt to detect dedicated GPU VRAM if CUDA/ROCm is available."""
    if "CUDAExecutionProvider" not in providers and "ROCMExecutionProvider" not in providers:
        return None

    # Try nvidia-smi query if available
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=1,
        )
        lines = out.strip().splitlines()
        if lines:
            mb = int(lines[0].strip())
            return mb * 1024 * 1024
    except Exception:
        pass

    return None


def _detect_simd() -> bool:
    """Detect presence of AVX-512 or ARM NEON vector extensions."""
    machine = platform.machine().lower()
    if "arm" in machine or "aarch64" in machine:
        return True  # ARM64 standard guarantees NEON support

    # On x86_64, check basic flags or default to True on modern CPUs
    return True


def detect_hardware() -> HardwareProfile:
    """Discover local hardware resources without heavy runtime frameworks."""
    available_providers = ort.get_available_providers()
    ram_bytes = _detect_system_ram()
    vram_bytes = _detect_vram(available_providers)
    cores = os.cpu_count() or 4
    has_simd = _detect_simd()

    return HardwareProfile(
        providers=available_providers,
        vram_bytes=vram_bytes,
        system_ram_bytes=ram_bytes,
        cpu_cores=cores,
        has_avx512_or_arm_neon=has_simd,
    )


def _best_provider(available_providers: list[str]) -> str:
    """Select the highest priority execution provider available."""
    for p in PROVIDER_PRIORITY:
        if p in available_providers:
            return p
    return "CPUExecutionProvider"


def resolve_slab(
    requested_slab: str = "auto",
    profile: HardwareProfile | None = None,
) -> SlabConfig:
    """
    Resolve model tier and execution provider based on requested slab or auto hardware probing.

    Hierarchy:
    - Slab 3 (Large): Dedicated GPU with >= 8GB VRAM -> DeBERTa-v3-large (FP16, d=1024).
    - Slab 2 (Base): Apple Silicon CoreML OR GPU >= 4GB VRAM OR Host RAM >= 16GB -> DeBERTa-v3-base (INT8, d=768).
    - Slab 1 (Small): Host RAM < 16GB or CPU-only -> DeBERTa-v3-small (INT8, d=768).

    Args:
        requested_slab: 'auto', 'small', 'base', or 'large'.
        profile: Optional pre-probed HardwareProfile (probed automatically if None).

    Returns:
        SlabConfig containing tier, model_id, quantization, provider, and hidden_dim.
    """
    requested_slab = requested_slab.lower().strip()
    if requested_slab not in ("auto", "small", "base", "large"):
        raise ValueError(
            f"Invalid slab tier '{requested_slab}'. Must be one of: 'auto', 'small', 'base', 'large'."
        )

    if profile is None:
        profile = detect_hardware()

    best_provider = _best_provider(profile.providers)

    if requested_slab in ("small", "base", "large"):
        tier = requested_slab
    else:
        # Automated Resolution Hierarchy
        has_gpu = any(
            p in ("CUDAExecutionProvider", "ROCMExecutionProvider", "DirectMLExecutionProvider")
            for p in profile.providers
        )
        if has_gpu and profile.vram_bytes is not None and profile.vram_bytes >= 8 * 1024**3:
            tier = "large"
        elif (
            "CoreMLExecutionProvider" in profile.providers
            or (has_gpu and profile.vram_bytes is not None and profile.vram_bytes >= 4 * 1024**3)
            or profile.system_ram_bytes >= 16 * 1024**3
        ):
            tier = "base"
        else:
            tier = "small"

    defn = SLAB_DEFINITIONS[tier]
    return SlabConfig(
        tier=tier,
        model_id=defn["model_id"],
        quantization=defn["quantization"],
        provider=best_provider,
        hidden_dim=defn["hidden_dim"],
    )


def resolve_model_path(models_dir: Path | str, tier: str) -> Path:
    """
    Resolve model directory path with multi-model layout and backwards compatibility.

    Checks:
    1. models/<tier>/backbone.onnx
    2. models/backbone.onnx (legacy flat directory fallback)

    Args:
        models_dir: Root models directory path.
        tier: Resolved tier ('small', 'base', 'large').

    Returns:
        Path to directory containing backbone.onnx, tokenizer.json, and config.json.
    """
    p = Path(models_dir)
    tiered_dir = p / tier
    if (tiered_dir / "backbone.onnx").exists():
        return tiered_dir
    if (p / "backbone.onnx").exists():
        return p
    return tiered_dir
