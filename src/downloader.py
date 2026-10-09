"""
src/downloader.py — Zero-Dependency Weights Downloader & Model Cache Manager
=============================================================================
Downloads pre-quantized ONNX model artifacts and tokenizer assets on demand
using Python standard library (urllib.request) with zero third-party dependencies.

Provides:
- Atomic file writing via temporary files (.tmp) and os.replace
- TTY-sensitive ASCII progress bar rendering (suppressed when piped or silent)
- Multi-file tier synchronization into standard OS cache directories
"""

from __future__ import annotations

import contextlib
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from src.resolver import get_default_cache_dir

TIER_REPOSITORIES = {
    "small": "cross-encoder/nli-deberta-v3-small",
    "base": "cross-encoder/nli-deberta-v3-base",
    "large": "cross-encoder/nli-deberta-v3-large",
}

REQUIRED_TIER_FILES = ["backbone.onnx", "tokenizer.json", "config.json"]
OPTIONAL_TIER_FILES = ["spm.model", "special_tokens_map.json", "tokenizer_config.json"]


class DownloadError(RuntimeError):
    """Raised when downloading model artifacts from remote registry fails."""

    pass


def get_file_url(tier: str, filename: str) -> str:
    """Build remote registry download URL for a model asset."""
    custom_url = os.environ.get("GAMAN_REGISTRY_URL")
    if custom_url:
        custom_url = custom_url.rstrip("/")
        if "{tier}" in custom_url:
            return custom_url.format(tier=tier, filename=filename)
        return f"{custom_url}/{tier}/{filename}"

    repo = TIER_REPOSITORIES.get(tier, f"cross-encoder/nli-deberta-v3-{tier}")
    return f"https://huggingface.co/{repo}/resolve/main/{filename}"


def _render_progress_bar(current: int, total: int, desc: str) -> None:
    """Render a lightweight ASCII progress bar to sys.stdout."""
    percent = min(100.0, (current / total) * 100.0)
    bar_len = 25
    filled = int(bar_len * current // total)
    bar = "=" * max(0, filled - 1) + (">" if filled > 0 else "")
    bar = bar.ljust(bar_len, " ")
    tot_mb = total / (1024 * 1024)
    sys.stdout.write(f"\r{desc} ({tot_mb:.0f} MB)... [{bar}] {percent:.0f}%")
    sys.stdout.flush()


def download_file(
    url: str,
    target_path: Path,
    silent: bool = False,
    desc: str = "Downloading",
    chunk_size: int = 1024 * 1024,
) -> None:
    """
    Download a single file atomically with integrity protection.

    Writes to target_path.tmp first and atomically moves to target_path upon completion.
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_name(f"{target_path.name}.tmp")

    show_progress = (not silent) and sys.stdout.isatty()

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "gaman-ai-downloader/0.2.0"},
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            total_bytes = int(response.headers.get("Content-Length", 0))
            downloaded = 0

            with open(tmp_path, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if show_progress and total_bytes > 0:
                        _render_progress_bar(downloaded, total_bytes, desc)

        if show_progress and total_bytes > 0:
            sys.stdout.write("\n")
            sys.stdout.flush()

        # Atomic move to prevent corrupted partial files
        os.replace(tmp_path, target_path)

    except Exception as exc:
        if tmp_path.exists():
            with contextlib.suppress(OSError):
                tmp_path.unlink()
        raise DownloadError(f"Failed to download from '{url}': {exc}") from exc


def ensure_model_tier(
    tier: str,
    cache_dir: Path | None = None,
    silent: bool = False,
) -> Path:
    """
    Ensure all required model artifacts for a Spec Slab tier exist on disk.

    Downloads missing files into the cache directory on demand.

    Args:
        tier: Spec Slab tier ('small', 'base', 'large').
        cache_dir: Optional root cache directory (defaults to get_default_cache_dir()).
        silent: If True, suppress progress output and logging.

    Returns:
        Path to the resolved model directory containing backbone.onnx.
    """
    base_cache = cache_dir if cache_dir is not None else get_default_cache_dir()
    tier_dir = base_cache / "models" / tier

    # Check if all required files already exist
    all_present = all((tier_dir / f).exists() for f in REQUIRED_TIER_FILES)
    if all_present:
        return tier_dir

    tier_dir.mkdir(parents=True, exist_ok=True)

    # Download required files
    for filename in REQUIRED_TIER_FILES:
        target_file = tier_dir / filename
        if target_file.exists():
            continue

        url = get_file_url(tier, filename)
        desc = f"Downloading Gaman AI '{tier}' {filename}"
        try:
            download_file(url, target_file, silent=silent, desc=desc)
        except DownloadError as exc:
            raise DownloadError(
                f"Failed to download model tier '{tier}' asset '{filename}' from {url}. "
                f"Check internet connection or firewall. Error: {exc}"
            ) from exc

    # Download optional files (silent failure if not found in repository)
    for filename in OPTIONAL_TIER_FILES:
        target_file = tier_dir / filename
        if target_file.exists():
            continue

        url = get_file_url(tier, filename)
        with contextlib.suppress(Exception):
            # Silently skip optional tokenizer/config files
            download_file(url, target_file, silent=True)

    return tier_dir
