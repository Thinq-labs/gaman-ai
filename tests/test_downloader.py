"""
tests/test_downloader.py — Unit and Integration Tests for Weights Downloader & Cache Resolver
=============================================================================================
Tests for:
1. Standard OS cache directory hierarchy and GAMAN_CACHE_DIR override.
2. Model lookup priority (Explicit > Local cwd > Global Cache).
3. Zero-dependency urllib downloader with atomic rename and .tmp cleanup.
4. DownloadError raising on network failure.
5. Progress bar suppression under silent=True or non-TTY.
6. CLI cache management commands (gaman cache --dir, gaman cache --clean).
"""

from __future__ import annotations

import io
import urllib.error
from pathlib import Path
from typing import Any

import pytest

from src.downloader import DownloadError, ensure_model_tier
from src.resolver import get_default_cache_dir, resolve_model_path

# ─── 1. Cache Directory Discovery ────────────────────────────────────────────


def test_get_default_cache_dir_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """GAMAN_CACHE_DIR environment variable takes precedence over OS defaults."""
    custom_cache = tmp_path / "custom_gaman_cache"
    monkeypatch.setenv("GAMAN_CACHE_DIR", str(custom_cache))
    assert get_default_cache_dir() == custom_cache


def test_get_default_cache_dir_os_standard(monkeypatch: pytest.MonkeyPatch):
    """When GAMAN_CACHE_DIR is unset, resolves standard OS path."""
    monkeypatch.delenv("GAMAN_CACHE_DIR", raising=False)
    cache_dir = get_default_cache_dir()
    assert isinstance(cache_dir, Path)
    assert "gaman" in str(cache_dir).lower()


# ─── 2. Model Lookup Priority Order ──────────────────────────────────────────


def test_resolve_model_path_explicit_priority(tmp_path: Path):
    """Explicit models_dir argument takes top priority."""
    explicit_dir = tmp_path / "explicit"
    explicit_tier = explicit_dir / "small"
    explicit_tier.mkdir(parents=True)
    (explicit_tier / "backbone.onnx").write_bytes(b"model_bytes")

    resolved = resolve_model_path(tier="small", models_dir=explicit_dir, auto_download=False)
    assert resolved == explicit_tier


def test_resolve_model_path_local_cwd_priority(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Local ./models directory in current working directory takes priority over global cache."""
    # Create fake local cwd
    local_cwd = tmp_path / "workspace"
    local_cwd.mkdir()
    local_models = local_cwd / "models" / "small"
    local_models.mkdir(parents=True)
    (local_models / "backbone.onnx").write_bytes(b"local_model")

    # Create global cache with model
    fake_cache = tmp_path / "global_cache"
    cache_models = fake_cache / "models" / "small"
    cache_models.mkdir(parents=True)
    (cache_models / "backbone.onnx").write_bytes(b"cache_model")

    monkeypatch.chdir(local_cwd)
    monkeypatch.setenv("GAMAN_CACHE_DIR", str(fake_cache))

    resolved = resolve_model_path(tier="small", models_dir=None, auto_download=False)
    assert resolved == Path("models") / "small"


def test_resolve_model_path_global_cache_fallback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """When local ./models doesn't exist, resolves from global OS cache."""
    empty_cwd = tmp_path / "empty_workspace"
    empty_cwd.mkdir()

    fake_cache = tmp_path / "global_cache"
    cache_tier = fake_cache / "models" / "small"
    cache_tier.mkdir(parents=True)
    (cache_tier / "backbone.onnx").write_bytes(b"cached_weights")

    monkeypatch.chdir(empty_cwd)
    monkeypatch.setenv("GAMAN_CACHE_DIR", str(fake_cache))

    resolved = resolve_model_path(tier="small", models_dir=None, auto_download=False)
    assert resolved == cache_tier


# ─── 3. Downloader & Atomic Writes ────────────────────────────────────────────


class FakeHTTPResponse:
    """Mock urllib HTTP response object supporting context manager and read chunks."""

    def __init__(self, data: bytes):
        self._data = data
        self._stream = io.BytesIO(data)
        self.headers = {"Content-Length": str(len(data))}

    def read(self, chunk_size: int = -1) -> bytes:
        return self._stream.read(chunk_size)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def test_downloader_atomic_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Successful download writes all required tier files atomically."""
    cache_dir = tmp_path / "cache"

    def mock_urlopen(req: Any, timeout: int = 30):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        return FakeHTTPResponse(f"mock-content-for-{url}".encode())

    monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)

    tier_dir = ensure_model_tier("small", cache_dir=cache_dir, silent=True)
    assert tier_dir == cache_dir / "models" / "small"
    assert (tier_dir / "backbone.onnx").exists()
    assert (tier_dir / "tokenizer.json").exists()
    assert (tier_dir / "config.json").exists()

    # Ensure no .tmp files remain
    tmp_files = list(tier_dir.glob("*.tmp"))
    assert len(tmp_files) == 0


def test_downloader_atomic_cleanup_on_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Network failure cleans up temporary files and raises DownloadError."""
    cache_dir = tmp_path / "cache"

    def mock_urlopen_fail(req: Any, timeout: int = 30):
        raise urllib.error.URLError("Network unreachable")

    monkeypatch.setattr("urllib.request.urlopen", mock_urlopen_fail)

    with pytest.raises(DownloadError) as exc_info:
        ensure_model_tier("small", cache_dir=cache_dir, silent=True)

    assert "Failed to download model tier 'small'" in str(exc_info.value)

    # Ensure no partial or tmp files remain
    tier_dir = cache_dir / "models" / "small"
    if tier_dir.exists():
        tmp_files = list(tier_dir.glob("*.tmp"))
        assert len(tmp_files) == 0


def test_downloader_silent_suppresses_progress_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    """When silent=True, no progress bar or text is printed to stdout."""
    cache_dir = tmp_path / "cache"

    def mock_urlopen(req: Any, timeout: int = 30):
        return FakeHTTPResponse(b"test data 12345")

    monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)

    ensure_model_tier("small", cache_dir=cache_dir, silent=True)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


# ─── 4. CLI Cache Subcommand ──────────────────────────────────────────────────


def test_cli_cache_dir_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    """gaman cache --dir prints the active cache path."""
    from src.cli import main

    monkeypatch.setenv("GAMAN_CACHE_DIR", str(tmp_path / "gaman_cache"))

    exit_code = main(["cache", "--dir"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert str(tmp_path / "gaman_cache") in captured.out


def test_cli_cache_clean_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    """gaman cache --clean wipes the cached models directory."""
    from src.cli import main

    cache_dir = tmp_path / "gaman_cache"
    models_dir = cache_dir / "models" / "small"
    models_dir.mkdir(parents=True)
    (models_dir / "backbone.onnx").write_bytes(b"cached")

    monkeypatch.setenv("GAMAN_CACHE_DIR", str(cache_dir))

    exit_code = main(["cache", "--clean"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Cache cleared" in captured.out
    assert not (cache_dir / "models").exists()
