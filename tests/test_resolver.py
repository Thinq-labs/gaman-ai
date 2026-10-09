"""Tests for Gaman AI Spec Slab Resolver (src/resolver.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.resolver import (
    HardwareProfile,
    detect_hardware,
    resolve_model_path,
    resolve_slab,
)


class TestHardwareDetection:
    def test_detect_hardware_returns_valid_profile(self) -> None:
        """Verifies that detect_hardware returns a populated HardwareProfile without errors."""
        profile = detect_hardware()
        assert isinstance(profile, HardwareProfile)
        assert isinstance(profile.providers, list)
        assert len(profile.providers) >= 1
        assert "CPUExecutionProvider" in profile.providers
        assert isinstance(profile.system_ram_bytes, int)
        assert profile.system_ram_bytes > 0
        assert isinstance(profile.cpu_cores, int)
        assert profile.cpu_cores > 0
        assert isinstance(profile.has_avx512_or_arm_neon, bool)


class TestSlabResolutionHierarchy:
    def test_auto_resolves_large_on_high_end_gpu(self) -> None:
        """GPU with >= 8GB VRAM must resolve to Slab 3 (Large)."""
        mock_profile = HardwareProfile(
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            vram_bytes=16 * 1024**3,  # 16 GB
            system_ram_bytes=32 * 1024**3,
            cpu_cores=16,
            has_avx512_or_arm_neon=True,
        )
        slab = resolve_slab(requested_slab="auto", profile=mock_profile)
        assert slab.tier == "large"
        assert slab.hidden_dim == 1024
        assert "large" in slab.model_id
        assert slab.provider == "CUDAExecutionProvider"

    def test_auto_resolves_base_on_mid_gpu(self) -> None:
        """GPU with 4GB to 8GB VRAM must resolve to Slab 2 (Base)."""
        mock_profile = HardwareProfile(
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            vram_bytes=6 * 1024**3,  # 6 GB
            system_ram_bytes=16 * 1024**3,
            cpu_cores=8,
            has_avx512_or_arm_neon=False,
        )
        slab = resolve_slab(requested_slab="auto", profile=mock_profile)
        assert slab.tier == "base"
        assert slab.hidden_dim == 768
        assert "base" in slab.model_id
        assert slab.provider == "CUDAExecutionProvider"

    def test_auto_resolves_base_on_coreml(self) -> None:
        """Apple Silicon with CoreML must resolve to Slab 2 (Base)."""
        mock_profile = HardwareProfile(
            providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
            vram_bytes=None,
            system_ram_bytes=16 * 1024**3,
            cpu_cores=8,
            has_avx512_or_arm_neon=True,
        )
        slab = resolve_slab(requested_slab="auto", profile=mock_profile)
        assert slab.tier == "base"
        assert slab.hidden_dim == 768
        assert "base" in slab.model_id
        assert slab.provider == "CoreMLExecutionProvider"

    def test_auto_resolves_base_on_high_ram_cpu(self) -> None:
        """CPU-only with >= 16GB RAM must resolve to Slab 2 (Base)."""
        mock_profile = HardwareProfile(
            providers=["CPUExecutionProvider"],
            vram_bytes=None,
            system_ram_bytes=32 * 1024**3,  # 32 GB
            cpu_cores=8,
            has_avx512_or_arm_neon=True,
        )
        slab = resolve_slab(requested_slab="auto", profile=mock_profile)
        assert slab.tier == "base"
        assert slab.hidden_dim == 768
        assert slab.provider == "CPUExecutionProvider"

    def test_auto_resolves_small_on_low_ram_cpu(self) -> None:
        """CPU-only with < 16GB RAM must resolve to Slab 1 (Small) edge fallback."""
        mock_profile = HardwareProfile(
            providers=["CPUExecutionProvider"],
            vram_bytes=None,
            system_ram_bytes=8 * 1024**3,  # 8 GB
            cpu_cores=4,
            has_avx512_or_arm_neon=False,
        )
        slab = resolve_slab(requested_slab="auto", profile=mock_profile)
        assert slab.tier == "small"
        assert slab.hidden_dim == 768
        assert "small" in slab.model_id
        assert slab.quantization == "int8"
        assert slab.provider == "CPUExecutionProvider"


class TestExplicitSlabOverrides:
    def test_explicit_slab_small_overrides_high_end_hardware(self) -> None:
        """User requesting 'small' must receive Slab 1 even on high-end CUDA hardware."""
        mock_profile = HardwareProfile(
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            vram_bytes=24 * 1024**3,
            system_ram_bytes=64 * 1024**3,
            cpu_cores=32,
            has_avx512_or_arm_neon=True,
        )
        slab = resolve_slab(requested_slab="small", profile=mock_profile)
        assert slab.tier == "small"
        assert slab.hidden_dim == 768
        assert "small" in slab.model_id
        # Best provider selected
        assert slab.provider == "CUDAExecutionProvider"

    def test_explicit_slab_large_on_cpu_system(self) -> None:
        """User requesting 'large' on CPU-only system uses CPU provider."""
        mock_profile = HardwareProfile(
            providers=["CPUExecutionProvider"],
            vram_bytes=None,
            system_ram_bytes=8 * 1024**3,
            cpu_cores=4,
            has_avx512_or_arm_neon=False,
        )
        slab = resolve_slab(requested_slab="large", profile=mock_profile)
        assert slab.tier == "large"
        assert slab.hidden_dim == 1024
        assert slab.provider == "CPUExecutionProvider"

    def test_invalid_slab_raises_value_error(self) -> None:
        """Invalid tier string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid slab tier"):
            resolve_slab(requested_slab="ultra_giant")


class TestModelPathResolution:
    def test_tiered_directory_layout(self, tmp_path: Path) -> None:
        """When models/<tier>/backbone.onnx exists, return models/<tier>."""
        tier_dir = tmp_path / "base"
        tier_dir.mkdir(parents=True)
        (tier_dir / "backbone.onnx").touch()

        resolved = resolve_model_path(tmp_path, "base")
        assert resolved == tier_dir

    def test_backward_compatibility_flat_models_dir(self, tmp_path: Path) -> None:
        """When only root models/backbone.onnx exists, fall back to root for small tier."""
        (tmp_path / "backbone.onnx").touch()

        resolved = resolve_model_path(tmp_path, "small")
        assert resolved == tmp_path

    def test_nonexistent_models_defaults_to_tiered_path(self, tmp_path: Path) -> None:
        """When neither exists, returns tiered path so caller gets expected path."""
        resolved = resolve_model_path(tmp_path, "large")
        assert resolved == tmp_path / "large"
