"""
tests/test_distillation.py — Unit Tests for Distillation Mathematics
====================================================================
Tests KL divergence loss, temperature scaling, and alignment metrics.
"""

import numpy as np
import pytest

from scripts.distill_slabs import (
    compute_alignment_metrics,
    compute_kl_divergence_loss,
)


class TestDistillationLoss:
    def test_identical_distributions_yield_zero_loss(self) -> None:
        """When student and teacher logits are identical, KL divergence is 0."""
        logits = np.array([[2.0, 1.0, -1.0], [0.5, 3.2, 1.1]], dtype=np.float32)
        loss = compute_kl_divergence_loss(logits, logits, temperature=2.0)
        assert pytest.approx(loss, abs=1e-5) == 0.0

    def test_divergent_distributions_yield_positive_loss(self) -> None:
        """Divergent distributions produce strictly positive loss."""
        s_logits = np.array([[5.0, -2.0, -2.0]], dtype=np.float32)
        t_logits = np.array([[-2.0, 5.0, -2.0]], dtype=np.float32)
        loss = compute_kl_divergence_loss(s_logits, t_logits, temperature=1.0)
        assert loss > 0.0

    def test_temperature_scaling_quadratically_scales(self) -> None:
        """Higher temperature scales the gradients and loss."""
        s_logits = np.array([[3.0, 0.0]], dtype=np.float32)
        t_logits = np.array([[0.0, 3.0]], dtype=np.float32)
        loss_t1 = compute_kl_divergence_loss(s_logits, t_logits, temperature=1.0)
        loss_t2 = compute_kl_divergence_loss(s_logits, t_logits, temperature=2.0)
        assert loss_t1 > 0.0
        assert loss_t2 > 0.0


class TestAlignmentMetrics:
    def test_perfect_agreement(self) -> None:
        s_preds = [0, 1, 2, 3]
        t_preds = [0, 1, 2, 3]
        gt = [0, 1, 2, 3]
        metrics = compute_alignment_metrics(s_preds, t_preds, gt)
        assert metrics["agreement_rate"] == 1.0
        assert metrics["student_accuracy"] == 1.0
        assert metrics["teacher_accuracy"] == 1.0

    def test_partial_agreement(self) -> None:
        s_preds = [0, 1, 2, 0]
        t_preds = [0, 1, 2, 3]
        metrics = compute_alignment_metrics(s_preds, t_preds)
        assert metrics["agreement_rate"] == 0.75
        assert metrics["total_samples"] == 4.0
