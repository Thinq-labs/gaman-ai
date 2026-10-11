#!/usr/bin/env python3
"""
scripts/distill_slabs.py — RLCD Knowledge Distillation Harness
================================================================
Distills decision intelligence from a large Teacher model (e.g. ModernBERT-large / Laya-421M)
into compact Student Spec Slabs (ModernBERT-base 86M / ModernBERT-small 44M).

Uses Temperature-Scaled Kullback-Leibler (KL) Divergence:
    L = (1 - alpha) * L_task + alpha * T^2 * KL(softmax(z_student / T) || softmax(z_teacher / T))

Dependencies (Developer only):
    pip install torch transformers optimum onnxruntime
This script is excluded from runtime `src/`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def compute_kl_divergence_loss(
    student_logits: np.ndarray,
    teacher_logits: np.ndarray,
    temperature: float = 2.0,
) -> float:
    """
    Compute temperature-scaled KL divergence loss in pure NumPy.

    D_KL(P || Q) = sum(P * log(P / Q))
    where P is student softmax distribution and Q is teacher target distribution.

    Args:
        student_logits: Array of shape (N, K).
        teacher_logits: Array of shape (N, K).
        temperature: Temperature scaling factor T > 0.

    Returns:
        Scalar KL divergence scaled by T^2.
    """
    # Scale logits by temperature
    s_scaled = student_logits / temperature
    t_scaled = teacher_logits / temperature

    # Compute stable Softmax
    s_exp = np.exp(s_scaled - np.max(s_scaled, axis=-1, keepdims=True))
    p_student = s_exp / np.sum(s_exp, axis=-1, keepdims=True)

    t_exp = np.exp(t_scaled - np.max(t_scaled, axis=-1, keepdims=True))
    q_teacher = t_exp / np.sum(t_exp, axis=-1, keepdims=True)

    # Numerical epsilon to prevent log(0)
    eps = 1e-12
    p_safe = np.clip(p_student, eps, 1.0)
    q_safe = np.clip(q_teacher, eps, 1.0)

    # KL(p_student || q_teacher)
    kl = np.sum(p_safe * (np.log(p_safe) - np.log(q_safe)), axis=-1)
    mean_kl = float(np.mean(kl))

    return mean_kl * (temperature**2)


def compute_alignment_metrics(
    student_preds: np.ndarray | list[int],
    teacher_preds: np.ndarray | list[int],
    ground_truth: np.ndarray | list[int] | None = None,
) -> dict[str, float]:
    """
    Compute alignment and fidelity metrics between Student and Teacher.

    Args:
        student_preds: Predicted class indices from student.
        teacher_preds: Predicted class indices from teacher.
        ground_truth: Optional true class labels.

    Returns:
        dict containing agreement_rate, student_acc, teacher_acc.
    """
    s = np.asarray(student_preds)
    t = np.asarray(teacher_preds)
    n = len(s)
    if n == 0:
        return {"agreement_rate": 0.0, "total_samples": 0}

    agreement = float(np.mean(s == t))
    metrics: dict[str, float] = {
        "agreement_rate": float(round(agreement, 4)),
        "total_samples": float(n),
    }

    if ground_truth is not None:
        gt = np.asarray(ground_truth)
        metrics["student_accuracy"] = float(round(float(np.mean(s == gt)), 4))
        metrics["teacher_accuracy"] = float(round(float(np.mean(t == gt)), 4))

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distill Teacher model decisions into lightweight Student Spec Slabs."
    )
    parser.add_argument(
        "--teacher_model_id",
        type=str,
        default="tasksource/ModernBERT-large-nli",
        help="Teacher model checkpoint or Hugging Face ID.",
    )
    parser.add_argument(
        "--student_model_id",
        type=str,
        default="tasksource/ModernBERT-base-nli",
        help="Student model checkpoint or Hugging Face ID.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="banking77_8class.csv",
        help="Path to training CSV containing text and category columns.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=2.0,
        help="Distillation temperature scaling factor (default: 2.0).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Weight of KL divergence loss vs task loss (default: 0.7).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Training batch size (default: 16).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models/distilled_student",
        help="Output directory for distilled student checkpoint.",
    )
    return parser.parse_args()


def _check_distill_deps() -> None:
    missing: list[str] = []
    for pkg in ("torch", "transformers"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"\n[ERROR] Missing distillation dependencies: {', '.join(missing)}", file=sys.stderr)
        print("  Install development dependencies to run PyTorch training:", file=sys.stderr)
        print("    pip install torch transformers\n", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    args = parse_args()
    print("=" * 66)
    print("  Gaman AI — RLCD Knowledge Distillation Harness")
    print(f"  Teacher Model : {args.teacher_model_id}")
    print(f"  Student Model : {args.student_model_id}")
    print(f"  Dataset       : {args.dataset}")
    print(f"  Temperature   : {args.temperature}")
    print(f"  Alpha (KL)    : {args.alpha}")
    print("=" * 66)

    # Check dependencies before attempting PyTorch training
    _check_distill_deps()

    import torch  # type: ignore[import]
    from transformers import (  # type: ignore[import]
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Active training accelerator: {device}")

    # Load Teacher and freeze weights
    print(f"[1/4] Loading Teacher: {args.teacher_model_id}...")
    teacher = AutoModelForSequenceClassification.from_pretrained(args.teacher_model_id).to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    # Load Student
    print(f"[2/4] Loading Student: {args.student_model_id}...")
    student = AutoModelForSequenceClassification.from_pretrained(args.student_model_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.student_model_id)

    # Load dataset
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"[ERROR] Dataset file not found: {dataset_path}", file=sys.stderr)
        sys.exit(1)

    import csv
    texts: list[str] = []
    with open(dataset_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "text" in row and row["text"].strip():
                texts.append(row["text"].strip())

    print(f"  Loaded {len(texts)} samples from {dataset_path.name}")

    print(f"[3/4] Distilling across {args.epochs} epochs (batch_size={args.batch_size})...")
    optimizer = torch.optim.AdamW(student.parameters(), lr=3e-5)
    loss_ce = torch.nn.CrossEntropyLoss()
    loss_kl = torch.nn.KLDivLoss(reduction="batchmean")

    student.train()
    total_steps = (len(texts) + args.batch_size - 1) // args.batch_size
    last_loss = 0.0

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for i in range(0, len(texts), args.batch_size):
            batch_texts = texts[i : i + args.batch_size]
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                t_logits = teacher(**inputs).logits

            s_logits = student(**inputs).logits

            # Soft targets: KL divergence scaled by T^2
            t_soft = torch.nn.functional.softmax(t_logits / args.temperature, dim=-1)
            s_log_soft = torch.nn.functional.log_softmax(s_logits / args.temperature, dim=-1)
            kl_loss = loss_kl(s_log_soft, t_soft) * (args.temperature**2)

            # Hard pseudo-labels from teacher's maximum prediction
            teacher_pseudo_labels = torch.argmax(t_logits, dim=-1)
            ce_loss = loss_ce(s_logits, teacher_pseudo_labels)

            loss = (1.0 - args.alpha) * ce_loss + args.alpha * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / max(1, total_steps)
        last_loss = avg_epoch_loss
        print(f"  Epoch {epoch + 1}/{args.epochs} - Loss: {avg_epoch_loss:.4f}")

    # [4/4] Compute alignment metrics across dataset
    print("\n[4/4] Evaluating Student-Teacher Agreement...")
    student.eval()
    all_s_preds: list[int] = []
    all_t_preds: list[int] = []

    with torch.no_grad():
        for i in range(0, len(texts), args.batch_size):
            batch_texts = texts[i : i + args.batch_size]
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(device)
            t_logits = teacher(**inputs).logits
            s_logits = student(**inputs).logits

            all_t_preds.extend(torch.argmax(t_logits, dim=-1).cpu().tolist())
            all_s_preds.extend(torch.argmax(s_logits, dim=-1).cpu().tolist())

    alignment_metrics = compute_alignment_metrics(all_s_preds, all_t_preds)
    print(f"  Agreement Rate : {alignment_metrics['agreement_rate'] * 100:.2f}%")
    print(f"  Total Samples  : {int(alignment_metrics['total_samples'])}")

    # Save distilled model
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    summary = {
        "teacher_model_id": args.teacher_model_id,
        "student_model_id": args.student_model_id,
        "dataset": args.dataset,
        "temperature": args.temperature,
        "alpha": args.alpha,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "final_loss": float(round(last_loss, 4)),
        "agreement_rate": alignment_metrics["agreement_rate"],
        "total_samples": alignment_metrics["total_samples"],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(out_dir / "distillation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[OK] Distillation complete! Saved to {out_dir}")


if __name__ == "__main__":
    main()
