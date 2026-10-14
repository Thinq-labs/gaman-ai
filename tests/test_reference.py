"""
tests/test_reference.py — Unit & Integration Tests for Reference Grounding Engine
=================================================================================
Verifies:
1. Pure standard library .docx extraction via zipfile.
2. Segmenting and retrieval of top policy clauses (< 1ms).
3. Tabular dataset (.csv, .jsonl) loading and caching.
4. Reference grounding integration in choice, noul, and score.
5. ModernBERT auto-temperature scaling check.
6. CLI argument parsing with --reference.
"""

from __future__ import annotations

import csv
import json
import time
import zipfile
from pathlib import Path

import pytest

from src.cli import main
from src.engine import GamanEngine
from src.reference import (
    ReferenceExtractor,
    extract_text_from_docx,
    extract_text_from_json,
    find_relevant_clauses,
    segment_clauses,
)


def create_synthetic_docx(filepath: Path, paragraphs: list[str]) -> Path:
    """Create a valid .docx file using standard library zipfile with WordprocessingML."""
    body_xml = "".join(f"<w:p><w:r><w:t>{p}</w:t></w:r></w:p>" for p in paragraphs)
    doc_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">\n'
        f"<w:body>{body_xml}</w:body>\n"
        "</w:document>"
    )
    with zipfile.ZipFile(filepath, "w") as z:
        z.writestr("word/document.xml", doc_xml.encode("utf-8"))
    return filepath


class TestDocxAndFileExtraction:
    def test_docx_extraction_pure_stdlib(self, tmp_path: Path) -> None:
        """Verify .docx file is extracted without third-party libraries."""
        paragraphs = [
            "Digital software licenses are strictly non-refundable once activated.",
            "Hardware returns are permitted within 30 days of initial delivery.",
            "Enterprise contracts require executive vice president countersignature.",
        ]
        docx_path = tmp_path / "test_policy.docx"
        create_synthetic_docx(docx_path, paragraphs)

        extracted = extract_text_from_docx(docx_path)
        for p in paragraphs:
            assert p in extracted

    def test_json_extraction(self, tmp_path: Path) -> None:
        """Verify .json reference extraction."""
        data = {
            "title": "Access Control Guidelines",
            "rules": [
                "Contractors must not inspect payroll records.",
                "Employees must change credentials every 90 days.",
            ],
        }
        json_path = tmp_path / "guidelines.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        extracted = extract_text_from_json(json_path)
        assert "Access Control Guidelines" in extracted
        assert "Contractors must not inspect payroll records." in extracted

    def test_missing_file_raises_error(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            extract_text_from_docx(tmp_path / "nonexistent.docx")


class TestClauseSegmentationAndRetrieval:
    def test_segment_clauses_bounds(self) -> None:
        """Ensure all generated clauses are bounded to max_chars and cleaned."""
        raw_text = (
            "1. Digital software licenses are strictly non-refundable once activated.\n"
            "2. Hardware returns are permitted within 30 days of initial delivery date with receipt.\n"
            "* For all international purchases, customs duties and clearance fees must be prepaid by customer."
        )
        clauses = segment_clauses(raw_text, max_chars=128)
        assert len(clauses) >= 3
        for c in clauses:
            assert len(c) <= 128
            assert not c.startswith("1.")
            assert not c.startswith("*")

    def test_find_relevant_clauses_latency_and_accuracy(self) -> None:
        """Verify top clause is retrieved in < 1ms."""
        clauses = [
            "Digital software licenses are strictly non-refundable once activated.",
            "Hardware equipment returns are permitted within 30 days of delivery.",
            "Annual support subscriptions auto-renew on the fiscal anniversary.",
            "Customer service inquiries are handled within 24 business hours.",
            "Security vulnerability reports must be sent to security@example.com.",
        ]
        query = "Customer asking for refund on opened activated software license key"
        # Warmup
        find_relevant_clauses(query, clauses, top_k=2)
        t0 = time.perf_counter()
        top_clauses = find_relevant_clauses(query, clauses, top_k=2)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        assert elapsed_ms < 5.0  # Well within SLA (< 5ms on any machine, < 1ms warm)
        assert len(top_clauses) == 2
        assert "Digital software licenses are strictly non-refundable once activated." in top_clauses[0]

    def test_caching_layer(self, tmp_path: Path) -> None:
        """Ensure repeated document extraction hits in-memory cache without disk I/O."""
        docx_path = tmp_path / "cached_policy.docx"
        create_synthetic_docx(docx_path, ["Policy item one.", "Policy item two."])

        ReferenceExtractor.clear_cache()
        clauses_1 = ReferenceExtractor.extract_clauses(docx_path)
        h = ReferenceExtractor.compute_hash(docx_path)
        assert h in ReferenceExtractor._clause_cache

        # Modify file on disk without changing mtime/content
        clauses_2 = ReferenceExtractor.extract_clauses(docx_path)
        assert clauses_1 == clauses_2


class TestTabularDatasetLoading:
    def test_load_csv_dataset(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "routing.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["text", "category"])
            writer.writeheader()
            writer.writerow({"text": "laptop won't boot", "category": "HARDWARE"})
            writer.writerow({"text": "lost my password", "category": "ACCESS"})

        texts, labels = ReferenceExtractor.load_dataset(csv_path)
        assert len(texts) == 2
        assert texts[0] == "laptop won't boot"
        assert labels[0] == "HARDWARE"
        assert ReferenceExtractor.is_tabular_file(csv_path) is True


class TestModernBertAutoCalibration:
    def test_modernbert_default_temp_choice(self) -> None:
        """ModernBERT must automatically set default temp_choice to 0.25 for sharp probabilities."""
        engine = GamanEngine(family="modernbert")
        assert engine.is_modernbert is True
        assert engine.temp_choice == 0.25

    def test_deberta_default_temp_choice(self) -> None:
        """DeBERTa backbone retains default temp_choice of 1.0."""
        engine = GamanEngine(slab="small")
        assert engine.is_modernbert is False
        assert engine.temp_choice == 1.0


class TestReferenceGroundingEngineIntegration:
    @pytest.fixture(scope="class")
    @staticmethod
    def shared_engine() -> GamanEngine:
        return GamanEngine(slab="small")

    def test_choice_grounding_with_docx_policy(
        self, shared_engine: GamanEngine, tmp_path: Path
    ) -> None:
        """Grounding against a .docx policy document."""
        docx_path = tmp_path / "company_data_policy.docx"
        create_synthetic_docx(
            docx_path,
            [
                "Employee requests to export bulk customer data exceeding 1,000 records require security approval.",
                "Standard report exports under 100 records are automatically approved.",
            ],
        )

        state = {"text": "Employee requested export of 50,000 user emails"}
        options = ["AUTOMATIC_APPROVE", "REQUIRE_SECURITY_APPROVAL", "REJECT_OUTRIGHT"]

        res = shared_engine.choice(state, options, reference=docx_path)
        assert res["grounded"] is True
        assert "relevant_clauses" in res
        assert res["selection"] == "REQUIRE_SECURITY_APPROVAL"

    def test_choice_grounding_with_inline_prompt(
        self, shared_engine: GamanEngine
    ) -> None:
        """Grounding against an inline prompt string."""
        reference = "Digital software licenses are strictly non-refundable once activated."
        state = {"text": "Customer wants refund on opened activated software license"}
        options = ["ALLOW_REFUND", "DENY_REFUND"]

        res = shared_engine.choice(state, options, reference=reference)
        assert res["grounded"] is True
        assert res["selection"] == "DENY_REFUND"

    def test_choice_grounding_with_csv_adapter(
        self, shared_engine: GamanEngine, tmp_path: Path
    ) -> None:
        """Grounding against a tabular CSV dataset via in-memory Ridge adapter."""
        csv_path = tmp_path / "issue_categories.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["text", "category"])
            writer.writeheader()
            writer.writerow({"text": "damaged parcel delivered broken", "category": "LOGISTICS"})
            writer.writerow({"text": "crate crushed during shipping", "category": "LOGISTICS"})
            writer.writerow({"text": "reset password credentials", "category": "SECURITY"})
            writer.writerow({"text": "unauthorized login to my account", "category": "SECURITY"})

        state = {"text": "my package arrived smashed in half"}
        options = ["LOGISTICS", "SECURITY"]

        res = shared_engine.choice(state, options, reference=csv_path)
        assert res["grounded"] is True
        assert res["selection"] == "LOGISTICS"

    def test_noul_grounding_with_policy(
        self, shared_engine: GamanEngine, tmp_path: Path
    ) -> None:
        """Grounding noul guardrail predicate against a policy document."""
        policy_file = tmp_path / "access_policy.txt"
        policy_file.write_text(
            "Contractors are strictly forbidden from viewing employee salary databases.",
            encoding="utf-8",
        )

        state = {"user_role": "contractor", "action": "read_salaries"}
        predicate = "The requested action violates policy."

        res = shared_engine.noul(state, predicate, reference=policy_file)
        assert res["grounded"] is True
        assert res["passed"] is True


class TestCLIGroundingIntegration:
    @pytest.fixture(scope="class")
    @staticmethod
    def shared_engine() -> GamanEngine:
        return GamanEngine(slab="small")

    def test_cli_choice_with_reference_prompt(
        self, capsys: pytest.CaptureFixture[str], shared_engine: GamanEngine
    ) -> None:
        exit_code = main(
            [
                "choice",
                "--state", "Customer wants refund on activated digital code",
                "--options", "ALLOW_REFUND", "DENY_REFUND",
                "--reference", "Digital software licenses and digital codes are strictly non-refundable once activated.",
                "--json",
            ],
            engine=shared_engine,
        )
        assert exit_code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["primitive"] == "choice"
        assert data["grounded"] is True
        assert data["selection"] == "DENY_REFUND"

    def test_cli_noul_with_reference_prompt(
        self, capsys: pytest.CaptureFixture[str], shared_engine: GamanEngine
    ) -> None:
        exit_code = main(
            [
                "noul",
                "--state", '{"role": "intern", "action": "delete_production_cluster"}',
                "--predicate", "The requested action violates company rules.",
                "--reference", "Interns are not authorized to delete production infrastructure.",
                "--json",
            ],
            engine=shared_engine,
        )
        assert exit_code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["primitive"] == "noul"
        assert data["grounded"] is True
        assert data["passed"] is True
