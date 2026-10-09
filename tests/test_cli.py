"""
tests/test_cli.py — Tests for Gaman AI Universal CLI
=====================================================
Verifies single-shot commands, --json schema outputs, error handling,
and streaming CSV/JSONL batch processing with metadata preservation.
Optimized to reuse a module-scoped GamanEngine instance to avoid
redundant 164MB ONNX reloads across tests.
"""

import csv
import json
from pathlib import Path
import pytest

from src.cli import main
from src.engine import GamanEngine

MODELS_DIR = Path("models")
BACKBONE_ONNX = MODELS_DIR / "backbone.onnx"

_requires_model = pytest.mark.skipif(
    not BACKBONE_ONNX.exists(),
    reason="Model artifact not found. Run export script first.",
)


@pytest.fixture(scope="module")
def shared_engine():
    """Module-scoped engine instance to optimize test execution time (<8s)."""
    return GamanEngine(models_dir=MODELS_DIR)


def test_cli_help(capsys: pytest.CaptureFixture[str]) -> None:
    """Verifies gaman --help prints usage cleanly."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "usage:" in captured.out.lower()


def test_cli_no_args_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main([])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "usage:" in captured.out.lower() or "gaman" in captured.out.lower()


@_requires_model
class TestSingleShotCLI:
    """Tests for single-shot commands using shared engine."""

    def test_choice_human_readable(
        self, capsys: pytest.CaptureFixture[str], shared_engine: GamanEngine
    ) -> None:
        exit_code = main(
            [
                "choice",
                "--state", '{"cpu_usage": 98, "memory_usage": 85}',
                "--options", "scale_up", "scale_down", "do_nothing",
            ],
            engine=shared_engine,
        )
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Selection:" in captured.out
        assert "scale_up" in captured.out
        assert "Confidence:" in captured.out
        assert "Latency:" in captured.out

    def test_choice_json_schema(
        self, capsys: pytest.CaptureFixture[str], shared_engine: GamanEngine
    ) -> None:
        exit_code = main(
            [
                "choice",
                "--state", '{"cpu_usage": 98, "memory_usage": 85}',
                "--options", "scale_up, scale_down, do_nothing",
                "--json",
            ],
            engine=shared_engine,
        )
        assert exit_code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["primitive"] == "choice"
        assert data["selection"] == "scale_up"
        assert "confidence" in data
        assert "latency_ms" in data

    def test_noul_human_readable(
        self, capsys: pytest.CaptureFixture[str], shared_engine: GamanEngine
    ) -> None:
        exit_code = main(
            [
                "noul",
                "--state", '{"user_id": 123, "action": "delete_all"}',
                "--predicate", "Is this a destructive action?",
            ],
            engine=shared_engine,
        )
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Predicate:" in captured.out
        assert "PASSED" in captured.out
        assert "Probability:" in captured.out
        assert "Latency:" in captured.out

    def test_noul_json_schema(
        self, capsys: pytest.CaptureFixture[str], shared_engine: GamanEngine
    ) -> None:
        exit_code = main(
            [
                "noul",
                "--state", '{"user_id": 123, "action": "delete_all"}',
                "--predicate", "Is this a destructive action?",
                "--json",
            ],
            engine=shared_engine,
        )
        assert exit_code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["primitive"] == "noul"
        assert data["passed"] is True
        assert "probability" in data
        assert "latency_ms" in data

    def test_score_human_readable(
        self, capsys: pytest.CaptureFixture[str], shared_engine: GamanEngine
    ) -> None:
        exit_code = main(
            [
                "score",
                "--state", '{"review": "The product broke after two days of use."}',
                "--criterion", "Severity of hardware failure",
            ],
            engine=shared_engine,
        )
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Score:" in captured.out
        assert "Latency:" in captured.out

    def test_score_json_schema(
        self, capsys: pytest.CaptureFixture[str], shared_engine: GamanEngine
    ) -> None:
        exit_code = main(
            [
                "score",
                "--state", '{"review": "The product broke after two days of use."}',
                "--criterion", "Severity of hardware failure",
                "--json",
            ],
            engine=shared_engine,
        )
        assert exit_code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["primitive"] == "score"
        assert "value" in data
        assert "latency_ms" in data


@_requires_model
class TestBatchCLI:
    """Tests for gaman batch processing using shared engine."""

    def test_batch_csv_with_state_column(
        self, tmp_path: Path, shared_engine: GamanEngine
    ) -> None:
        input_csv = tmp_path / "input.csv"
        output_csv = tmp_path / "output.csv"

        rows = [
            {"id": "msg_001", "author": "alice", "text": "Deploying emergency patch"},
            {"id": "msg_002", "author": "bob", "text": "Database read replicas healthy"},
        ]
        with open(input_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "author", "text"])
            writer.writeheader()
            writer.writerows(rows)

        exit_code = main(
            [
                "batch",
                "--input", str(input_csv),
                "--output", str(output_csv),
                "--primitive", "choice",
                "--state-column", "text",
                "--options", "deployment", "database", "general",
            ],
            engine=shared_engine,
        )
        assert exit_code == 0
        assert output_csv.exists()

        with open(output_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            out_rows = list(reader)

        assert len(out_rows) == 2
        # Verify metadata preserved
        assert out_rows[0]["id"] == "msg_001"
        assert out_rows[0]["author"] == "alice"
        assert out_rows[0]["text"] == "Deploying emergency patch"
        # Verify augmented columns
        assert "gaman_selection" in out_rows[0]
        assert "gaman_confidence" in out_rows[0]
        assert "gaman_latency_ms" in out_rows[0]
        assert out_rows[0]["gaman_selection"] in ["deployment", "database", "general"]

    def test_batch_csv_full_row_mode(
        self, tmp_path: Path, shared_engine: GamanEngine
    ) -> None:
        input_csv = tmp_path / "server_metrics.csv"
        output_csv = tmp_path / "scored_metrics.csv"

        rows = [
            {"host": "node-1", "cpu": "95", "temp": "85"},
            {"host": "node-2", "cpu": "12", "temp": "40"},
        ]
        with open(input_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["host", "cpu", "temp"])
            writer.writeheader()
            writer.writerows(rows)

        exit_code = main(
            [
                "batch",
                "--input", str(input_csv),
                "--output", str(output_csv),
                "--primitive", "score",
                "--criterion", "System overload and thermal risk",
            ],
            engine=shared_engine,
        )
        assert exit_code == 0
        assert output_csv.exists()

        with open(output_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            out_rows = list(reader)

        assert len(out_rows) == 2
        assert out_rows[0]["host"] == "node-1"
        assert "gaman_score" in out_rows[0]
        assert "gaman_latency_ms" in out_rows[0]

    def test_batch_jsonl_mode(
        self, tmp_path: Path, shared_engine: GamanEngine
    ) -> None:
        input_jsonl = tmp_path / "events.jsonl"
        output_jsonl = tmp_path / "evaluated_events.jsonl"

        events = [
            {"event_id": 101, "payload": {"user": "admin", "cmd": "rm -rf /"}},
            {"event_id": 102, "payload": {"user": "analyst", "cmd": "SELECT 1"}},
        ]
        with open(input_jsonl, "w", encoding="utf-8") as f:
            for ev in events:
                f.write(json.dumps(ev) + "\n")

        exit_code = main(
            [
                "batch",
                "--input", str(input_jsonl),
                "--output", str(output_jsonl),
                "--primitive", "noul",
                "--state-column", "payload",
                "--predicate", "Is this a destructive system command?",
            ],
            engine=shared_engine,
        )
        assert exit_code == 0
        assert output_jsonl.exists()

        with open(output_jsonl, "r", encoding="utf-8") as f:
            out_lines = [json.loads(line) for line in f if line.strip()]

        assert len(out_lines) == 2
        assert out_lines[0]["event_id"] == 101
        assert "gaman" in out_lines[0]
        assert out_lines[0]["gaman"]["primitive"] == "noul"
        assert isinstance(out_lines[0]["gaman"]["passed"], bool)
        assert isinstance(out_lines[0]["gaman"]["probability"], float)

    def test_batch_missing_file_returns_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        exit_code = main([
            "batch",
            "--input", "nonexistent_file.csv",
            "--primitive", "noul",
            "--predicate", "test",
        ])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Input file not found" in captured.err

    def test_e2e_default_engine_initialization(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """End-to-end test verifying default engine initialization when engine=None."""
        exit_code = main([
            "choice",
            "--state", '{"status": "ok"}',
            "--options", "healthy", "unhealthy",
            "--json",
        ])
        assert exit_code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["primitive"] == "choice"
        assert data["selection"] in ["healthy", "unhealthy"]
