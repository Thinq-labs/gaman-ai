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
    engine = GamanEngine(models_dir=MODELS_DIR)
    engine.temp_choice = 1.0
    engine.temp_noul = 1.0
    engine.temp_score = 1.0
    return engine


@pytest.fixture(autouse=True)
def _reset_shared_engine_temps(shared_engine: GamanEngine) -> None:
    shared_engine.temp_choice = 1.0
    shared_engine.temp_noul = 1.0
    shared_engine.temp_score = 1.0


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

        with open(output_csv, encoding="utf-8") as f:
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

        with open(output_csv, encoding="utf-8") as f:
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

        with open(output_jsonl, encoding="utf-8") as f:
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


@_requires_model
class TestCalibrateCLI:
    """Tests for the calibrate CLI subcommand."""

    def test_calibrate_choice_csv(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        shared_engine: GamanEngine,
    ) -> None:
        csv_file = tmp_path / "val_choice.csv"
        out_cal = tmp_path / "calibration.json"

        # Create validation dataset
        with open(csv_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["state", "target"])
            writer.writeheader()
            writer.writerow({"state": '{"cpu": 99, "mem": 90}', "target": "scale_up"})
            writer.writerow({"state": '{"cpu": 10, "mem": 15}', "target": "scale_down"})
            writer.writerow({"state": '{"cpu": 50, "mem": 50}', "target": "do_nothing"})

        exit_code = main(
            [
                "calibrate",
                "--data", str(csv_file),
                "--primitive", "choice",
                "--state-column", "state",
                "--target-column", "target",
                "--options", "scale_up", "scale_down", "do_nothing",
                "--output", str(out_cal),
            ],
            engine=shared_engine,
        )
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Calibration Complete for [choice]" in captured.out
        assert "Optimal Temp (T*):" in captured.out
        assert out_cal.exists()

        with open(out_cal, encoding="utf-8") as f:
            data = json.load(f)
        assert "temperatures" in data
        assert 0.1 <= data["temperatures"]["choice"] <= 10.0

    def test_calibrate_noul_csv(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        shared_engine: GamanEngine,
    ) -> None:
        csv_file = tmp_path / "val_noul.csv"
        out_cal = tmp_path / "calibration.json"

        with open(csv_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["cmd", "label"])
            writer.writeheader()
            writer.writerow({"cmd": "DROP TABLE users;", "label": "1"})
            writer.writerow({"cmd": "SELECT * FROM users;", "label": "0"})

        exit_code = main(
            [
                "calibrate",
                "--data", str(csv_file),
                "--primitive", "noul",
                "--state-column", "cmd",
                "--target-column", "label",
                "--predicate", "Is this action destructive?",
                "--output", str(out_cal),
            ],
            engine=shared_engine,
        )
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Calibration Complete for [noul]" in captured.out
        assert out_cal.exists()

    def test_calibrate_missing_file_returns_error(
        self, capsys: pytest.CaptureFixture[str], shared_engine: GamanEngine
    ) -> None:
        exit_code = main(
            [
                "calibrate",
                "--data", "nonexistent_val.csv",
                "--primitive", "choice",
                "--target-column", "target",
                "--options", "opt1", "opt2",
            ],
            engine=shared_engine,
        )
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Dataset not found" in captured.err


class TestInfoCLI:
    """Tests for gaman info subcommand and --slab parameterization."""

    def test_gaman_info_human_readable(self, capsys: pytest.CaptureFixture[str]) -> None:
        exit_code = main(["info"])
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Hardware Profile & Spec Slab Resolution" in captured.out
        assert "System RAM:" in captured.out
        assert "Resolved Tier:" in captured.out

    def test_gaman_info_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        exit_code = main(["info", "--json"])
        assert exit_code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert "hardware" in data
        assert "slab" in data
        assert "system_ram_gb" in data["hardware"]
        assert data["slab"]["resolved_tier"] in ["small", "base", "large"]

    def test_gaman_info_explicit_slab_small(self, capsys: pytest.CaptureFixture[str]) -> None:
        exit_code = main(["--slab", "small", "info", "--json"])
        assert exit_code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["slab"]["requested"] == "small"
        assert data["slab"]["resolved_tier"] == "small"
        assert data["slab"]["hidden_dim"] == 768


class TestFitAndPredictCLI:
    """Tests for gaman fit and gaman predict commands."""

    def test_fit_and_predict_pipeline(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        shared_engine: GamanEngine,
    ) -> None:
        csv_file = tmp_path / "train.csv"
        head_file = tmp_path / "heads" / "issue_router.json"
        head_file.parent.mkdir(parents=True, exist_ok=True)

        with open(csv_file, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["text", "label"])
            writer.writeheader()
            writer.writerow({"text": "network connection reset", "label": "network"})
            writer.writerow({"text": "dns resolution timeout", "label": "network"})
            writer.writerow({"text": "database disk full", "label": "storage"})
            writer.writerow({"text": "cannot write to volume", "label": "storage"})

        # 1. Fit adapter head
        exit_code = main(
            [
                "fit",
                "--data", str(csv_file),
                "--state-column", "text",
                "--target-column", "label",
                "--name", "issue_router",
                "--output", str(head_file),
            ],
            engine=shared_engine,
        )
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Linear Adapter Head Fitted: 'issue_router'" in captured.out
        assert "Samples (N):        4" in captured.out
        assert head_file.exists()

        # Cache the head in engine for predict test
        from src.heads import CustomLinearHead
        shared_engine._loaded_heads["issue_router"] = CustomLinearHead.load(head_file)

        # 2. Predict human-readable
        exit_code = main(
            [
                "predict",
                "--state", '{"text": "network connection reset"}',
                "--head", "issue_router",
            ],
            engine=shared_engine,
        )
        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Head:        issue_router" in captured.out
        assert "Selection:   network" in captured.out
        assert "Confidence:  " in captured.out

        # 3. Predict JSON output
        exit_code = main(
            [
                "predict",
                "--state", '{"text": "database disk full"}',
                "--head", "issue_router",
                "--json",
            ],
            engine=shared_engine,
        )
        assert exit_code == 0
        captured = capsys.readouterr()
        pred_json = json.loads(captured.out)
        assert pred_json["primitive"] == "choice"
        assert pred_json["head"] == "issue_router"
        assert pred_json["selection"] == "storage"
        assert "confidence" in pred_json
        assert "probabilities" in pred_json
        assert "latency_ms" in pred_json

    def test_fit_missing_dataset(
        self,
        capsys: pytest.CaptureFixture[str],
        shared_engine: GamanEngine,
    ) -> None:
        exit_code = main(
            [
                "fit",
                "--data", "nonexistent.csv",
                "--target-column", "label",
                "--name", "dummy",
            ],
            engine=shared_engine,
        )
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Dataset not found" in captured.err

    def test_predict_missing_head(
        self,
        capsys: pytest.CaptureFixture[str],
        shared_engine: GamanEngine,
    ) -> None:
        exit_code = main(
            [
                "predict",
                "--state", '{"msg": "hi"}',
                "--head", "nonexistent_head_xyz",
            ],
            engine=shared_engine,
        )
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Error during prediction" in captured.err

