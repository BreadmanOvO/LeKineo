from pathlib import Path
import subprocess
import sys


def test_action_horizon_dry_run_protocol(tmp_path: Path):
    csv_path = tmp_path / "horizon.csv"
    json_path = tmp_path / "horizon.json"
    subprocess.run([
        sys.executable, "scripts/evaluate_action_horizon.py", "--mode", "dry-run",
        "--horizons", "5", "10", "--episodes", "2",
        "--output-csv", str(csv_path), "--output-json", str(json_path),
    ], check=True)
    import json
    payload = json.loads(json_path.read_text())
    assert payload["status"] == "PROVISIONAL_DRY_RUN"
    assert payload["rows"] == 8
    assert csv_path.exists()
