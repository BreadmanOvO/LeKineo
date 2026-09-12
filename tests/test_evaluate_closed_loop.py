from pathlib import Path
import csv
import json
import subprocess
import sys


def test_day11_dry_run_protocol(tmp_path: Path):
    csv_path = tmp_path / "closed_loop.csv"
    json_path = tmp_path / "closed_loop.json"
    command = [sys.executable, "scripts/evaluate_closed_loop.py", "--output-csv", str(csv_path), "--output-json", str(json_path)]
    subprocess.run(command, check=True)
    report = json.loads(json_path.read_text())
    assert report["status"] == "PROVISIONAL_DRY_RUN"
    assert report["rows"] == 4 * (3 + 5)
    rows = list(csv.DictReader(csv_path.open()))
    assert len(rows) == 32
    assert {row["system"] for row in rows} == {"baseline", "finetuned", "planner_finetuned", "planner_verifier_recovery"}
    assert all(row["seed"] == "20260912" for row in rows)
