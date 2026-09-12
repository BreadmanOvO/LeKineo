from pathlib import Path
import json
import subprocess
import sys


def test_prepare_split_manifest_resolves_expected_shards(tmp_path: Path):
    split = {
        "repo_id": "HuggingFaceVLA/libero",
        "revision": "86958911c0f959db2bbbdb107eb3e17c5f9c798e",
        "episode_ids": {"train": [1261], "val": [1262], "test": [1290]},
    }
    split_path = tmp_path / "split.json"
    manifest_path = tmp_path / "manifest.json"
    split_path.write_text(json.dumps(split))
    subprocess.run([sys.executable, "scripts/prepare_split_data.py", "--split", str(split_path), "--output", str(manifest_path)], check=True)
    payload = json.loads(manifest_path.read_text())
    assert payload["file_count"] == 2
    assert {item["file_index"] for item in payload["files"]} == {55, 56}
