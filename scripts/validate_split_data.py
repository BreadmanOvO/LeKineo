#!/usr/bin/env python3
"""Validate that local parquet shards cover every split episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    split = json.loads(args.split.read_text(encoding="utf-8"))
    observed: set[int] = set()
    rows = 0
    task_ids: set[int] = set()
    for item in manifest["files"]:
        if not item.get("path") or not Path(item["path"]).exists():
            raise SystemExit(f"missing local shard: {item}")
        table = pq.read_table(item["path"], columns=["episode_index", "task_index"])
        observed.update(int(value) for value in table["episode_index"].to_pylist())
        task_ids.update(int(value) for value in table["task_index"].to_pylist())
        rows += table.num_rows
    expected = set(int(value) for group in split["episode_ids"].values() for value in group)
    missing = sorted(expected - observed)
    overlap = {name: len(set(int(value) for value in values) & observed) for name, values in split["episode_ids"].items()}
    payload = {"status": "PASS" if not missing else "FAIL", "rows": rows, "observed_episode_count": len(observed), "expected_episode_count": len(expected), "missing_episode_ids": missing, "split_coverage": overlap, "task_ids": sorted(task_ids), "revision": split["revision"]}
    print(json.dumps(payload, indent=2))
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
