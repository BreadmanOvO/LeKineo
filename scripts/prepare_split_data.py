#!/usr/bin/env python3
"""Resolve the parquet shards required by a pinned task split.

This downloads only the data files containing the requested train/val/test
episodes. The files remain in the Hugging Face cache and are never committed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--file-range", nargs=2, type=int, metavar=("START", "END"), help="actual data file numbers to resolve, END inclusive")
    args = parser.parse_args()
    split = json.loads(args.split.read_text(encoding="utf-8"))
    revision = split["revision"]
    repo_id = split["repo_id"]
    metadata = hf_hub_download(repo_id, "meta/episodes/chunk-000/file-000.parquet", revision=revision, repo_type="dataset")
    episodes = pq.read_table(metadata, columns=["episode_index", "data/chunk_index", "data/file_index", "length", "tasks"]).to_pylist()
    requested = {name: set(int(value) for value in values) for name, values in split["episode_ids"].items()}
    records = []
    file_indexes: set[int] = set()
    for row in episodes:
        episode = int(row["episode_index"])
        memberships = [name for name, ids in requested.items() if episode in ids]
        if not memberships:
            continue
        # Some historical snapshots expose a metadata file_index that does not
        # match the public data/chunk-000/file-XXX path. A caller may provide
        # the verified actual range; otherwise retain metadata behavior.
        file_index = int(row["data/file_index"])
        file_indexes.add(file_index)
        records.append({"episode_index": episode, "file_index": file_index, "length": int(row["length"]), "tasks": row["tasks"], "splits": memberships})
    if args.file_range:
        file_indexes = set(range(args.file_range[0], args.file_range[1] + 1))
    files = []
    for file_index in sorted(file_indexes):
        filename = f"data/chunk-000/file-{file_index:03d}.parquet"
        path = None
        if args.download:
            path = hf_hub_download(repo_id, filename, revision=revision, repo_type="dataset")
        files.append({"file_index": file_index, "filename": filename, "path": path})
    payload = {
        "schema_version": 1,
        "repo_id": repo_id,
        "revision": revision,
        "requested_episode_counts": {name: len(ids) for name, ids in requested.items()},
        "resolved_episode_counts": {name: sum(name in record["splits"] for record in records) for name in requested},
        "file_count": len(files),
        "files": files,
        "episodes": records,
        "status": "READY" if args.download and all(item["path"] for item in files) else "MANIFEST_ONLY",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
