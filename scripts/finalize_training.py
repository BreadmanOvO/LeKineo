#!/usr/bin/env python3
"""Validate and freeze Day 7 training artifacts without copying model weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(repo: Path) -> str:
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True, help="Day 6 output directory")
    parser.add_argument("--evidence", type=Path, required=True, help="Windows-mounted evidence directory")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data-revision", default="HuggingFaceVLA/libero@86958911c0f959db2bbbdb107eb3e17c5f9c798e")
    args = parser.parse_args()

    source_summary = args.output / "training_summary.json"
    if not source_summary.exists():
        raise FileNotFoundError(source_summary)
    summary = json.loads(source_summary.read_text(encoding="utf-8"))
    required_steps = [2000, 5000, 10000]
    checkpoints = []
    for step in required_steps:
        directory = args.output / "checkpoints" / f"step-{step:07d}"
        files = {name: directory / name for name in ("trainable_state.pt", "optimizer.pt", "metadata.json", "config.json")}
        missing = [name for name, path in files.items() if not path.exists()]
        if missing:
            raise RuntimeError(f"checkpoint {step} missing: {missing}")
        metadata = json.loads(files["metadata.json"].read_text(encoding="utf-8"))
        checkpoints.append({
            "step": step,
            "path": str(directory),
            "loss": metadata.get("loss"),
            "loss_mean_last_10": metadata.get("loss_mean_last_10"),
            "trainable_parameter_count": metadata.get("trainable_parameter_count"),
            "files": {name: {"bytes": path.stat().st_size, "sha256": sha256(path)} for name, path in files.items()},
        })

    frozen = {
        "schema_version": 1,
        "status": "ready_for_day8_offline_evaluation",
        "source_summary": str(source_summary),
        "training": summary,
        "data_revision": args.data_revision,
        "canonical_repo": str(args.repo),
        "canonical_repo_commit": git_revision(args.repo),
        "checkpoint_candidates": checkpoints,
        "evaluation_protocol": {
            "validation_split": "task_split.json::validation",
            "same_samples_across_checkpoints": True,
            "metrics": ["validation_loss", "action_l1", "action_mse", "action_smoothness"],
            "normalization": "reuse training adapter and policy normalization; no per-checkpoint refit",
        },
        "artifact_policy": "weights and optimizer states remain local under outputs/ and are not committed",
    }
    args.evidence.mkdir(parents=True, exist_ok=True)
    target = args.evidence / "training_summary.json"
    target.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(target), "checkpoint_steps": required_steps, "canonical_repo_commit": frozen["canonical_repo_commit"]}, indent=2))


if __name__ == "__main__":
    main()
