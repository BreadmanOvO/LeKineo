#!/usr/bin/env python3
"""Day 11 closed-loop evaluation harness.

The default mode is an explicit deterministic dry-run. Real LIBERO control is
kept behind an adapter boundary until the complete validation snapshot and
task-specific action mapping are available.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent.evaluation import EpisodeMetrics  # noqa: E402


SYSTEMS = [
    ("baseline", "lerobot/smolvla_base"),
    ("finetuned", "step-0002000"),
    ("planner_finetuned", "step-0002000"),
    ("planner_verifier_recovery", "step-0002000"),
]


def run_dry_episode(system: str, split: str, task_id: int, episode: int, seed: int, horizon: int) -> EpisodeMetrics:
    # Deterministic behavior makes the accounting and downstream tables testable.
    if system == "baseline":
        success, steps, retries, reason = False, 80, 0, "policy_timeout"
    elif system == "finetuned":
        success, steps, retries, reason = True, 42, 0, None
    elif system == "planner_finetuned":
        success, steps, retries, reason = True, 45, 0, None
    else:
        success, steps, retries, reason = True, 52, 1, None
    return EpisodeMetrics(
        system=system,
        checkpoint=SYSTEMS[[name for name, _ in SYSTEMS].index(system)][1],
        task_suite="libero_spatial",
        task_id=task_id,
        split=split,
        seed=seed,
        episode=episode,
        success=success,
        episode_steps=steps,
        early_termination=not success,
        invalid_action=False,
        retries=retries,
        recovery_success=bool(success and retries > 0),
        n_action_steps=horizon,
        mean_inference_ms=130.9,
        peak_vram_gb=0.905,
        status="provisional_dry_run",
        failure_reason=reason,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--seen-episodes", type=int, default=3)
    parser.add_argument("--unseen-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--n-action-steps", type=int, default=10)
    parser.add_argument("--mode", choices=("dry-run", "real"), default="dry-run")
    args = parser.parse_args()
    if args.mode == "real":
        raise SystemExit("real mode is blocked until the complete validation snapshot and LIBERO executor adapter are supplied")
    rows: list[EpisodeMetrics] = []
    for system, _ in SYSTEMS:
        for episode in range(args.seen_episodes):
            rows.append(run_dry_episode(system, "seen", 31, episode, args.seed, args.n_action_steps))
        for episode in range(args.unseen_episodes):
            rows.append(run_dry_episode(system, "unseen", 39, episode, args.seed, args.n_action_steps))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].to_row()))
        writer.writeheader()
        writer.writerows(row.to_row() for row in rows)
    summary: dict[str, dict[str, object]] = {}
    for system, _ in SYSTEMS:
        subset = [row for row in rows if row.system == system]
        summary[system] = {
            "episodes": len(subset),
            "seen_episodes": sum(row.split == "seen" for row in subset),
            "unseen_episodes": sum(row.split == "unseen" for row in subset),
            "success_rate": sum(row.success for row in subset) / len(subset),
            "mean_episode_steps": sum(row.episode_steps for row in subset) / len(subset),
            "early_termination_rate": sum(row.early_termination for row in subset) / len(subset),
            "invalid_action_rate": sum(row.invalid_action for row in subset) / len(subset),
            "mean_retries": sum(row.retries for row in subset) / len(subset),
            "recovery_success_rate": sum(row.recovery_success for row in subset) / len(subset),
        }
    report = {
        "status": "PROVISIONAL_DRY_RUN",
        "mode": args.mode,
        "reason": "Deterministic fake executor; not a real LIBERO success-rate result.",
        "protocol": {"seed": args.seed, "seen_episodes_per_system": args.seen_episodes, "unseen_episodes_per_system": args.unseen_episodes, "n_action_steps": args.n_action_steps},
        "systems": summary,
        "rows": len(rows),
        "csv": str(args.output_csv),
    }
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
