#!/usr/bin/env python3
"""Day 12: compare execution horizons with one fixed SmolVLA checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.agent import AgentRuntime, RecoveryPolicy, TaskSpec  # noqa: E402
from src.agent.libero_executor import RealLiberoExecutor  # noqa: E402
from scripts.evaluate_real_closed_loop import load_policy  # noqa: E402


TASKS = {
    31: "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
    39: "pick up the black bowl on the wooden cabinet and place it on the plate",
}


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def dry_row(horizon: int, task_id: int, episode: int, seed: int) -> dict:
    return {
        "status": "PROVISIONAL_DRY_RUN", "checkpoint": "step-0002000", "horizon": horizon,
        "task_id": task_id, "split": "seen" if task_id == 31 else "unseen", "episode": episode,
        "seed": seed, "success": False, "episode_steps": 400, "inference_calls": 400 // horizon + int(400 % horizon != 0),
        "attempts": 1, "total_environment_steps": 400, "total_inference_calls": 400 // horizon + int(400 % horizon != 0),
        "mean_inference_ms": 0.0, "action_smoothness_l1": 0.0, "peak_vram_gb": 0.0,
        "retries": 0, "recovery_success": False, "failure_reason": "dry_run",
    }


def run_real(args: argparse.Namespace) -> list[dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict] = []
    if args.resume and args.output_csv.exists():
        rows = list(csv.DictReader(args.output_csv.open(encoding="utf-8", newline="")))
    completed = {(int(row["horizon"]), int(row["task_id"]), int(row["episode"]), int(row["seed"])) for row in rows}
    for horizon in args.horizons:
        policy = load_policy("step-0002000", args.evidence_root, device)
        for task_id, prompt in TASKS.items():
            split = "seen" if task_id == 31 else "unseen"
            for episode in range(args.episodes):
                seed = args.seed + episode
                key = (horizon, task_id, episode, seed)
                if key in completed:
                    print(f"resume_skip horizon={horizon} task={task_id} episode={episode} seed={seed}", flush=True)
                    continue
                task = TaskSpec(f"libero_task_{task_id}", task_id, prompt, max_steps=args.max_steps,
                                n_action_steps=horizon, seed=seed)
                executor = RealLiberoExecutor(policy, device=device)
                result = AgentRuntime(recovery=RecoveryPolicy(1)).run(task, executor)
                details = result.verification.get("details", {})
                # Runtime verification intentionally keeps a small state
                # contract; collect executor-owned performance counters here
                # so retries and action smoothness are not silently dropped.
                rows.append({
                    "status": "REAL_LIBERO", "checkpoint": "step-0002000", "horizon": horizon,
                    "task_id": task_id, "split": split, "episode": episode, "seed": seed,
                    "success": result.success, "episode_steps": details.get("step_count", 0),
                    "inference_calls": details.get("inference_calls", 0), "retries": result.retries,
                    "total_inference_calls": details.get("inference_calls", 0),
                    "attempts": result.attempts,
                    "total_environment_steps": int(executor.total_environment_steps),
                    "mean_inference_ms": float(1000.0 * executor.inference_seconds / max(1, executor.inference_calls)),
                    "action_smoothness_l1": float(executor.action_delta_sum / max(1, executor.action_delta_count)),
                    "peak_vram_gb": (torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0),
                    "recovery_success": bool(result.success and result.retries > 0),
                    "failure_reason": result.failure_reason or "",
                })
                write_rows(args.output_csv, rows)
                completed.add(key)
                print(f"saved horizon={horizon} task={task_id} episode={episode} success={result.success}", flush=True)
                executor.close()
        del policy
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, default=Path("/mnt/d/Code/Learn/LeKineo_docs"))
    parser.add_argument("--horizons", nargs="+", type=int, default=[5, 10, 25, 50])
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--mode", choices=("real", "dry-run"), default="real")
    parser.add_argument("--resume", action="store_true", help="keep completed CSV rows and skip matching episodes")
    args = parser.parse_args()
    if any(h <= 0 for h in args.horizons) or args.episodes <= 0:
        raise SystemExit("horizons and episodes must be positive")
    rows = run_real(args) if args.mode == "real" else [dry_row(h, task, ep, args.seed + ep)
        for h in args.horizons for task in TASKS for ep in range(args.episodes)]
    write_rows(args.output_csv, rows)
    payload = {
        "status": "REAL_LIBERO" if args.mode == "real" else "PROVISIONAL_DRY_RUN",
        "protocol": {"checkpoint": "step-0002000", "horizons": args.horizons, "tasks": sorted(TASKS),
                     "episodes_per_task_per_horizon": args.episodes, "max_steps": args.max_steps, "seed": args.seed},
        "rows": len(rows), "csv": str(args.output_csv),
    }
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
