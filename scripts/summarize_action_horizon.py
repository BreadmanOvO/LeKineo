#!/usr/bin/env python3
"""Merge and validate Day 12 real horizon runs, then produce aggregates."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def as_bool(value: object) -> bool:
    return str(value).lower() == "true"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-chart", type=Path)
    parser.add_argument("--expected-horizons", nargs="+", type=int, default=[5, 10, 25, 50])
    parser.add_argument("--expected-tasks", nargs="+", type=int, default=[31, 39])
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    for path in args.inputs:
        rows.extend(csv.DictReader(path.open(encoding="utf-8", newline="")))
    keys = [(int(r["horizon"]), int(r["task_id"]), int(r["episode"]), int(r["seed"])) for r in rows]
    expected = {(h, task, episode) for h in args.expected_horizons for task in args.expected_tasks
                for episode in range(args.episodes)}
    actual = {(h, task, episode) for h, task, episode, _ in keys}
    required = {"success", "attempts", "total_environment_steps", "total_inference_calls",
                "mean_inference_ms", "action_smoothness_l1", "peak_vram_gb"}
    checks = {
        "real_status": all(r["status"] == "REAL_LIBERO" for r in rows),
        "row_count": len(rows) == len(expected),
        "unique_rows": len(keys) == len(set(keys)),
        "complete_grid": actual == expected,
        "required_metrics": bool(rows) and required.issubset(rows[0]),
        "finite_nonnegative_metrics": all(float(r[name]) >= 0 for r in rows for name in required - {"success"}),
    }

    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["horizon"])].append(row)
    by_horizon = {}
    for horizon in sorted(grouped):
        group = grouped[horizon]
        by_horizon[str(horizon)] = {
            "episodes": len(group),
            "success_rate": mean(as_bool(r["success"]) for r in group),
            "mean_attempts": mean(int(r["attempts"]) for r in group),
            "mean_total_environment_steps": mean(int(r["total_environment_steps"]) for r in group),
            "mean_total_inference_calls": mean(int(r["total_inference_calls"]) for r in group),
            "mean_inference_ms": mean(float(r["mean_inference_ms"]) for r in group),
            "mean_action_smoothness_l1": mean(float(r["action_smoothness_l1"]) for r in group),
            "peak_vram_gb": max(float(r["peak_vram_gb"]) for r in group),
            "recovery_success_rate": mean(as_bool(r["recovery_success"]) for r in group),
        }
    ranked = sorted(by_horizon, key=lambda h: (-by_horizon[h]["success_rate"], by_horizon[h]["mean_total_inference_calls"]))
    best = int(ranked[0]) if ranked else None
    all_zero = all(item["success_rate"] == 0 for item in by_horizon.values())
    payload = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "protocol": {"horizons": args.expected_horizons, "tasks": args.expected_tasks,
                     "episodes_per_task_per_horizon": args.episodes},
        "checks": checks,
        "rows": len(rows),
        "by_horizon": by_horizon,
        "engineering_default_horizon": best,
        "selection_confidence": "LOW_ALL_ZERO_SUCCESS" if all_zero else "MEASURED_SUCCESS_DIFFERENCE",
        "selection_note": ("All horizons had zero task success; the default only minimizes model calls and is not a capability claim."
                           if all_zero else "Selected by success rate first, then model calls."),
    }
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if args.output_chart:
        import matplotlib.pyplot as plt
        horizons = sorted(grouped)
        calls = [by_horizon[str(h)]["mean_total_inference_calls"] for h in horizons]
        smooth = [by_horizon[str(h)]["mean_action_smoothness_l1"] for h in horizons]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].bar([str(h) for h in horizons], calls)
        axes[0].set(title="Model calls per logical episode", xlabel="Execution horizon", ylabel="Calls")
        axes[1].plot(horizons, smooth, marker="o")
        axes[1].set(title="Action smoothness (lower is smoother)", xlabel="Execution horizon", ylabel="Mean L1 delta")
        fig.tight_layout()
        args.output_chart.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output_chart, dpi=160)
    print(json.dumps(payload, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
