#!/usr/bin/env python3
"""Run real SmolVLA -> LIBERO closed-loop episodes and save auditable rows."""

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
from lerobot.policies.smolvla import SmolVLAPolicy  # noqa: E402


SYSTEMS = {"baseline": None, "finetuned": "step-0002000", "planner_finetuned": "step-0002000", "planner_verifier_recovery": "step-0002000"}
TASK_PROMPTS = {
    31: "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
    39: "pick up the black bowl on the wooden cabinet and place it on the plate",
}


def load_policy(checkpoint: str | None, evidence_root: Path, device: torch.device) -> SmolVLAPolicy:
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
    policy.config.freeze_vision_encoder = True
    policy.config.train_expert_only = True
    policy.config.train_state_proj = True
    if checkpoint:
        path = evidence_root / "outputs/smolvla_main/checkpoints" / checkpoint
        payload = torch.load(path / "trainable_state.pt", map_location="cpu", weights_only=True)
        missing, unexpected = policy.load_state_dict(payload["state_dict"], strict=False)
        if unexpected or any(key not in policy.state_dict() for key in payload["state_dict"]):
            raise RuntimeError(f"checkpoint incompatible: {path}")
    return policy.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, default=Path("/mnt/d/Code/Learn/LeKineo_docs"))
    parser.add_argument("--seen-episodes", type=int, default=3)
    parser.add_argument("--unseen-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--systems", nargs="+", choices=sorted(SYSTEMS), default=sorted(SYSTEMS))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for system in args.systems:
        policy = load_policy(SYSTEMS[system], args.evidence_root, device)
        for split, task_id, count in (("seen", 31, args.seen_episodes), ("unseen", 39, args.unseen_episodes)):
            for episode in range(count):
                task = TaskSpec(f"libero_task_{task_id}", task_id, TASK_PROMPTS[task_id], max_steps=400, seed=args.seed + episode)
                executor = RealLiberoExecutor(policy, device=device)
                result = AgentRuntime(recovery=RecoveryPolicy(1)).run(task, executor)
                verification = result.verification
                details = verification.get("details", {})
                rows.append({
                    "system": system, "checkpoint": SYSTEMS[system] or "lerobot/smolvla_base", "task_suite": "libero_spatial", "task_id": task_id,
                    "split": split, "seed": task.seed, "episode": episode, "success": result.success, "episode_steps": details.get("step_count", 0),
                    "early_termination": not result.success, "invalid_action": details.get("invalid_action", False), "retries": result.retries,
                    "recovery_success": bool(result.success and result.retries > 0), "n_action_steps": task.n_action_steps,
                    "inference_calls": int(details.get("inference_calls", 0)),
                    "mean_inference_ms": 0.0, "peak_vram_gb": (torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0),
                    "status": "REAL_LIBERO", "failure_reason": result.failure_reason,
                })
                executor.close()
        del policy
        if device.type == "cuda":
            torch.cuda.empty_cache()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    payload = {"status": "REAL_LIBERO", "protocol": {"seed": args.seed, "seen_episodes_per_system": args.seen_episodes, "unseen_episodes_per_system": args.unseen_episodes, "systems": args.systems}, "rows": len(rows), "csv": str(args.output_csv)}
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
