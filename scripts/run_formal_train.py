#!/usr/bin/env python3
"""Run resumable SmolVLA fine-tuning with periodic checkpoints and JSONL logs."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from run_smoke_train import LiberoAdapter, encode_task_prompts, load_config, set_seed, split_inputs, trainable_snapshot


def save_checkpoint(
    checkpoint_dir: Path,
    step: int,
    policy: SmolVLAPolicy,
    optimizer: torch.optim.Optimizer,
    losses: list[float],
    step_times: list[float],
    grad_norms: list[float],
    peak_memory: int,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    trainable_state = {
        name: parameter.detach().cpu()
        for name, parameter in policy.named_parameters()
        if parameter.requires_grad
    }
    torch.save({"format": "trainable_state_dict_v2", "state_dict": trainable_state}, checkpoint_dir / "trainable_state.pt")
    torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
    policy.config.save_pretrained(checkpoint_dir)
    metadata = {
        "step": step,
        "loss": losses[-1] if losses else None,
        "loss_mean_last_10": float(np.mean(losses[-10:])) if losses else None,
        "step_time_mean_seconds": float(np.mean(step_times[-100:])) if step_times else None,
        "grad_norm_mean_last_100": float(np.mean(grad_norms[-100:])) if grad_norms else None,
        "peak_memory_bytes": peak_memory,
        "peak_memory_gib": peak_memory / (1024**3),
        "trainable_parameter_count": sum(parameter.numel() for parameter in policy.parameters() if parameter.requires_grad),
        "rng_state": {
            "python": repr(random.getstate()),
            "numpy": repr(np.random.get_state()),
            "torch": torch.get_rng_state().tolist(),
            "cuda": torch.cuda.get_rng_state_all()[0].tolist() if torch.cuda.is_available() else None,
        },
    }
    (checkpoint_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def load_checkpoint(
    checkpoint_dir: Path,
    policy: SmolVLAPolicy,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    payload = torch.load(checkpoint_dir / "trainable_state.pt", map_location="cpu", weights_only=True)
    missing, unexpected = policy.load_state_dict(payload["state_dict"], strict=False)
    missing_trainable = [key for key in payload["state_dict"] if key not in policy.state_dict()]
    if unexpected or missing_trainable:
        raise RuntimeError(
            f"checkpoint incompatible: unexpected={len(unexpected)}, missing_trainable={len(missing_trainable)}"
        )
    optimizer.load_state_dict(torch.load(checkpoint_dir / "optimizer.pt", map_location=device, weights_only=False))
    metadata = json.loads((checkpoint_dir / "metadata.json").read_text(encoding="utf-8"))
    return int(metadata["step"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/smolvla_main.yaml"))
    parser.add_argument("--steps", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    total_steps = args.steps or int(config["training"]["steps"])
    output = args.output_dir or Path(config["output"]["directory"])
    checkpoints = output / "checkpoints"
    output.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    set_seed(int(config["training"]["seed"]))

    device = torch.device(config["model"].get("device", "cuda") if torch.cuda.is_available() else "cpu")
    policy = SmolVLAPolicy.from_pretrained(config["model"]["pretrained"])
    policy.config.freeze_vision_encoder = True
    policy.config.train_expert_only = True
    policy.config.train_state_proj = True
    policy.to(device).train()
    trainable_vision = [name for name, parameter in policy.named_parameters() if "vision_model" in name.lower() and parameter.requires_grad]
    if trainable_vision:
        raise RuntimeError(f"Vision Encoder unexpectedly trainable: {trainable_vision[:3]}")

    processor = policy.model.vlm_with_expert.processor
    episode_ids, task_prompts = split_inputs(config, "train")
    encoded_tokens, encoded_attention = encode_task_prompts(processor, task_prompts, config["data"]["language"])
    adapter = LiberoAdapter(
        config["data"]["parquet"],
        list(config["data"]["state_indices"]),
        list(config["data"]["action_indices"]),
        policy.config.chunk_size,
        episode_ids=episode_ids,
        task_prompts=task_prompts,
    )
    optimizer = torch.optim.AdamW(
        [parameter for parameter in policy.parameters() if parameter.requires_grad],
        lr=float(config["training"]["learning_rate"]),
        weight_decay=1e-10,
    )
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, policy, optimizer, device)
        print(f"resumed_from_step={start_step} checkpoint={args.resume}")
    if total_steps <= start_step:
        raise ValueError(f"--steps must be greater than resume step {start_step}, got {total_steps}")

    before = trainable_snapshot(policy)
    losses: list[float] = []
    step_times: list[float] = []
    grad_norms: list[float] = []
    peak_memory = 0
    log_every = int(config["training"].get("log_every", 50))
    checkpoint_steps = set(int(value) for value in config["training"].get("checkpoint_steps", [2000, 5000, 10000]))
    checkpoint_steps.add(total_steps)
    log_path = output / "train.jsonl"
    with log_path.open("a", encoding="utf-8") as log_handle:
        for step_index in range(start_step, total_steps):
            step = step_index + 1
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            batch = adapter.batch(step_index, device, encoded_tokens, encoded_attention)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss, _ = policy(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}: {loss.item()}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), float(config["training"]["grad_clip_norm"]))
            optimizer.step()
            step_time = time.perf_counter() - started
            loss_value = float(loss.detach().cpu())
            grad_value = float(grad_norm)
            losses.append(loss_value)
            step_times.append(step_time)
            grad_norms.append(grad_value)
            if device.type == "cuda":
                peak_memory = max(peak_memory, torch.cuda.max_memory_allocated(device))
            record = {
                "step": step,
                "loss": loss_value,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "grad_norm": grad_value,
                "step_time_seconds": step_time,
                "peak_memory_gib": peak_memory / (1024**3),
            }
            log_handle.write(json.dumps(record) + "\n")
            log_handle.flush()
            if step == 1 or step % log_every == 0 or step == total_steps:
                print(
                    f"step={step}/{total_steps} loss={loss_value:.6f} lr={record['learning_rate']:.2e} "
                    f"grad_norm={grad_value:.4f} step_time={step_time:.3f}s peak_vram={record['peak_memory_gib']:.3f}GiB"
                )
            if step in checkpoint_steps:
                save_checkpoint(checkpoints / f"step-{step:07d}", step, policy, optimizer, losses, step_times, grad_norms, peak_memory)
                print(f"checkpoint_saved=step-{step:07d}")

    final_state = {name: parameter.detach().cpu() for name, parameter in policy.named_parameters() if parameter.requires_grad}
    expert_delta = max(
        [float((parameter.detach().cpu() - before[name]).abs().max()) for name, parameter in policy.named_parameters() if name in before and ("lm_expert" in name or any(token in name for token in ("action_in_proj", "action_out_proj", "action_time_mlp")))],
        default=0.0,
    )
    state_delta = max(
        [float((parameter.detach().cpu() - before[name]).abs().max()) for name, parameter in policy.named_parameters() if name in before and "state_proj" in name],
        default=0.0,
    )
    metrics = {
        "steps": total_steps,
        "start_step": start_step,
        "loss_first_10_mean": float(np.mean(losses[:10])),
        "loss_last_10_mean": float(np.mean(losses[-10:])),
        "loss_last": losses[-1],
        "loss_finite": bool(np.isfinite(losses).all()),
        "action_expert_updated": expert_delta > 0.0,
        "state_projector_updated": state_delta > 0.0,
        "action_expert_max_abs_delta": expert_delta,
        "state_projector_max_abs_delta": state_delta,
        "peak_memory_gib": peak_memory / (1024**3),
        "step_time_mean_seconds": float(np.mean(step_times)),
        "step_time_p95_seconds": float(np.percentile(step_times, 95)),
        "samples_per_second": 1.0 / float(np.mean(step_times)),
        "checkpoint_steps_present": sorted(int(path.name.split("-")[-1]) for path in checkpoints.glob("step-*")),
        "output": str(output),
        "config": str(args.config),
    }
    (output / "training_summary.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
