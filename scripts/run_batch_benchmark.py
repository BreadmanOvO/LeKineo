#!/usr/bin/env python3
"""Benchmark SmolVLA batch sizes using the same real-data adapter as Day 4."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from lerobot.policies.smolvla import SmolVLAPolicy
from run_smoke_train import LiberoAdapter, set_seed


def load_config(path: Path) -> dict:
    def expand(value):
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        if isinstance(value, list):
            return [expand(item) for item in value]
        if isinstance(value, str):
            return os.path.expandvars(value)
        return value

    return expand(yaml.safe_load(path.read_text(encoding="utf-8")))


def gpu_utilization() -> float | None:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip().splitlines()[0]
        return float(output)
    except (FileNotFoundError, IndexError, ValueError, subprocess.CalledProcessError):
        return None


def make_batch(adapter, indices, device, tokens, attention):
    batches = [adapter.batch(index, device, tokens, attention) for index in indices]
    result = {}
    for key in batches[0]:
        result[key] = torch.cat([item[key] for item in batches], dim=0)
    return result


def benchmark(config: dict, batch_size: int, steps: int) -> dict:
    set_seed(int(config["training"]["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = SmolVLAPolicy.from_pretrained(config["model"]["pretrained"])
    policy.to(device).train()
    if any(parameter.requires_grad for name, parameter in policy.named_parameters() if "vision_model" in name.lower()):
        raise RuntimeError("Vision Encoder is not frozen")
    processor = policy.model.vlm_with_expert.processor
    encoded = processor.tokenizer(config["data"]["language"], return_tensors="pt")
    adapter = LiberoAdapter(
        config["data"]["parquet"], list(config["data"]["state_indices"]), list(config["data"]["action_indices"]), policy.config.chunk_size
    )
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=float(config["training"]["learning_rate"]), weight_decay=1e-10)
    amp_enabled = device.type == "cuda"
    losses, step_times, utils = [], [], []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    for step in range(steps + 10):
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        indices = [(step * batch_size + offset) % len(adapter.rows) for offset in range(batch_size)]
        batch = make_batch(adapter, indices, device, encoded["input_ids"], encoded["attention_mask"])
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
            loss, _ = policy(batch)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss for batch {batch_size}, step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), float(config["training"]["grad_clip_norm"]))
        optimizer.step()
        if step >= 10:
            losses.append(float(loss.detach().cpu()))
            step_times.append(time.perf_counter() - started)
            value = gpu_utilization()
            if value is not None:
                utils.append(value)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    peak_bytes = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    mean_time = float(np.mean(step_times))
    return {
        "batch_size": batch_size,
        "steps": steps,
        "status": "PASS",
        "loss_first_10_mean": float(np.mean(losses[:10])),
        "loss_last_10_mean": float(np.mean(losses[-10:])),
        "loss_finite": bool(np.isfinite(losses).all()),
        "peak_memory_bytes": int(peak_bytes),
        "peak_memory_gib": peak_bytes / (1024**3),
        "step_time_mean_seconds": mean_time,
        "step_time_p95_seconds": float(np.percentile(step_times, 95)),
        "samples_per_second": batch_size / mean_time,
        "gpu_utilization_mean_percent": float(np.mean(utils)) if utils else None,
        "gpu_utilization_samples": len(utils),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/smolvla_main.yaml"))
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    rows = []
    for batch_size in args.batch_sizes:
        try:
            result = benchmark(config, batch_size, args.steps)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            result = {"batch_size": batch_size, "steps": args.steps, "status": "OOM_OR_ERROR", "error": str(exc)}
        print(json.dumps(result, ensure_ascii=False))
        rows.append(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
