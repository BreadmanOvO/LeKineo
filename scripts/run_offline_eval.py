#!/usr/bin/env python3
"""Compare SmolVLA checkpoints on a deterministic LIBERO validation subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from run_smoke_train import LiberoAdapter, load_config, set_seed  # noqa: E402
from lerobot.policies.smolvla import SmolVLAPolicy  # noqa: E402
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS  # noqa: E402


def load_trainable(policy: SmolVLAPolicy, checkpoint: Path) -> None:
    payload = torch.load(checkpoint / "trainable_state.pt", map_location="cpu", weights_only=True)
    missing, unexpected = policy.load_state_dict(payload["state_dict"], strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected keys in {checkpoint}: {unexpected[:3]}")
    missing_trainable = [key for key in payload["state_dict"] if key not in policy.state_dict()]
    if missing_trainable:
        raise RuntimeError(f"missing trainable keys in {checkpoint}: {missing_trainable[:3]}")


def sample_digest(indices: list[int]) -> str:
    return hashlib.sha256(json.dumps(indices, separators=(",", ":")).encode()).hexdigest()


def evaluate_policy(
    name: str,
    checkpoint: Path | None,
    config: dict,
    adapter: LiberoAdapter,
    indices: list[int],
    device: torch.device,
    tokens: torch.Tensor,
    attention: torch.Tensor,
    seed: int,
) -> dict:
    policy = SmolVLAPolicy.from_pretrained(config["model"]["pretrained"])
    policy.config.freeze_vision_encoder = True
    policy.config.train_expert_only = True
    policy.config.train_state_proj = True
    if checkpoint is not None:
        load_trainable(policy, checkpoint)
    policy.to(device).eval()
    losses: list[float] = []
    l1_values: list[float] = []
    mse_values: list[float] = []
    smooth_values: list[float] = []
    prediction_shapes: set[str] = set()
    with torch.no_grad():
        for position, row_index in enumerate(indices):
            batch = adapter.batch(row_index, device, tokens, attention)
            torch.manual_seed(seed + position)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed + position)
            loss, _ = policy(batch)
            policy.reset()
            torch.manual_seed(seed + position)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed + position)
            prediction = policy.predict_action_chunk(batch)
            target = batch["action"]
            if prediction.shape != target.shape:
                raise RuntimeError(f"{name} shape mismatch: prediction={prediction.shape} target={target.shape}")
            prediction_shapes.add(str(tuple(prediction.shape)))
            losses.append(float(loss.detach().cpu()))
            error = prediction.float() - target.float()
            l1_values.append(float(error.abs().mean().cpu()))
            mse_values.append(float(error.square().mean().cpu()))
            smooth_values.append(float(prediction.float().diff(dim=1).abs().mean().cpu()))
    return {
        "name": name,
        "checkpoint": f"outputs/smolvla_main/checkpoints/step-{int(name):07d}" if name.isdigit() else "lerobot/smolvla_base",
        "samples": len(indices),
        "validation_loss_mean": float(np.mean(losses)),
        "validation_loss_p95": float(np.percentile(losses, 95)),
        "action_l1_mean": float(np.mean(l1_values)),
        "action_mse_mean": float(np.mean(mse_values)),
        "action_smoothness_mean": float(np.mean(smooth_values)),
        "prediction_shapes": sorted(prediction_shapes),
        "finite": bool(np.isfinite(losses + l1_values + mse_values + smooth_values).all()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=SCRIPT_DIR.parent / "configs/smolvla_main.yaml")
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()
    config = load_config(args.config)
    split = json.loads(args.split.read_text(encoding="utf-8"))
    requested_val_episodes = set(split["episode_ids"]["val"])
    adapter = LiberoAdapter(config["data"]["parquet"], list(config["data"]["state_indices"]), list(config["data"]["action_indices"]), 50)
    available_episodes = sorted({row["episode_index"] for row in adapter.rows})
    val_episodes = requested_val_episodes.intersection(available_episodes)
    split_status = "pinned"
    if not val_episodes:
        # The cached parquet can be a small shard whose local episode numbering
        # differs from the full task split. Keep the fallback explicit in output.
        val_episodes = {available_episodes[-1]}
        split_status = "provisional_fallback"
    indices = [index for index, row in enumerate(adapter.rows) if row["episode_index"] in val_episodes]
    indices = indices[: args.samples]
    if len(indices) < args.samples:
        raise RuntimeError(f"validation rows available={len(indices)} < requested={args.samples}")
    device = torch.device(config["model"].get("device", "cuda") if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    base_policy = SmolVLAPolicy.from_pretrained(config["model"]["pretrained"])
    processor = base_policy.model.vlm_with_expert.processor
    encoded = processor.tokenizer(config["data"]["language"], return_tensors="pt")
    del base_policy
    checkpoint_root = Path(os.environ.get("LEKINEO_EVIDENCE", ".")) / "outputs/smolvla_main/checkpoints"
    candidates: list[tuple[str, Path | None]] = [("baseline", None)]
    for step in (2000, 5000, 10000):
        candidates.append((str(step), checkpoint_root / f"step-{step:07d}"))
    results = [evaluate_policy(name, checkpoint, config, adapter, indices, device, encoded["input_ids"], encoded["attention_mask"], args.seed) for name, checkpoint in candidates]
    payload = {
        "schema_version": 1,
        "data_revision": f"{split['repo_id']}@{split['revision']}",
        "validation_episode_ids": sorted(val_episodes),
        "requested_validation_episode_ids": sorted(requested_val_episodes),
        "available_episode_ids": available_episodes,
        "split_status": split_status,
        "validation_row_indices": indices,
        "validation_sample_digest": sample_digest(indices),
        "samples": len(indices),
        "language": config["data"]["language"],
        "schema_adapter": {"state_indices": config["data"]["state_indices"], "action_indices": config["data"]["action_indices"], "camera3_source": config["data"]["camera3_source"]},
        "metrics_definition": {"validation_loss": "mean policy flow-matching loss", "action_l1": "mean absolute error over action chunk", "action_mse": "mean squared error over action chunk", "action_smoothness": "mean absolute adjacent-step delta of predicted action chunk"},
        "normalization": "reuse policy and training adapter; no per-checkpoint refit",
        "results": results,
        "finite_all": all(item["finite"] for item in results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
