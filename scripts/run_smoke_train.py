#!/usr/bin/env python3
"""Run a small, auditable SmolVLA fine-tuning smoke test on LIBERO parquet data."""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from io import BytesIO
import pyarrow.parquet as pq

from lerobot.policies.smolvla import SmolVLAPolicy
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    def expand(value):
        if isinstance(value, dict):
            return {key: expand(item) for key, item in value.items()}
        if isinstance(value, list):
            return [expand(item) for item in value]
        if isinstance(value, str):
            return os.path.expandvars(value)
        return value

    return expand(config)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LiberoAdapter:
    """Loads real rows and applies explicit LIBERO -> SmolVLA schema adapters."""

    def __init__(self, parquet_path: str, state_indices: list[int], action_indices: list[int], chunk_size: int,
                 episode_ids: set[int] | None = None, task_prompts: dict[int, str] | None = None):
        paths = sorted(glob.glob(parquet_path)) or [parquet_path]
        columns = ["observation.images.image", "observation.images.image2", "observation.state", "action", "episode_index", "frame_index", "task_index"]
        self.rows = []
        for path in paths:
            table = pq.read_table(path, columns=columns)
            self.rows.extend(table.to_pylist())
        if episode_ids is not None:
            self.rows = [row for row in self.rows if int(row["episode_index"]) in episode_ids]
        self.rows.sort(key=lambda row: (int(row["episode_index"]), int(row["frame_index"])))
        if not self.rows:
            raise ValueError(f"no rows found for parquet={parquet_path!r}, episode_ids={episode_ids}")
        self.state_indices = state_indices
        self.action_indices = action_indices
        self.chunk_size = chunk_size
        self.task_prompts = task_prompts or {}
        self.images: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.states: list[torch.Tensor] = []
        self.actions: list[torch.Tensor] = []
        self._decode_rows()

    def _decode_image(self, value: dict) -> torch.Tensor:
        image = Image.open(BytesIO(value["bytes"])).convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    def _decode_rows(self) -> None:
        for index, row in enumerate(self.rows):
            image1 = self._decode_image(row["observation.images.image"])
            image2 = self._decode_image(row["observation.images.image2"])
            self.images.append((image1, image2))
            self.states.append(torch.tensor(row["observation.state"], dtype=torch.float32)[self.state_indices])
            action = torch.tensor(row["action"], dtype=torch.float32)[self.action_indices]
            window: list[torch.Tensor] = []
            episode = row["episode_index"]
            for offset in range(self.chunk_size):
                candidate = min(index + offset, len(self.rows) - 1)
                if self.rows[candidate]["episode_index"] != episode:
                    candidate = index
                window.append(torch.tensor(self.rows[candidate]["action"], dtype=torch.float32)[self.action_indices])
            self.actions.append(torch.stack(window))

    def batch(self, index: int, device: torch.device, tokens: torch.Tensor | dict[int, torch.Tensor], attention: torch.Tensor | dict[int, torch.Tensor]) -> dict[str, torch.Tensor]:
        image1, image2 = self.images[index % len(self.images)]
        task_id = int(self.rows[index % len(self.rows)].get("task_index", -1))
        selected_tokens = tokens.get(task_id, next(iter(tokens.values()))) if isinstance(tokens, dict) else tokens
        selected_attention = attention.get(task_id, next(iter(attention.values()))) if isinstance(attention, dict) else attention
        return {
            "observation.images.camera1": image1.unsqueeze(0).to(device),
            "observation.images.camera2": image2.unsqueeze(0).to(device),
            "observation.images.camera3": image2.unsqueeze(0).to(device),
            "observation.state": self.states[index % len(self.states)].unsqueeze(0).to(device),
            "action": self.actions[index % len(self.actions)].unsqueeze(0).to(device),
            OBS_LANGUAGE_TOKENS: selected_tokens.to(device),
            OBS_LANGUAGE_ATTENTION_MASK: selected_attention.bool().to(device),
        }


class LazyLiberoAdapter:
    """Episode-indexed adapter that decodes only the requested episode."""

    def __init__(self, parquet_path: str, state_indices: list[int], action_indices: list[int], chunk_size: int,
                 episode_ids: set[int] | None = None, task_prompts: dict[int, str] | None = None):
        self.paths = sorted(glob.glob(parquet_path)) or [parquet_path]
        self.state_indices = state_indices
        self.action_indices = action_indices
        self.chunk_size = chunk_size
        self.task_prompts = task_prompts or {}
        self.rows: list[dict] = []
        for path in self.paths:
            table = pq.read_table(path, columns=["episode_index", "frame_index", "task_index"])
            for row_index, row in enumerate(table.to_pylist()):
                episode = int(row["episode_index"])
                if episode_ids is None or episode in episode_ids:
                    self.rows.append({**row, "episode_index": episode, "frame_index": int(row["frame_index"]),
                                      "_path": path, "_row_index": row_index})
        self.rows.sort(key=lambda row: (row["episode_index"], row["frame_index"]))
        if not self.rows:
            raise ValueError(f"no rows found for parquet={parquet_path!r}, episode_ids={episode_ids}")
        self._episode_cache: dict[int, list[dict]] = {}

    @staticmethod
    def _decode_image(value: dict) -> torch.Tensor:
        image = Image.open(BytesIO(value["bytes"])).convert("RGB")
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    def _load_episode(self, episode: int) -> list[dict]:
        if episode in self._episode_cache:
            return self._episode_cache[episode]
        source = next(row["_path"] for row in self.rows if row["episode_index"] == episode)
        columns = ["observation.images.image", "observation.images.image2", "observation.state", "action",
                   "episode_index", "frame_index", "task_index"]
        table = pq.read_table(source, columns=columns, filters=[("episode_index", "=", episode)])
        rows = sorted(table.to_pylist(), key=lambda row: int(row["frame_index"]))
        # Keep a one-episode cache: validation iteration must not accumulate
        # decoded JPEG tensors for every episode in the split.
        self._episode_cache.clear()
        self._episode_cache[episode] = rows
        return rows

    def batch(self, index: int, device: torch.device, tokens: torch.Tensor | dict[int, torch.Tensor],
              attention: torch.Tensor | dict[int, torch.Tensor]) -> dict[str, torch.Tensor]:
        meta = self.rows[index % len(self.rows)]
        episode_rows = self._load_episode(meta["episode_index"])
        local_index = next(i for i, row in enumerate(episode_rows) if int(row["frame_index"]) == meta["frame_index"])
        row = episode_rows[local_index]
        image1 = self._decode_image(row["observation.images.image"])
        image2 = self._decode_image(row["observation.images.image2"])
        state = torch.tensor(row["observation.state"], dtype=torch.float32)[self.state_indices]
        window = []
        for offset in range(self.chunk_size):
            candidate = min(local_index + offset, len(episode_rows) - 1)
            window.append(torch.tensor(episode_rows[candidate]["action"], dtype=torch.float32)[self.action_indices])
        action = torch.stack(window)
        task_id = int(row.get("task_index", -1))
        selected_tokens = tokens.get(task_id, next(iter(tokens.values()))) if isinstance(tokens, dict) else tokens
        selected_attention = attention.get(task_id, next(iter(attention.values()))) if isinstance(attention, dict) else attention
        return {
            "observation.images.camera1": image1.unsqueeze(0).to(device),
            "observation.images.camera2": image2.unsqueeze(0).to(device),
            "observation.images.camera3": image2.unsqueeze(0).to(device),
            "observation.state": state.unsqueeze(0).to(device),
            "action": action.unsqueeze(0).to(device),
            OBS_LANGUAGE_TOKENS: selected_tokens.to(device),
            OBS_LANGUAGE_ATTENTION_MASK: selected_attention.bool().to(device),
        }


def split_inputs(config: dict, split_name: str) -> tuple[set[int] | None, dict[int, str]]:
    split_path = config["data"].get("task_split")
    if not split_path or not Path(split_path).exists():
        return None, {}
    payload = json.loads(Path(split_path).read_text(encoding="utf-8"))
    ids = set(int(value) for value in payload.get("episode_ids", {}).get(split_name, []))
    prompts = {int(key): value for key, value in payload.get("task_names", {}).items()}
    return ids, prompts


def encode_task_prompts(processor, prompts: dict[int, str], fallback: str) -> tuple[dict[int, torch.Tensor] | torch.Tensor, dict[int, torch.Tensor] | torch.Tensor]:
    if not prompts:
        encoded = processor.tokenizer(fallback, return_tensors="pt")
        return encoded["input_ids"], encoded["attention_mask"]
    encoded = {task_id: processor.tokenizer(text, return_tensors="pt")["input_ids"] for task_id, text in prompts.items()}
    attention = {task_id: processor.tokenizer(text, return_tensors="pt")["attention_mask"] for task_id, text in prompts.items()}
    return encoded, attention


def trainable_snapshot(policy: SmolVLAPolicy) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().cpu().clone() for name, parameter in policy.named_parameters() if parameter.requires_grad}


def group_delta(before: dict[str, torch.Tensor], policy: SmolVLAPolicy, predicate) -> float:
    deltas = []
    for name, parameter in policy.named_parameters():
        if predicate(name) and name in before:
            deltas.append(float((parameter.detach().cpu() - before[name]).abs().max()))
    return max(deltas, default=0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/smolvla_smoke.yaml"))
    parser.add_argument("--steps", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    steps = args.steps or int(config["training"]["steps"])
    output = Path(config["output"]["directory"])
    checkpoint = Path(config["output"]["checkpoint"])
    output.mkdir(parents=True, exist_ok=True)
    checkpoint.mkdir(parents=True, exist_ok=True)
    set_seed(int(config["training"]["seed"]))

    device = torch.device(config["model"].get("device", "cuda") if torch.cuda.is_available() else "cpu")
    policy = SmolVLAPolicy.from_pretrained(config["model"]["pretrained"])
    policy.config.freeze_vision_encoder = True
    policy.config.train_expert_only = True
    policy.config.train_state_proj = True
    policy.to(device)
    policy.train()
    frozen_vision = [name for name, parameter in policy.named_parameters() if "vision_model" in name.lower() and parameter.requires_grad]
    if frozen_vision:
        raise RuntimeError(f"Vision parameters unexpectedly trainable: {frozen_vision[:3]}")

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
    before = trainable_snapshot(policy)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in policy.parameters() if parameter.requires_grad],
        lr=float(config["training"]["learning_rate"]),
        weight_decay=1e-10,
    )
    amp_enabled = bool(config["training"].get("use_amp", True)) and device.type == "cuda"
    scaler_enabled = False
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
    losses: list[float] = []
    step_times: list[float] = []
    peak_memory = 0
    log_every = int(config["training"].get("log_every", 10))
    for step in range(steps):
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        batch = adapter.batch(step, device, encoded_tokens, encoded_attention)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
            loss, _ = policy(batch)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step + 1}: {loss.item()}")
        if scaler_enabled:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), float(config["training"]["grad_clip_norm"]))
        if scaler_enabled:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        value = float(loss.detach().cpu())
        losses.append(value)
        step_times.append(time.perf_counter() - started)
        if device.type == "cuda":
            peak_memory = max(peak_memory, torch.cuda.max_memory_allocated(device))
        if (step + 1) % log_every == 0 or step == 0 or step + 1 == steps:
            print(f"step={step + 1}/{steps} loss={value:.6f} grad_norm={float(grad_norm):.4f} step_time={step_times[-1]:.3f}s")

    vision_delta = group_delta(before, policy, lambda name: "vision_model" in name.lower())
    expert_delta = group_delta(before, policy, lambda name: "lm_expert" in name or any(x in name for x in ("action_in_proj", "action_out_proj", "action_time_mlp")))
    state_delta = group_delta(before, policy, lambda name: "state_proj" in name)
    trainable_state = {name: parameter.detach().cpu() for name, parameter in policy.named_parameters() if parameter.requires_grad}
    torch.save({"format": "trainable_state_dict_v1", "state_dict": trainable_state}, checkpoint / "trainable_state.pt")
    policy.config.save_pretrained(checkpoint)
    reloaded = SmolVLAPolicy.from_pretrained(config["model"]["pretrained"])
    payload = torch.load(checkpoint / "trainable_state.pt", map_location="cpu", weights_only=True)
    missing, unexpected = reloaded.load_state_dict(payload["state_dict"], strict=False)
    missing_trainable = [key for key in payload["state_dict"] if key not in reloaded.state_dict()]
    reload_ok = len(unexpected) == 0 and len(missing_trainable) == 0
    metrics = {
        "steps": steps,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_first_10_mean": float(np.mean(losses[: min(10, len(losses))])),
        "loss_last_10_mean": float(np.mean(losses[-min(10, len(losses)) :])),
        "loss_finite": bool(all(np.isfinite(losses))),
        "loss_downward_trend": bool(np.mean(losses[-min(10, len(losses)) :]) < np.mean(losses[: min(10, len(losses))])),
        "vision_encoder_max_abs_delta": vision_delta,
        "action_expert_max_abs_delta": expert_delta,
        "state_projector_max_abs_delta": state_delta,
        "vision_encoder_frozen": vision_delta == 0.0,
        "action_expert_updated": expert_delta > 0.0,
        "state_projector_updated": state_delta > 0.0,
        "checkpoint": str(checkpoint),
        "checkpoint_reload_ok": reload_ok,
        "reload_missing_count": len(missing),
        "reload_missing_trainable_count": len(missing_trainable),
        "reload_unexpected_count": len(unexpected),
        "peak_memory_bytes": peak_memory,
        "peak_memory_gib": peak_memory / (1024**3),
        "step_time_mean_seconds": float(np.mean(step_times)),
        "step_time_p95_seconds": float(np.percentile(step_times, 95)),
        "schema_adapter": {
            "state_source_dim": 8,
            "state_target_dim": 6,
            "state_indices": config["data"]["state_indices"],
            "action_source_dim": 7,
            "action_target_dim": 6,
            "action_indices": config["data"]["action_indices"],
            "camera3_source": config["data"]["camera3_source"],
        },
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (output / "losses.json").write_text(json.dumps(losses) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    if not (metrics["loss_finite"] and metrics["vision_encoder_frozen"] and metrics["action_expert_updated"] and metrics["checkpoint_reload_ok"]):
        raise RuntimeError("Day 4 smoke acceptance criteria failed")


if __name__ == "__main__":
    main()
