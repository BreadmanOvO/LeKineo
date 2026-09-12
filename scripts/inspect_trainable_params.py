#!/usr/bin/env python3
"""Audit SmolVLA freezing and trainable parameter groups."""

import argparse
import json
from pathlib import Path

import torch
from lerobot.policies.smolvla import SmolVLAPolicy


def build_policy(model_name: str) -> SmolVLAPolicy:
    policy = SmolVLAPolicy.from_pretrained(model_name)
    policy.config.freeze_vision_encoder = True
    policy.config.train_expert_only = True
    policy.config.train_state_proj = True
    return policy


def classify(name: str) -> str:
    lower = name.lower()
    if "vision_model" in lower:
        return "vision_encoder"
    if "lm_expert" in lower or any(token in lower for token in ("action_in_proj", "action_out_proj", "action_time_mlp")):
        return "action_expert"
    if "state_proj" in lower:
        return "state_projector"
    return "frozen_vlm_or_other"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lerobot/smolvla_base")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    policy = build_policy(args.model)
    groups: dict[str, dict[str, object]] = {}
    for name, parameter in policy.named_parameters():
        group = classify(name)
        entry = groups.setdefault(group, {"parameter_count": 0, "trainable_count": 0, "parameters": []})
        entry["parameter_count"] += parameter.numel()
        entry["trainable_count"] += int(parameter.requires_grad) * parameter.numel()
        entry["parameters"].append({"name": name, "shape": list(parameter.shape), "requires_grad": parameter.requires_grad})

    result = {
        "model": args.model,
        "config": {
            "freeze_vision_encoder": policy.config.freeze_vision_encoder,
            "train_expert_only": policy.config.train_expert_only,
            "train_state_proj": policy.config.train_state_proj,
        },
        "total_parameters": sum(parameter.numel() for parameter in policy.parameters()),
        "total_trainable_parameters": sum(parameter.numel() for parameter in policy.parameters() if parameter.requires_grad),
        "groups": groups,
        "vision_trainable_parameters": [name for name, p in policy.named_parameters() if "vision_model" in name.lower() and p.requires_grad],
    }
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
