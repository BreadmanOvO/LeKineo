#!/usr/bin/env python3
"""Create the deterministic 4-train/2-validation/1-held-out task split."""
import argparse, json
from pathlib import Path
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

DEFAULT_REPO = "HuggingFaceVLA/libero"
DEFAULT_REVISION = "86958911c0f959db2bbbdb107eb3e17c5f9c798e"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--repo", default=DEFAULT_REPO); ap.add_argument("--revision", default=DEFAULT_REVISION); ap.add_argument("--out", required=True); args = ap.parse_args()
    meta = LeRobotDatasetMetadata(args.repo, revision=args.revision)
    task_names = {int(r.task_index): str(task) for task, r in meta.tasks.iterrows()}
    train_ids, val_ids, test_ids = [30, 32, 34, 36], [31, 33], [39]
    selected = train_ids + val_ids + test_ids
    episode_ids = {"train": [], "val": [], "test": []}
    for row in meta.episodes:
        tid = next((i for i, name in task_names.items() if name in (row["tasks"] or [])), None)
        if tid not in selected: continue
        split = "train" if tid in train_ids else "val" if tid in val_ids else "test"
        episode_ids[split].append(int(row["episode_index"]))
    result = {"repo_id": args.repo, "revision": args.revision, "strategy": "task-level deterministic split",
              "train_task_ids": train_ids, "val_task_ids": val_ids, "test_task_ids": test_ids,
              "held_out_task_id": 39, "unused_task_ids": [x for x in sorted(task_names) if x not in selected],
              "task_names": {str(k): task_names[k] for k in sorted(task_names)}, "episode_ids": episode_ids}
    sets = {k: set(v) for k, v in episode_ids.items()}
    result["checks"] = {"selected_tasks_assigned_once": len(set(selected)) == len(selected), "four_training_tasks": len(train_ids) == 4,
                        "episode_sets_disjoint": not (sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"]),
                        "held_out_not_train": 39 not in train_ids}
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result["checks"], indent=2)); print({k: len(v) for k, v in episode_ids.items()})

if __name__ == "__main__": main()

