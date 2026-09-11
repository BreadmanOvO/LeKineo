#!/usr/bin/env python3
"""Audit a frozen LeRobot LIBERO dataset using metadata only."""
import argparse, json
from collections import Counter, defaultdict
from pathlib import Path
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

DEFAULT_REPO = "HuggingFaceVLA/libero"
DEFAULT_REVISION = "86958911c0f959db2bbbdb107eb3e17c5f9c798e"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--revision", default=DEFAULT_REVISION)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    meta = LeRobotDatasetMetadata(args.repo, revision=args.revision)
    task_names = {int(r.task_index): str(task) for task, r in meta.tasks.iterrows()}
    counts, frames, episode_ids = Counter(), Counter(), defaultdict(list)
    for row in meta.episodes:
        for task in row["tasks"] or []:
            tid = next((k for k, v in task_names.items() if v == task), None)
            if tid is not None:
                counts[tid] += 1; frames[tid] += int(row["length"]); episode_ids[tid].append(int(row["episode_index"]))
    features = {k: {"dtype": str(v.get("dtype")), "shape": list(v.get("shape", []))} for k, v in meta.features.items()}
    stats = {"repo_id": args.repo, "revision": args.revision, "total_episodes": meta.total_episodes,
             "total_frames": meta.total_frames, "total_tasks": meta.total_tasks, "fps": meta.fps,
             "features": features,
             "task_counts": {str(k): {"task": task_names[k], "episodes": counts[k], "frames": frames[k]} for k in sorted(task_names)},
             "episode_ids_by_task": {str(k): sorted(v) for k, v in episode_ids.items()}}
    stats["checks"] = {"state_dim": features.get("observation.state", {}).get("shape") == [8],
                       "action_dim": features.get("action", {}).get("shape") == [7],
                       "two_cameras": all(k in features for k in ("observation.images.image", "observation.images.image2")),
                       "episode_count_matches": len(meta.episodes) == meta.total_episodes}
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats["checks"], indent=2)); print(f"tasks={meta.total_tasks} episodes={meta.total_episodes} frames={meta.total_frames}")

if __name__ == "__main__": main()

