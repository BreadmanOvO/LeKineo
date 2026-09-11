#!/usr/bin/env python3
"""Inspect one frozen parquet shard and render 20 image/action audit cards."""
import argparse, io, json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image, ImageDraw

DEFAULT_REPO = "HuggingFaceVLA/libero"
DEFAULT_REVISION = "86958911c0f959db2bbbdb107eb3e17c5f9c798e"

def decode_image(value):
    if isinstance(value, dict) and value.get("bytes") is not None: return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
    if isinstance(value, dict) and value.get("path"): return Image.open(value["path"]).convert("RGB")
    if isinstance(value, (bytes, bytearray)): return Image.open(io.BytesIO(value)).convert("RGB")
    raise TypeError(f"Unsupported image cell: {type(value)}")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--repo", default=DEFAULT_REPO); ap.add_argument("--revision", default=DEFAULT_REVISION)
    ap.add_argument("--shard", default="data/chunk-000/file-000.parquet"); ap.add_argument("--out-dir", required=True); ap.add_argument("--report", required=True); args = ap.parse_args()
    path = hf_hub_download(args.repo, args.shard, repo_type="dataset", revision=args.revision)
    table = pq.read_table(path)
    required = ["observation.images.image", "observation.images.image2", "observation.state", "action", "timestamp", "frame_index", "episode_index", "task_index"]
    missing = [x for x in required if x not in table.column_names]
    if missing: raise RuntimeError(f"missing columns: {missing}")
    df = table.select(required).to_pandas(); state_shapes = {tuple(np.asarray(v).shape) for v in df["observation.state"]}; action_shapes = {tuple(np.asarray(v).shape) for v in df["action"]}
    temporal_ok, bad = True, []
    for ep, rows in df.groupby("episode_index", sort=False):
        frame, ts = rows["frame_index"].to_numpy(), rows["timestamp"].to_numpy()
        if not (np.array_equal(frame, np.arange(len(frame))) and np.all(np.diff(ts) > 0)): temporal_ok = False; bad.append(int(ep))
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    for n, idx in enumerate(np.linspace(0, len(df)-1, 20, dtype=int)):
        row = df.iloc[int(idx)]; a = decode_image(row["observation.images.image"]).resize((256,256)); b = decode_image(row["observation.images.image2"]).resize((256,256))
        canvas = Image.new("RGB", (512,300), "white"); canvas.paste(a,(0,0)); canvas.paste(b,(256,0)); draw = ImageDraw.Draw(canvas)
        draw.text((8,264), f"ep={int(row.episode_index)} frame={int(row.frame_index)} task={int(row.task_index)}", fill="black")
        draw.text((8,280), "action=" + np.array2string(np.asarray(row.action), precision=2), fill="black"); canvas.save(out/f"sample_{n:02d}.jpg", quality=90)
    report = {"repo_id": args.repo, "revision": args.revision, "shard": args.shard, "rows_checked": len(df), "episodes_checked": int(df.episode_index.nunique()),
              "state_shapes": [list(x) for x in sorted(state_shapes)], "action_shapes": [list(x) for x in sorted(action_shapes)],
              "checks": {"required_columns": not missing, "state_dim_8": state_shapes == {(8,)}, "action_dim_7": action_shapes == {(7,)}, "frame_and_timestamp_monotonic": temporal_ok, "twenty_samples_saved": len(list(out.glob("sample_*.jpg"))) == 20}, "bad_episodes": bad}
    dst = Path(args.report); dst.parent.mkdir(parents=True, exist_ok=True); dst.write_text(json.dumps(report, indent=2), encoding="utf-8"); print(json.dumps(report, indent=2))

if __name__ == "__main__": main()

