#!/usr/bin/env python3
"""Download the verified data shards containing split episodes 1261-1692."""
from huggingface_hub import hf_hub_download

REVISION = "86958911c0f959db2bbbdb107eb3e17c5f9c798e"

for index in range(308, 377):
    filename = f"data/chunk-000/file-{index:03d}.parquet"
    print(f"START {filename}", flush=True)
    path = hf_hub_download("HuggingFaceVLA/libero", filename, revision=REVISION, repo_type="dataset")
    print(f"READY {path}", flush=True)
print("DONE", flush=True)
