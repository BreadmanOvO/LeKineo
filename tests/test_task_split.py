import json
import os
from pathlib import Path

import pytest

def test_generated_split_has_no_leakage():
    manifest = os.environ.get("LEKINEO_TASK_SPLIT")
    if not manifest:
        pytest.skip("set LEKINEO_TASK_SPLIT to an acceptance manifest")
    path = Path(manifest)
    assert path.is_file(), f"manifest does not exist: {path}"
    split = json.loads(path.read_text(encoding="utf-8"))
    train, val, test = (set(split["episode_ids"][x]) for x in ("train", "val", "test"))
    assert not train & val and not train & test and not val & test
    assert split["held_out_task_id"] not in split["train_task_ids"]
    assert len(split["train_task_ids"]) == 4
