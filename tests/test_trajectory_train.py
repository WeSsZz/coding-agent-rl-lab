import hashlib
import json

import pytest

from coding_agent_rl_lab.trajectory_sft import TrajectoryError
from coding_agent_rl_lab.trajectory_train import load_training_data, select_smoke_row


MODEL = {"id": "test/model", "revision": "fixed"}


def write_data(tmp_path, *, labels=None, ids=None):
    row = {"instance_id": "repo-1", "input_ids": ids or [1, 2, 3],
           "labels": labels or [-100, -100, 3], "attention_mask": [1, 1, 1]}
    raw = (json.dumps(row) + "\n").encode()
    (tmp_path / "train.jsonl").write_bytes(raw)
    manifest = {"schema": "complete-trajectory-tokens-v1", "model": MODEL["id"],
                "revision": MODEL["revision"], "data_sha256": hashlib.sha256(raw).hexdigest(),
                "trajectory_count": 1, "max_length": 10, "total_tokens": 3,
                "total_supervised_tokens": 1}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))


def test_valid_data_loads_without_torch(tmp_path):
    write_data(tmp_path)
    rows, _ = load_training_data(tmp_path, MODEL)
    assert rows[0]["labels"] == [-100, -100, 3]


def test_changed_data_rejected(tmp_path):
    write_data(tmp_path)
    with (tmp_path / "train.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(TrajectoryError, match="hash_mismatch"):
        load_training_data(tmp_path, MODEL)


@pytest.mark.parametrize("labels,reason", [([-100, -100, 7], "invalid_training_mask"),
                                            ([-100, -100, -100], "no_causal_targets")])
def test_bad_labels_rejected_even_with_matching_file_hash(tmp_path, labels, reason):
    write_data(tmp_path, labels=labels)
    with pytest.raises(TrajectoryError, match=reason):
        load_training_data(tmp_path, MODEL)


def test_wrong_model_rejected(tmp_path):
    write_data(tmp_path)
    with pytest.raises(TrajectoryError, match="model_mismatch"):
        load_training_data(tmp_path, {"id": "other", "revision": "fixed"})


def test_smoke_selection_uses_longest_complete_row_under_cap():
    rows = [{"instance_id": "short", "input_ids": [0] * 10},
            {"instance_id": "medium", "input_ids": [0] * 20},
            {"instance_id": "long", "input_ids": [0] * 30}]
    assert select_smoke_row(rows, 25)["instance_id"] == "medium"
    assert select_smoke_row(rows)["instance_id"] == "long"


def test_smoke_selection_rejects_cap_without_complete_row():
    rows = [{"instance_id": "long", "input_ids": [0] * 30}]
    with pytest.raises(TrajectoryError, match="no_complete_trajectory_under_smoke_cap"):
        select_smoke_row(rows, 25)


@pytest.mark.parametrize("cap", [0, 1, True, "24576"])
def test_smoke_selection_rejects_invalid_cap(cap):
    with pytest.raises(TrajectoryError, match="invalid_smoke_token_cap"):
        select_smoke_row([{"input_ids": [0] * 3}], cap)
