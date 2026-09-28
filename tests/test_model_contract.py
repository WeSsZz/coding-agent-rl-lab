import hashlib
import json
import struct
from pathlib import Path

import pytest

from scripts.model_contract import validate_snapshot, selected_model


def snapshot(tmp_path):
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        (tmp_path / name).write_text('{}', encoding='utf-8')
    shard = 'model-00001-of-00001.safetensors'
    header = json.dumps({'weight': {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}}).encode()
    (tmp_path / shard).write_bytes(struct.pack('<Q', len(header)) + header + b'\0' * 4)
    (tmp_path / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'weight': shard}}))
    model = {'id': 'fixture', 'revision': 'a' * 40,
             'config_json_sha256': hashlib.sha256(b'{}').hexdigest(),
             'tokenizer_config_json_sha256': hashlib.sha256(b'{}').hexdigest()}
    inventory = {'id': model['id'], 'sha': model['revision'], 'siblings': []}
    for path in tmp_path.iterdir():
        raw = path.read_bytes()
        inventory['siblings'].append({'rfilename': path.name, 'blobId': hashlib.sha1(
            b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()})
    return model, inventory, tmp_path / shard


def test_mainline_is_qwen25():
    assert selected_model()['id'] == 'Qwen/Qwen2.5-Coder-7B-Instruct'


def test_qwen3_manifest_is_separate_and_pinned():
    manifest = Path(__file__).resolve().parents[1] / 'manifests/route-v3-qwen3-model.json'
    model = selected_model(manifest)
    assert model['id'] == 'Qwen/Qwen3-8B'
    assert model['revision'] == 'b968826d9c46dd6066d109eabc6255188de91218'
    assert model['thinking_mode'].startswith('non-thinking:')


def test_complete_snapshot(tmp_path):
    model, inventory, _ = snapshot(tmp_path)
    assert validate_snapshot(tmp_path, model, inventory)['verified']


def test_truncated_shard_rejected(tmp_path):
    model, inventory, shard = snapshot(tmp_path)
    shard.write_bytes(shard.read_bytes()[:-1])
    with pytest.raises(ValueError, match='truncated'):
        validate_snapshot(tmp_path, model, inventory)


def test_other_weights_with_same_config_rejected(tmp_path):
    model, inventory, shard = snapshot(tmp_path)
    shard.write_bytes(shard.read_bytes()[:-1] + b'1')
    with pytest.raises(ValueError, match='hash mismatch'):
        validate_snapshot(tmp_path, model, inventory)


def test_no_provenance_rejected(tmp_path):
    model, _, _ = snapshot(tmp_path)
    with pytest.raises(ValueError, match='inventory required'):
        validate_snapshot(tmp_path, model)


def test_other_revision_rejected(tmp_path):
    model, inventory, _ = snapshot(tmp_path)
    inventory['sha'] = 'b' * 40
    with pytest.raises(ValueError, match='revision mismatch'):
        validate_snapshot(tmp_path, model, inventory)
