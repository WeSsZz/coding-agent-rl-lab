import copy
import json
from pathlib import Path

import pytest

from coding_agent_rl_lab.trajectory_sft import (
    TrajectoryError, assistant_examples, encode_example, encode_trajectory, pad_training_batch, normalize, prepare,
)


TOOLS = [{"type": "function", "function": {"name": "execute_bash", "parameters": {
    "type": "object", "required": ["command"], "properties": {"command": {"type": "string"}}}}}]


def sample():
    return {"trajectory_id": "t1", "instance_id": "other__repo-1", "repo": "other/repo", "tools": TOOLS,
            "resolved": 1, "exit_status": "submit", "model_patch": "diff --git a/a b/a",
            "trajectory": [{"role": "system", "content": "system"},
                           {"role": "user", "content": "issue"},
                           {"role": "assistant", "content": None, "tool_calls": [
                               {"type": "function", "id": "c1", "function": {
                                   "name": "execute_bash", "arguments": '{"command":"pwd"}'}}]},
                           {"role": "tool", "content": "/repo", "tool_call_id": "c1"},
                           {"role": "assistant", "content": "done"}]}


def test_normalize_preserves_observation_and_does_not_mutate():
    row = sample()
    original = copy.deepcopy(row)
    result = normalize(row, TOOLS, set())
    assert row == original
    assert result[2]["tool_calls"][0]["function"]["arguments"] == {"command": "pwd"}
    assert result[3]["content"] == "/repo"
    assert len(list(assistant_examples(result))) == 2


@pytest.mark.parametrize("change,reason", [
    (lambda r: r.update(resolved=0), "not_successful_submission"),
    (lambda r: r.update(model_patch=""), "empty_patch"),
    (lambda r: r["trajectory"][3].update(tool_call_id="bad"), "orphan_tool_result"),
    (lambda r: r["trajectory"].pop(3), "missing_tool_result"),
    (lambda r: r["trajectory"][2]["tool_calls"][0]["function"].update(arguments='"string"'), "arguments_not_object"),
    (lambda r: r["trajectory"][2]["tool_calls"][0]["function"].update(arguments='{"command":123}'), "invalid_argument_schema"),
    (lambda r: r.update(tools=[]), "tool_schema_mismatch"),
])
def test_bad_trajectories_rejected(change, reason):
    row = sample()
    change(row)
    with pytest.raises(TrajectoryError, match=reason):
        normalize(row, TOOLS, set())


def test_repository_exclusion():
    with pytest.raises(TrajectoryError, match="evaluation_repository"):
        normalize(sample(), TOOLS, {"other/repo"})


class Tokenizer:
    def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
        return "".join(m["role"] + ":" + m["content"] + ";" for m in messages) + (
            "assistant:" if add_generation_prompt else "")

    def encode(self, text, **kwargs):
        return list(text.encode())


def test_only_completion_supervised_and_overlength_rejected():
    example = {"prompt": [{"role": "tool", "content": "SECRET_OBSERVATION"}],
               "completion": [{"role": "assistant", "content": "answer"}]}
    encoded = encode_example(Tokenizer(), example, [], 1000)
    supervised = bytes(x for x in encoded["labels"] if x != -100)
    assert supervised == b"answer;"
    with pytest.raises(TrajectoryError, match="over_max_length"):
        encode_example(Tokenizer(), example, [], 2)


def test_prefix_mismatch_fails():
    class BadTokenizer(Tokenizer):
        def apply_chat_template(self, messages, *, add_generation_prompt, **kwargs):
            return "wrong" if add_generation_prompt else "answer"
    with pytest.raises(TrajectoryError, match="token_prefix_mismatch"):
        encode_example(BadTokenizer(), {"prompt": [], "completion": []}, [], 100)


def test_complete_conversation_mask_excludes_all_observations():
    messages = [{"role": "system", "content": "policy"},
                {"role": "user", "content": "problem"},
                {"role": "assistant", "content": "first"},
                {"role": "tool", "content": "observation"},
                {"role": "assistant", "content": "finish"}]
    encoded = encode_trajectory(Tokenizer(), messages, [], 1000)
    assert bytes(t for t in encoded["labels"] if t != -100) == b"first;finish;"
    with pytest.raises(TrajectoryError, match="over_max_length"):
        encode_trajectory(Tokenizer(), messages, [], 12)


def test_padding_never_supervised():
    a = {"input_ids": [1, 2], "labels": [-100, 2], "attention_mask": [1, 1]}
    b = {"input_ids": [3], "labels": [3], "attention_mask": [1]}
    padded = pad_training_batch([a, b], 3)
    assert padded["input_ids"][1] == [3, 3]
    assert padded["labels"][1] == [3, -100]
    assert padded["attention_mask"][1] == [1, 0]


def test_fast_offset_mask_matches_prefix_mask():
    class FastTokenizer(Tokenizer):
        is_fast = True

        def __call__(self, text, **kwargs):
            return {"input_ids": list(text.encode()), "offset_mapping": [(i, i + 1) for i in range(len(text))]}
    messages = [{"role": "user", "content": "issue"},
                {"role": "assistant", "content": "act"},
                {"role": "tool", "content": "observation"},
                {"role": "assistant", "content": "finish"}]
    assert encode_trajectory(FastTokenizer(), messages, [], 1000) == encode_trajectory(Tokenizer(), messages, [], 1000)


def test_fast_mask_rejects_boundary_crossing_token():
    class CrossingTokenizer(Tokenizer):
        is_fast = True

        def __call__(self, text, **kwargs):
            return {"input_ids": [1], "offset_mapping": [(0, len(text))]}
    with pytest.raises(TrajectoryError, match="token_boundary_mismatch"):
        encode_trajectory(CrossingTokenizer(), [{"role": "assistant", "content": "answer"}], [], 1000)


def test_qwen3_nonthinking_tool_trajectory_keeps_prefixes():
    from jinja2 import Environment

    class Qwen3TemplateTokenizer(Tokenizer):
        chat_template = (Path(__file__).resolve().parents[1] / "envs/qwen3-8b-chat-template.jinja").read_text(encoding="utf-8")

        def apply_chat_template(self, messages, *, tools, add_generation_prompt, **kwargs):
            env = Environment()
            env.filters["tojson"] = lambda value, **_: json.dumps(value, ensure_ascii=False)
            return env.from_string(self.chat_template).render(
                messages=messages, tools=tools, add_generation_prompt=add_generation_prompt,
                enable_thinking=kwargs["enable_thinking"])

        def encode(self, text, **kwargs):
            return [ord(char) for char in text]

    messages = normalize(sample(), TOOLS, set())
    original = copy.deepcopy(messages)
    encoded = encode_trajectory(Qwen3TemplateTokenizer(), messages, TOOLS, 100000)
    supervised = "".join(chr(token) for token in encoded["labels"] if token != -100)
    assert messages == original
    assert "<tool_call>" in supervised and "done" in supervised
    assert "<tool_response>" not in supervised
    assert supervised.count("<think>\n\n</think>") == 0


def test_qwen3_rollout_template_matches_nonthinking_sft_context():
    from jinja2 import Environment
    from coding_agent_rl_lab.trajectory_sft import nonthinking_messages

    root = Path(__file__).resolve().parents[1]
    training = (root / "envs/qwen3-8b-chat-template.jinja").read_text(encoding="utf-8")
    rollout = (root / "envs/qwen3-8b-nonthinking-rollout.jinja").read_text(encoding="utf-8")
    env = Environment()
    env.filters["tojson"] = lambda value, **_: json.dumps(value, ensure_ascii=False)

    class Qwen3:
        chat_template = training

    messages = normalize(sample(), TOOLS, set())
    cases = [messages[:2], messages[:3], messages[:4], messages,
             messages + [{"role": "user", "content": "follow up"}, {"role": "assistant", "content": "reply"}]]
    for conversation in cases:
        for generation_prompt in (False, True):
            expected = env.from_string(training).render(
                messages=nonthinking_messages(Qwen3(), conversation, False), tools=TOOLS,
                add_generation_prompt=generation_prompt, enable_thinking=False)
            actual = env.from_string(rollout).render(
                messages=conversation, tools=TOOLS,
                add_generation_prompt=generation_prompt, enable_thinking=False)
            assert actual == expected
    assert "<think>\n\n</think>" in actual


def test_report_duplicates_and_no_false_readiness(tmp_path):
    source, split, tools = [tmp_path / s for s in ("input.jsonl", "split.json", "tools.json")]
    source.write_text((json.dumps(sample()) + "\n") * 2)
    split.write_text(json.dumps({"instances": [{"repo": "eval/repo", "split": "final-test"}]}))
    tools.write_text(json.dumps(TOOLS))
    report = prepare(source, split, tools, tmp_path / "out")
    assert report["accepted"] == 1
    assert report["rejections"] == {"duplicate_trajectory": 1}
    assert not report["training_ready"]
    with pytest.raises(TrajectoryError, match="output_already_exists"):
        prepare(source, split, tools, tmp_path / "out")
