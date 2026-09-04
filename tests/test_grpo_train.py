from __future__ import annotations

import json
import os
import stat
import tempfile
import unittest
from pathlib import Path

from coding_agent_rl_lab.grpo_train import (
    GRPOTrainingError,
    _metric_number,
    build_parser,
    configure_prompt_rows_tool_format,
    configure_tool_response_parsing,
    is_valid_bare_json_tool_probe,
    load_prompt_rows,
    probe_bare_json_tool_parsing,
    read_worker_token,
    run_bare_json_tool_parsing_probe,
)


class GRPOTrainTests(unittest.TestCase):
    def test_parser_accepts_sft_adapter_initialization(self) -> None:
        args = build_parser().parse_args(
            [
                "--model-path",
                "/models/base",
                "--adapter-path",
                "/models/adapter",
                "--prompt-rows",
                "prompts.jsonl",
                "--worker-token-file",
                "token",
            ]
        )

        self.assertEqual(args.adapter_path, "/models/adapter")

    def test_load_prompt_rows_validates_and_limits(self) -> None:
        rows = [
            {
                "task_id": f"task-{index}",
                "prompt": [{"role": "user", "content": "fix it"}],
            }
            for index in range(2)
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            self.assertEqual(load_prompt_rows(path, limit=1), rows[:1])

    def test_load_prompt_rows_rejects_invalid_message(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text(
                json.dumps({"task_id": "task", "prompt": [{"role": "tool", "content": "x"}]}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(GRPOTrainingError, "invalid message"):
                load_prompt_rows(path)

    def test_worker_token_is_never_returned_when_too_short(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "token"
            path.write_text("short", encoding="utf-8")
            if os.name != "nt":
                path.chmod(stat.S_IRUSR | stat.S_IWUSR)
            with self.assertRaisesRegex(GRPOTrainingError, "at least 32"):
                read_worker_token(path)

    def test_training_metrics_are_normalized_without_inventing_values(self) -> None:
        self.assertEqual(_metric_number("0.25"), 0.25)
        self.assertEqual(_metric_number(2), 2.0)
        self.assertIsNone(_metric_number(None))
        self.assertIsNone(_metric_number("unavailable"))

    def test_bare_json_tool_parser_is_explicit_and_does_not_change_chat_template(self) -> None:
        class Tokenizer:
            chat_template = "original-chat-template"
            response_template = {"original": True}

        tokenizer = Tokenizer()
        configured = configure_tool_response_parsing(tokenizer, bare_json_tool_calls=True)

        self.assertIs(configured, tokenizer)
        self.assertEqual(tokenizer.chat_template, "original-chat-template")
        tool_field = tokenizer.response_template["fields"]["tool_calls"]
        self.assertEqual(tool_field["content"], "json")
        self.assertEqual(
            tool_field["transform"],
            [{"type": "function", "function": "{content}"}],
        )

    def test_default_tool_parser_is_preserved(self) -> None:
        class Tokenizer:
            response_template = {"original": True}

        tokenizer = Tokenizer()
        configure_tool_response_parsing(tokenizer, bare_json_tool_calls=False)
        self.assertEqual(tokenizer.response_template, {"original": True})

    def test_bare_json_prompt_does_not_contradict_parser(self) -> None:
        rows = [
            {
                "task_id": "task",
                "prompt": [
                    {
                        "role": "system",
                        "content": (
                            "Use tools.\n\nEvery assistant turn must contain exactly one tool call.\n"
                            "<tool_call>{}</tool_call>\nNever answer with a plain JSON action object."
                        ),
                    },
                    {"role": "user", "content": "fix it"},
                ],
            }
        ]

        configured = configure_prompt_rows_tool_format(rows, bare_json_tool_calls=True)

        system_prompt = configured[0]["prompt"][0]["content"]
        self.assertIn('{"name":"search_text"', system_prompt)
        self.assertIn("Never use Markdown fences", system_prompt)
        self.assertNotIn("must contain exactly one tool call", system_prompt)
        self.assertNotIn("Never answer with a plain JSON", system_prompt)
        self.assertNotEqual(configured, rows)

    def test_bare_json_parser_probe_requires_exact_tool_shape(self) -> None:
        class Tokenizer:
            def encode(self, text, *, add_special_tokens):
                self.last_text = text
                self.add_special_tokens = add_special_tokens
                return [1, 2, 3]

        tokenizer = Tokenizer()

        def parse_response(_tokenizer, ids, *, prefix):
            self.assertEqual(ids, [1, 2, 3])
            self.assertEqual(prefix, [1, 2, 3])
            return {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "list_files", "arguments": {}},
                    }
                ],
            }

        self.assertTrue(probe_bare_json_tool_parsing(tokenizer, parse_response))
        self.assertFalse(tokenizer.add_special_tokens)

        parsed = run_bare_json_tool_parsing_probe(tokenizer, parse_response)
        self.assertTrue(is_valid_bare_json_tool_probe(parsed))

    def test_bare_json_parser_probe_rejects_plain_content(self) -> None:
        class Tokenizer:
            def encode(self, _text, *, add_special_tokens):
                return [1]

        self.assertFalse(
            probe_bare_json_tool_parsing(
                Tokenizer(),
                lambda *_args, **_kwargs: {"role": "assistant", "content": "{}"},
            )
        )

    def test_bare_json_parser_probe_preserves_bounded_error_type(self) -> None:
        class Tokenizer:
            def encode(self, _text, *, add_special_tokens):
                return [1]

        result = run_bare_json_tool_parsing_probe(
            Tokenizer(),
            lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad parse")),
        )
        self.assertEqual(result["probe_error"], "ValueError")
        self.assertEqual(result["probe_message"], "bad parse")


if __name__ == "__main__":
    unittest.main()
