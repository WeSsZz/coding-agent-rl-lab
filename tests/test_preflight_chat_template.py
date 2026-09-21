from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.preflight_chat_template import (  # noqa: E402
    PreflightError,
    main,
    render,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = REPOSITORY_ROOT / "envs/qwen3-8b-chat-template.jinja"
TOOLS = REPOSITORY_ROOT / "envs/openhands-v0.54-tools.json"


class PinnedArtefactTests(unittest.TestCase):
    def test_the_pinned_artefacts_exist(self) -> None:
        self.assertTrue(TEMPLATE.is_file(), f"missing {TEMPLATE}")
        self.assertTrue(TOOLS.is_file(), f"missing {TOOLS}")

    def test_the_manifest_hash_matches_the_pinned_template(self) -> None:
        import hashlib

        manifest = json.loads(
            (REPOSITORY_ROOT / "envs/qwen3-8b-chat-template.manifest.json").read_text(encoding="utf-8")
        )

        actual = hashlib.sha256(TEMPLATE.read_bytes()).hexdigest()

        self.assertEqual(manifest["template"]["sha256"], actual)

    def test_the_manifest_source_hash_matches_the_tokenizer_config(self) -> None:
        import hashlib

        manifest = json.loads(
            (REPOSITORY_ROOT / "envs/qwen3-8b-chat-template.manifest.json").read_text(encoding="utf-8")
        )
        config = REPOSITORY_ROOT / "envs/qwen3-8b-tokenizer_config.json"

        self.assertEqual(
            manifest["source"]["local_copy_sha256"],
            hashlib.sha256(config.read_bytes()).hexdigest(),
        )

    def test_the_tool_schema_is_the_adopted_one(self) -> None:
        tools = json.loads(TOOLS.read_text(encoding="utf-8"))
        names = [entry["function"]["name"] for entry in tools]

        self.assertEqual(
            names,
            ["execute_bash", "think", "finish", "task_tracker", "str_replace_editor"],
        )


class PreflightTests(unittest.TestCase):
    def test_the_full_preflight_passes_on_the_pinned_artefacts(self) -> None:
        self.assertEqual(main([]), 0)

    def test_a_template_without_a_tools_branch_is_rejected(self) -> None:
        source = TEMPLATE.read_text(encoding="utf-8")
        stripped = source.replace("{%- if tools %}", "{%- if false %}", 1)
        tools = json.loads(TOOLS.read_text(encoding="utf-8"))

        with tempfile.TemporaryDirectory() as directory:
            broken = Path(directory) / "t.jinja"
            broken.write_text(stripped, encoding="utf-8")
            code = main(["--template", str(broken), "--tools", str(TOOLS)])

        self.assertEqual(code, 1)
        _ = tools

    def test_an_unknown_variable_is_an_error_rather_than_empty_text(self) -> None:
        # StrictUndefined matters: a typo in a template should fail loudly, not silently drop
        # the system prompt and train on a conversation the model never saw.
        with self.assertRaises(PreflightError):
            render("{{ nonexistent_variable }}", [{"role": "user", "content": "x"}], None, enable_thinking=False)

    def test_arguments_survive_as_an_object_not_a_string(self) -> None:
        source = TEMPLATE.read_text(encoding="utf-8")
        tools = json.loads(TOOLS.read_text(encoding="utf-8"))
        rendered = render(
            source,
            [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "u"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": "execute_bash", "arguments": {"command": "ls"}},
                        }
                    ],
                },
            ],
            tools,
            enable_thinking=False,
        )

        self.assertIn('"arguments": {"command": "ls"}', rendered)
        self.assertNotIn('"arguments": "', rendered)

    def test_a_string_argument_payload_is_passed_through_unchanged(self) -> None:
        # OpenHands trajectories store arguments as a JSON string; the template must not
        # double-encode it, or every replayed SFT turn gains a layer of escaping.
        source = TEMPLATE.read_text(encoding="utf-8")
        tools = json.loads(TOOLS.read_text(encoding="utf-8"))
        rendered = render(
            source,
            [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "u"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": "execute_bash", "arguments": '{"command": "ls"}'},
                        }
                    ],
                },
            ],
            tools,
            enable_thinking=False,
        )

        self.assertIn('"arguments": {"command": "ls"}', rendered)


if __name__ == "__main__":
    unittest.main()
