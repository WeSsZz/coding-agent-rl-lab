from __future__ import annotations

import hashlib
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
TOOLS = REPOSITORY_ROOT / "envs/openhands-v0.54-tools.json"

#: The mainline model first, then the recorded alternative. Both must keep working: the pinned
#: alternative is what the project falls back to if the mainline choice is revisited.
TEMPLATES = {
    "qwen25-coder-7b": (
        REPOSITORY_ROOT / "envs/qwen25-coder-7b-chat-template.jinja",
        REPOSITORY_ROOT / "envs/qwen25-coder-7b-tokenizer_config.json",
    ),
    "qwen3-8b": (
        REPOSITORY_ROOT / "envs/qwen3-8b-chat-template.jinja",
        REPOSITORY_ROOT / "envs/qwen3-8b-tokenizer_config.json",
    ),
}


class PinnedArtefactTests(unittest.TestCase):
    def test_every_pinned_artefact_exists(self) -> None:
        for name, (template, config) in TEMPLATES.items():
            with self.subTest(template=name):
                self.assertTrue(template.is_file(), f"missing {template}")
                self.assertTrue(config.is_file(), f"missing {config}")
        self.assertTrue(TOOLS.is_file(), f"missing {TOOLS}")

    def test_each_manifest_hash_matches_its_template(self) -> None:
        for name, (template, _) in TEMPLATES.items():
            with self.subTest(template=name):
                manifest = json.loads(template.with_suffix(".manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(
                    manifest["template"]["sha256"], hashlib.sha256(template.read_bytes()).hexdigest()
                )

    def test_each_manifest_source_hash_matches_its_tokenizer_config(self) -> None:
        for name, (template, config) in TEMPLATES.items():
            with self.subTest(template=name):
                manifest = json.loads(template.with_suffix(".manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(
                    manifest["source"]["local_copy_sha256"],
                    hashlib.sha256(config.read_bytes()).hexdigest(),
                )

    def test_each_manifest_records_a_concrete_revision(self) -> None:
        for name, (template, _) in TEMPLATES.items():
            with self.subTest(template=name):
                manifest = json.loads(template.with_suffix(".manifest.json").read_text(encoding="utf-8"))
                revision = manifest["revision"]
                self.assertEqual(len(revision), 40, f"{name}: revision {revision!r} is not a commit sha")

    def test_the_tool_schema_is_the_adopted_one(self) -> None:
        tools = json.loads(TOOLS.read_text(encoding="utf-8"))
        names = [entry["function"]["name"] for entry in tools]

        self.assertEqual(
            names,
            ["execute_bash", "think", "finish", "task_tracker", "str_replace_editor"],
        )


class PreflightTests(unittest.TestCase):
    def test_every_pinned_template_passes_its_own_preflight(self) -> None:
        for name, (template, _) in TEMPLATES.items():
            with self.subTest(template=name):
                self.assertEqual(main(["--template", str(template)]), 0)

    def test_a_template_without_a_tools_branch_is_rejected(self) -> None:
        source = TEMPLATES["qwen25-coder-7b"][0].read_text(encoding="utf-8")
        stripped = source.replace("{%- if tools %}", "{%- if false %}", 1)

        with tempfile.TemporaryDirectory() as directory:
            broken = Path(directory) / "t.jinja"
            broken.write_text(stripped, encoding="utf-8")
            code = main(["--template", str(broken), "--tools", str(TOOLS)])

        self.assertEqual(code, 1)

    def test_an_unknown_variable_is_an_error_rather_than_empty_text(self) -> None:
        # StrictUndefined matters: a typo in a template should fail loudly, not silently drop the
        # system prompt and train on a conversation the model never saw.
        with self.assertRaises(PreflightError):
            render("{{ nonexistent_variable }}", [{"role": "user", "content": "x"}], None, enable_thinking=False)


class TemplateCapabilityTests(unittest.TestCase):
    """The two families differ in ways that change what the data pipeline must do."""

    def test_qwen3_has_a_thinking_switch_and_qwen25_does_not(self) -> None:
        qwen3 = TEMPLATES["qwen3-8b"][0].read_text(encoding="utf-8")
        qwen25 = TEMPLATES["qwen25-coder-7b"][0].read_text(encoding="utf-8")

        self.assertIn("enable_thinking", qwen3)
        self.assertNotIn("enable_thinking", qwen25)

    def test_only_qwen3_guards_string_argument_payloads(self) -> None:
        qwen3 = TEMPLATES["qwen3-8b"][0].read_text(encoding="utf-8")
        qwen25 = TEMPLATES["qwen25-coder-7b"][0].read_text(encoding="utf-8")

        self.assertIn("tool_call.arguments is string", qwen3)
        self.assertNotIn("tool_call.arguments is string", qwen25)

    def test_the_mainline_template_double_encodes_string_arguments(self) -> None:
        # This is the constraint the preflight warns about. Pinning it as a test makes sure the
        # SFT pipeline requirement cannot be forgotten silently.
        source = TEMPLATES["qwen25-coder-7b"][0].read_text(encoding="utf-8")
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

        self.assertIn('"arguments": "{\\"command\\"', rendered)

    def test_the_qwen3_template_passes_string_arguments_through(self) -> None:
        source = TEMPLATES["qwen3-8b"][0].read_text(encoding="utf-8")
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

    def test_dict_arguments_work_on_both_templates(self) -> None:
        tools = json.loads(TOOLS.read_text(encoding="utf-8"))
        for name, (template, _) in TEMPLATES.items():
            with self.subTest(template=name):
                rendered = render(
                    template.read_text(encoding="utf-8"),
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


if __name__ == "__main__":
    unittest.main()
