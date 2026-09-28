"""Checks the evaluation prompt contract and repeated-command cutoff."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from io import BytesIO
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "evaluate_openhands_candidate", ROOT / "scripts/evaluate_openhands_candidate.py"
)
assert SPEC and SPEC.loader
EVAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVAL)


class EvaluationRuntimeTest(unittest.TestCase):
    def test_qwen3_request_disables_thinking_without_changing_other_models(self) -> None:
        class Response(BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *_):
                self.close()

        reply = b'{"choices":[{"message":{"role":"assistant","content":"ok"}}]}'
        with patch.object(EVAL.urllib.request, "urlopen", return_value=Response(reply)) as opened:
            EVAL.api_chat("http://localhost:8000", "base", [{"role": "user", "content": "x"}],
                          non_thinking=True, max_output_tokens=4096, tool_choice="auto")
            payload = json.loads(opened.call_args.args[0].data)
            self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
            self.assertEqual(payload["tool_choice"], "auto")
            self.assertEqual(payload["max_tokens"], 4096)

    def test_http_error_keeps_replayable_request(self) -> None:
        failure = EVAL.urllib.error.HTTPError(
            "http://localhost:8000/v1/chat/completions", 400, "bad request", {},
            BytesIO(b'{"error":"invalid JSON"}'),
        )
        with tempfile.TemporaryDirectory() as directory:
            request_path = Path(directory) / "failed-request.json"
            with patch.object(EVAL.urllib.request, "urlopen", side_effect=failure):
                with self.assertRaisesRegex(RuntimeError, "inference HTTP 400"):
                    EVAL.api_chat("http://localhost:8000", "base",
                                  [{"role": "user", "content": "issue"}],
                                  failed_request_path=request_path)
            payload = json.loads(request_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["messages"][0]["content"], "issue")
            self.assertEqual(payload["model"], "base")

    def test_prompt_uses_preinstalled_environment_without_install_instruction(self) -> None:
        template = json.loads(
            (ROOT / "envs/openhands-eval-template-v2.json").read_text(encoding="utf-8-sig")
        )
        prompt = template["prefix"] + "Example issue" + template["suffix"]
        prompt = prompt.replace(template["source_workspace"], "/workspace/example__repo__1.0")
        prompt = prompt.replace(template["source_repo_dir"], "example__repo__1.0")
        prompt = prompt.replace(template["source_base_commit"], "abc123")
        self.assertIn("do not install packages", prompt)
        self.assertIn("/opt/miniconda3/envs/testbed/bin/pytest", prompt)
        self.assertIn("/workspace/example__repo__1.0", prompt)
        self.assertIn("abc123", prompt)
        self.assertNotIn("Install the environment", prompt)
        self.assertNotIn("install and run the tests", prompt)
        self.assertNotIn("test.py", prompt.replace("do not assume test.py exists", ""))

    def test_identical_commands_are_counted_even_when_they_succeed(self) -> None:
        previous, count = None, 0
        for expected in (1, 2, 3):
            previous, count = EVAL.repeated_command(
                previous, count, "execute_bash", {"command": "pip install x"},
                "exit_code=1\nnetwork unavailable",
            )
            self.assertEqual(count, expected)
            previous, count = EVAL.repeated_command(
                previous, count, "think", {"thought": "inspect"}, "Thought recorded."
            )
            self.assertEqual(count, expected)
        previous, count = EVAL.repeated_command(
            previous, count, "execute_bash", {"command": "rg x"}, "exit_code=0\nfound"
        )
        self.assertEqual((previous, count), ("rg x", 1))
        previous, count = EVAL.repeated_command(
            previous, count, "execute_bash", {"command": "rg x"}, "exit_code=0\nfound"
        )
        self.assertEqual((previous, count), ("rg x", 2))
        previous, count = EVAL.repeated_command(
            previous, count, "str_replace_editor", {"command": "str_replace"},
            "exit_code=0\nDone",
        )
        self.assertEqual((previous, count), (None, 0))
        previous, count = EVAL.repeated_command(
            "sleep 3", 2, "execute_bash", {"command": "sleep 3"},
            "exit_code=-1\ncommand still running",
        )
        self.assertEqual((previous, count), ("sleep 3", 2))

    def test_task_tracker_keeps_plan_and_stops_repeated_views(self) -> None:
        task_list = []
        plan = [{"id": "1", "title": "Inspect source", "status": "in_progress"}]
        observation, done = EVAL.tool_result(
            "unused", "task_tracker", {"command": "plan", "task_list": plan},
            "/testbed", None, {}, task_list,
        )
        self.assertFalse(done)
        self.assertEqual(json.loads(observation)["task_list"], plan)
        for expected in (1, 2, 3):
            observation, _ = EVAL.tool_result(
                "unused", "task_tracker", {"command": "view"},
                "/testbed", None, {}, task_list,
            )
            self.assertEqual(json.loads(observation)["task_list"], plan)
            previous, count = EVAL.repeated_command(
                "task_tracker:view" if expected > 1 else None,
                expected - 1, "task_tracker", {"command": "view"}, observation,
            )
            self.assertEqual((previous, count), ("task_tracker:view", expected))

    def test_v3_prompt_moves_from_planning_to_repository_inspection(self) -> None:
        template = json.loads(
            (ROOT / "envs/openhands-eval-template-v3.json").read_text(encoding="utf-8")
        )
        self.assertIn("inspect relevant code with execute_bash", template["system"])
        self.assertNotIn("Use this tool REGULARLY", template["system"])
        self.assertIn("Begin by inspecting the relevant files", template["suffix"])

    def test_v4_prompt_requires_source_read_after_search(self) -> None:
        template = json.loads(
            (ROOT / "envs/openhands-eval-template-v4.json").read_text(encoding="utf-8")
        )
        self.assertIn("open the most relevant source file", template["suffix"])
        self.assertIn("never alternate between two failed searches", template["suffix"])

    def test_long_tool_output_keeps_beginning_and_end(self) -> None:
        output = "START" + "a" * 100 + "END"
        clipped = EVAL.clip_output(output, 30)
        self.assertTrue(clipped.startswith("START"))
        self.assertTrue(clipped.endswith("END"))
        self.assertIn("characters clipped", clipped)

    def test_search_feedback_moves_from_paths_to_source_and_limits_empty_loops(self) -> None:
        paths = "exit_code=0\n/testbed/moto/dynamodb/models.py\n"
        observation, previous, count, stop = EVAL.search_feedback(paths, None, 0)
        self.assertIn("Read a relevant source file", observation)
        self.assertFalse(stop)
        observation, previous, count, stop = EVAL.search_feedback(paths, previous, count)
        self.assertIn("no new information", observation)
        self.assertFalse(stop)
        previous, count = None, 0
        for expected in (1, 2, 3, 4):
            observation, previous, count, stop = EVAL.search_feedback(
                "exit_code=1\n", previous, count,
            )
            self.assertEqual(count, expected)
        self.assertTrue(stop)

    def test_history_compaction_preserves_complete_tool_turns(self) -> None:
        messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
        for index in range(9):
            messages.extend([
                {"role": "assistant", "tool_calls": [{"id": str(index)}]},
                {"role": "tool", "tool_call_id": str(index), "content": "result"},
            ])
        compacted = EVAL.compact_messages(messages, keep_turns=6)
        self.assertEqual(compacted[:2], messages[:2])
        self.assertEqual(len(compacted), 14)
        self.assertEqual(compacted[2]["tool_calls"][0]["id"], "3")
        self.assertEqual(compacted[-1]["tool_call_id"], "8")

    def test_view_feedback_limits_same_file_without_blocking_other_files(self) -> None:
        previous, count = None, 0
        for expected in range(1, 5):
            observation, previous, count, stop = EVAL.view_feedback(
                "str_replace_editor", {"command": "view", "path": "/testbed/a.py"},
                "exit_code=0\ncode", previous, count,
            )
            self.assertEqual(count, expected)
        self.assertTrue(stop)
        observation, previous, count, stop = EVAL.view_feedback(
            "str_replace_editor", {"command": "view", "path": "/testbed/b.py"},
            "exit_code=0\nother", previous, count,
        )
        self.assertEqual(count, 1)
        self.assertFalse(stop)


if __name__ == "__main__":
    unittest.main()
