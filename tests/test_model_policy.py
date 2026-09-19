from __future__ import annotations

import json
import unittest
from typing import Any, Mapping

from coding_agent_rl_lab.contracts import AgentAction, ActionKind, CodingTask, DatasetSplit, TrajectoryStep
from coding_agent_rl_lab.model_policy import (
    ModelTransportError,
    OpenAICompatiblePolicy,
    OpenAICompatiblePolicyConfig,
)


class FakeTransport:
    def __init__(self, responses: list[dict[str, Any] | Exception]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, Mapping[str, Any], Mapping[str, str], float]] = []

    def post_json(
        self,
        url: str,
        payload: Mapping[str, Any],
        *,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        self.calls.append((url, payload, headers, timeout_seconds))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _task() -> CodingTask:
    return CodingTask(
        task_id="getmoto__moto-7365",
        issue="Fix decimal arithmetic.",
        fixture_path=None,
        base_commit="a" * 40,
        test_command=("pytest",),
        split=DatasetSplit.DEVELOPMENT,
        provenance="test",
        metadata={
            "repo": "getmoto/moto",
            "version": "5.0",
            "test_patch": "PRIVATE-TEST-PATCH",
        },
    )


def _response(content: str) -> dict[str, Any]:
    return {
        "id": "request-1",
        "choices": [
            {
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
    }


class OpenAICompatiblePolicyTests(unittest.TestCase):
    def test_valid_json_action_records_exact_io_and_usage_without_secret_metadata(self) -> None:
        transport = FakeTransport([_response('{"kind":"list_files","arguments":{}}')])
        policy = OpenAICompatiblePolicy(
            OpenAICompatiblePolicyConfig(model="example/coder"),
            api_key="secret-token",
            transport=transport,
        )

        decision = policy.next_action(
            _task(),
            (),
            seed=123,
            initial_observation="Tests failed: test_decimal expected Decimal('11.7')",
        )

        self.assertEqual(decision.action.kind, ActionKind.LIST_FILES)
        self.assertIsNone(decision.violation)
        self.assertEqual(decision.metadata["usage"]["total_tokens"], 120)
        self.assertEqual(decision.metadata["seed"], 123)
        self.assertEqual(decision.output_text, '{"kind":"list_files","arguments":{}}')
        recorded_prompt = json.dumps(decision.input_messages)
        self.assertNotIn("PRIVATE-TEST-PATCH", recorded_prompt)
        self.assertNotIn("secret-token", json.dumps(policy.manifest.metadata))
        self.assertIn("test_decimal", recorded_prompt)
        self.assertEqual(policy.manifest.metadata["prompt_version"], "coding-tools-json-v21")
        self.assertIn(
            "reserve at least a third of the remaining steps",
            decision.input_messages[0]["content"],
        )

        url, payload, headers, timeout = transport.calls[0]
        self.assertEqual(url, "http://127.0.0.1:8000/v1/chat/completions")
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(payload["seed"], 123)
        self.assertEqual(headers["Authorization"], "Bearer secret-token")
        self.assertEqual(timeout, 180.0)

    def test_invalid_model_output_retries_then_becomes_a_recorded_violation(self) -> None:
        transport = FakeTransport([_response("not-json"), _response('{"kind":"read_file","arguments":{}}')])
        policy = OpenAICompatiblePolicy(
            OpenAICompatiblePolicyConfig(model="example/coder", max_attempts=2),
            transport=transport,
        )

        decision = policy.next_action(_task(), (), seed=5)

        self.assertEqual(decision.action.kind, ActionKind.FINISH)
        self.assertEqual(decision.violation, "policy_protocol_error")
        self.assertEqual(len(decision.metadata["errors"]), 2)
        self.assertEqual(len(transport.calls), 2)
        second_messages = transport.calls[1][1]["messages"]
        self.assertIn("response was invalid", second_messages[-1]["content"])

    def test_first_action_can_read_exact_path_from_initial_verifier_output(self) -> None:
        transport = FakeTransport(
            [_response('{"kind":"read_file","arguments":{"path":"tests/test_decimal.py"}}')]
        )
        policy = OpenAICompatiblePolicy(
            OpenAICompatiblePolicyConfig(model="example/coder", max_attempts=2),
            transport=transport,
        )

        decision = policy.next_action(
            _task(),
            (),
            seed=11,
            initial_observation="Tests failed: tests/test_decimal.py::test_add",
        )

        self.assertEqual(decision.action.kind, ActionKind.READ_FILE)
        self.assertIsNone(decision.violation)
        self.assertEqual(len(transport.calls), 1)

    def test_read_file_accepts_a_bounded_line_range(self) -> None:
        transport = FakeTransport(
            [
                _response(
                    '{"kind":"read_file","arguments":'
                    '{"path":"moto/ec2/models/vpcs.py","start_line":120,"end_line":180}}'
                )
            ]
        )
        policy = OpenAICompatiblePolicy(
            OpenAICompatiblePolicyConfig(model="example/coder"),
            transport=transport,
        )

        decision = policy.next_action(_task(), (), seed=16)

        self.assertEqual(decision.action.kind, ActionKind.READ_FILE)
        self.assertEqual(decision.action.arguments["start_line"], 120)
        self.assertEqual(decision.action.arguments["end_line"], 180)
        self.assertIsNone(decision.violation)

    def test_replace_lines_accepts_a_small_line_range(self) -> None:
        transport = FakeTransport(
            [
                _response(
                    '{"kind":"replace_lines","arguments":'
                    '{"path":"moto/ec2/models/vpcs.py","start_line":120,'
                    '"end_line":124,"new":"replacement"}}'
                )
            ]
        )
        policy = OpenAICompatiblePolicy(
            OpenAICompatiblePolicyConfig(model="example/coder"),
            transport=transport,
        )

        decision = policy.next_action(_task(), (), seed=19)

        self.assertEqual(decision.action.kind, ActionKind.REPLACE_LINES)
        self.assertEqual(decision.action.arguments["start_line"], 120)
        self.assertEqual(decision.action.arguments["end_line"], 124)
        self.assertIsNone(decision.violation)

    def test_replace_lines_rejects_more_than_eighty_lines(self) -> None:
        transport = FakeTransport(
            [
                _response(
                    '{"kind":"replace_lines","arguments":'
                    '{"path":"moto/ec2/models/vpcs.py","start_line":1,'
                    '"end_line":81,"new":"replacement"}}'
                )
            ]
        )
        policy = OpenAICompatiblePolicy(
            OpenAICompatiblePolicyConfig(model="example/coder", max_attempts=1),
            transport=transport,
        )

        decision = policy.next_action(_task(), (), seed=20)

        self.assertEqual(decision.action.kind, ActionKind.FINISH)
        self.assertEqual(decision.violation, "policy_protocol_error")
        self.assertIn("cannot replace more than 80 lines", decision.metadata["errors"][0])

    def test_read_file_rejects_an_oversized_line_range(self) -> None:
        transport = FakeTransport(
            [
                _response(
                    '{"kind":"read_file","arguments":'
                    '{"path":"moto/ec2/models/vpcs.py","start_line":1,"end_line":401}}'
                )
            ]
        )
        policy = OpenAICompatiblePolicy(
            OpenAICompatiblePolicyConfig(model="example/coder", max_attempts=1),
            transport=transport,
        )

        decision = policy.next_action(_task(), (), seed=17)

        self.assertEqual(decision.action.kind, ActionKind.FINISH)
        self.assertEqual(decision.violation, "policy_protocol_error")
        self.assertIn("cannot exceed 400 lines", decision.metadata["errors"][0])

    def test_read_file_rejects_too_little_context(self) -> None:
        transport = FakeTransport(
            [
                _response(
                    '{"kind":"read_file","arguments":'
                    '{"path":"moto/ec2/models/vpcs.py","start_line":85,"end_line":85}}'
                )
            ]
        )
        policy = OpenAICompatiblePolicy(
            OpenAICompatiblePolicyConfig(model="example/coder", max_attempts=1),
            transport=transport,
        )

        decision = policy.next_action(_task(), (), seed=18)

        self.assertEqual(decision.action.kind, ActionKind.FINISH)
        self.assertEqual(decision.violation, "policy_protocol_error")
        self.assertIn("must include at least 20 lines", decision.metadata["errors"][0])

    def test_environment_receives_identical_action_and_owns_loop_recovery(self) -> None:
        failed_action = AgentAction(ActionKind.READ_FILE, {"path": "missing.py"})
        history = (
            TrajectoryStep(
                sequence=1,
                action=failed_action,
                observation="Tool error: file does not exist: missing.py",
                terminated=False,
            ),
        )
        transport = FakeTransport(
            [_response('{"kind":"read_file","arguments":{"path":"missing.py"}}')]
        )
        policy = OpenAICompatiblePolicy(
            OpenAICompatiblePolicyConfig(model="example/coder", max_attempts=2),
            transport=transport,
        )

        decision = policy.next_action(_task(), history, seed=12)

        self.assertEqual(decision.action, failed_action)
        self.assertEqual(len(transport.calls), 1)
        self.assertIsNone(decision.violation)

    def test_environment_receives_repeated_search_and_owns_loop_recovery(self) -> None:
        previous_action = AgentAction(ActionKind.SEARCH_TEXT, {"query": "Decimal"})
        history = (
            TrajectoryStep(
                sequence=1,
                action=previous_action,
                observation="moto/dynamodb/models.py:10:Decimal",
                terminated=False,
            ),
        )
        transport = FakeTransport(
            [_response('{"kind":"search_text","arguments":{"query":"Decimal"}}')]
        )
        policy = OpenAICompatiblePolicy(
            OpenAICompatiblePolicyConfig(model="example/coder", max_attempts=2),
            transport=transport,
        )

        decision = policy.next_action(_task(), history, seed=13)

        self.assertEqual(decision.action, previous_action)
        self.assertEqual(len(transport.calls), 1)
        self.assertIsNone(decision.violation)

    def test_environment_receives_reread_and_owns_loop_recovery(self) -> None:
        previous_action = AgentAction(ActionKind.READ_FILE, {"path": "tests/test_decimal.py"})
        history = (
            TrajectoryStep(
                sequence=1,
                action=previous_action,
                observation="def test_decimal(): ...",
                terminated=False,
            ),
        )
        transport = FakeTransport(
            [_response('{"kind":"read_file","arguments":{"path":"tests/test_decimal.py"}}')]
        )
        policy = OpenAICompatiblePolicy(
            OpenAICompatiblePolicyConfig(model="example/coder", max_attempts=2),
            transport=transport,
        )

        decision = policy.next_action(_task(), history, seed=14)

        self.assertEqual(decision.action, previous_action)
        self.assertEqual(len(transport.calls), 1)
        self.assertIsNone(decision.violation)

    def test_transport_failure_is_distinct_from_invalid_model_json(self) -> None:
        transport = FakeTransport([ModelTransportError("offline")])
        policy = OpenAICompatiblePolicy(
            OpenAICompatiblePolicyConfig(model="example/coder", max_attempts=1),
            transport=transport,
        )

        decision = policy.next_action(_task(), (), seed=7)

        self.assertEqual(decision.action.kind, ActionKind.FINISH)
        self.assertEqual(decision.violation, "policy_transport_error")
        self.assertEqual(decision.metadata["errors"], ["offline"])

    def test_oversized_prompt_is_refused_before_the_request_is_sent(self) -> None:
        transport = FakeTransport([_response('{"kind":"finish","arguments":{}}')])
        policy = OpenAICompatiblePolicy(
            OpenAICompatiblePolicyConfig(
                model="example/coder",
                max_attempts=2,
                max_tokens=1024,
                max_observation_chars=30_000,
                context_window_tokens=8192,
            ),
            transport=transport,
        )

        decision = policy.next_action(
            _task(),
            (),
            seed=21,
            initial_observation="x" * 30_000,
        )

        self.assertEqual(decision.action.kind, ActionKind.FINISH)
        self.assertEqual(decision.violation, "policy_transport_error")
        self.assertEqual(decision.metadata["attempts"], 0)
        self.assertIn("--max-model-len", decision.metadata["errors"][0])
        self.assertEqual(transport.calls, [])

    def test_context_window_preflight_passes_a_prompt_that_fits(self) -> None:
        transport = FakeTransport([_response('{"kind":"finish","arguments":{}}')])
        policy = OpenAICompatiblePolicy(
            OpenAICompatiblePolicyConfig(
                model="example/coder",
                max_observation_chars=30_000,
                context_window_tokens=32_768,
            ),
            transport=transport,
        )

        decision = policy.next_action(_task(), (), seed=22, initial_observation="x" * 30_000)

        self.assertEqual(decision.action.kind, ActionKind.FINISH)
        self.assertIsNone(decision.violation)
        self.assertEqual(len(transport.calls), 1)
        self.assertGreater(decision.metadata["prompt_token_estimate"], 10_000)

    def test_context_window_must_exceed_max_tokens(self) -> None:
        with self.assertRaisesRegex(ValueError, "context_window_tokens"):
            OpenAICompatiblePolicyConfig(
                model="example/coder",
                max_tokens=1024,
                context_window_tokens=1024,
            )

    def test_history_observations_share_a_total_budget_and_keep_recent_content(self) -> None:
        history = tuple(
            TrajectoryStep(
                sequence=index,
                action=AgentAction(ActionKind.SEARCH_TEXT, {"query": str(index)}),
                observation=character * 10,
                terminated=False,
            )
            for index, character in enumerate(("a", "b", "c"), start=1)
        )
        transport = FakeTransport([_response('{"kind":"finish","arguments":{}}')])
        policy = OpenAICompatiblePolicy(
            OpenAICompatiblePolicyConfig(
                model="example/coder",
                max_observation_chars=10,
                max_history_chars=15,
            ),
            transport=transport,
        )

        decision = policy.next_action(_task(), history, seed=15)
        payload = json.loads(decision.input_messages[1]["content"])
        observations = [step["observation"] for step in payload["history"]]

        self.assertEqual(observations, ["", "b" * 5, "c" * 10])
        self.assertEqual(sum(map(len, observations)), 15)
        self.assertEqual(policy.manifest.metadata["max_history_chars"], 15)

    def test_api_base_rejects_embedded_credentials(self) -> None:
        with self.assertRaisesRegex(ValueError, "credentials"):
            OpenAICompatiblePolicyConfig(
                model="example/coder",
                api_base="https://user:password@example.invalid/v1",
            )

    def test_sampling_penalties_are_optional_and_recorded_in_the_manifest(self) -> None:
        transport = FakeTransport([_response('{"kind":"finish","arguments":{}}')])
        policy = OpenAICompatiblePolicy(
            OpenAICompatiblePolicyConfig(
                model="example/coder",
                repetition_penalty=1.1,
                frequency_penalty=0.2,
            ),
            transport=transport,
        )

        policy.next_action(_task(), (), seed=23)

        payload = transport.calls[0][1]
        self.assertEqual(payload["repetition_penalty"], 1.1)
        self.assertEqual(payload["frequency_penalty"], 0.2)
        self.assertNotIn("presence_penalty", payload)
        self.assertEqual(policy.manifest.metadata["repetition_penalty"], 1.1)
        self.assertIsNone(policy.manifest.metadata["presence_penalty"])

    def test_penalty_defaults_leave_the_payload_untouched(self) -> None:
        transport = FakeTransport([_response('{"kind":"finish","arguments":{}}')])
        policy = OpenAICompatiblePolicy(
            OpenAICompatiblePolicyConfig(model="example/coder"),
            transport=transport,
        )

        policy.next_action(_task(), (), seed=24)

        payload = transport.calls[0][1]
        self.assertNotIn("repetition_penalty", payload)
        self.assertNotIn("frequency_penalty", payload)

    def test_invalid_penalties_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "repetition_penalty"):
            OpenAICompatiblePolicyConfig(model="example/coder", repetition_penalty=0.0)
        with self.assertRaisesRegex(ValueError, "frequency_penalty"):
            OpenAICompatiblePolicyConfig(model="example/coder", frequency_penalty=2.5)


if __name__ == "__main__":
    unittest.main()
