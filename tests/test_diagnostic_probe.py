"""Pin the A/B/C runner's contract: ordering, seeds, and an honest auxiliary message.

The load-bearing properties are that the three conditions differ *only* by one extra user message,
that the message is never dressed up as a tool result, and that a trajectory records which
condition and which auxiliary text produced it - otherwise an assisted pass could be read as
autonomous capability.
"""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from coding_agent_rl_lab.contracts import (
    AgentAction,
    ActionKind,
    PolicyManifest,
    RewardVector,
    Trajectory,
    TrajectoryStep,
    VerifierBreakdown,
)
from coding_agent_rl_lab.diagnostic_probe import (
    AuxiliaryContextPolicy,
    DiagnosticProbeError,
    load_oracle_spec,
    plan_trials,
    summarize,
    trial_record,
)
from coding_agent_rl_lab.model_policy import (
    OpenAICompatiblePolicy,
    OpenAICompatiblePolicyConfig,
)


def _task() -> object:
    from coding_agent_rl_lab.contracts import CodingTask, DatasetSplit

    return CodingTask(
        task_id="getmoto__moto-7365",
        issue="the tests fail",
        fixture_path=None,
        base_commit="0" * 40,
        test_command=("python", "-m", "pytest", "-q", "tests/test_x.py"),
        split=DatasetSplit.DEVELOPMENT,
        provenance="test",
        max_steps=24,
    )


def _config() -> OpenAICompatiblePolicyConfig:
    return OpenAICompatiblePolicyConfig(model="test-model", api_base="http://127.0.0.1:1/v1")


def _step(sequence: int, prompt_tokens: int, completion_tokens: int) -> TrajectoryStep:
    return TrajectoryStep(
        sequence=sequence,
        action=AgentAction(ActionKind.READ_FILE, {"path": "pkg/mod.py"}),
        observation="1: line",
        terminated=False,
        policy_metadata={
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
        },
    )


class PlanTrialsTests(unittest.TestCase):
    def test_interleaves_conditions_inside_one_task_and_repetition(self) -> None:
        plan = plan_trials(
            ["a", "b"],
            repetitions=2,
            conditions=("A", "B", "C"),
            base_seed=140001,
        )
        self.assertEqual(len(plan), 12)
        self.assertEqual(
            [(entry["repetition"], entry["task_id"], entry["condition"]) for entry in plan[:6]],
            [
                (1, "a", "A"),
                (1, "a", "B"),
                (1, "a", "C"),
                (1, "b", "A"),
                (1, "b", "B"),
                (1, "b", "C"),
            ],
        )
        self.assertEqual(plan[6]["repetition"], 2)

    def test_the_three_conditions_of_one_cell_share_a_seed(self) -> None:
        plan = plan_trials(
            ["a", "b"],
            repetitions=2,
            conditions=("A", "B", "C"),
            base_seed=140001,
        )
        seeds = {}
        for entry in plan:
            seeds.setdefault((entry["task_id"], entry["repetition"]), set()).add(entry["seed"])
        self.assertEqual(seeds[("a", 1)], {140001})
        self.assertEqual(seeds[("b", 1)], {160001})
        self.assertEqual(seeds[("a", 2)], {150001})
        self.assertEqual(seeds[("b", 2)], {170001})

    def test_conditions_do_not_collide_on_a_seed(self) -> None:
        plan = plan_trials(["a"], repetitions=1, conditions=("A", "B", "C"), base_seed=140001)
        self.assertEqual({entry["seed"] for entry in plan}, {140001})


class OracleSpecTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = TemporaryDirectory()
        self.path = Path(self._temporary.name) / "oracle.jsonl"

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def write(self, rows: list[dict]) -> None:
        self.path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )

    def row(self, task_id: str = "t", condition: str = "B", message: str = "hint") -> dict:
        return {
            "task_id": task_id,
            "condition": condition,
            "auxiliary_message": message,
            "auxiliary_sha256": hashlib.sha256(message.encode("utf-8")).hexdigest(),
        }

    def test_loads_a_well_formed_spec(self) -> None:
        self.write([self.row(condition="A", message=""), self.row(condition="B")])
        spec = load_oracle_spec(self.path)
        self.assertEqual(spec[("t", "A")].auxiliary_message, "")
        self.assertEqual(spec[("t", "B")].auxiliary_sha256, hashlib.sha256(b"hint").hexdigest())

    def test_rejects_a_row_whose_hash_does_not_match_its_message(self) -> None:
        row = self.row()
        row["auxiliary_sha256"] = "0" * 64
        self.write([row])
        with self.assertRaises(DiagnosticProbeError):
            load_oracle_spec(self.path)

    def test_rejects_a_duplicate_condition(self) -> None:
        self.write([self.row(), self.row()])
        with self.assertRaises(DiagnosticProbeError):
            load_oracle_spec(self.path)


class AuxiliaryContextPolicyTests(unittest.TestCase):
    def test_untouched_when_the_condition_has_no_auxiliary_text(self) -> None:
        stock = OpenAICompatiblePolicy(_config())
        policy = AuxiliaryContextPolicy(
            _config(), condition_id="A", auxiliary_message=""
        )
        task = _task()
        history = [_step(1, 10, 2)]
        self.assertEqual(policy._messages(task, history, "baseline"), stock._messages(task, history, "baseline"))

    def test_the_assist_is_a_labelled_field_inside_the_task_payload(self) -> None:
        stock = OpenAICompatiblePolicy(_config())
        policy = AuxiliaryContextPolicy(
            _config(), condition_id="C", auxiliary_message="the window"
        )
        task = _task()
        history = [_step(1, 10, 2)]
        baseline = stock._messages(task, history, "baseline")
        messages = policy._messages(task, history, "baseline")

        # Same shape as the stock prompt: nothing is appended after the task payload.
        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertEqual([message["role"] for message in baseline], ["system", "user"])
        self.assertEqual(messages[0], baseline[0])

        payload = json.loads(messages[1]["content"])
        stock_payload = json.loads(baseline[1]["content"])
        assist = payload.pop("diagnostic_auxiliary_input")
        self.assertEqual(payload, stock_payload, "only the assist field may differ")
        self.assertEqual(assist["text"], "the window")
        self.assertEqual(assist["sha256"], hashlib.sha256(b"the window").hexdigest())
        self.assertIn("not produced by any tool call", assist["provenance"])
        self.assertIn("C", assist["label"])

    def test_the_assist_is_not_the_last_thing_the_model_reads(self) -> None:
        policy = AuxiliaryContextPolicy(
            _config(), condition_id="B", auxiliary_message="paths"
        )
        payload = json.loads(policy._messages(_task(), [], "baseline")[1]["content"])
        keys = list(payload)
        self.assertLess(keys.index("diagnostic_auxiliary_input"), keys.index("history"))
        self.assertEqual(keys[-1], "history")

    def test_the_auxiliary_text_is_never_a_tool_result_or_a_history_entry(self) -> None:
        policy = AuxiliaryContextPolicy(
            _config(), condition_id="B", auxiliary_message="paths"
        )
        messages = policy._messages(_task(), [], "baseline")
        payload = json.loads(messages[1]["content"])
        self.assertEqual(payload["history"], [])
        self.assertNotIn("paths", json.dumps(payload["history"]))
        self.assertEqual(messages[0]["role"], "system")

    def test_the_manifest_identifies_the_condition_and_the_text(self) -> None:
        policy = AuxiliaryContextPolicy(
            _config(), condition_id="C", auxiliary_message="window text"
        )
        metadata = policy.manifest.metadata
        self.assertEqual(metadata["diagnostic_condition"], "C")
        self.assertEqual(metadata["auxiliary_chars"], len("window text"))
        self.assertEqual(metadata["auxiliary_placement"], "task_payload_field")
        self.assertEqual(
            metadata["auxiliary_sha256"], hashlib.sha256(b"window text").hexdigest()
        )

    def test_two_conditions_do_not_share_a_manifest(self) -> None:
        first = AuxiliaryContextPolicy(_config(), condition_id="A", auxiliary_message="")
        second = AuxiliaryContextPolicy(_config(), condition_id="B", auxiliary_message="paths")
        self.assertNotEqual(first.manifest, second.manifest)


class TrialRecordTests(unittest.TestCase):
    def trajectory(self) -> Trajectory:
        return Trajectory(
            trajectory_id="traj-1",
            task_id="getmoto__moto-7365",
            repetition=1,
            seed=140001,
            policy=PolicyManifest(
                policy_id="openai-compatible:test-model",
                version="1",
                policy_type="openai_compatible_chat_json",
                model="test-model",
                metadata={"prompt_version": "coding-tools-json-v25", "diagnostic_condition": "B"},
            ),
            steps=(
                _step(1, 100, 20),
                _step(2, 200, 30),
            ),
            reward=RewardVector(
                task_success=False,
                tests_passed=False,
                regression_free=False,
                patch_created=True,
                tool_calls=2,
                steps=2,
                violations=("policy_protocol_error",),
                loop_rejections=3,
            ),
            changed_files=("pkg/mod.py",),
            baseline_tests_passed=False,
            final_tests_passed=False,
            initial_observation="baseline",
            verifier=VerifierBreakdown(
                fail_to_pass_total=3,
                fail_to_pass_resolved=1,
                pass_to_pass_total=7,
                pass_to_pass_regressed=0,
                failed_nodes=("tests/test_x.py::test_a",),
            ),
            training_reward=-0.25,
        )

    def test_sums_tokens_and_carries_the_grader_numbers(self) -> None:
        record = trial_record(
            self.trajectory(),
            {"condition": "B", "repetition": 1},
            auxiliary_sha256="a" * 64,
            auxiliary_chars=42,
            elapsed_seconds=12.5,
        )
        self.assertEqual(record["prompt_tokens"], 300)
        self.assertEqual(record["completion_tokens"], 50)
        self.assertEqual(record["fail_to_pass_resolved"], 1)
        self.assertEqual(record["fail_to_pass_total"], 3)
        self.assertEqual(record["failed_nodes"], ["tests/test_x.py::test_a"])
        self.assertEqual(record["steps"], 2)
        self.assertEqual(record["loop_rejections"], 3)
        self.assertFalse(record["strict_success"])
        self.assertEqual(record["condition"], "B")
        self.assertEqual(record["auxiliary_chars"], 42)
        self.assertIsNone(record["infrastructure_failure"])
        self.assertEqual(record["policy_condition"], "B")

    def test_records_an_infrastructure_failure_instead_of_hiding_it(self) -> None:
        record = trial_record(
            self.trajectory(),
            {"condition": "A", "repetition": 2},
            auxiliary_sha256=hashlib.sha256(b"").hexdigest(),
            auxiliary_chars=0,
            elapsed_seconds=1.0,
            infrastructure_failure="trial_timeout: exceeded",
        )
        self.assertEqual(record["infrastructure_failure"], "trial_timeout: exceeded")
        self.assertEqual(record["condition"], "A")


class SummarizeTests(unittest.TestCase):
    def row(self, condition: str, **overrides: object) -> dict:
        base = {
            "task_id": "t",
            "condition": condition,
            "repetition": 1,
            "strict_success": False,
            "tests_passed": False,
            "changed_files": [],
            "patch_created": False,
            "violations": [],
            "steps": 4,
            "training_reward": 0.0,
            "prompt_tokens": 100,
            "infrastructure_failure": None,
        }
        base.update(overrides)
        return base

    def test_reports_per_condition_totals_rather_than_one_headline(self) -> None:
        trials = [
            self.row("A"),
            self.row("A", strict_success=True, changed_files=["x"], patch_created=True, steps=6),
            self.row("B", changed_files=["y"], patch_created=True),
            self.row("C", violations=["policy_transport_error"], infrastructure_failure="x"),
        ]
        summary = summarize(trials)
        self.assertEqual(summary["by_condition"]["A"]["strict_success_count"], 1)
        self.assertEqual(summary["by_condition"]["A"]["trials_with_a_changed_file"], 1)
        self.assertEqual(summary["by_condition"]["A"]["mean_steps"], 5.0)
        self.assertEqual(summary["by_condition"]["B"]["strict_success_count"], 0)
        self.assertEqual(summary["by_condition"]["C"]["transport_error_count"], 1)
        self.assertEqual(summary["by_condition"]["C"]["infrastructure_failure_count"], 1)
        self.assertEqual(set(summary["by_condition"]), {"A", "B", "C"})
        self.assertEqual(summary["by_condition"]["B"]["meaning"].startswith("oracle-file"), True)
        self.assertEqual(
            sorted(summary["per_task"]["t"]), ["A", "B", "C"]
        )

    def test_keeps_each_task_separate(self) -> None:
        trials = [
            self.row("A", task_id="one", strict_success=True),
            self.row("A", task_id="two"),
        ]
        summary = summarize(trials)
        self.assertEqual(sorted(summary["per_task"]), ["one", "two"])
        self.assertEqual(len(summary["per_task"]["one"]["A"]), 1)
        self.assertEqual(summary["by_condition"]["A"]["trial_count"], 2)


if __name__ == "__main__":
    unittest.main()
