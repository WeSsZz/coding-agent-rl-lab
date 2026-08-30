from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from coding_agent_rl_lab.contracts import ActionKind, AgentAction
from coding_agent_rl_lab.evaluation import evaluate, load_builtin_tasks
from coding_agent_rl_lab.model import ModelActionError, ModelConfigurationError, OpenAICompatibleConfig
from coding_agent_rl_lab.policies import ModelCodingPolicy


class FakeActionModel:
    model_name = "fake-model"

    def complete_action(self, *, system_prompt: str, user_prompt: str) -> AgentAction:
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        return AgentAction(ActionKind.LIST_FILES)


class FailingActionModel:
    model_name = "failing-model"

    def complete_action(self, *, system_prompt: str, user_prompt: str) -> AgentAction:
        del system_prompt, user_prompt
        raise ModelActionError("invalid JSON")


class ModelPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parents[1]

    def test_policy_exposes_issue_but_not_hidden_verifier_data(self) -> None:
        model = FakeActionModel()
        policy = ModelCodingPolicy(model)
        task = load_builtin_tasks(self.root)[0]
        action = policy.next_action(task, ())
        self.assertEqual(action.kind, ActionKind.LIST_FILES)
        self.assertIn(task.issue, model.user_prompt)
        self.assertNotIn("reference_actions", model.user_prompt)
        self.assertEqual(policy.manifest.metadata["prompt_id"], "repository-repair-v1")
        self.assertEqual(len(policy.manifest.metadata["prompt_sha256"]), 64)

    def test_model_failure_becomes_a_failed_closed_trial(self) -> None:
        report, trajectories = evaluate(
            self.root,
            policy_name="model",
            repetitions=1,
            action_model=FailingActionModel(),
        )
        self.assertEqual(report["pass_at_1"], 0.0)
        self.assertEqual(report["violation_count"], 2)
        self.assertTrue(all(item.reward.violations == ("policy_error:ModelActionError",) for item in trajectories))

    def test_config_reads_key_from_environment_without_exposing_it_in_repr(self) -> None:
        with patch.dict(os.environ, {"CODING_AGENT_API_KEY": "top-secret"}, clear=False):
            config = OpenAICompatibleConfig.from_env(
                base_url="https://example.com/v1",
                model="model-v1",
            )
        self.assertNotIn("top-secret", repr(config))

    def test_missing_key_is_rejected(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ModelConfigurationError):
                OpenAICompatibleConfig.from_env(base_url="https://example.com/v1", model="model-v1")


if __name__ == "__main__":
    unittest.main()
