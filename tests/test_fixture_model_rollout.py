from __future__ import annotations

import unittest

from coding_agent_rl_lab.fixture_model_rollout import build_parser


class FixtureModelRolloutCommandTests(unittest.TestCase):
    def test_curriculum_defaults_use_multiple_stochastic_repetitions(self) -> None:
        args = build_parser().parse_args(["--model", "example/coder"])

        self.assertEqual(args.repetitions, 4)
        self.assertEqual(args.temperature, 0.7)
        self.assertEqual(args.max_steps, 8)

    def test_repetitions_must_be_positive(self) -> None:
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["--model", "example/coder", "--repetitions", "0"])


if __name__ == "__main__":
    unittest.main()
