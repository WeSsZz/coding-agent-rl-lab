from __future__ import annotations

import unittest

from coding_agent_rl_lab.fixture_grpo_prepare import build_parser


class FixtureGRPOPrepareTests(unittest.TestCase):
    def test_defaults_write_separate_curriculum_prompts(self) -> None:
        args = build_parser().parse_args([])

        self.assertEqual(args.output, "work/fixture-grpo-prompts.jsonl")
        self.assertEqual(args.max_steps, 8)


if __name__ == "__main__":
    unittest.main()
