from __future__ import annotations

import unittest

from coding_agent_rl_lab.lora_inference import LoRAInferenceError, _base_parameter_name


class LoRAInferenceTests(unittest.TestCase):
    def test_maps_peft_prefix_to_base_parameter(self) -> None:
        parameters = {"model.layers.0.mlp.up_proj.weight": object()}
        self.assertEqual(
            _base_parameter_name(
                "base_model.model.model.layers.0.mlp.up_proj", parameters
            ),
            "model.layers.0.mlp.up_proj.weight",
        )

    def test_rejects_unknown_base_parameter(self) -> None:
        with self.assertRaisesRegex(LoRAInferenceError, "no base parameter"):
            _base_parameter_name("base_model.model.missing", {})


if __name__ == "__main__":
    unittest.main()
