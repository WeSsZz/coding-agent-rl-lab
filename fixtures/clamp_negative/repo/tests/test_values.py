from __future__ import annotations

import unittest

from values import clamp_non_negative


class ClampNonNegativeTests(unittest.TestCase):
    def test_negative_value_is_clamped(self) -> None:
        self.assertEqual(clamp_non_negative(-7), 0)

    def test_zero_and_positive_values_are_preserved(self) -> None:
        self.assertEqual(clamp_non_negative(0), 0)
        self.assertEqual(clamp_non_negative(9), 9)


if __name__ == "__main__":
    unittest.main()
