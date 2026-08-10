from __future__ import annotations

import unittest

from calculator import inclusive_range


class InclusiveRangeTests(unittest.TestCase):
    def test_includes_upper_endpoint(self) -> None:
        self.assertEqual(inclusive_range(1, 3), [1, 2, 3])

    def test_single_value_range(self) -> None:
        self.assertEqual(inclusive_range(4, 4), [4])


if __name__ == "__main__":
    unittest.main()
