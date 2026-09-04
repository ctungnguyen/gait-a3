from __future__ import annotations

import hashlib
from pathlib import Path
import unittest


class PreservedReferenceTests(unittest.TestCase):
    def test_uploaded_gridworld_was_preserved_byte_for_byte(self):
        path = Path(__file__).parents[1] / "part1_reference" / "gridworld.py"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(digest, "353fe42879bfa74490e5492b057a6b8a4c27b9fb21e2d5c12351bf24ea9e2752")


if __name__ == "__main__":
    unittest.main()
