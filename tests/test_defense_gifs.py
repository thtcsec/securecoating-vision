"""Defense GIFs are looping presentation artifacts from checked-in stills."""

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import generate_defense_gifs


class TestDefenseGifs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        generate_defense_gifs.build_all()
        cls.rgb = ROOT / "reports/defense_gifs/rgb_hold_replay.gif"
        cls.libad = ROOT / "reports/defense_gifs/libad_gate.gif"

    def test_gifs_are_looping_gif89a(self):
        for path in (self.rgb, self.libad):
            payload = path.read_bytes()
            self.assertTrue(payload.startswith(b"GIF89a"), path.as_posix())
            self.assertGreater(path.stat().st_size, 20_000)
            self.assertLess(path.stat().st_size, 4_000_000)

    def test_captions_stay_honest(self):
        source = (ROOT / "scripts/generate_defense_gifs.py").read_text(encoding="utf-8")
        self.assertIn("comparable_to_paper=false", source)
        self.assertIn("not a factory PASS", source)
        self.assertIn("protocol fixture", source)
        self.assertNotIn("99.4%", source)
        self.assertNotIn("0.02 ppm", source)


if __name__ == "__main__":
    unittest.main()
