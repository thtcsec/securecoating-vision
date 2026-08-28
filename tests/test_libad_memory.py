"""PatchCore FPS vs DA-Core density-aware FPS attribution and behavior."""

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from libad.memory import select_coreset
from libad.protocol import LIBAD_CITATION
from libad.scorer import MemoryAnomalyScorer


class TestLibadMemory(unittest.TestCase):
    def test_da_core_is_attributed_to_the_paper_authors(self):
        self.assertIn("Sui", " ".join(LIBAD_CITATION["authors"]))
        self.assertIn("does not claim DA-Core", LIBAD_CITATION["da_core_attribution"])

    def test_coreset_is_smaller_than_the_candidate_set(self):
        rng = np.random.default_rng(7)
        features = rng.normal(size=(40, 8)).astype(np.float32)
        features /= np.linalg.norm(features, axis=1, keepdims=True)
        memory = select_coreset(features, ratio=0.25, method="density_fps", seed=7)
        self.assertEqual(len(memory), 10)

    def test_anomaly_image_scores_higher_than_a_matched_normal(self):
        rng = np.random.default_rng(11)
        normals = [np.full((32, 32), 140, dtype=np.uint8) for _ in range(6)]
        for index, image in enumerate(normals):
            image[:] = np.clip(image.astype(np.int16) + rng.integers(-4, 5, image.shape), 0, 255)
        scorer = MemoryAnomalyScorer(
            method="fps",
            coreset_ratio=0.5,
            patch_size=8,
            stride=8,
            image_size=(32, 32),
            seed=11,
        )
        scorer.fit(vis_images=normals)
        clean = normals[0]
        defect = clean.copy()
        defect[8:24, 8:24] = 0
        clean_score = scorer.score_sample("clean", vis_a=clean).vis_score
        defect_score = scorer.score_sample("defect", vis_a=defect).vis_score
        self.assertIsNotNone(clean_score)
        self.assertGreater(defect_score, clean_score)


if __name__ == "__main__":
    unittest.main()
