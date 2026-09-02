"""Local numpy patch descriptor stays 20-D and L2-normalized."""

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from libad.features import FEATURE_DIM, extract_image_features, extract_patch_features


class TestLibadFeatures(unittest.TestCase):
    def test_empty_stack_keeps_descriptor_width(self):
        features, index = extract_image_features([])
        self.assertEqual(features.shape, (0, FEATURE_DIM))
        self.assertEqual(index.shape, (0,))

    def test_patch_matrix_is_normalized_20d(self):
        image = np.full((64, 64), 140, dtype=np.uint8)
        image[20:28, 10:54] = 0
        features = extract_patch_features(image, patch_size=16, stride=8, image_size=(64, 64))
        self.assertEqual(features.shape[1], FEATURE_DIM)
        self.assertGreater(features.shape[0], 1)
        norms = np.linalg.norm(features, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_scratch_changes_orientation_bins(self):
        clean = np.full((64, 64), 140, dtype=np.uint8)
        scratch = clean.copy()
        scratch[30, :] = 0
        clean_feat = extract_patch_features(clean).mean(axis=0)
        scratch_feat = extract_patch_features(scratch).mean(axis=0)
        self.assertGreater(float(np.linalg.norm(scratch_feat - clean_feat)), 1e-4)


if __name__ == "__main__":
    unittest.main()
