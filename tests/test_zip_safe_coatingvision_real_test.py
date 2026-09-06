"""ZIP-safe CoatingVision test bundle and DINOv2 CSV packing contracts."""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_evaluate_module():
    path = ROOT / "scripts" / "evaluate_coatingvision_real.py"
    spec = importlib.util.spec_from_file_location("evaluate_coatingvision_real", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # Avoid importing ultralytics at collection time by stubbing if needed.
    if "ultralytics" not in sys.modules:
        fake = mock.MagicMock()
        sys.modules["ultralytics"] = fake
    spec.loader.exec_module(module)
    return module


class TestZipSafeCoatingVisionAndDinov2Packing(unittest.TestCase):
    def test_coatingvision_real_test_has_88_labels_and_honest_readme(self):
        bundle = ROOT / "data" / "coatingvision_real_test"
        self.assertTrue(bundle.is_dir())
        labels = sorted((bundle / "labels" / "test").glob("*.txt"))
        images = sorted((bundle / "images" / "test").glob("*.jpg"))
        self.assertEqual(len(labels), 88)
        self.assertEqual(len(images), 88)
        self.assertEqual(
            {path.stem for path in labels},
            {path.stem for path in images},
        )
        readme = (bundle / "README.md").read_text(encoding="utf-8")
        self.assertIn("held-out image-disjoint test split", readme.lower())
        self.assertIn("reports/coatingvision_real_test_metrics.json", readme)
        self.assertIn("not", readme.lower())
        self.assertIn("roll-disjoint", readme.lower())
        self.assertIn("stub", readme.lower())
        self.assertIn("not training data", readme.lower())
        manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(manifest["used_for_defense_rgb_metrics"])
        self.assertFalse(manifest["factory_roll_disjoint"])
        self.assertEqual(manifest["seed"], 71)
        self.assertEqual(manifest["evidence_class"], "public_real_optical_image_split")
        self.assertEqual(len(manifest["test_image_names"]), 88)
        self.assertEqual(len(manifest["label_sha256"]), 88)
        self.assertFalse(manifest["used_for_defense_rgb_metrics_by_split"]["train"])
        self.assertFalse(manifest["used_for_defense_rgb_metrics_by_split"]["val"])
        self.assertTrue(manifest["used_for_defense_rgb_metrics_by_split"]["test"])
        self.assertEqual(len(manifest["splits"]["train"]), 1)
        self.assertEqual(len(manifest["splits"]["val"]), 1)
        yaml_text = (ROOT / "configs" / "coatingvision_real_test.yaml").read_text(
            encoding="utf-8"
        )
        self.assertIn("path: data/coatingvision_real_test", yaml_text)

    def test_evaluate_script_prefers_zip_safe_root(self):
        module = _load_evaluate_module()
        root, data_yaml = module.resolve_dataset(None, None)
        self.assertEqual(root, (ROOT / "data" / "coatingvision_real_test").resolve())
        self.assertEqual(
            data_yaml, (ROOT / "configs" / "coatingvision_real_test.yaml").resolve()
        )
        source = (ROOT / "scripts" / "evaluate_coatingvision_real.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("final_artifact_provenance", source)
        self.assertIn("zip_safe_dataset", source)
        self.assertIn("--dataset-root", source)
        self.assertIn("--data", source)

        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "absent"
            # When zip-safe is missing, fall back to full detect root.
            original = module.ZIP_SAFE_ROOT
            try:
                module.ZIP_SAFE_ROOT = missing
                fallback_root, fallback_yaml = module.resolve_dataset(None, None)
                self.assertEqual(
                    fallback_root, module.FULL_DETECT_ROOT.resolve()
                )
                self.assertEqual(
                    fallback_yaml, module.FULL_DETECT_YAML.resolve()
                )
            finally:
                module.ZIP_SAFE_ROOT = original

    def test_dinov2_interim_source_csv_is_relative_and_packed(self):
        interim = ROOT / "reports" / "libad" / "official_dinov2_dacore_interim.json"
        payload = json.loads(interim.read_text(encoding="utf-8"))
        source_csv = payload["source_csv"]
        self.assertEqual(
            source_csv, "reports/libad/official_dinov2_dacore_interim_results.csv"
        )
        self.assertFalse(Path(source_csv).is_absolute())
        self.assertNotIn(":\\", source_csv)
        self.assertTrue((ROOT / source_csv).is_file())
        builder = (ROOT / "scripts" / "build_submission.py").read_text(encoding="utf-8")
        self.assertIn(
            "reports/libad/official_dinov2_dacore_interim_results.csv", builder
        )

    def test_builder_packs_zip_safe_test_and_excludes_full_detect_train(self):
        from scripts.build_submission import TREE_DIRS, is_blacklisted

        tree_srcs = {src for src, _dst in TREE_DIRS}
        self.assertIn("data/coatingvision_real_test", tree_srcs)
        self.assertNotIn("data/coatingvision_real_detect", tree_srcs)
        self.assertFalse(
            is_blacklisted("data/coatingvision_real_test/labels/test/image_25.txt")
        )
        self.assertFalse(
            is_blacklisted("data/coatingvision_real_test/images/test/image_25.jpg")
        )
        self.assertTrue(
            is_blacklisted(
                "data/coatingvision_real_detect/labels/train/image_4.txt"
            )
        )
        self.assertTrue(
            is_blacklisted(
                "data/coatingvision_real_detect/images/train/image_4.jpg"
            )
        )
        self.assertTrue(is_blacklisted("data/coating_defects/labels/train/x.txt"))
        source = (ROOT / "scripts" / "build_submission.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        whitelist = None
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "WHITELIST"
                for target in node.targets
            ):
                whitelist = node.value
                break
        self.assertIsInstance(whitelist, ast.List)
        # Conditional whitelist entries use IfExp; also accept string presence in source.
        self.assertIn("configs/coatingvision_real_test.yaml", source)
        self.assertIn("data/coatingvision_real_test", source)


if __name__ == "__main__":
    unittest.main()
