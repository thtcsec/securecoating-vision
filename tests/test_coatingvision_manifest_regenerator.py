"""Round-trip: regenerating the ZIP-safe CoatingVision manifest must stay valid."""

from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from evaluation.dataset_manifest import validate_detection_dataset_manifest  # noqa: E402
from scripts.build_coatingvision_test_bundle_manifest import (  # noqa: E402
    CANONICAL_MANIFEST_SHA256,
    LF_DRIFT_MANIFEST_SHA256,
    STUB_TRAIN,
    STUB_VAL,
    build_manifest,
    write_manifest,
)


class TestCoatingVisionManifestRegenerator(unittest.TestCase):
    def test_regenerate_matches_shipped_schema_and_validates(self):
        bundle = ROOT / "data" / "coatingvision_real_test"
        before = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))

        regenerated = build_manifest(bundle)
        self.assertEqual(regenerated["schema_version"], "coatingvision-real-test/v1")
        self.assertEqual(regenerated["splits"]["train"], [STUB_TRAIN])
        self.assertEqual(regenerated["splits"]["val"], [STUB_VAL])
        self.assertEqual(len(regenerated["test_image_names"]), 88)
        self.assertEqual(len(regenerated["label_sha256"]), 88)
        self.assertIn("tree_hashes", regenerated)
        self.assertIn("images_and_labels", regenerated["tree_hashes"])
        self.assertEqual(
            regenerated["used_for_defense_rgb_metrics_by_split"],
            {"train": False, "val": False, "test": True},
        )
        self.assertEqual(
            set(regenerated["split_roles"]),
            {"train", "val", "test"},
        )
        # Semantic content bit-stable against the checked-in authoritative manifest.
        self.assertEqual(regenerated["tree_hashes"], before["tree_hashes"])
        self.assertEqual(regenerated["label_sha256"], before["label_sha256"])
        self.assertEqual(regenerated["test_image_names"], before["test_image_names"])
        self.assertEqual(regenerated["splits"], before["splits"])

        write_manifest(bundle)
        raw = (bundle / "manifest.json").read_bytes()
        self.assertIn(b"\r\n", raw)
        # After stripping CRLF pairs, no bare LF may remain.
        self.assertNotIn(b"\n", raw.replace(b"\r\n", b""))
        after_sha = hashlib.sha256(raw).hexdigest()
        self.assertEqual(after_sha, CANONICAL_MANIFEST_SHA256)

        # LF-only serialization is a known OS drift; regenerator must not emit it.
        lf_only = json.dumps(regenerated, indent=2).replace("\r\n", "\n") + "\n"
        lf_sha = hashlib.sha256(lf_only.encode("utf-8")).hexdigest()
        self.assertEqual(lf_sha, LF_DRIFT_MANIFEST_SHA256)
        self.assertNotEqual(lf_sha, CANONICAL_MANIFEST_SHA256)

        result = validate_detection_dataset_manifest(
            bundle / "manifest.json", bundle
        )
        self.assertEqual(result["splits"]["test"], 88)
        self.assertEqual(result["splits"]["train"], 1)
        self.assertEqual(result["splits"]["val"], 1)
        self.assertEqual(len(result["dataset_tree_sha256"]), 64)

    def test_gitattributes_forces_crlf_checkout(self):
        attrs = (ROOT / ".gitattributes").read_text(encoding="utf-8")
        self.assertIn("data/coatingvision_real_test/manifest.json", attrs)
        self.assertIn("eol=crlf", attrs)

    def test_script_does_not_emit_legacy_stub_names(self):
        source = (
            ROOT / "scripts" / "build_coatingvision_test_bundle_manifest.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn('"stub_train.jpg"', source)
        self.assertNotIn('"stub_val.jpg"', source)
        self.assertNotIn("'stub_train.jpg'", source)
        self.assertNotIn("'stub_val.jpg'", source)
        self.assertIn("_zip_stub_train.jpg", source)
        self.assertIn("test_image_names", source)
        self.assertIn("label_sha256", source)
        self.assertIn("tree_hashes", source)
        self.assertIn("CANONICAL_MANIFEST_SHA256", source)


if __name__ == "__main__":
    unittest.main()
