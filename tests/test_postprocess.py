import os
import sys
import numpy as np
import unittest

# Add src to python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from inference.postprocess import extract_defects_from_mask, grade_coating

class TestPostProcess(unittest.TestCase):
    def test_extract_defects(self):
        # Generate mock segmentation mask (100x100)
        seg_mask = np.zeros((100, 100), dtype=np.uint8)
        # Add a thicker scratch (class 1) from (10, 10) to (10, 30) (width=3, height=21)
        seg_mask[10:31, 9:12] = 1
        
        # Add a blister (class 3) block at (50, 50) to (55, 55) (size 6x6)
        seg_mask[50:56, 50:56] = 3
        
        # Generate mock height map
        height_map = np.zeros((100, 100), dtype=np.float32)
        height_map[50:56, 50:56] = 150.0  # Height peak
        
        defects = extract_defects_from_mask(seg_mask, height_map, pixel_to_mm_ratio=0.5)
        
        # Verify defects found
        self.assertGreaterEqual(len(defects), 2)
        
        # Find scratch and blister
        scratch = next(d for d in defects if d["class_name"] == "scratch")
        blister = next(d for d in defects if d["class_name"] == "blister")
        
        # Validate scratch metrics
        # Length should be around max(w, h) in mm. Pixel height is 21 pixels. 21 * 0.5 = 10.5 mm
        self.assertAlmostEqual(scratch["length_mm"], 10.5, delta=0.1)
        self.assertEqual(scratch["peak_height_um"], 0.0)
        
        # Validate blister metrics
        self.assertEqual(blister["peak_height_um"], 150.0)
        # Area: 6x6 block coordinates span from index 50 to 55 which is a distance of 5.
        # Enclosed contour area is 5 * 5 = 25 pixels. 25 * (0.5^2) = 6.25 mm2
        self.assertAlmostEqual(blister["area_mm2"], 6.25, delta=0.1)

    def test_grade_coating(self):
        defects = [
            {
                "defect_id": "scratch_0",
                "class_id": 1,
                "class_name": "scratch",
                "length_mm": 6.5,
                "width_mm": 0.5,
                "area_mm2": 3.25,
                "peak_height_um": 0.0
            },
            {
                "defect_id": "blister_0",
                "class_id": 3,
                "class_name": "blister",
                "length_mm": 3.0,
                "width_mm": 3.0,
                "area_mm2": 9.0,
                "peak_height_um": 35.0
            }
        ]
        
        grading_config = {
            "scratch": {"max_allowable_length_mm": 5.0},
            "blister": {"max_allowable_height_um": 50.0}
        }
        
        # Scratch length 6.5 > 5.0 -> FAIL
        res1 = grade_coating(defects, grading_config)
        self.assertFalse(res1["passed"])
        self.assertEqual(len(res1["reject_reasons"]), 1)
        self.assertIn("scratch", res1["reject_reasons"][0].lower())
        
        # Update scratch to valid length -> PASS
        defects[0]["length_mm"] = 4.0
        res2 = grade_coating(defects, grading_config)
        self.assertTrue(res2["passed"])
        self.assertEqual(len(res2["reject_reasons"]), 0)

    def test_grade_violation_cap_does_not_change_total(self):
        defects = [
            {"class_name": "scratch", "length_mm": 6.0, "area_mm2": 0.0,
             "peak_height_um": 0.0}
            for _ in range(12)
        ]
        result = grade_coating(defects, {})
        self.assertFalse(result["passed"])
        self.assertEqual(result["total_violations"], 12)
        self.assertEqual(len(result["reject_reasons"]), 11)
        self.assertIn("2 additional", result["reject_reasons"][-1])

    def test_unknown_defect_class_fails_closed(self):
        result = grade_coating([
            {"class_name": "unexpected", "length_mm": 0.0, "area_mm2": 0.0,
             "peak_height_um": 0.0}
        ], {})
        self.assertFalse(result["passed"])
        self.assertEqual(result["total_violations"], 1)

if __name__ == "__main__":
    unittest.main()
