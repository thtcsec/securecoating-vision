"""Create traceable visual evidence views from one real CoatingVision sample.

The outputs are deterministic transformations of dataset image_565 and its
published segmentation label: an optical contrast view and a ground-truth
annotation overlay. They are for presentation inspection only, not model input
or a production performance claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/coatingvision_visual_evidence"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-id", default="565", help="CoatingVision image id, e.g. 1548")
    parser.add_argument("--model-overlay", help="Optional path to a real model overlay for a candidate zoom.")
    parser.add_argument("--model-json", help="Optional path to the matching raw model-output JSON.")
    args = parser.parse_args()
    image_id = str(args.image_id)
    image_path = ROOT / f"data/external/coatingvision/segmentation/images/image_{image_id}.jpg"
    mask_path = ROOT / f"data/external/coatingvision/segmentation/masks/image_{image_id}.png"
    OUT.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if image is None or mask is None:
        raise FileNotFoundError(f"Missing CoatingVision image_{image_id} or its segmentation mask")

    # Contrast-limited adaptive histogram equalization makes the captured
    # surface texture easier to inspect without creating or adding defects.
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # CoatingVision masks are published annotations. Preserve their geometry
    # and render a 50% red overlay strictly for visual verification.
    if mask.ndim == 3:
        foreground = np.any(mask > 0, axis=2)
    else:
        foreground = mask > 0
    gt_overlay = image.copy()
    red = np.zeros_like(image)
    red[:, :, 2] = 255
    gt_overlay[foreground] = cv2.addWeighted(
        image[foreground], 0.45, red[foreground], 0.55, 0
    )

    cv2.imwrite(str(OUT / f"image_{image_id}_optical_raw.png"), image)
    cv2.imwrite(str(OUT / f"image_{image_id}_contrast_clahe.png"), enhanced)
    cv2.imwrite(str(OUT / f"image_{image_id}_ground_truth_overlay.png"), gt_overlay)

    if args.model_overlay and args.model_json:
        overlay = cv2.imread(str(ROOT / args.model_overlay), cv2.IMREAD_COLOR)
        payload = json.loads((ROOT / args.model_json).read_text(encoding="utf-8"))
        detections = payload.get("detections", [])
        if overlay is None or not detections:
            raise ValueError("Expected a readable model overlay with at least one raw detection")
        x1, y1, x2, y2 = [int(v) for v in detections[0]["box"]]
        h, w = overlay.shape[:2]
        left, right = max(0, x1 - 150), min(w, x2 + 180)
        top, bottom = max(0, y1 - 110), min(h, y2 + 70)
        zoom = overlay[top:bottom, left:right]
        cv2.imwrite(str(OUT / f"image_{image_id}_yolo_candidate_zoom.png"), zoom)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
