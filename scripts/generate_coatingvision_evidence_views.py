"""Create traceable visual evidence views from one real CoatingVision sample.

The outputs are deterministic transformations of dataset image_565 and its
published segmentation label: an optical contrast view and a ground-truth
annotation overlay. They are for presentation inspection only, not model input
or a production performance claim.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
IMAGE = ROOT / "data/external/coatingvision/segmentation/images/image_565.jpg"
MASK = ROOT / "data/external/coatingvision/segmentation/masks/image_565.png"
OUT = ROOT / "reports/coatingvision_visual_evidence"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(IMAGE), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(MASK), cv2.IMREAD_UNCHANGED)
    if image is None or mask is None:
        raise FileNotFoundError("Missing CoatingVision image_565 or its segmentation mask")

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

    cv2.imwrite(str(OUT / "image_565_optical_raw.png"), image)
    cv2.imwrite(str(OUT / "image_565_contrast_clahe.png"), enhanced)
    cv2.imwrite(str(OUT / "image_565_ground_truth_overlay.png"), gt_overlay)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
