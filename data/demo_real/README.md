# Attributed real optical demo subset

This directory contains the **held-out CoatingVision detection test split**:
unmodified JPEG optical surface frames. They are real inspection frames, not
synthetic renders and not ordinary overview photographs of the manufacturing
line.

The dashboard Dataset Library is this test-split subset. It is **not** the full
imported archive:

| Tree (local, mostly gitignored) | Role |
|---|---|
| `data/external/coatingvision/detection/images` | Public Figshare detection archive (~2227 JPEGs) |
| `data/coatingvision_real_detect` | Deterministic train/val/test copy with labels (581 pairs, gitignored) |
| `data/demo_real` | Test-split optical frames checked into git for the dashboard/API demo |

- Source: CoatingVision, Figshare DOI `10.6084/m9.figshare.29260121.v1`
- License: CC BY 4.0
- Split: `test` from `scripts/prepare_coatingvision_detection_dataset.py` (seed 71)
- Intended use: bounded dashboard/inference demonstration only
- Annotation status: source labels are deliberately not embedded here; model
  overlays are predictions, not ground truth. Do not publish these frames as
  an independent factory metric set.

`manifest.json` records the source identity and SHA-256 of every checked-in
sample. The API returns this provenance as metadata without embedding image
payloads in its operations snapshot.
