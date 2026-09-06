# Synthetic Evaluation Fixture

This directory is a small, synthetic four-class fixture for reproducing the
local evaluation pipeline inside the submission ZIP.

## What is included

- `images/` + `labels/` — fixture test split (ground-truth labels for evaluator
  reproducibility only; not training labels)
- `reference/images/train|val/` — unique stub images used as the leakage-audit
  reference tree (ZIP-safe; no `coating_defects` dependency)
- This README — honesty boundary for judges

## What this is NOT

- Not the CoatingVision real-image RGB detector report
- Not an independent / roll-disjoint factory benchmark
- Not semantic or instance-segmentation evidence

The test imagery **intentionally reuses development images**. Reports must keep:

- `evidence_class: SYNTHETIC_EVALUATOR_FIXTURE`
- `used_for_defense_rgb_metrics: false`
- `development_image_reuse: true`
- `cross_taxonomy_run: true` (fixture is 4-class; runtime ONNX detector is 2-class)

Detection-only mask values are **bbox-derived proxy masks**.

## Reproduce (ZIP-safe)

From the repository root (or unzipped submission archive):

```text
python scripts/build_synthetic_evaluation_manifest.py --output reports/synthetic_evaluation_manifest.json
python scripts/run_evaluation.py --dataset-dir data/evaluation --model-path outputs/model.onnx --reference-dataset-dir data/evaluation/reference --dataset-manifest reports/synthetic_evaluation_manifest.json --iou 0.50 --conf 0.25 --seed 42
```

Authoritative RGB defense metrics remain in
`reports/coatingvision_real_test_metrics.json`.
