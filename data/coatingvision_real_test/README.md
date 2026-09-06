# CoatingVision ZIP-safe real test split

This directory is the **held-out image-disjoint test split** used to reproduce
`reports/coatingvision_real_test_metrics.json` (authoritative RGB defense metrics,
mAP50 ≈ 0.633).

## What this is

- **88** real optical test images + matching YOLO detection labels
- Fixed split from `scripts/prepare_coatingvision_detection_dataset.py` (**seed 71**)
- Source: CoatingVision, Figshare DOI `10.6084/m9.figshare.29260121.v1`, CC BY 4.0
- `used_for_defense_rgb_metrics=true` applies to the **test** split only
- `factory_roll_disjoint=false` — public metadata does not support factory roll IDs

## What this is not

- **Not** roll-disjoint factory validation
- **Not** the full `data/coatingvision_real_detect` train/val set (that tree stays gitignored and out of the submission ZIP)
- **train/** and **val/** contain one Ultralytics stub image+label each so YOLO dataset YAML loading works. Stubs are not training data and are not metrics evidence.

## Reproduce metrics

```bash
python scripts/evaluate_coatingvision_real.py --weights outputs/best.pt --imgsz 512 --device cpu
```

Default dataset root is `data/coatingvision_real_test` when present.
