# Test Dataset

50 sample images with ground-truth segmentation annotations for evaluation.

## Format
- `images/` — Input images (640×640 JPEG)
- `labels/` — YOLO segmentation polygon annotations (class_id x1 y1 x2 y2 ... xn yn)

## Classes
| ID | Name | Description |
|----|------|-------------|
| 0 | scratch | Linear surface cracks/scratches |
| 1 | void | Sub-surface pores and inclusions |
| 2 | blister | Raised bumps in coating |
| 3 | delamination | Coating peeling/separation |

## Usage
```bash
# Run validation on test set
yolo val task=segment model=outputs/model.onnx data=configs/dataset.yaml split=val
```
