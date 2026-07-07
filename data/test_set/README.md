# Test Dataset (Inference Samples)

50 sample coating surface images for demonstrating inference capabilities.

## Format
- `images/` — Input images (640x640 JPEG, metallic coating surfaces with defects)

## Classes (detected by model)
| ID | Name | Description |
|----|------|-------------|
| 0 | scratch | Linear surface cracks/scratches |
| 1 | void | Sub-surface pores and inclusions |
| 2 | blister | Raised bumps in coating |
| 3 | delamination | Coating peeling/separation |

## Usage
```bash
# Run inference on test images
python scripts/run_evaluation.py

# Or use the dashboard: upload any image from images/ folder
streamlit run dashboard/app.py
```

## Note
These images are provided for inference demonstration only.
Ground-truth annotations are held separately for validation and are not included in this package.
Full validation metrics are reproduced via `scripts/run_evaluation.py` using the training validation split.
