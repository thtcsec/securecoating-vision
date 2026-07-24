"""
Prepare Synthetic Coating Defect Dataset (Entry Point Alias)

Generates synthetic coating defect dataset with contour-derived YOLOv8-seg 
polygons, negative baseline samples, and data.yaml configuration.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_synthetic_coating_defects import generate_dataset

if __name__ == "__main__":
    generate_dataset(num_train=500, num_val=100, overwrite=True)
