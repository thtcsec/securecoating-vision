"""
SecureCoating-Vision: Core Model Evaluation Pipeline
====================================================
Computes real detection and segmentation metrics by running model inference
on the evaluation dataset and comparing predictions against ground-truth labels.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from run_evaluation import run_evaluation


def evaluate():
    """Execute reproducible evaluation on the benchmark evaluation dataset."""
    return run_evaluation()


if __name__ == "__main__":
    evaluate()
