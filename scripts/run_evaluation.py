"""
SecureCoating-Vision: Reproducible Evaluation Script
=====================================================
Computes real detection metrics by running ONNX model inference on the
validation dataset and comparing predictions against ground-truth labels.

Metrics computed:
- Per-class and overall Precision, Recall, F1
- Detection rate (IoU-based matching)
- Inference latency statistics
- Full pipeline throughput

Usage:
    python scripts/run_evaluation.py

Prerequisites:
    - outputs/model.onnx must exist (run training + export first)
    - data/coating_defects/images/val/ and labels/val/ must exist
"""

import os
import sys
import time
import numpy as np
import cv2
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

CLASS_NAMES = {0: "scratch", 1: "void", 2: "blister", 3: "delamination"}


def load_gt_labels(label_path, img_h=640, img_w=640):
    """Parse YOLO segmentation label file into class IDs and bounding boxes."""
    labels = []
    if not os.path.exists(label_path):
        return labels
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(parts[0])
            # Parse polygon points, compute bounding box
            coords = [float(x) for x in parts[1:]]
            xs = [coords[i] * img_w for i in range(0, len(coords), 2)]
            ys = [coords[i] * img_h for i in range(1, len(coords), 2)]
            if not xs or not ys:
                continue
            x1, y1 = min(xs), min(ys)
            x2, y2 = max(xs), max(ys)
            labels.append({
                "class_id": class_id,
                "bbox": [x1, y1, x2 - x1, y2 - y1],  # x, y, w, h
                "area": (x2 - x1) * (y2 - y1)
            })
    return labels


def compute_iou(box1, box2):
    """Compute IoU between two boxes [x, y, w, h]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0] + box1[2], box2[0] + box2[2])
    y2 = min(box1[1] + box1[3], box2[1] + box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - inter

    return inter / (union + 1e-6)


def match_detections(gt_labels, detections, iou_threshold=0.5):
    """Match detections to ground truth using IoU threshold."""
    tp = 0
    fp = 0
    fn = 0
    matched_gt = set()
    class_results = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    # Sort detections by confidence (highest first)
    sorted_dets = sorted(detections, key=lambda d: d.get("confidence", 0), reverse=True)

    for det in sorted_dets:
        det_bbox = det.get("box", None)
        if det_bbox is None:
            continue
        # Convert from [x1, y1, x2, y2] to [x, y, w, h] if needed
        if len(det_bbox) == 4:
            if det_bbox[2] > det_bbox[0] and det_bbox[3] > det_bbox[1]:
                # x1, y1, x2, y2 format
                det_bbox = [det_bbox[0], det_bbox[1],
                           det_bbox[2] - det_bbox[0], det_bbox[3] - det_bbox[1]]

        det_class = det.get("class_id", 0)
        best_iou = 0
        best_gt_idx = -1

        for i, gt in enumerate(gt_labels):
            if i in matched_gt:
                continue
            if gt["class_id"] != det_class:
                continue
            iou = compute_iou(det_bbox, gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i

        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp += 1
            matched_gt.add(best_gt_idx)
            class_results[det_class]["tp"] += 1
        else:
            fp += 1
            class_results[det_class]["fp"] += 1

    # Count false negatives (unmatched GT)
    for i, gt in enumerate(gt_labels):
        if i not in matched_gt:
            fn += 1
            class_results[gt["class_id"]]["fn"] += 1

    return tp, fp, fn, class_results


def run_evaluation():
    """Main evaluation routine."""
    from inference.onnx_engine import InferenceEngine

    print()
    print("=" * 70)
    print("  SecureCoating-Vision: Reproducible Model Evaluation")
    print("  Computing real metrics against ground-truth validation labels")
    print("=" * 70)

    # Check prerequisites
    model_path = "outputs/model.onnx"
    val_img_dir = "data/coating_defects/images/val"
    val_lbl_dir = "data/coating_defects/labels/val"

    if not os.path.exists(model_path):
        print(f"\n  [ERROR] Model not found: {model_path}")
        print("  Run: python src/training/train_yolo.py train --epochs 50")
        sys.exit(1)

    if not os.path.exists(val_img_dir) or not os.path.exists(val_lbl_dir):
        print(f"\n  [ERROR] Validation data not found.")
        print("  Run: python scripts/prepare_real_dataset.py")
        sys.exit(1)

    # Load engine
    engine = InferenceEngine(model_path, imgsz=640, conf_thresh=0.5)
    print(f"\n  Model: {model_path} ({os.path.getsize(model_path)/1024/1024:.1f} MB)")
    print(f"  Provider: {engine.active_provider}")

    # Get validation images
    images = sorted([f for f in os.listdir(val_img_dir) if f.endswith(('.jpg', '.png'))])
    print(f"  Validation images: {len(images)}")
    print(f"\n  Running inference + evaluation...")

    # Evaluate
    total_tp, total_fp, total_fn = 0, 0, 0
    all_class_results = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    latencies = []
    total_gt = 0
    total_det = 0

    for img_name in images:
        img_path = os.path.join(val_img_dir, img_name)
        lbl_name = img_name.rsplit('.', 1)[0] + '.txt'
        lbl_path = os.path.join(val_lbl_dir, lbl_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]

        # Load ground truth
        gt_labels = load_gt_labels(lbl_path, h, w)
        total_gt += len(gt_labels)

        # Run inference
        result = engine.infer(img)
        latencies.append(result["latency_ms"])
        detections = result.get("detections", [])
        total_det += len(detections)

        # Match predictions to ground truth
        tp, fp, fn, class_results = match_detections(gt_labels, detections, iou_threshold=0.5)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        for cls_id, counts in class_results.items():
            all_class_results[cls_id]["tp"] += counts["tp"]
            all_class_results[cls_id]["fp"] += counts["fp"]
            all_class_results[cls_id]["fn"] += counts["fn"]

    # Compute metrics
    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    # Print results
    print("\n" + "=" * 70)
    print("  EVALUATION RESULTS (IoU threshold = 0.50)")
    print("=" * 70)

    print(f"\n  Overall (across {len(images)} images, {total_gt} GT instances):")
    print(f"  {'─' * 50}")
    print(f"  {'Metric':<25} {'Value':<15} {'Target':<15}")
    print(f"  {'─' * 50}")
    print(f"  {'Precision':<25} {precision*100:.1f}%")
    print(f"  {'Recall':<25} {recall*100:.1f}%{'≥98.2%':>15}")
    print(f"  {'F1-Score':<25} {f1*100:.1f}%")
    print(f"  {'Total Detections':<25} {total_det}")
    print(f"  {'True Positives':<25} {total_tp}")
    print(f"  {'False Positives':<25} {total_fp}")
    print(f"  {'False Negatives':<25} {total_fn}")

    print(f"\n  Per-Class Breakdown:")
    print(f"  {'─' * 60}")
    print(f"  {'Class':<15} {'TP':<6} {'FP':<6} {'FN':<6} {'Precision':<12} {'Recall':<12}")
    print(f"  {'─' * 60}")
    for cls_id in sorted(all_class_results.keys()):
        r = all_class_results[cls_id]
        cls_p = r["tp"] / (r["tp"] + r["fp"] + 1e-6)
        cls_r = r["tp"] / (r["tp"] + r["fn"] + 1e-6)
        name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
        print(f"  {name:<15} {r['tp']:<6} {r['fp']:<6} {r['fn']:<6} {cls_p*100:<12.1f} {cls_r*100:<12.1f}")

    print(f"\n  Latency Statistics ({engine.active_provider}):")
    print(f"  {'─' * 50}")
    print(f"  {'Mean':<25} {np.mean(latencies):.1f} ms")
    print(f"  {'Median':<25} {np.median(latencies):.1f} ms")
    print(f"  {'P95':<25} {np.percentile(latencies, 95):.1f} ms")
    print(f"  {'Min':<25} {np.min(latencies):.1f} ms")
    print(f"  {'Max':<25} {np.max(latencies):.1f} ms")
    print(f"  {'Throughput':<25} {1000.0/np.mean(latencies):.1f} FPS")

    print(f"\n  Model Specifications:")
    print(f"  {'─' * 50}")
    print(f"  {'Architecture':<25} YOLOv8n-seg")
    print(f"  {'Parameters':<25} 3,258,844")
    print(f"  {'GFLOPs':<25} 11.3")
    print(f"  {'Input Size':<25} 640x640")
    print(f"  {'Export Format':<25} ONNX (opset 12)")
    print(f"  {'File Size':<25} {os.path.getsize(model_path)/1024/1024:.1f} MB")

    print("\n" + "=" * 70)
    print("  EVALUATION COMPLETE — Results are reproducible by re-running this script")
    print("=" * 70)
    print()

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "avg_latency_ms": np.mean(latencies),
    }


if __name__ == "__main__":
    run_evaluation()
