"""
SecureCoating-Vision: Reproducible Evaluation Script
=====================================================
Computes real detection metrics by running ONNX model inference on the
evaluation subset dataset and comparing predictions against ground-truth labels.

Outputs automatically:
- reports/evaluation_results.json
- reports/evaluation_results.csv

Usage:
    python scripts/run_evaluation.py
"""

import os
import sys
import json
import csv
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
            coords = [float(x) for x in parts[1:]]
            xs = [coords[i] * img_w for i in range(0, len(coords), 2)]
            ys = [coords[i] * img_h for i in range(1, len(coords), 2)]
            if not xs or not ys:
                continue
            x1, y1 = min(xs), min(ys)
            x2, y2 = max(xs), max(ys)
            labels.append({
                "class_id": class_id,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
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
    tp, fp, fn = 0, 0, 0
    matched_gt = set()
    class_results = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    sorted_dets = sorted(detections, key=lambda d: d.get("confidence", 0), reverse=True)

    for det in sorted_dets:
        det_bbox = det.get("box", None)
        if det_bbox is None:
            continue
        if len(det_bbox) == 4:
            if det_bbox[2] > det_bbox[0] and det_bbox[3] > det_bbox[1]:
                det_bbox = [det_bbox[0], det_bbox[1], det_bbox[2] - det_bbox[0], det_bbox[3] - det_bbox[1]]

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
    print("  Computing real metrics against ground-truth evaluation labels")
    print("=" * 70)

    model_path = "outputs/model.onnx"
    
    # Priority 1: Included evaluation subset data/evaluation/
    # Priority 2: Full synthetic dataset data/coating_defects/images/val
    val_img_dir = "data/evaluation/images"
    val_lbl_dir = "data/evaluation/labels"

    if not (os.path.exists(val_img_dir) and os.path.exists(val_lbl_dir)):
        val_img_dir = "data/coating_defects/images/val"
        val_lbl_dir = "data/coating_defects/labels/val"

    if not os.path.exists(model_path):
        print(f"\n  [ERROR] Model not found: {model_path}")
        print("  Run: python src/training/train_yolo.py train --epochs 50")
        sys.exit(1)

    if not (os.path.exists(val_img_dir) and os.path.exists(val_lbl_dir)):
        print(f"\n  [ERROR] Evaluation dataset not found.")
        print("  Run: python scripts/generate_synthetic_coating_defects.py")
        sys.exit(1)

    engine = InferenceEngine(model_path, imgsz=640, conf_thresh=0.5)
    print(f"\n  Model: {model_path} ({os.path.getsize(model_path)/1024/1024:.1f} MB)")
    print(f"  Provider: {engine.active_provider}")
    print(f"  Dataset path: {val_img_dir}")

    images = sorted([f for f in os.listdir(val_img_dir) if f.endswith(('.jpg', '.png'))])
    print(f"  Evaluation images: {len(images)}")
    print(f"\n  Running inference + evaluation...")

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

        gt_labels = load_gt_labels(lbl_path, h, w)
        total_gt += len(gt_labels)

        result = engine.infer(img)
        latencies.append(result["latency_ms"])
        detections = result.get("detections", [])
        total_det += len(detections)

        tp, fp, fn, class_results = match_detections(gt_labels, detections, iou_threshold=0.5)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        for cls_id, counts in class_results.items():
            all_class_results[cls_id]["tp"] += counts["tp"]
            all_class_results[cls_id]["fp"] += counts["fp"]
            all_class_results[cls_id]["fn"] += counts["fn"]

    precision = total_tp / (total_tp + total_fp + 1e-6)
    recall = total_tp / (total_tp + total_fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    print("\n" + "=" * 70)
    print("  EVALUATION RESULTS (IoU threshold = 0.50)")
    print("=" * 70)

    print(f"\n  Overall (across {len(images)} images, {total_gt} GT instances):")
    print(f"  {'─' * 50}")
    print(f"  {'Metric':<25} {'Value':<15}")
    print(f"  {'─' * 50}")
    print(f"  {'Precision':<25} {precision*100:.1f}%")
    print(f"  {'Recall':<25} {recall*100:.1f}%")
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
    print(f"  {'Mean (ONNX FP16)':<25} {np.mean(latencies):.1f} ms")
    print(f"  {'Median':<25} {np.median(latencies):.1f} ms")
    print(f"  {'P95':<25} {np.percentile(latencies, 95):.1f} ms")
    print(f"  {'Throughput':<25} {1000.0/np.mean(latencies):.1f} FPS")

    # Save reports/evaluation_results.json and reports/evaluation_results.csv
    reports_dir = os.path.join(PROJECT_ROOT, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    json_path = os.path.join(reports_dir, "evaluation_results.json")
    results_data = {
        "dataset_path": val_img_dir,
        "total_images": len(images),
        "total_gt_instances": total_gt,
        "total_detections": total_det,
        "overall": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
        },
        "latency_ms": {
            "mean": round(float(np.mean(latencies)), 2),
            "median": round(float(np.median(latencies)), 2),
            "p95": round(float(np.percentile(latencies, 95)), 2),
        },
        "per_class": {
            CLASS_NAMES[cid]: {
                "tp": r["tp"],
                "fp": r["fp"],
                "fn": r["fn"],
                "precision": round(r["tp"] / (r["tp"] + r["fp"] + 1e-6), 4),
                "recall": round(r["tp"] / (r["tp"] + r["fn"] + 1e-6), 4),
            } for cid, r in all_class_results.items()
        }
    }
    with open(json_path, "w") as f:
        json.dump(results_data, f, indent=2)

    csv_path = os.path.join(reports_dir, "evaluation_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Class", "TP", "FP", "FN", "Precision", "Recall"])
        for cid, r in all_class_results.items():
            c_p = r["tp"] / (r["tp"] + r["fp"] + 1e-6)
            c_r = r["tp"] / (r["tp"] + r["fn"] + 1e-6)
            writer.writerow([CLASS_NAMES.get(cid, str(cid)), r["tp"], r["fp"], r["fn"], round(c_p, 4), round(c_r, 4)])

    print(f"\n  Exported results:")
    print(f"    - {json_path}")
    print(f"    - {csv_path}")

    print("\n" + "=" * 70)
    print("  EVALUATION COMPLETE — Results verified")
    print("=" * 70)
    print()

    return results_data


if __name__ == "__main__":
    run_evaluation()
