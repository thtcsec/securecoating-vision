"""
SecureCoating-Vision: Unified Benchmark & True Segmentation Evaluation Engine
=============================================================================
Computes mathematically rigorous, reproducible metrics by running ONNX/YOLO
inference on the evaluation dataset and comparing predictions against ground-truth
polygons using both Box IoU and pixel-level Mask IoU.

Addresses:
1. Strict YOLO Segmentation Polygon Parsing (even coordinates, >=6 floats)
2. True Pixel-Level Mask IoU & Polygon Rasterization (alongside BBox IoU)
3. Standardized Bounding Box Coordinates (xyxy vs xywh)
4. Direct Integration of Ultralytics Box & Mask mAP50 / mAP50-95
5. Division-by-Zero & Empty Evaluation Protections
6. CLI Configurable Thresholds (--iou, --conf, --seed)
7. Deterministic Random Seeds (seed=42)

Outputs:
- reports/evaluation_results.json (unified schema)
- reports/evaluation_results.csv (per-class breakdown)
- reports/ultralytics_validation_results.json

Usage:
    python scripts/run_evaluation.py --iou 0.5 --conf 0.25 --seed 42
"""

import os
import sys
import json
import csv
import time
import argparse
import hashlib
from datetime import datetime, timezone
from collections import defaultdict
import numpy as np
import cv2

# Set project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)
if os.path.join(PROJECT_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

CLASS_NAMES = {0: "scratch", 1: "void", 2: "blister", 3: "delamination"}


def parse_args():
    parser = argparse.ArgumentParser(description="SecureCoating-Vision Unified Evaluation Engine")
    parser.add_argument("--dataset-dir", type=str, default="data/evaluation",
                        help="Path to evaluation dataset root (containing images/ and labels/)")
    parser.add_argument("--model-path", type=str, default="outputs/model.onnx",
                        help="Path to ONNX model weights")
    parser.add_argument("--iou", type=float, default=0.50,
                        help="IoU threshold for positive detection/segmentation match (default: 0.50)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold for predictions (default: 0.25)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for deterministic evaluation (default: 42)")
    parser.add_argument("--ultralytics-model-path", default=None,
                        help="Optional explicit .pt artifact for official Ultralytics metrics")
    parser.add_argument("--ultralytics-data-config", default=None,
                        help="Dataset YAML matching --dataset-dir; required with --ultralytics-model-path")
    parser.add_argument("--reference-dataset-dir", default="data/coating_defects",
                        help="Training dataset root used for mandatory hash-overlap audit")
    return parser.parse_args()


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_no_dataset_overlap(evaluation_images_dir: str, reference_dataset_dir: str) -> None:
    """Reject evaluation data that overlaps train or validation artifacts by content hash."""
    reference_hashes = {}
    for split in ("train", "val"):
        split_dir = os.path.join(reference_dataset_dir, "images", split)
        if not os.path.isdir(split_dir):
            continue
        for name in os.listdir(split_dir):
            path = os.path.join(split_dir, name)
            if os.path.isfile(path):
                reference_hashes[_file_sha256(path)] = path
    overlaps = []
    for name in os.listdir(evaluation_images_dir):
        path = os.path.join(evaluation_images_dir, name)
        if os.path.isfile(path):
            matching = reference_hashes.get(_file_sha256(path))
            if matching:
                overlaps.append((path, matching))
    if overlaps:
        examples = "; ".join(f"{a} == {b}" for a, b in overlaps[:5])
        raise RuntimeError(
            f"Evaluation leakage detected: {len(overlaps)} image(s) overlap train/val by SHA-256. {examples}"
        )


def load_gt_labels(label_path: str, img_h: int = 640, img_w: int = 640):
    """
    Parse YOLO segmentation label file into class IDs, bounding boxes, and rasterized binary masks.
    
    YOLO segmentation format:
        <class_id> <x1> <y1> <x2> <y2> <x3> <y3> ... (normalized [0, 1])
    """
    labels = []
    if not os.path.exists(label_path):
        raise FileNotFoundError(
            f"Ground-truth label is missing: {label_path}. Use an empty file for a verified negative sample."
        )

    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            # YOLO polygon requires class_id and at least 3 vertices (>= 6 coordinate floats)
            if not parts:
                continue
            if len(parts) < 7:
                raise ValueError(f"Malformed polygon label in {label_path}: {line.strip()}")

            try:
                class_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]
            except ValueError as exc:
                raise ValueError(f"Non-numeric label in {label_path}: {line.strip()}") from exc

            # Must have even number of coordinate values
            if len(coords) % 2 != 0:
                raise ValueError(f"Odd polygon coordinate count in {label_path}: {line.strip()}")

            polygon_pts = []
            for i in range(0, len(coords), 2):
                px = np.clip(coords[i] * img_w, 0, img_w - 1)
                py = np.clip(coords[i + 1] * img_h, 0, img_h - 1)
                polygon_pts.append([px, py])

            polygon_np = np.array(polygon_pts, dtype=np.int32)

            # Rasterize ground-truth mask
            gt_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            cv2.fillPoly(gt_mask, [polygon_np], 1)

            # Tight bounding box
            xs = [p[0] for p in polygon_pts]
            ys = [p[1] for p in polygon_pts]
            x1, y1 = max(0.0, min(xs)), max(0.0, min(ys))
            x2, y2 = min(float(img_w), max(xs)), min(float(img_h), max(ys))

            labels.append({
                "class_id": class_id,
                "bbox_xyxy": [x1, y1, x2, y2],
                "bbox_xywh": [x1, y1, x2 - x1, y2 - y1],
                "mask": gt_mask,
                "area_px": float(np.sum(gt_mask)),
                "polygon": polygon_pts
            })

    return labels


def compute_box_iou(box1: list, box2: list) -> float:
    """Compute IoU between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, (box1[2] - box1[0])) * max(0.0, (box1[3] - box1[1]))
    area2 = max(0.0, (box2[2] - box2[0])) * max(0.0, (box2[3] - box2[1]))
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return float(inter_area / (union_area + 1e-6))


def compute_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute pixel-level Mask IoU between two binary masks."""
    intersection = np.logical_and(mask1 > 0, mask2 > 0).sum()
    union = np.logical_or(mask1 > 0, mask2 > 0).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def evaluate_sample_predictions(gt_labels: list, detections: list, seg_mask: np.ndarray, iou_threshold: float = 0.50):
    """
    Match predictions to ground truth evaluating both BBox and Mask IoU.
    """
    img_h, img_w = seg_mask.shape[:2]
    class_results = defaultdict(lambda: {
        "box_tp": 0, "box_fp": 0, "box_fn": 0,
        "mask_tp": 0, "mask_fp": 0, "mask_fn": 0,
        "mask_ious": []
    })

    matched_gt_box = set()
    matched_gt_mask = set()

    # Sort detections by confidence descending
    sorted_dets = sorted(detections, key=lambda d: d.get("confidence", 0.0), reverse=True)

    for det in sorted_dets:
        det_class = det.get("class_id", 0)
        raw_box = det.get("box", [0, 0, 0, 0])

        if len(raw_box) != 4:
            raise ValueError(f"Detection has invalid box: {raw_box}")
        box_format = det.get("box_format")
        if box_format == "xyxy":
            det_xyxy = list(map(float, raw_box))
        elif box_format == "xywh":
            x, y, width, height = map(float, raw_box)
            det_xyxy = [x, y, x + width, y + height]
        else:
            raise ValueError(f"Detection is missing a supported box_format: {box_format}")

        # Extract detection binary mask for this class
        # Extract detection binary mask for this instance
        det_binary_mask = det.get("mask", None)
        if det_binary_mask is None:
            det_binary_mask = (seg_mask == (det_class + 1)).astype(np.uint8)
        elif det_binary_mask.shape != (img_h, img_w):
            det_binary_mask = cv2.resize(det_binary_mask.astype(np.uint8), (img_w, img_h), interpolation=cv2.INTER_NEAREST)

        # 1. BBox Matching
        best_box_iou = 0.0
        best_gt_box_idx = -1
        for i, gt in enumerate(gt_labels):
            if i in matched_gt_box or gt["class_id"] != det_class:
                continue
            b_iou = compute_box_iou(det_xyxy, gt["bbox_xyxy"])
            if b_iou > best_box_iou:
                best_box_iou = b_iou
                best_gt_box_idx = i

        if best_box_iou >= iou_threshold and best_gt_box_idx >= 0:
            matched_gt_box.add(best_gt_box_idx)
            class_results[det_class]["box_tp"] += 1
        else:
            class_results[det_class]["box_fp"] += 1

        # 2. Mask IoU Matching
        best_mask_iou = 0.0
        best_gt_mask_idx = -1
        for i, gt in enumerate(gt_labels):
            if i in matched_gt_mask or gt["class_id"] != det_class:
                continue
            m_iou = compute_mask_iou(det_binary_mask, gt["mask"])
            if m_iou > best_mask_iou:
                best_mask_iou = m_iou
                best_gt_mask_idx = i

        # Record best mask IoU found
        if best_mask_iou > 0:
            class_results[det_class]["mask_ious"].append(best_mask_iou)

        if best_mask_iou >= iou_threshold and best_gt_mask_idx >= 0:
            matched_gt_mask.add(best_gt_mask_idx)
            class_results[det_class]["mask_tp"] += 1
        else:
            class_results[det_class]["mask_fp"] += 1

    # Count False Negatives for unmatched GTs
    for i, gt in enumerate(gt_labels):
        cid = gt["class_id"]
        if i not in matched_gt_box:
            class_results[cid]["box_fn"] += 1
        if i not in matched_gt_mask:
            class_results[cid]["mask_fn"] += 1

    return class_results


def run_evaluation(
    dataset_dir: str = "data/evaluation",
    model_path: str = "outputs/model.onnx",
    iou_threshold: float = 0.50,
    conf_threshold: float = 0.25,
    seed: int = 42,
    ultralytics_model_path: str = None,
    ultralytics_data_config: str = None,
    reference_dataset_dir: str = "data/coating_defects",
):
    """
    Main evaluation pipeline: runs inference on all images, evaluates Box/Mask metrics,
    and merges Ultralytics mAP validation into a unified report.
    """
    np.random.seed(seed)
    print("=" * 75)
    print("  SecureCoating-Vision: Unified Benchmark & Segmentation Evaluation Engine")
    print(f"  Configuration: IoU Threshold={iou_threshold:.2f} | Conf Threshold={conf_threshold:.2f} | Seed={seed}")
    print("=" * 75)

    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Evaluation images directory not found: {images_dir}")
    assert_no_dataset_overlap(images_dir, reference_dataset_dir)

    # Initialize ONNX inference engine
    from inference.onnx_engine import InferenceEngine
    engine = InferenceEngine(
        model_path=model_path,
        conf_thresh=conf_threshold,
        iou_thresh=0.45,
        device="auto"
    )
    if not engine.is_loaded:
        raise RuntimeError(f"Evaluation model did not load: {model_path}")

    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")
    image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(valid_extensions)])

    if not image_files:
        raise RuntimeError(f"No valid image files found under: {images_dir}")

    print(f"\n  Active Execution Provider : {engine.active_provider}")
    print(f"  Evaluation Dataset Path   : {images_dir}")
    print(f"  Total Images to Evaluate  : {len(image_files)}")

    latencies = []
    total_gt_count = 0
    total_det_count = 0
    aggregated_results = defaultdict(lambda: {
        "box_tp": 0, "box_fp": 0, "box_fn": 0,
        "mask_tp": 0, "mask_fp": 0, "mask_fn": 0,
        "mask_ious": []
    })

    # Warmup runs to stabilize GPU latency timing
    dummy_warmup = np.zeros((640, 640, 3), dtype=np.uint8)
    for _ in range(5):
        engine.infer(dummy_warmup)

    # Evaluation loop
    for idx, fname in enumerate(image_files):
        img_path = os.path.join(images_dir, fname)
        base_name = os.path.splitext(fname)[0]
        label_path = os.path.join(labels_dir, base_name + ".txt")

        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not decode evaluation image: {img_path}")

        h, w = img.shape[:2]
        gt_labels = load_gt_labels(label_path, img_h=h, img_w=w)
        total_gt_count += len(gt_labels)

        # Time inference
        t0 = time.perf_counter()
        result = engine.infer(img)
        t_elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(t_elapsed_ms)

        detections = result.get("detections", [])
        seg_mask = result.get("segmentation_mask", np.zeros((h, w), dtype=np.uint8))
        total_det_count += len(detections)

        # Match sample
        sample_results = evaluate_sample_predictions(
            gt_labels=gt_labels,
            detections=detections,
            seg_mask=seg_mask,
            iou_threshold=iou_threshold
        )

        for cid, res in sample_results.items():
            aggregated_results[cid]["box_tp"] += res["box_tp"]
            aggregated_results[cid]["box_fp"] += res["box_fp"]
            aggregated_results[cid]["box_fn"] += res["box_fn"]
            aggregated_results[cid]["mask_tp"] += res["mask_tp"]
            aggregated_results[cid]["mask_fp"] += res["mask_fp"]
            aggregated_results[cid]["mask_fn"] += res["mask_fn"]
            aggregated_results[cid]["mask_ious"].extend(res["mask_ious"])

    if not latencies:
        raise RuntimeError("Evaluation failed: No valid images were successfully processed.")

    # Calculate overall aggregate metrics
    tot_box_tp = sum(r["box_tp"] for r in aggregated_results.values())
    tot_box_fp = sum(r["box_fp"] for r in aggregated_results.values())
    tot_box_fn = sum(r["box_fn"] for r in aggregated_results.values())

    box_prec = tot_box_tp / (tot_box_tp + tot_box_fp + 1e-6)
    box_rec = tot_box_tp / (tot_box_tp + tot_box_fn + 1e-6)
    box_f1 = (2 * box_prec * box_rec) / (box_prec + box_rec + 1e-6)

    tot_mask_tp = sum(r["mask_tp"] for r in aggregated_results.values())
    tot_mask_fp = sum(r["mask_fp"] for r in aggregated_results.values())
    tot_mask_fn = sum(r["mask_fn"] for r in aggregated_results.values())

    mask_prec = tot_mask_tp / (tot_mask_tp + tot_mask_fp + 1e-6)
    mask_rec = tot_mask_tp / (tot_mask_tp + tot_mask_fn + 1e-6)
    mask_f1 = (2 * mask_prec * mask_rec) / (mask_prec + mask_rec + 1e-6)

    all_mask_ious = []
    for r in aggregated_results.values():
        all_mask_ious.extend(r["mask_ious"])
    overall_mean_mask_iou = round(float(np.mean(all_mask_ious)), 4) if all_mask_ious else 0.0

    mean_lat = max(float(np.mean(latencies)), 1e-6)
    p50_lat = float(np.median(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    throughput_fps = 1000.0 / mean_lat

    # Run Ultralytics mAP validation if available to merge formal mAP50 / mAP50-95
    ultralytics_map = {}
    try:
        if bool(ultralytics_model_path) != bool(ultralytics_data_config):
            raise ValueError(
                "--ultralytics-model-path and --ultralytics-data-config must be provided together"
            )
        if not ultralytics_model_path:
            raise RuntimeError("explicit Ultralytics model/config not provided")
        from run_ultralytics_validation import run_ultralytics_validation
        val_res = run_ultralytics_validation(
            model_pt=ultralytics_model_path,
            config_yaml=ultralytics_data_config,
        )
        if val_res:
            ultralytics_map = {
                "box_map50": val_res.get("box_map50", None),
                "box_map50_95": val_res.get("box_map50_95", None),
                "mask_map50": val_res.get("mask_map50", None),
                "mask_map50_95": val_res.get("mask_map50_95", None)
            }
    except Exception as e:
        print(f"  [NOTE] Ultralytics direct validation skipped: {e}")

    # Output formatted report to stdout
    print("\n" + "=" * 75)
    print("  EVALUATION METRIC RESULTS")
    print("=" * 75)
    print(f"  Evaluated Samples       : {len(latencies)} images ({total_gt_count} GT instances)")
    print(f"  Total Predictions       : {total_det_count} detections")
    print(f"  Box Detection Precision : {box_prec * 100.0:.1f}%")
    print(f"  Box Detection Recall    : {box_rec * 100.0:.1f}%")
    print(f"  Box Detection F1-Score  : {box_f1 * 100.0:.1f}%")
    print(f"  Mask Segmentation Prec  : {mask_prec * 100.0:.1f}%")
    print(f"  Mask Segmentation Recall: {mask_rec * 100.0:.1f}%")
    print(f"  Mask Segmentation F1    : {mask_f1 * 100.0:.1f}%")
    print(f"  Overall Mean Mask IoU   : {overall_mean_mask_iou * 100.0:.2f}%")
    if ultralytics_map:
        print(f"  Ultralytics Mask mAP@50 : {ultralytics_map.get('mask_map50', 0)*100.0:.2f}%")
        print(f"  Ultralytics Box mAP@50  : {ultralytics_map.get('box_map50', 0)*100.0:.2f}%")
    print(f"  Inference Latency (Mean): {mean_lat:.2f} ms (P50: {p50_lat:.2f} ms, P95: {p95_lat:.2f} ms)")
    print(f"  Inference Throughput    : {throughput_fps:.1f} FPS")

    print("\n  Per-Class Breakdown:")
    print("  " + "-" * 71)
    print(f"  {'Class':<14} {'Box TP':<8} {'Box FP':<8} {'Box FN':<8} {'Box Prec':<10} {'Box Rec':<10} {'Mask F1':<10}")
    print("  " + "-" * 71)
    for cid in sorted(CLASS_NAMES.keys()):
        r = aggregated_results[cid]
        c_name = CLASS_NAMES[cid]
        c_p = r["box_tp"] / (r["box_tp"] + r["box_fp"] + 1e-6) * 100.0
        c_r = r["box_tp"] / (r["box_tp"] + r["box_fn"] + 1e-6) * 100.0
        m_p = r["mask_tp"] / (r["mask_tp"] + r["mask_fp"] + 1e-6)
        m_r = r["mask_tp"] / (r["mask_tp"] + r["mask_fn"] + 1e-6)
        m_f1 = (2 * m_p * m_r) / (m_p + m_r + 1e-6) * 100.0
        print(f"  {c_name:<14} {r['box_tp']:<8} {r['box_fp']:<8} {r['box_fn']:<8} {c_p:<9.1f}% {c_r:<9.1f}% {m_f1:<9.1f}%")
    print("  " + "-" * 71)

    # Save to JSON and CSV reports
    reports_dir = os.path.join(PROJECT_ROOT, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    json_path = os.path.join(reports_dir, "evaluation_results.json")
    csv_path = os.path.join(reports_dir, "evaluation_results.csv")

    export_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": engine.active_provider,
        "dataset_path": dataset_dir,
        "total_images": len(latencies),
        "total_gt_instances": total_gt_count,
        "total_detections": total_det_count,
        "iou_threshold": iou_threshold,
        "confidence_threshold": conf_threshold,
        "seed": seed,
        "detection_metrics": {
            "box_precision": round(box_prec, 4),
            "box_recall": round(box_rec, 4),
            "box_f1_score": round(box_f1, 4),
        },
        "segmentation_metrics": {
            "mask_precision": round(mask_prec, 4),
            "mask_recall": round(mask_rec, 4),
            "mask_f1_score": round(mask_f1, 4),
            "overall_mean_mask_iou": overall_mean_mask_iou,
        },
        "ultralytics_map_metrics": ultralytics_map,
        "latency_ms": {
            "mean": round(mean_lat, 2),
            "median": round(p50_lat, 2),
            "p95": round(p95_lat, 2),
            "throughput_fps": round(throughput_fps, 1)
        },
        "per_class": {
            CLASS_NAMES[cid]: {
                "box_tp": aggregated_results[cid]["box_tp"],
                "box_fp": aggregated_results[cid]["box_fp"],
                "box_fn": aggregated_results[cid]["box_fn"],
                "box_precision": round(aggregated_results[cid]["box_tp"] / (aggregated_results[cid]["box_tp"] + aggregated_results[cid]["box_fp"] + 1e-6), 4),
                "box_recall": round(aggregated_results[cid]["box_tp"] / (aggregated_results[cid]["box_tp"] + aggregated_results[cid]["box_fn"] + 1e-6), 4),
                "mask_tp": aggregated_results[cid]["mask_tp"],
                "mask_fp": aggregated_results[cid]["mask_fp"],
                "mask_fn": aggregated_results[cid]["mask_fn"],
                "mean_mask_iou": round(float(np.mean(aggregated_results[cid]["mask_ious"])), 4) if aggregated_results[cid]["mask_ious"] else 0.0
            } for cid in CLASS_NAMES
        }
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Class", "Box_TP", "Box_FP", "Box_FN", "Box_Precision", "Box_Recall", "Mask_TP", "Mask_FP", "Mask_FN", "Mean_Mask_IoU"])
        for cid in sorted(CLASS_NAMES.keys()):
            r = aggregated_results[cid]
            b_p = r["box_tp"] / (r["box_tp"] + r["box_fp"] + 1e-6)
            b_r = r["box_tp"] / (r["box_tp"] + r["box_fn"] + 1e-6)
            m_iou = float(np.mean(r["mask_ious"])) if r["mask_ious"] else 0.0
            writer.writerow([
                CLASS_NAMES[cid],
                r["box_tp"], r["box_fp"], r["box_fn"], round(b_p, 4), round(b_r, 4),
                r["mask_tp"], r["mask_fp"], r["mask_fn"], round(m_iou, 4)
            ])

    print(f"\n  Exported results:")
    print(f"    - {json_path}")
    print(f"    - {csv_path}")
    print("=" * 75 + "\n")
    return export_data


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(
        dataset_dir=args.dataset_dir,
        model_path=args.model_path,
        iou_threshold=args.iou,
        conf_threshold=args.conf,
        seed=args.seed,
        ultralytics_model_path=args.ultralytics_model_path,
        ultralytics_data_config=args.ultralytics_data_config,
        reference_dataset_dir=args.reference_dataset_dir,
    )
