"""
SecureCoating-Vision: YOLOv8 Segmentation Training Script
=========================================================
Optimized for RTX 4050 6GB VRAM. Uses YOLOv8n-seg or YOLOv8s-seg for 
efficient training on coating defect detection datasets.

Usage:
    python src/training/train_yolo.py --model yolov8n-seg --epochs 50 --imgsz 640
    python src/training/train_yolo.py --model yolov8s-seg --epochs 30 --imgsz 512 --batch 4
"""

import os
import sys
import argparse
import yaml
import logging
from pathlib import Path
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def check_gpu_memory():
    """Check available GPU memory and recommend settings."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"GPU detected: {gpu_name} ({total_mem:.1f} GB)")
            
            if total_mem <= 4:
                return {"model": "yolov8n-seg", "batch": 2, "imgsz": 512}
            elif total_mem <= 6:
                return {"model": "yolov8n-seg", "batch": 4, "imgsz": 640}
            elif total_mem <= 8:
                return {"model": "yolov8s-seg", "batch": 4, "imgsz": 640}
            else:
                return {"model": "yolov8s-seg", "batch": 8, "imgsz": 640}
        else:
            logger.warning("No CUDA GPU detected. Training will use CPU (very slow).")
            return {"model": "yolov8n-seg", "batch": 2, "imgsz": 512}
    except ImportError:
        return {"model": "yolov8n-seg", "batch": 2, "imgsz": 512}


def setup_dataset_structure(data_root: str):
    """Creates the expected YOLO dataset folder structure if it doesn't exist."""
    dirs = [
        os.path.join(data_root, "images", "train"),
        os.path.join(data_root, "images", "val"),
        os.path.join(data_root, "images", "test"),
        os.path.join(data_root, "labels", "train"),
        os.path.join(data_root, "labels", "val"),
        os.path.join(data_root, "labels", "test"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        
    logger.info(f"Dataset structure verified at: {data_root}")
    return True


def generate_synthetic_samples(data_root: str, num_train: int = 100, num_val: int = 20):
    """
    Generate synthetic training samples for development/testing purposes.
    In production, replace with actual CoatingVision 2026 dataset images.
    """
    import numpy as np
    import cv2
    
    class_names = ["scratch", "void", "blister", "delamination"]
    
    for split, num_samples in [("train", num_train), ("val", num_val)]:
        img_dir = os.path.join(data_root, "images", split)
        lbl_dir = os.path.join(data_root, "labels", split)
        
        existing = len([f for f in os.listdir(img_dir) if f.endswith('.png') or f.endswith('.jpg')])
        if existing >= num_samples:
            logger.info(f"[{split}] Already has {existing} images, skipping generation.")
            continue
        
        logger.info(f"Generating {num_samples} synthetic samples for [{split}] split...")
        
        for i in range(num_samples):
            # Generate base surface texture (simulated coating)
            img = np.random.randint(170, 200, (640, 640, 3), dtype=np.uint8)
            # Add subtle noise texture
            noise = np.random.randint(-10, 10, (640, 640, 3), dtype=np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            # Randomly select defect class
            class_id = np.random.randint(0, len(class_names))
            class_name = class_names[class_id]
            
            # Generate polygon mask coordinates (normalized YOLO format)
            polygon_points = []
            
            if class_name == "scratch":
                # Long thin line defect
                y_start = np.random.uniform(0.2, 0.5)
                x_start = np.random.uniform(0.1, 0.3)
                length = np.random.uniform(0.3, 0.6)
                width = np.random.uniform(0.01, 0.03)
                angle = np.random.uniform(-15, 15) * np.pi / 180
                
                # Draw on image
                pt1 = (int(x_start * 640), int(y_start * 640))
                pt2 = (int((x_start + length * np.cos(angle)) * 640), 
                       int((y_start + length * np.sin(angle)) * 640))
                cv2.line(img, pt1, pt2, (40, 40, 40), thickness=np.random.randint(3, 8))
                
                # Create polygon for YOLO seg annotation
                dx = width * np.sin(angle) / 2
                dy = width * np.cos(angle) / 2
                polygon_points = [
                    x_start - dx, y_start + dy,
                    x_start + length * np.cos(angle) - dx, y_start + length * np.sin(angle) + dy,
                    x_start + length * np.cos(angle) + dx, y_start + length * np.sin(angle) - dy,
                    x_start + dx, y_start - dy,
                ]
                
            elif class_name == "void":
                # Circular sub-surface void
                cx = np.random.uniform(0.25, 0.75)
                cy = np.random.uniform(0.25, 0.75)
                radius = np.random.uniform(0.03, 0.08)
                
                cv2.circle(img, (int(cx*640), int(cy*640)), int(radius*640), (80, 60, 80), -1)
                
                # Approximate circle as polygon (12 points)
                for k in range(12):
                    angle = 2 * np.pi * k / 12
                    polygon_points.extend([cx + radius * np.cos(angle), cy + radius * np.sin(angle)])
                    
            elif class_name == "blister":
                # Raised bump
                cx = np.random.uniform(0.2, 0.8)
                cy = np.random.uniform(0.2, 0.8)
                rx = np.random.uniform(0.04, 0.1)
                ry = np.random.uniform(0.03, 0.08)
                
                cv2.ellipse(img, (int(cx*640), int(cy*640)), (int(rx*640), int(ry*640)), 
                           0, 0, 360, (200, 210, 200), -1)
                
                # Approximate ellipse as polygon
                for k in range(16):
                    angle = 2 * np.pi * k / 16
                    polygon_points.extend([cx + rx * np.cos(angle), cy + ry * np.sin(angle)])
                    
            elif class_name == "delamination":
                # Irregular patch
                cx = np.random.uniform(0.3, 0.7)
                cy = np.random.uniform(0.3, 0.7)
                num_pts = np.random.randint(6, 10)
                
                pts_img = []
                for k in range(num_pts):
                    angle = 2 * np.pi * k / num_pts
                    r = np.random.uniform(0.05, 0.12)
                    px = cx + r * np.cos(angle)
                    py = cy + r * np.sin(angle)
                    polygon_points.extend([np.clip(px, 0, 1), np.clip(py, 0, 1)])
                    pts_img.append([int(px * 640), int(py * 640)])
                    
                pts_arr = np.array(pts_img, dtype=np.int32)
                cv2.fillPoly(img, [pts_arr], (100, 90, 110))
            
            # Clip polygon coords to [0, 1]
            polygon_points = [max(0.0, min(1.0, p)) for p in polygon_points]
            
            # Save image
            img_path = os.path.join(img_dir, f"sample_{i:05d}.png")
            cv2.imwrite(img_path, img)
            
            # Save YOLO segmentation label (class_id followed by polygon coordinates)
            lbl_path = os.path.join(lbl_dir, f"sample_{i:05d}.txt")
            coords_str = " ".join([f"{p:.6f}" for p in polygon_points])
            with open(lbl_path, 'w') as f:
                f.write(f"{class_id} {coords_str}\n")
    
    logger.info("Synthetic dataset generation complete.")


def train(args):
    """Main training function using Ultralytics YOLOv8."""
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("Ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)
    
    # Load dataset config
    dataset_config_path = os.path.abspath(args.data)
    with open(dataset_config_path, 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    data_root = dataset_config.get("path", "data/coating_defects")
    if not os.path.isabs(data_root):
        data_root = os.path.join(os.path.dirname(dataset_config_path), "..", data_root)
        data_root = os.path.abspath(data_root)
    
    # Setup directory structure
    setup_dataset_structure(data_root)
    
    # Never silently replace an empty real dataset with synthetic samples.
    train_dir = os.path.join(data_root, "images", "train")
    if len(os.listdir(train_dir)) == 0:
        if not args.allow_synthetic:
            raise RuntimeError(
                "Training dataset is empty. Generate/approve synthetic data explicitly "
                "with --allow-synthetic or provide a real dataset manifest."
            )
        logger.warning("Generating explicitly requested synthetic development samples")
        np.random.seed(args.seed)
        generate_synthetic_samples(data_root, num_train=200, num_val=40)
    
    # GPU memory check and auto-config
    recommended = check_gpu_memory()
    
    model_name = args.model
    batch_size = args.batch or recommended["batch"]
    imgsz = args.imgsz or recommended["imgsz"]
    
    logger.info("=" * 60)
    logger.info("  SECURECOATING-VISION: YOLOv8 SEGMENTATION TRAINING")
    logger.info("=" * 60)
    logger.info(f"  Model:      {model_name}")
    logger.info(f"  Epochs:     {args.epochs}")
    logger.info(f"  Batch Size: {batch_size}")
    logger.info(f"  Image Size: {imgsz}")
    logger.info(f"  Dataset:    {dataset_config_path}")
    logger.info(f"  Output:     outputs/yolo_training/")
    logger.info("=" * 60)
    
    # Initialize model
    model_source = model_name if model_name.endswith(".pt") else f"{model_name}.pt"
    model = YOLO(model_source)
    resolved_device = args.device
    if args.device == "auto":
        resolved_device = 0 if torch.cuda.is_available() else "cpu"
    
    # Train with optimized settings for 6GB VRAM
    results = model.train(
        data=dataset_config_path,
        epochs=args.epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=resolved_device,
        seed=args.seed,
        deterministic=True,
        project="outputs/yolo_training",
        name=f"{model_name.replace('.pt', '')}_{args.epochs}ep",
        # Memory optimization for RTX 4050 6GB
        workers=4,
        cache=False,  # Disable caching to save RAM
        amp=True,  # Mixed precision (FP16) for VRAM savings
        # Augmentation for coating-specific data
        hsv_h=0.01,  # Minimal hue shift (coatings have consistent color)
        hsv_s=0.3,
        hsv_v=0.3,
        degrees=5.0,  # Small rotation (line scans are roughly aligned)
        translate=0.1,
        scale=0.3,
        fliplr=0.5,
        flipud=0.2,
        mosaic=0.8,
        # Save best + last checkpoints
        save=True,
        save_period=10,
        patience=15,  # Early stopping patience
        verbose=True,
    )
    
    logger.info("Training completed successfully!")
    logger.info(f"Best model saved at: {results.save_dir}/weights/best.pt")
    
    return results


def export_onnx(args):
    """Export trained YOLOv8 model to ONNX format for ONNX Runtime inference."""
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("Ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)
    
    weights_path = args.weights
    if not os.path.exists(weights_path):
        logger.error(f"Weights file not found: {weights_path}")
        sys.exit(1)
    
    logger.info(f"Loading model from: {weights_path}")
    model = YOLO(weights_path)
    
    # Export to ONNX
    output_path = model.export(
        format="onnx",
        imgsz=args.imgsz or 640,
        simplify=True,
        opset=12,
        dynamic=False,  # Static shapes for optimal ONNX Runtime performance
    )
    
    # Copy to standard location
    os.makedirs("outputs", exist_ok=True)
    import shutil
    final_path = os.path.join("outputs", "model.onnx")
    shutil.copy2(output_path, final_path)
    
    logger.info(f"ONNX model exported successfully!")
    logger.info(f"  Source: {output_path}")
    logger.info(f"  Copied to: {final_path}")
    logger.info(f"  Ready for ONNX Runtime inference in FastAPI backend.")
    
    return final_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SecureCoating-Vision YOLOv8 Segmentation Training & Export"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train YOLOv8 segmentation model")
    train_parser.add_argument("--model", type=str, required=True,
                             help="Explicit local weights path or exact Ultralytics model identifier")
    train_parser.add_argument("--data", type=str, default="configs/dataset.yaml",
                             help="Path to dataset YAML config")
    train_parser.add_argument("--epochs", type=int, default=50,
                             help="Number of training epochs")
    train_parser.add_argument("--batch", type=int, default=None,
                             help="Batch size (auto-detected based on VRAM)")
    train_parser.add_argument("--imgsz", type=int, default=None,
                             help="Training image size (auto-detected based on VRAM)")
    train_parser.add_argument("--device", type=str, default="auto",
                             help="Device: auto, 0, 1, cpu")
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--allow-synthetic", action="store_true",
                              help="Explicitly permit generation of synthetic development data")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export model to ONNX format")
    export_parser.add_argument("--weights", type=str, required=True,
                              help="Path to trained .pt weights file")
    export_parser.add_argument("--imgsz", type=int, default=640,
                              help="Export image size")
    
    args = parser.parse_args()
    
    if args.command == "train":
        train(args)
    elif args.command == "export":
        export_onnx(args)
    else:
        parser.print_help()
