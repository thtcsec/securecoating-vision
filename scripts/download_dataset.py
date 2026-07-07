"""
Download a real surface defect detection dataset from Roboflow Universe.
This uses the public NEU Steel Surface Defect dataset in YOLOv8 segmentation format.

Usage:
    python scripts/download_dataset.py

The dataset will be placed in data/coating_defects/ ready for training.
If you have a Roboflow API key, set it as ROBOFLOW_API_KEY env variable.
Otherwise, you can download manually from:
    https://universe.roboflow.com/surfacedefectdetectiondataset/surface-defect-6cbia

Alternative free datasets for coating/surface inspection:
    - NEU-DET: https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database
    - MVTec AD: https://www.mvtec.com/company/research/datasets/mvtec-ad
"""

import os
import sys
import shutil

def download_from_roboflow():
    """Download using Roboflow Python SDK."""
    try:
        from roboflow import Roboflow
    except ImportError:
        print("Install roboflow first: pip install roboflow")
        sys.exit(1)

    # Use API key from env or prompt
    api_key = os.environ.get("ROBOFLOW_API_KEY", "")
    if not api_key:
        print("=" * 60)
        print("  ROBOFLOW DATASET DOWNLOAD")
        print("=" * 60)
        print()
        print("To download, you need a free Roboflow API key:")
        print("  1. Go to https://app.roboflow.com/ and sign up (free)")
        print("  2. Go to Settings -> API Key")
        print("  3. Copy your API key")
        print()
        api_key = input("Paste your Roboflow API key (or press Enter to skip): ").strip()
        
        if not api_key:
            print("\nSkipped. Using synthetic data instead.")
            print("You can also download datasets manually from:")
            print("  https://universe.roboflow.com/surfacedefectdetectiondataset/surface-defect-6cbia")
            return False

    rf = Roboflow(api_key=api_key)
    
    # Surface Defect dataset (349 images, 4 classes, YOLOv8 seg format)
    print("Downloading Surface Defect dataset from Roboflow Universe...")
    project = rf.workspace("surfacedefectdetectiondataset").project("surface-defect-6cbia")
    version = project.version(2)
    dataset = version.download("yolov8", location="data/roboflow_download")
    
    print(f"Dataset downloaded to: {dataset.location}")
    
    # Move to expected structure
    src = dataset.location
    dst = "data/coating_defects"
    
    # Copy images and labels to our expected structure
    for split in ["train", "valid", "test"]:
        src_split = os.path.join(src, split)
        if not os.path.exists(src_split):
            continue
            
        # Map 'valid' -> 'val'
        dst_split = "val" if split == "valid" else split
        
        img_dst = os.path.join(dst, "images", dst_split)
        lbl_dst = os.path.join(dst, "labels", dst_split)
        os.makedirs(img_dst, exist_ok=True)
        os.makedirs(lbl_dst, exist_ok=True)
        
        # Copy images
        img_src = os.path.join(src_split, "images")
        if os.path.exists(img_src):
            for f in os.listdir(img_src):
                shutil.copy2(os.path.join(img_src, f), os.path.join(img_dst, f))
                
        # Copy labels
        lbl_src = os.path.join(src_split, "labels")
        if os.path.exists(lbl_src):
            for f in os.listdir(lbl_src):
                shutil.copy2(os.path.join(lbl_src, f), os.path.join(lbl_dst, f))

    # Count files
    train_imgs = len(os.listdir(os.path.join(dst, "images", "train"))) if os.path.exists(os.path.join(dst, "images", "train")) else 0
    val_imgs = len(os.listdir(os.path.join(dst, "images", "val"))) if os.path.exists(os.path.join(dst, "images", "val")) else 0
    
    print(f"\nDataset ready at: {dst}")
    print(f"  Train images: {train_imgs}")
    print(f"  Val images: {val_imgs}")
    print(f"\nTo train: python src/training/train_yolo.py train --epochs 50")
    
    return True


if __name__ == "__main__":
    download_from_roboflow()
