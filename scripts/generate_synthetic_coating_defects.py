"""
Synthetic Coating Defect Generator (YOLOv8-seg Format)

Features:
- Mask-based rendering with contour-derived polygon annotations.
- Generator retry loop (up to 20 attempts per instance) ensuring 100% class quota fulfillment.
- Single-defect placement per positive image with collision-ready mask infrastructure.
- Stratified class targeting with explicit quota verification assertions.
- Balanced preview generation (16 positive images with polygons + 4 clean negative images).
- Seed Python and NumPy RNGs for reproducible generation (--seed).
- Portable relative data.yaml configuration.
"""

import argparse
import os
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "coating_defects"

CLASSES = ["scratch", "void", "blister", "delamination"]


def set_reproducibility_seed(seed: int = 42):
    """Seed Python and NumPy RNGs for reproducible generation."""
    random.seed(seed)
    np.random.seed(seed)


def create_base_coating_texture(h: int = 640, w: int = 640, domain_shift: bool = False) -> np.ndarray:
    """Generate metallic/electrode coating background texture."""
    base_val = random.randint(110, 210) if domain_shift else random.randint(130, 190)
    img = np.full((h, w, 3), base_val, dtype=np.uint8)

    # Color tint
    tint = np.array([random.randint(-15, 8), random.randint(-8, 15), random.randint(-8, 8)])
    img = np.clip(img.astype(np.int16) + tint, 0, 255).astype(np.uint8)

    # Directional texture lines
    direction = random.choice(["horizontal", "vertical"])
    spacing = random.randint(2, 9)
    for pos in range(0, h if direction == "horizontal" else w, spacing):
        intensity = random.randint(-12, 12)
        if direction == "horizontal":
            img[pos:pos+1, :] = np.clip(img[pos:pos+1, :].astype(np.int16) + intensity, 0, 255).astype(np.uint8)
        else:
            img[:, pos:pos+1] = np.clip(img[:, pos:pos+1].astype(np.int16) + intensity, 0, 255).astype(np.uint8)

    # Lighting gradient
    grad_angle = random.uniform(0, 2 * np.pi)
    x_grad = np.linspace(-0.10, 0.10, w) * np.cos(grad_angle)
    y_grad = np.linspace(-0.10, 0.10, h) * np.sin(grad_angle)
    gradient = 1.0 + x_grad[np.newaxis, :] + y_grad[:, np.newaxis]
    for c in range(3):
        img[:, :, c] = np.clip(img[:, :, c].astype(np.float32) * gradient, 0, 255).astype(np.uint8)

    # Gaussian noise
    noise_sigma = 4.5 if domain_shift else 3.0
    noise = np.random.normal(0, noise_sigma, (h, w, 3)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    if random.random() < 0.4:
        img = cv2.GaussianBlur(img, (3, 3), 0.5)

    return img


def add_benign_artifacts(img: np.ndarray) -> np.ndarray:
    """Add non-defect surface features (lens glare, dust specs)."""
    h, w = img.shape[:2]
    if random.random() < 0.35:
        cx, cy = random.randint(50, w - 50), random.randint(50, h - 50)
        radius = random.randint(30, 110)
        overlay = img.copy()
        cv2.circle(overlay, (cx, cy), radius, (245, 245, 245), -1)
        img = cv2.addWeighted(overlay, 0.10, img, 0.90, 0)

    if random.random() < 0.45:
        for _ in range(random.randint(2, 5)):
            dx, dy = random.randint(10, w - 10), random.randint(10, h - 10)
            cv2.circle(img, (dx, dy), 1, (30, 30, 30), -1)

    return img


def extract_polygon_from_mask(mask: np.ndarray, epsilon_ratio: float = 0.001) -> Optional[List[float]]:
    """Extract normalized polygon from binary mask with contour fallback."""
    h, w = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 8.0:
        return None

    epsilon = epsilon_ratio * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    
    # Fallback to raw contour points if RDP simplification reduces vertices below 3
    pts_to_use = approx if len(approx) >= 3 else cnt

    poly_coords = []
    for pt in pts_to_use:
        px, py = pt[0]
        nx = np.clip(px / float(w), 0.0, 1.0)
        ny = np.clip(py / float(h), 0.0, 1.0)
        poly_coords.extend([round(nx, 6), round(ny, 6)])

    return poly_coords if len(poly_coords) >= 6 else None


def generate_scratch_instance(img: np.ndarray, canvas_mask: np.ndarray) -> Optional[Tuple[List[float], np.ndarray]]:
    """Generate scratch defect instance."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    num_pts = random.randint(10, 22)
    if random.random() < 0.5:
        x1, x2 = random.randint(20, w // 3), random.randint(2 * w // 3, w - 20)
        y1, y2 = random.randint(40, h - 40), random.randint(40, h - 40)
    else:
        y1, y2 = random.randint(20, h // 3), random.randint(2 * h // 3, h - 20)
        x1, x2 = random.randint(40, w - 40), random.randint(40, w - 40)

    t = np.linspace(0, 1, num_pts)
    x_pts = x1 + (x2 - x1) * t + np.random.normal(0, 2.0, num_pts)
    y_pts = y1 + (y2 - y1) * t + np.random.normal(0, 2.0, num_pts)

    pts = np.column_stack([x_pts.astype(int), y_pts.astype(int)]).reshape((-1, 1, 2))
    thickness = random.randint(3, 7)

    cv2.polylines(mask, [pts], False, 255, thickness)

    if np.any((canvas_mask > 0) & (mask > 0)):
        return None

    img[mask > 0] = (30, 30, 30)

    poly = extract_polygon_from_mask(mask)
    return (poly, mask) if poly else None


def generate_void_instance(img: np.ndarray, canvas_mask: np.ndarray) -> Optional[Tuple[List[float], np.ndarray]]:
    """Generate void instance."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    cx, cy = random.randint(50, w - 50), random.randint(50, h - 50)
    rx, ry = random.randint(12, 28), random.randint(12, 28)
    angle = random.randint(0, 180)

    num_pts = random.randint(14, 24)
    angles = np.linspace(0, 2 * np.pi, num_pts, endpoint=False)
    r_noise = np.random.uniform(0.75, 1.25, num_pts)

    pts = []
    cos_a, sin_a = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    for a, r_mod in zip(angles, r_noise):
        px = rx * r_mod * np.cos(a)
        py = ry * r_mod * np.sin(a)
        rx_rot = px * cos_a - py * sin_a + cx
        ry_rot = px * sin_a + py * cos_a + cy
        pts.append([int(rx_rot), int(ry_rot)])

    pts_arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts_arr], 255)

    if np.any((canvas_mask > 0) & (mask > 0)):
        return None

    img[mask > 0] = (20, 20, 20)

    poly = extract_polygon_from_mask(mask)
    return (poly, mask) if poly else None


def generate_blister_instance(img: np.ndarray, canvas_mask: np.ndarray) -> Optional[Tuple[List[float], np.ndarray]]:
    """Generate 3D blister instance."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    cx, cy = random.randint(60, w - 60), random.randint(60, h - 60)
    rx, ry = random.randint(18, 42), random.randint(18, 42)

    cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)

    if np.any((canvas_mask > 0) & (mask > 0)):
        return None

    y_indices, x_indices = np.where(mask > 0)
    norm_x = (x_indices - cx) / float(rx)
    norm_y = (y_indices - cy) / float(ry)
    dist = np.sqrt(norm_x**2 + norm_y**2)
    dist = np.clip(dist, 0.0, 1.0)

    height = np.sqrt(1.0 - dist**2)
    shading = (height * 80.0 + 130.0).astype(np.uint8)

    for i, (yy, xx) in enumerate(zip(y_indices, x_indices)):
        val = shading[i]
        img[yy, xx] = (val, val, val)

    poly = extract_polygon_from_mask(mask)
    return (poly, mask) if poly else None


def generate_delamination_instance(img: np.ndarray, canvas_mask: np.ndarray) -> Optional[Tuple[List[float], np.ndarray]]:
    """Generate delamination patch instance."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    cx, cy = random.randint(80, w - 80), random.randint(80, h - 80)
    size = random.randint(28, 65)

    num_verts = random.randint(7, 13)
    angles = sorted([random.uniform(0, 2 * np.pi) for _ in range(num_verts)])
    radii = [size * random.uniform(0.65, 1.35) for _ in range(num_verts)]

    pts = []
    for a, r in zip(angles, radii):
        px = int(cx + r * np.cos(a))
        py = int(cy + r * np.sin(a))
        pts.append([px, py])

    pts_arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts_arr], 255)

    if np.any((canvas_mask > 0) & (mask > 0)):
        return None

    patch_color = (random.randint(45, 75), random.randint(45, 75), random.randint(45, 75))
    img[mask > 0] = patch_color

    poly = extract_polygon_from_mask(mask)
    return (poly, mask) if poly else None


def generate_dataset(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    train_size: int = 500,
    val_size: int = 100,
    overwrite: bool = True,
    seed: int = 42,
):
    """Generate synthetic dataset with retry loop, exact class quota, and preview split."""
    set_reproducibility_seed(seed)

    images_train = output_dir / "images" / "train"
    images_val = output_dir / "images" / "val"
    labels_train = output_dir / "labels" / "train"
    labels_val = output_dir / "labels" / "val"
    previews_dir = output_dir / "previews"

    if not overwrite and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory '{output_dir}' exists and is not empty. Use --overwrite to replace."
        )

    for d in [images_train, images_val, labels_train, labels_val, previews_dir]:
        d.mkdir(parents=True, exist_ok=True)
        if overwrite:
            for f in d.glob("*.*"):
                f.unlink()

    generators = {
        0: generate_scratch_instance,
        1: generate_void_instance,
        2: generate_blister_instance,
        3: generate_delamination_instance,
    }

    print("=" * 70)
    print("Generating Synthetic Coating Defect Dataset (Strict Quota Engine)")
    print(f"Seed: {seed} | Output: {output_dir}")
    print("=" * 70)

    stats = {"train": {c: 0 for c in CLASSES}, "val": {c: 0 for c in CLASSES}, "negative": 0}
    pos_previews = 0
    neg_previews = 0

    for split, count, img_dir, lbl_dir in [
        ("train", train_size, images_train, labels_train),
        ("val", val_size, images_val, labels_val),
    ]:
        print(f"Generating {split} set ({count} samples)...")
        num_negative = int(count * 0.20)
        num_positive = count - num_negative
        domain_shift = (split == "val")

        class_allocation = [i % len(CLASSES) for i in range(num_positive)]
        random.shuffle(class_allocation)

        for i in range(count):
            img = create_base_coating_texture(domain_shift=domain_shift)
            img = add_benign_artifacts(img)
            canvas_mask = np.zeros((640, 640), dtype=np.uint8)
            label_lines: List[str] = []

            if i >= num_negative:
                target_cls = class_allocation[i - num_negative]

                res = None
                for _ in range(20):
                    cand_img = img.copy()
                    cand_mask = canvas_mask.copy()
                    res = generators[target_cls](cand_img, cand_mask)
                    if res:
                        img = cand_img
                        poly, inst_mask = res
                        canvas_mask[inst_mask > 0] = 255
                        coords_str = " ".join(f"{p:.6f}" for p in poly)
                        label_lines.append(f"{target_cls} {coords_str}")
                        stats[split][CLASSES[target_cls]] += 1
                        break

                if not res:
                    raise RuntimeError(
                        f"Failed to generate defect class {CLASSES[target_cls]} after 20 attempts!"
                    )
            else:
                stats["negative"] += 1

            img_name = f"defect_{split}_{i:05d}.jpg"
            img_path = img_dir / img_name
            cv2.imwrite(str(img_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

            lbl_name = f"defect_{split}_{i:05d}.txt"
            lbl_path = lbl_dir / lbl_name
            with open(lbl_path, "w") as f:
                if label_lines:
                    f.write("\n".join(label_lines) + "\n")

            if label_lines and pos_previews < 16:
                preview_img = img.copy()
                for line in label_lines:
                    parts = line.split()
                    cls_id = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                    pts = np.array(coords).reshape((-1, 2))
                    pts[:, 0] *= 640
                    pts[:, 1] *= 640
                    pts_int = pts.astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(preview_img, [pts_int], True, (0, 255, 0), 2)
                    cv2.putText(
                        preview_img,
                        CLASSES[cls_id],
                        (int(pts[0][0]), int(pts[0][1]) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                    )
                cv2.imwrite(str(previews_dir / f"pos_preview_{pos_previews:02d}.jpg"), preview_img)
                pos_previews += 1
            elif not label_lines and neg_previews < 4:
                cv2.imwrite(str(previews_dir / f"neg_preview_{neg_previews:02d}.jpg"), img)
                neg_previews += 1

        actual_positives = sum(stats[split].values())
        if actual_positives != num_positive:
            raise RuntimeError(
                f"Quota mismatch in {split}: Expected {num_positive} positives, generated {actual_positives}"
            )

    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write("# Ultralytics YOLOv8 Dataset Config\n")
        f.write("path: .\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n\n")
        f.write("names:\n")
        for idx, name in enumerate(CLASSES):
            f.write(f"  {idx}: {name}\n")

    print("\nDataset Generation Complete!")
    print(f"  - Config written: {yaml_path}")
    print(f"  - Previews saved: {previews_dir} (16 positive overlays + 4 clean negatives)")
    print(f"  - Class Stats (Train): {stats['train']}")
    print(f"  - Class Stats (Val):   {stats['val']}")
    print(f"  - Negative Baseline Samples: {stats['negative']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Synthetic Coating Defect Dataset")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--train-size", type=int, default=500, help="Train split size")
    parser.add_argument("--val-size", type=int, default=100, help="Val split size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--no-overwrite", action="store_true", help="Do not overwrite existing files")
    args = parser.parse_args()

    generate_dataset(
        output_dir=Path(args.output_dir),
        train_size=args.train_size,
        val_size=args.val_size,
        overwrite=not args.no_overwrite,
        seed=args.seed,
    )
