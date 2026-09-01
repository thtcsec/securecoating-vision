import cv2
import numpy as np
import logging

logger = logging.getLogger("SecureCoatingVision.Postprocess")

def extract_defects_from_mask(
    seg_mask,
    height_map=None,
    pixel_to_mm_ratio=0.1,
    class_names=None,
):
    """
    Identifies defect regions from the raw segmentation mask and calculates physical size properties.
    Args:
        seg_mask (np.ndarray): Integer segmentation mask where 0=background, 1..N=defect classes.
        height_map (np.ndarray): Physical 3D heightmap aligned with the coordinate system.
        pixel_to_mm_ratio (float): Calibration scale to translate pixels into millimeters.
    Returns:
        list of dicts containing localized defect statistics.
    """
    defects = []
    num_classes = np.max(seg_mask) + 1
    
    # Class ID mapping
    default_class_map = {
        0: "background",
        1: "scratch",
        2: "void",
        3: "blister",
        4: "delamination"
    }
    if class_names:
        class_map = {0: "background"}
        class_map.update({int(class_id) + 1: str(name) for class_id, name in class_names.items()})
    else:
        class_map = default_class_map

    # Analyze classes 1 to N
    for class_id in range(1, int(num_classes)):
        class_name = class_map.get(class_id, f"unknown_{class_id}")
        binary_mask = (seg_mask == class_id).astype(np.uint8)
        
        # Find contours using OpenCV
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for idx, contour in enumerate(contours):
            area_pixels = cv2.contourArea(contour)
            if area_pixels < 5:  # Filter out tiny noise contours
                continue
                
            # Compute bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Translate pixel sizes to physical metrics (mm)
            width_mm = w * pixel_to_mm_ratio
            height_mm = h * pixel_to_mm_ratio
            area_mm2 = area_pixels * (pixel_to_mm_ratio ** 2)
            length_mm = max(width_mm, height_mm)
            
            # Fetch height offset metrics if 3D data is available
            peak_height_um = 0.0
            if height_map is not None:
                # Mask out height values inside the specific contour boundary
                mask_contour = np.zeros(seg_mask.shape, dtype=np.uint8)
                cv2.drawContours(mask_contour, [contour], -1, 1, thickness=-1)
                contour_heights = height_map[mask_contour == 1]
                if len(contour_heights) > 0:
                    peak_height_um = float(np.max(contour_heights))
                    
            defects.append({
                "defect_id": f"{class_name}_{idx}",
                "class_id": class_id,
                "class_name": class_name,
                "bbox": [x, y, w, h],
                "length_mm": float(round(length_mm, 2)),
                "width_mm": float(round(width_mm, 2)),
                "area_mm2": float(round(area_mm2, 2)),
                "peak_height_um": float(round(peak_height_um, 2))
            })
            
    return defects

def grade_coating(defects, grading_config):
    """
    Applies the industrial quality rubric thresholds to determine if the part is a PASS or FAIL.
    Args:
        defects (list): List of detected defect specifications from extract_defects_from_mask.
        grading_config (dict): Rule set specifying maximum limits.
    Returns:
        dict: Grading evaluation details (passed: bool, reject_reason: str).
    """
    violations = []
    max_reasons = 10

    for defect in defects:
        cls_name = defect["class_name"]
        limits = grading_config.get(cls_name, {})

        if limits.get("reject_any", False):
            violations.append(f"Detected {cls_name.replace('_', ' ')} requires rejection")
            continue

        if cls_name == "scratch":
            max_len = limits.get("max_allowable_length_mm", 5.0)
            if defect["length_mm"] > max_len:
                violations.append(f"Scratch length {defect['length_mm']}mm exceeds limit {max_len}mm")
        elif cls_name == "void":
            max_area = limits.get("max_allowable_area_mm2", 2.0)
            if defect["area_mm2"] > max_area:
                violations.append(f"Void area {defect['area_mm2']}mm² exceeds limit {max_area}mm²")
        elif cls_name == "blister":
            max_height = limits.get("max_allowable_height_um", 50.0)
            if defect["peak_height_um"] > max_height:
                violations.append(f"Blister height {defect['peak_height_um']}um exceeds limit {max_height}um")
        elif cls_name == "delamination":
            max_area = limits.get("max_allowable_area_mm2", 10.0)
            if defect["area_mm2"] > max_area:
                violations.append(f"Delamination area {defect['area_mm2']}mm² exceeds limit {max_area}mm²")
        else:
            violations.append(f"Unrecognized defect class '{cls_name}' requires rejection")

    total_violations = len(violations)
    reject_reasons = violations[:max_reasons]
    if total_violations > max_reasons:
        reject_reasons.append(f"... and {total_violations - max_reasons} additional violations")
    passed = total_violations == 0

    return {
        "passed": passed,
        "reject_reasons": reject_reasons,
        "total_violations": total_violations,
        "action": "PASS" if passed else "REJECT"
    }
