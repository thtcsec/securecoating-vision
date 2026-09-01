"""
SecureCoating-Vision: ONNX Runtime Inference Engine
====================================================
ONNX Runtime engine for the configured post-NMS detector output, with backward
compatibility for the repository's legacy synthetic YOLO segmentation format.

Designed for production deployment with:
- CUDAExecutionProvider for GPU acceleration
- Automatic fallback to CPU if GPU unavailable
- Postprocessing for post-NMS detection and legacy segmentation outputs
- Thread-safe singleton pattern for multi-worker FastAPI
"""

import os
import time
import logging
import numpy as np
import cv2
from typing import Optional, Tuple, Dict, List

logger = logging.getLogger("SecureCoatingVision.ONNXEngine")


class InferenceEngine:
    """
    ONNX Runtime inference engine for detection and legacy segmentation models.
    
    Loads a pre-exported .onnx model and provides:
    - Image preprocessing (resize, normalize, pad)
    - Model inference via ONNX Runtime (GPU/CPU)
    - Postprocessing: detection parsing or legacy mask extraction
    
    Usage:
        engine = InferenceEngine("outputs/model.onnx", imgsz=512, conf_thresh=0.5)
        result = engine.infer(bgr_image)
        # result["segmentation_mask"] -> np.ndarray (H, W) with class IDs
    """

    # Class-level mapping for defect classes
    CLASS_NAMES = {
        0: "scratch",
        1: "void",
        2: "blister",
        3: "delamination"
    }

    def __init__(
        self,
        model_path: str = "outputs/model.onnx",
        imgsz: int = 640,
        conf_thresh: float = 0.50,
        iou_thresh: float = 0.45,
        device: str = "auto",
        num_classes: int = 4,
        class_names: Optional[Dict[int, str]] = None,
    ):
        """
        Initialize the ONNX inference engine.
        
        Args:
            model_path: Path to the ONNX model file.
            imgsz: Input image size the model expects.
            conf_thresh: Confidence threshold for detection filtering.
            iou_thresh: IoU threshold for Non-Maximum Suppression.
            device: 'auto' (try GPU then CPU), 'cuda', or 'cpu'.
        """
        self.model_path = model_path
        self.imgsz = imgsz
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = device
        self.num_classes = int(num_classes)
        if self.num_classes <= 0:
            raise ValueError("num_classes must be positive")
        self.class_names = dict(class_names or self.CLASS_NAMES)
        self.session = None
        self.input_name = None
        self.output_names = None
        self._model_loaded = False

        self._load_model()

    def _load_model(self):
        """Load ONNX model with appropriate execution provider."""
        try:
            import onnxruntime as ort
        except ImportError:
            logger.error("onnxruntime not installed. Install with: pip install onnxruntime (or pip install onnxruntime-gpu for CUDA)")
            raise ImportError("onnxruntime is required for ONNX inference. Install with: pip install onnxruntime")

        if not os.path.exists(self.model_path):
            logger.warning(
                f"ONNX model not found at '{self.model_path}'. "
                "Engine will operate in mock mode until model is available."
            )
            self._model_loaded = False
            return

        # Configure execution providers based on device setting
        providers = self._get_providers()

        logger.info(f"Loading ONNX model: {self.model_path}")
        logger.info(f"Requested providers: {providers}")

        # Session options for performance
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4
        sess_options.inter_op_num_threads = 2

        try:
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=sess_options,
                providers=providers
            )
        except Exception as e:
            logger.warning(f"Failed to load with preferred providers: {e}")
            logger.info("Falling back to CPUExecutionProvider...")
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=sess_options,
                providers=["CPUExecutionProvider"]
            )

        # Cache input/output metadata
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        outputs = self.session.get_outputs()
        if len(outputs) >= 2:
            det_shape = outputs[0].shape
            proto_shape = outputs[1].shape
            if (
                len(det_shape) == 3 and len(proto_shape) == 4
                and isinstance(det_shape[1], int) and isinstance(proto_shape[1], int)
            ):
                inferred_classes = det_shape[1] - 4 - proto_shape[1]
                if inferred_classes != self.num_classes:
                    raise RuntimeError(
                        "ONNX output schema/class mismatch: "
                        f"artifact={inferred_classes}, configured={self.num_classes}"
                    )
        elif len(outputs) == 1:
            output_shape = outputs[0].shape
            if len(output_shape) == 3:
                feature_dims = [dim for dim in output_shape[1:] if isinstance(dim, int)]
                expected = 4 + self.num_classes
                # Width 6 is the post-NMS export contract (xyxy, score, class)
                # and does not encode the class count in the feature width.
                if 6 not in feature_dims and expected not in feature_dims:
                    raise RuntimeError(
                        "ONNX detection output schema/class mismatch: "
                        f"shape={output_shape}, expected_feature_width={expected}"
                    )
        self._model_loaded = True

        active_provider = self.session.get_providers()[0]
        logger.info(f"ONNX model loaded successfully on: {active_provider}")
        logger.info(f"  Input: {self.input_name} shape={self.session.get_inputs()[0].shape}")
        logger.info(f"  Outputs: {self.output_names}")

    def _get_providers(self) -> List[str]:
        """Determine execution providers based on device setting."""
        import onnxruntime as ort
        available = ort.get_available_providers()

        if self.device == "cuda" or self.device == "auto":
            if "CUDAExecutionProvider" in available:
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]
            elif "TensorrtExecutionProvider" in available:
                return ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                if self.device == "cuda":
                    logger.warning("CUDA requested but not available. Falling back to CPU.")
                return ["CPUExecutionProvider"]
        else:
            return ["CPUExecutionProvider"]

    @property
    def is_loaded(self) -> bool:
        """Check if a real ONNX model is loaded."""
        return self._model_loaded

    @property
    def active_provider(self) -> str:
        """Return the currently active execution provider."""
        if self.session:
            return self.session.get_providers()[0]
        return "None (Mock Mode)"

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Preprocess image for YOLOv8 inference.
        
        Applies letterbox resize, BGR->RGB conversion, normalization to [0,1],
        and HWC->CHW transpose.
        
        Args:
            image: Input BGR image (H, W, 3) uint8.
            
        Returns:
            Tuple of (preprocessed tensor [1,3,H,W] float32, preprocessing metadata)
        """
        orig_h, orig_w = image.shape[:2]

        # Letterbox resize maintaining aspect ratio
        scale = min(self.imgsz / orig_h, self.imgsz / orig_w)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create padded canvas
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, dtype=np.uint8)
        pad_top = (self.imgsz - new_h) // 2
        pad_left = (self.imgsz - new_w) // 2
        canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

        # BGR -> RGB, normalize to [0, 1], transpose to CHW
        blob = canvas[:, :, ::-1].astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)  # HWC -> CHW
        blob = np.expand_dims(blob, axis=0)  # Add batch dim -> (1, 3, H, W)
        blob = np.ascontiguousarray(blob)

        meta = {
            "orig_shape": (orig_h, orig_w),
            "scale": scale,
            "pad_top": pad_top,
            "pad_left": pad_left,
            "new_shape": (new_h, new_w),
        }

        return blob, meta

    def postprocess(
        self,
        outputs: List[np.ndarray],
        meta: dict,
        num_classes: Optional[int] = None
    ) -> Dict:
        """
        Postprocess legacy synthetic YOLO segmentation outputs into masks.
        
        Legacy segmentation outputs:
        - output0: Detection predictions (1, num_dets, 4+nc+nm) - boxes + class scores + mask coefficients
        - output1: Prototype masks (1, nm, mask_h, mask_w)
        
        Args:
            outputs: Raw ONNX session outputs.
            meta: Preprocessing metadata from preprocess().
            num_classes: Number of defect classes.
            
        Returns:
            Dict with segmentation_mask, detections, class_scores.
        """
        orig_h, orig_w = meta["orig_shape"]
        num_classes = self.num_classes if num_classes is None else int(num_classes)

        # Handle the retained legacy synthetic segmentation output format.
        if len(outputs) >= 2:
            # Legacy dual output: [detection_output, proto_masks]
            det_output = outputs[0]  # (1, 4+nc+nm, num_anchors) or transposed
            proto_masks = outputs[1]  # (1, nm, mask_h, mask_w)
            return self._process_yolo_seg(det_output, proto_masks, meta, num_classes)
        elif len(outputs) == 1:
            # Single output - might be detection-only or custom format
            return self._process_single_output(outputs[0], meta, num_classes)
        else:
            logger.warning("Unexpected output count. Returning empty mask.")
            return self._empty_result(orig_h, orig_w)

    def _process_yolo_seg(
        self,
        det_output: np.ndarray,
        proto_masks: np.ndarray,
        meta: dict,
        num_classes: int
    ) -> Dict:
        """Process the retained legacy synthetic dual-output format."""
        orig_h, orig_w = meta["orig_shape"]
        pad_top, pad_left = meta["pad_top"], meta["pad_left"]
        scale = meta["scale"]

        # det_output shape: (1, 4+nc+nm, num_anchors) -> transpose to (num_anchors, 4+nc+nm)
        if det_output.ndim == 3:
            preds = det_output[0].T  # (num_anchors, 4+nc+nm)
        else:
            preds = det_output

        # Split: boxes (4) + class_scores (nc) + mask_coefficients (nm)
        nm = proto_masks.shape[1] if proto_masks.ndim == 4 else 32
        inferred_classes = preds.shape[1] - 4 - nm
        if inferred_classes != num_classes:
            raise ValueError(
                "YOLO segmentation output class mismatch: "
                f"artifact={inferred_classes}, configured={num_classes}"
            )
        boxes = preds[:, :4]  # cx, cy, w, h
        class_scores = preds[:, 4:4 + num_classes]
        mask_coeffs = preds[:, 4 + num_classes:4 + num_classes + nm]

        # Get confidence and class per detection
        confidences = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)

        # Filter by confidence
        mask = confidences > self.conf_thresh
        if not np.any(mask):
            return self._empty_result(orig_h, orig_w)

        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]
        mask_coeffs = mask_coeffs[mask]

        # Convert cx,cy,w,h to x1,y1,x2,y2
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        xyxy_boxes = np.stack([x1, y1, x2, y2], axis=1)

        # Class-aware NMS: overlapping defects of different classes must not
        # suppress each other.
        keep_indices = []
        for class_id in np.unique(class_ids):
            class_indices = np.where(class_ids == class_id)[0]
            class_keep = self._nms(
                xyxy_boxes[class_indices], confidences[class_indices], self.iou_thresh
            )
            keep_indices.extend(class_indices[class_keep].tolist())
        keep_indices = sorted(keep_indices, key=lambda idx: confidences[idx], reverse=True)
        if len(keep_indices) == 0:
            return self._empty_result(orig_h, orig_w)

        final_boxes = xyxy_boxes[keep_indices]
        final_scores = confidences[keep_indices]
        final_classes = class_ids[keep_indices]
        final_coeffs = mask_coeffs[keep_indices]

        # Generate instance masks from prototype masks
        # proto shape: (1, nm, mask_h, mask_w) -> (nm, mask_h, mask_w)
        protos = proto_masks[0] if proto_masks.ndim == 4 else proto_masks
        mask_h, mask_w = protos.shape[1], protos.shape[2]

        # Matrix multiply: (num_dets, nm) @ (nm, mask_h*mask_w) -> (num_dets, mask_h*mask_w)
        protos_flat = protos.reshape(nm, -1)  # (nm, mask_h * mask_w)
        instance_masks = final_coeffs @ protos_flat  # (num_dets, mask_h*mask_w)
        instance_masks = self._sigmoid(instance_masks)
        instance_masks = instance_masks.reshape(-1, mask_h, mask_w)

        # Build full segmentation mask at original resolution
        seg_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        detections = []

        for i in range(len(final_boxes)):
            # Resize mask to input size then crop to original
            mask_resized = cv2.resize(instance_masks[i], (self.imgsz, self.imgsz))

            # YOLO instance masks are valid only inside their detection box.
            # Crop in the letterboxed input coordinate system before removing padding.
            input_box = final_boxes[i]
            bx1 = max(0, min(self.imgsz, int(np.floor(input_box[0]))))
            by1 = max(0, min(self.imgsz, int(np.floor(input_box[1]))))
            bx2 = max(0, min(self.imgsz, int(np.ceil(input_box[2]))))
            by2 = max(0, min(self.imgsz, int(np.ceil(input_box[3]))))
            box_crop = np.zeros_like(mask_resized, dtype=np.float32)
            if bx2 > bx1 and by2 > by1:
                box_crop[by1:by2, bx1:bx2] = 1.0
            mask_resized *= box_crop

            # Remove padding
            nh, nw = meta["new_shape"]
            mask_cropped = mask_resized[pad_top:pad_top + nh, pad_left:pad_left + nw]

            # Resize to original image size
            mask_orig = cv2.resize(mask_cropped, (orig_w, orig_h))

            # Threshold mask
            binary_mask = (mask_orig > 0.5).astype(np.uint8)

            # Apply class to segmentation mask (later detections override earlier ones)
            # Class IDs in seg_mask are 1-indexed (0 = background)
            class_id_1indexed = int(final_classes[i]) + 1
            seg_mask[(binary_mask == 1) & (seg_mask == 0)] = class_id_1indexed

            # Scale box to original coordinates
            box = final_boxes[i].copy()
            box[0] = (box[0] - pad_left) / scale
            box[1] = (box[1] - pad_top) / scale
            box[2] = (box[2] - pad_left) / scale
            box[3] = (box[3] - pad_top) / scale
            box = np.clip(box, 0, [orig_w, orig_h, orig_w, orig_h])

            detections.append({
                "box": box.tolist(),
                "box_format": "xyxy",
                "confidence": float(final_scores[i]),
                "class_id": int(final_classes[i]),
                "class_name": getattr(self, "class_names", self.CLASS_NAMES).get(
                    int(final_classes[i]), f"class_{final_classes[i]}"
                ),
                "mask": binary_mask,
            })

        return {
            "segmentation_mask": seg_mask,
            "detections": detections,
            "num_defects": len(detections),
        }

    def _process_single_output(self, output: np.ndarray, meta: dict, num_classes: int) -> Dict:
        """Handle single-output models (detection only, no mask prototypes)."""
        orig_h, orig_w = meta["orig_shape"]

        # Attempt to interpret as semantic segmentation direct output
        if output.ndim == 4 and output.shape[1] == num_classes + 1:
            # Direct semantic segmentation: (1, nc+1, H, W)
            seg_logits = output[0]  # (nc+1, H, W)
            seg_mask = np.argmax(seg_logits, axis=0).astype(np.uint8)

            # Resize to original
            seg_mask = cv2.resize(seg_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            return {
                "segmentation_mask": seg_mask,
                "detections": [],
                "num_defects": int(np.max(seg_mask) > 0),
            }

        # Ultralytics detection export: (1, 4+nc, anchors), with boxes in
        # letterboxed cx/cy/w/h coordinates and per-class confidence scores.
        if output.ndim == 3:
            raw = output[0]
            expected_width = 4 + num_classes
            # Current YOLO26 detection exports return a bounded post-NMS tensor
            # shaped (1, 300, 6): x1, y1, x2, y2, confidence, class_id.
            if (
                raw.shape[1] == 6
                and raw.shape[0] <= 1000
                and np.allclose(raw[:, 5], np.round(raw[:, 5]), atol=1e-4)
            ):
                selected = raw[:, 4] > self.conf_thresh
                rows = raw[selected]
                scale = float(meta["scale"])
                pad_top = float(meta["pad_top"])
                pad_left = float(meta["pad_left"])
                seg_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
                detections = []
                for row in rows:
                    class_id = int(round(float(row[5])))
                    if not (0 <= class_id < num_classes):
                        continue
                    box = row[:4].astype(np.float32).copy()
                    box[[0, 2]] = (box[[0, 2]] - pad_left) / scale
                    box[[1, 3]] = (box[[1, 3]] - pad_top) / scale
                    box = np.clip(box, 0, [orig_w, orig_h, orig_w, orig_h])
                    x1, y1, x2, y2 = [int(round(value)) for value in box]
                    if x2 <= x1 or y2 <= y1:
                        continue
                    binary_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
                    binary_mask[y1:y2, x1:x2] = 1
                    seg_mask[(binary_mask == 1) & (seg_mask == 0)] = class_id + 1
                    detections.append(
                        {
                            "box": box.tolist(),
                            "box_format": "xyxy",
                            "confidence": float(row[4]),
                            "class_id": class_id,
                            "class_name": getattr(self, "class_names", self.CLASS_NAMES).get(
                                class_id, f"class_{class_id}"
                            ),
                            "mask": binary_mask,
                        }
                    )
                return {
                    "segmentation_mask": seg_mask,
                    "detections": detections,
                    "num_defects": len(detections),
                }

            if raw.shape[0] == expected_width:
                preds = raw.T
            elif raw.shape[1] == expected_width:
                preds = raw
            else:
                return self._empty_result(orig_h, orig_w)

            boxes = preds[:, :4]
            class_scores = preds[:, 4:4 + num_classes]
            confidences = np.max(class_scores, axis=1)
            class_ids = np.argmax(class_scores, axis=1)
            selected = confidences > self.conf_thresh
            if not np.any(selected):
                return self._empty_result(orig_h, orig_w)
            boxes = boxes[selected]
            confidences = confidences[selected]
            class_ids = class_ids[selected]

            xyxy = np.stack(
                [
                    boxes[:, 0] - boxes[:, 2] / 2,
                    boxes[:, 1] - boxes[:, 3] / 2,
                    boxes[:, 0] + boxes[:, 2] / 2,
                    boxes[:, 1] + boxes[:, 3] / 2,
                ],
                axis=1,
            )
            keep_indices = []
            for class_id in np.unique(class_ids):
                class_indices = np.where(class_ids == class_id)[0]
                class_keep = self._nms(
                    xyxy[class_indices], confidences[class_indices], self.iou_thresh
                )
                keep_indices.extend(class_indices[class_keep].tolist())
            keep_indices = sorted(
                keep_indices, key=lambda index: confidences[index], reverse=True
            )

            scale = float(meta["scale"])
            pad_top = float(meta["pad_top"])
            pad_left = float(meta["pad_left"])
            seg_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            detections = []
            for index in keep_indices:
                box = xyxy[index].copy()
                box[[0, 2]] = (box[[0, 2]] - pad_left) / scale
                box[[1, 3]] = (box[[1, 3]] - pad_top) / scale
                box = np.clip(box, 0, [orig_w, orig_h, orig_w, orig_h])
                x1, y1, x2, y2 = [int(round(value)) for value in box]
                if x2 <= x1 or y2 <= y1:
                    continue
                binary_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
                binary_mask[y1:y2, x1:x2] = 1
                class_id = int(class_ids[index])
                seg_mask[(binary_mask == 1) & (seg_mask == 0)] = class_id + 1
                detections.append(
                    {
                        "box": box.tolist(),
                        "box_format": "xyxy",
                        "confidence": float(confidences[index]),
                        "class_id": class_id,
                        "class_name": getattr(self, "class_names", self.CLASS_NAMES).get(
                            class_id, f"class_{class_id}"
                        ),
                        "mask": binary_mask,
                    }
                )
            return {
                "segmentation_mask": seg_mask,
                "detections": detections,
                "num_defects": len(detections),
            }

        # Default: return empty
        return self._empty_result(orig_h, orig_w)

    def _empty_result(self, h: int, w: int) -> Dict:
        """Return an empty detection result."""
        return {
            "segmentation_mask": np.zeros((h, w), dtype=np.uint8),
            "detections": [],
            "num_defects": 0,
        }

    def _mock_inference(self, image: np.ndarray) -> Dict:
        """
        Mock inference when no real model is loaded.
        Returns a realistic-looking but randomized segmentation for demo purposes.
        """
        h, w = image.shape[:2]
        seg_mask = np.zeros((h, w), dtype=np.uint8)

        # Occasionally produce a mock defect (30% chance)
        if np.random.random() < 0.3:
            class_id = np.random.randint(1, self.num_classes + 1)
            cx, cy = np.random.randint(w // 4, 3 * w // 4), np.random.randint(h // 4, 3 * h // 4)
            radius = np.random.randint(30, 100)
            cv2.circle(seg_mask, (cx, cy), radius, int(class_id), -1)

        return {
            "segmentation_mask": seg_mask,
            "detections": [],
            "num_defects": int(np.max(seg_mask) > 0),
        }

    def infer(self, image: np.ndarray) -> Dict:
        """
        Run full inference pipeline on a single BGR image.
        
        Args:
            image: Input BGR image (H, W, 3) uint8.
            
        Returns:
            Dict with:
                - segmentation_mask: (H, W) uint8 array, 0=background, 1-4=defect classes
                - detections: List of detection dicts with box, confidence, class info
                - num_defects: Number of detected defects
                - latency_ms: Inference time in milliseconds
                - provider: Active execution provider
        """
        start_time = time.time()

        if not self._model_loaded:
            raise RuntimeError("ONNX inference requested without a loaded model artifact")

        # Preprocess
        blob, meta = self.preprocess(image)

        # Run ONNX inference
        outputs = self.session.run(self.output_names, {self.input_name: blob})

        # Postprocess
        result = self.postprocess(outputs, meta)
        result["latency_ms"] = (time.time() - start_time) * 1000.0
        result["provider"] = self.active_provider

        return result

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
        """
        Non-Maximum Suppression implementation.
        
        Args:
            boxes: (N, 4) array of [x1, y1, x2, y2] boxes.
            scores: (N,) array of confidence scores.
            iou_threshold: IoU threshold for suppression.
            
        Returns:
            List of indices to keep.
        """
        if len(boxes) == 0:
            return []

        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        order = scores.argsort()[::-1]
        keep = []

        while len(order) > 0:
            i = order[0]
            keep.append(i)

            if len(order) == 1:
                break

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            inter_w = np.maximum(0, xx2 - xx1)
            inter_h = np.maximum(0, yy2 - yy1)
            intersection = inter_w * inter_h

            union = areas[i] + areas[order[1:]] - intersection
            iou = intersection / (union + 1e-6)

            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return keep
