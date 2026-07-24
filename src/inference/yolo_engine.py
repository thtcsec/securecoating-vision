"""
Ultralytics YOLOv8-seg engine — preferred GPU path on CUDA-enabled machines.
Returns the same result schema as InferenceEngine (ONNX).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger("SecureCoatingVision.YOLOEngine")


class YOLOEngine:
    CLASS_NAMES = {
        0: "scratch",
        1: "void",
        2: "blister",
        3: "delamination",
    }

    def __init__(
        self,
        model_path: str = "outputs/best.pt",
        imgsz: int = 640,
        conf_thresh: float = 0.35,
        iou_thresh: float = 0.45,
        device: str = "auto",
    ):
        self.model_path = model_path
        self.imgsz = imgsz
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = device
        self.model = None
        self._resolved_device = "cpu"
        self._model_loaded = False
        self._load_model()

    @property
    def is_loaded(self) -> bool:
        return self._model_loaded

    @property
    def active_provider(self) -> str:
        if not self._model_loaded:
            return "None"
        return f"Ultralytics/{self._resolved_device}"

    def _resolve_device(self) -> str:
        if self.device == "cpu":
            return "cpu"
        try:
            import torch

            if self.device in ("auto", "cuda") and torch.cuda.is_available():
                return "0"
        except Exception:
            pass
        if self.device == "cuda":
            logger.warning("CUDA requested for YOLO but unavailable; using CPU.")
        return "cpu"

    def _load_model(self):
        if not os.path.isfile(self.model_path):
            logger.warning(f"YOLO weights not found: {self.model_path}")
            return
        try:
            from ultralytics import YOLO

            self._resolved_device = self._resolve_device()
            self.model = YOLO(self.model_path)
            # Warmup so first user request is fast
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            self.model.predict(
                dummy,
                imgsz=self.imgsz,
                conf=self.conf_thresh,
                iou=self.iou_thresh,
                device=self._resolved_device,
                verbose=False,
            )
            self._model_loaded = True
            logger.info(
                f"YOLO engine loaded: {self.model_path} on {self.active_provider}"
            )
        except Exception as e:
            logger.warning(f"Failed to load YOLO engine: {e}")
            self.model = None
            self._model_loaded = False

    def infer(self, image: np.ndarray) -> Dict:
        start = time.time()
        if not self._model_loaded:
            h, w = image.shape[:2]
            return {
                "segmentation_mask": np.zeros((h, w), dtype=np.uint8),
                "detections": [],
                "num_defects": 0,
                "latency_ms": 0.0,
                "provider": "None",
            }

        results = self.model.predict(
            image,
            imgsz=self.imgsz,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            device=self._resolved_device,
            verbose=False,
        )
        result = results[0]
        h, w = image.shape[:2]
        seg_mask = np.zeros((h, w), dtype=np.uint8)
        detections: List[dict] = []

        boxes = result.boxes
        masks = result.masks
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            mask_data = None
            if masks is not None and masks.data is not None:
                mask_data = masks.data.cpu().numpy()

            for i in range(len(boxes)):
                class_id = int(clss[i])
                class_id_1 = class_id + 1
                conf = float(confs[i])
                box = xyxy[i].tolist()

                if mask_data is not None and i < len(mask_data):
                    m = mask_data[i]
                    m_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
                    binary = (m_resized > 0.5).astype(np.uint8)
                    seg_mask[binary == 1] = class_id_1
                else:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(seg_mask, (x1, y1), (x2, y2), class_id_1, -1)

                detections.append(
                    {
                        "box": box,
                        "confidence": conf,
                        "class_id": class_id,
                        "class_name": self.CLASS_NAMES.get(class_id, f"class_{class_id}"),
                    }
                )

        return {
            "segmentation_mask": seg_mask,
            "detections": detections,
            "num_defects": len(detections),
            "latency_ms": (time.time() - start) * 1000.0,
            "provider": self.active_provider,
        }
