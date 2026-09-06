import logging
import os
import time
from typing import Dict, List

import cv2
import numpy as np

logger = logging.getLogger("securecoating.yolo_engine")


class YOLOEngine:
    """
    Ultralytics YOLO inference wrapper for battery coating defect inspection.
    Provides segmentation masks and bounding box detections.
    """

    def __init__(
        self,
        model_path: str = "outputs/best.pt",
        imgsz: int = 640,
        conf_thresh: float = 0.35,
        iou_thresh: float = 0.45,
        device: str = "auto",
        precision: str = "fp32",
        num_classes: int = 4,
    ):
        self.model_path = model_path
        self.imgsz = imgsz
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = device
        self.precision = precision.lower()
        if self.precision not in {"fp32", "fp16"}:
            raise ValueError("YOLO precision must be fp32 or fp16")
        self.num_classes = int(num_classes)
        self.model = None
        self._model_loaded = False
        self.class_names: Dict[int, str] = {
            0: "scratch",
            1: "void",
            2: "blister",
            3: "delamination",
        }
        self._load_model()

    @property
    def is_loaded(self) -> bool:
        return self._model_loaded

    @property
    def active_provider(self) -> str:
        if not self._model_loaded:
            return "None"
        try:
            import torch
            if torch.cuda.is_available() and self._resolved_device not in ("cpu", "CPU"):
                return f"CUDA:{torch.cuda.current_device()} ({torch.cuda.get_device_name(0)})"
            return "CPU"
        except Exception:
            return "CPU"

    @property
    def active_precision(self) -> str:
        if self._model_loaded and self._resolved_device not in ("cpu", "CPU"):
            return self.precision
        return "fp32"

    def _load_model(self):
        if not os.path.exists(self.model_path):
            logger.warning(
                f"YOLO model weights not found at: {self.model_path}. Engine inactive."
            )
            return

        try:
            from ultralytics import YOLO
            import torch

            if self.device == "auto":
                self._resolved_device = "0" if torch.cuda.is_available() else "cpu"
            else:
                self._resolved_device = self.device

            self.model = YOLO(self.model_path)
            if hasattr(self.model, "names") and self.model.names:
                self.class_names = {
                    int(class_id): str(class_name)
                    for class_id, class_name in self.model.names.items()
                }
            if len(self.class_names) != self.num_classes:
                raise RuntimeError(
                    f"YOLO artifact exposes {len(self.class_names)} classes; "
                    f"expected {self.num_classes}"
                )
            # Warmup so first user request is fast
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            self.model.predict(
                dummy,
                imgsz=self.imgsz,
                conf=self.conf_thresh,
                iou=self.iou_thresh,
                device=self._resolved_device,
                half=self.precision == "fp16" and self._resolved_device not in ("cpu", "CPU"),
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
            half=self.precision == "fp16" and self._resolved_device not in ("cpu", "CPU"),
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

                binary = None
                if mask_data is not None and i < len(mask_data):
                    m = mask_data[i]
                    m_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
                    binary = (m_resized > 0.5).astype(np.uint8)
                    seg_mask[binary == 1] = class_id_1
                else:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(seg_mask, (x1, y1), (x2, y2), class_id_1, -1)
                    binary = np.zeros((h, w), dtype=np.uint8)
                    binary[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = 1

                detections.append(
                    {
                        "box": box,
                        "box_format": "xyxy",
                        "confidence": conf,
                        "class_id": class_id,
                        "class_name": self.class_names.get(class_id, f"class_{class_id}"),
                        "mask": binary,
                    }
                )

        return {
            "segmentation_mask": seg_mask,
            "detections": detections,
            "num_defects": len(detections),
            "latency_ms": (time.time() - start) * 1000.0,
            "provider": self.active_provider,
        }
