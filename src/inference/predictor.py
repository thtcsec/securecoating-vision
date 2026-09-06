import os
import time
import logging
import numpy as np
import cv2
import hashlib
import re
import torch
import torch.nn as nn

from inference.hardware_profile import resolve_inference_profile

logger = logging.getLogger("SecureCoatingVision.Predictor")

# Project root (…/securecoating-vision), independent of process cwd
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SimpleFusionNetwork(nn.Module):
    """
    Lightweight 5-channel fusion network (RGB + Thermal + Height).
    Used only when YOLO / ONNX engines are unavailable.
    """

    def __init__(self, in_channels=5, num_classes=5):
        super(SimpleFusionNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, num_classes),
        )
        self.segmentor = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

    def forward(self, x):
        x = self.relu(self.conv1(x))
        features = self.relu(self.conv2(x))
        pooled = self.pool(features)
        cls_out = self.classifier(pooled)
        seg_out = self.segmentor(pooled)
        return cls_out, seg_out


class CoatingPredictor:
    """
    Inference priority:
      1. Configured Ultralytics detector (.pt) on CUDA — primary runtime
      2. Matching ONNX detector — portable / submission path
      3. PyTorch SimpleFusionNetwork stub
    """

    def __init__(self, model_config):
        self.config = model_config

        self.profile = resolve_inference_profile(self.config)
        self.device = torch.device(self.profile.device)
        self.requested_device = self.profile.device
        self.imgsz = self.profile.imgsz
        self.requested_precision = self.profile.requested_precision

        self.num_classes = int(self.config.get("model", {}).get("num_classes", 5))
        self.artifact_num_classes = int(
            self.config.get("model", {}).get("artifact_num_classes", self.num_classes - 1)
        )
        if self.artifact_num_classes <= 0:
            raise ValueError("model.artifact_num_classes must be positive")
        raw_names = self.config.get("model", {}).get("class_names", {}) or {}
        self.class_names = {int(key): str(value) for key, value in raw_names.items()}
        self.model_version = str(self.config.get("model", {}).get("version", "2.0.0"))
        self.model = SimpleFusionNetwork(in_channels=5, num_classes=self.num_classes)
        self.model.to(self.device)
        self.model.eval()

        self.yolo_engine = None
        self.onnx_engine = None
        for engine_name in self.profile.engine_priority:
            if engine_name == "yolo":
                self._init_yolo_engine()
            else:
                self._init_onnx_engine()

        self.h_thermal_to_rgb = np.eye(3, dtype=np.float32)
        self.h_depth_to_rgb = np.eye(3, dtype=np.float32)

        logger.info(f"CoatingPredictor initialized on device: {self.device}")
        if self.yolo_available:
            logger.info(
                f"Primary engine: {self.yolo_engine.active_provider} "
                f"(model v{self.model_version})"
            )
        elif self.onnx_available:
            logger.info(
                f"Primary engine: ONNX {self.onnx_engine.active_provider} "
                f"(model v{self.model_version})"
            )
        else:
            logger.warning(
                "No YOLO/ONNX weights loaded — using PyTorch fusion fallback. "
                "Place outputs/best.pt or outputs/model.onnx."
            )

    @property
    def yolo_available(self) -> bool:
        return self.yolo_engine is not None and self.yolo_engine.is_loaded

    @property
    def onnx_available(self) -> bool:
        return self.onnx_engine is not None and self.onnx_engine.is_loaded

    def _resolve_path(self, configured: str, defaults: list) -> str:
        candidates = [
            configured,
            os.path.join(_PROJECT_ROOT, configured) if configured else None,
            os.path.join(os.getcwd(), configured) if configured else None,
        ]
        candidates.extend(defaults)
        for path in candidates:
            if path and os.path.isfile(path):
                return os.path.abspath(path)
        return os.path.abspath(os.path.join(_PROJECT_ROOT, configured or defaults[0]))

    def _verify_artifact(self, path: str, config_key: str) -> None:
        expected = str(self.config.get("model", {}).get(config_key, "")).lower()
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise RuntimeError(f"model.{config_key} must contain a SHA-256 digest")
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise RuntimeError(f"Model artifact hash mismatch: {path}")

    def _init_yolo_engine(self):
        try:
            from inference.yolo_engine import YOLOEngine

            weights = self.config.get("model", {}).get("weights_path", "outputs/best.pt")
            path = self._resolve_path(
                weights,
                [
                    os.path.join(_PROJECT_ROOT, "outputs", "best.pt"),
                    os.path.join(
                        _PROJECT_ROOT,
                        "runs",
                        "segment",
                        "outputs",
                        "yolo_training",
                        "yolov8n-seg_50ep",
                        "weights",
                        "best.pt",
                    ),
                ],
            )
            self._verify_artifact(path, "weights_sha256")
            conf = self.config.get("inference", {}).get("confidence_threshold", 0.35)
            iou = self.config.get("inference", {}).get("nms_threshold", 0.45)
            imgsz = self.imgsz
            self.yolo_engine = YOLOEngine(
                model_path=path,
                imgsz=imgsz,
                conf_thresh=conf,
                iou_thresh=iou,
                device=self.requested_device,
                precision=self.requested_precision,
                num_classes=self.artifact_num_classes,
            )
        except Exception as e:
            logger.info(f"YOLO engine not available: {e}")
            self.yolo_engine = None

    def _init_onnx_engine(self):
        try:
            from inference.onnx_engine import InferenceEngine

            onnx_path = self._resolve_path(
                self.config.get("model", {}).get("onnx_path", "outputs/model.onnx"),
                [os.path.join(_PROJECT_ROOT, "outputs", "model.onnx")],
            )
            self._verify_artifact(onnx_path, "onnx_sha256")
            conf = self.config.get("inference", {}).get("confidence_threshold", 0.35)
            iou = self.config.get("inference", {}).get("nms_threshold", 0.45)
            imgsz = self.imgsz
            self.onnx_engine = InferenceEngine(
                model_path=onnx_path,
                imgsz=imgsz,
                conf_thresh=conf,
                iou_thresh=iou,
                device=self.requested_device,
                num_classes=self.artifact_num_classes,
                class_names=self.class_names,
            )
            if self.onnx_engine.is_loaded:
                # Warmup CUDA kernels so first API call is not multi-second
                dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
                self.onnx_engine.infer(dummy)
            else:
                logger.warning(f"ONNX file missing or unloadable: {onnx_path}")
        except Exception as e:
            logger.info(f"ONNX engine not available, skipping: {e}")
            self.onnx_engine = None

    def align_sensors(self, optical, thermal, height):
        h, w = optical.shape[:2]
        aligned_thermal = (
            cv2.warpPerspective(thermal, self.h_thermal_to_rgb, (w, h))
            if thermal is not None
            else None
        )
        aligned_height = (
            cv2.warpPerspective(height, self.h_depth_to_rgb, (w, h))
            if height is not None
            else None
        )
        return aligned_thermal, aligned_height

    def _sensor_fallback_flag(self, thermal, height) -> bool:
        return thermal is None or height is None

    def preprocess(self, optical, thermal, height):
        h, w = optical.shape[:2]
        fallback_active = False
        norm_optical = optical.astype(np.float32) / 255.0

        if thermal is not None:
            norm_thermal = np.clip((thermal.astype(np.float32) - 15.0) / 60.0, 0.0, 1.0)
        else:
            logger.warning("Thermal sensor signal missing! Activating fallback degradation mode.")
            norm_thermal = np.zeros((h, w), dtype=np.float32)
            fallback_active = True

        if height is not None:
            norm_height = np.clip(height.astype(np.float32) / 500.0, 0.0, 1.0)
        else:
            logger.warning("3D Depth profile signal missing! Activating fallback degradation mode.")
            norm_height = np.zeros((h, w), dtype=np.float32)
            fallback_active = True

        if norm_thermal.ndim == 2:
            norm_thermal = np.expand_dims(norm_thermal, axis=-1)
        if norm_height.ndim == 2:
            norm_height = np.expand_dims(norm_height, axis=-1)

        fused = np.concatenate([norm_optical, norm_thermal, norm_height], axis=-1)
        tensor_data = torch.tensor(fused).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return tensor_data, fallback_active

    @staticmethod
    def _probs_from_detections(detections, num_classes: int):
        probs = [0.0] * num_classes
        if not detections:
            probs[0] = 1.0
            return probs, 0

        best_by_yolo = {}
        for det in detections:
            cid = int(det.get("class_id", 0))
            conf = float(det.get("confidence", 0.0))
            best_by_yolo[cid] = max(best_by_yolo.get(cid, 0.0), conf)

        for yolo_id, conf in best_by_yolo.items():
            idx = yolo_id + 1
            if 0 < idx < num_classes:
                probs[idx] = conf

        probs[0] = max(0.0, 1.0 - sum(probs[1:]))
        return probs, int(np.argmax(probs))

    def _pack_engine_result(self, engine_result, fallback_active, engine_label, start_time):
        detections = engine_result.get("detections", [])
        class_probs, predicted_class = self._probs_from_detections(
            detections, self.num_classes
        )
        if not detections:
            predicted_class = int(np.max(engine_result["segmentation_mask"]))
        latency_ms = (time.time() - start_time) * 1000.0
        return {
            "class_probabilities": class_probs,
            "predicted_class_id": predicted_class,
            "segmentation_mask": engine_result["segmentation_mask"],
            "detections": detections,
            "latency_ms": latency_ms,
            "fallback_active": fallback_active,
            "status": "Degraded Mode" if fallback_active else "Optimal",
            "engine": engine_label,
            "model_version": self.model_version,
        }

    def predict(self, optical, thermal=None, height=None):
        start_time = time.time()
        aligned_thermal, aligned_height = self.align_sensors(optical, thermal, height)
        fallback_active = self._sensor_fallback_flag(aligned_thermal, aligned_height)
        if fallback_active:
            if aligned_thermal is None:
                logger.warning("Thermal sensor signal missing! Activating fallback degradation mode.")
            if aligned_height is None:
                logger.warning("3D Depth profile signal missing! Activating fallback degradation mode.")

        for engine_name in self.profile.engine_priority:
            if engine_name == "yolo" and self.yolo_available:
                result = self.yolo_engine.infer(optical)
                return self._pack_engine_result(
                    result,
                    fallback_active,
                    f"YOLO ({result.get('provider', 'Ultralytics')})",
                    start_time,
                )
            if engine_name == "onnx" and self.onnx_available:
                result = self.onnx_engine.infer(optical)
                return self._pack_engine_result(
                    result,
                    fallback_active,
                    f"ONNX Runtime ({result.get('provider', 'unknown')})",
                    start_time,
                )

        # 3) Untrained fusion stub
        input_tensor, fallback_active = self.preprocess(
            optical, aligned_thermal, aligned_height
        )
        orig_h, orig_w = optical.shape[:2]
        with torch.no_grad():
            cls_out, seg_out = self.model(input_tensor)
            class_probs = torch.softmax(cls_out, dim=1).cpu().numpy()[0]
            predicted_class = int(np.argmax(class_probs))
            seg_probs = torch.softmax(seg_out, dim=1).cpu().numpy()[0]
            seg_mask = np.argmax(seg_probs, axis=0).astype(np.uint8)
            if seg_mask.shape[0] != orig_h or seg_mask.shape[1] != orig_w:
                seg_mask = cv2.resize(
                    seg_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
                )

        return {
            "class_probabilities": class_probs.tolist(),
            "predicted_class_id": predicted_class,
            "segmentation_mask": seg_mask,
            "detections": [],
            "latency_ms": (time.time() - start_time) * 1000.0,
            "fallback_active": fallback_active,
            "untrained_fallback": True,
            "status": "Degraded Mode",
            "engine": "PyTorch (Fusion Fallback)",
            "model_version": self.model_version,
        }

    def runtime_info(self) -> dict:
        """Expose requested versus active runtime without overstating acceleration."""
        info = self.profile.as_dict()
        preferred = self.profile.engine_priority[0]
        if preferred == "yolo" and self.yolo_available:
            active_engine = "yolo"
            active_provider = self.yolo_engine.active_provider
            active_precision = self.yolo_engine.active_precision
        elif preferred == "onnx" and self.onnx_available:
            active_engine = "onnx"
            active_provider = self.onnx_engine.active_provider
            active_precision = self.onnx_engine.active_precision
        elif self.onnx_available:
            active_engine = "onnx"
            active_provider = self.onnx_engine.active_provider
            active_precision = self.onnx_engine.active_precision
        elif self.yolo_available:
            active_engine = "yolo"
            active_provider = self.yolo_engine.active_provider
            active_precision = self.yolo_engine.active_precision
        else:
            active_engine = "untrained_fallback"
            active_provider = str(self.device)
            active_precision = "fp32"
        info.update(
            {
                "active_engine": active_engine,
                "active_provider": active_provider,
                "active_precision": active_precision,
                "model_version": self.model_version,
            }
        )
        if active_engine != preferred and not info["fallback_reason"]:
            info["fallback_reason"] = f"preferred engine {preferred} is unavailable"
        return info
