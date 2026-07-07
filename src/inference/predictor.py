import os
import time
import logging
import numpy as np
import cv2
import torch
import torch.nn as nn

logger = logging.getLogger("SecureCoatingVision.Predictor")

class SimpleFusionNetwork(nn.Module):
    """
    A lightweight fusion network model structure supporting 5 channels (RGB + Thermal + 3D Height).
    """
    def __init__(self, in_channels=5, num_classes=5):
        super(SimpleFusionNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Output layers for box classification and segmentation mask
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )
        # Reconstruction/Segmentation branch outputting class logit maps
        self.segmentor = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

    def forward(self, x):
        # x shape: (Batch, 5, Height, Width)
        x = self.relu(self.conv1(x))
        features = self.relu(self.conv2(x))
        pooled = self.pool(features)
        
        cls_out = self.classifier(pooled)
        seg_out = self.segmentor(features)
        return cls_out, seg_out

class CoatingPredictor:
    def __init__(self, model_config):
        self.config = model_config
        
        # Resolve device with graceful CUDA fallback
        requested_device = self.config.get("inference", {}).get("device", "cpu")
        if requested_device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(requested_device)
        
        # Instantiate network
        self.num_classes = self.config.get("model", {}).get("num_classes", 5)
        self.model = SimpleFusionNetwork(in_channels=5, num_classes=self.num_classes)
        self.model.to(self.device)
        self.model.eval()
        
        # Try to load ONNX engine for accelerated inference
        self.onnx_engine = None
        self._init_onnx_engine()
        
        # Setup homography calibration targets (mock values)
        # Used to warp Thermal and 3D Height profiles to align with RGB camera frame
        self.h_thermal_to_rgb = np.eye(3, dtype=np.float32)
        self.h_depth_to_rgb = np.eye(3, dtype=np.float32)
        
        logger.info(f"CoatingPredictor initialized on device: {self.device}")
        if self.onnx_engine and self.onnx_engine.is_loaded:
            logger.info(f"ONNX Engine active: {self.onnx_engine.active_provider}")

    def _init_onnx_engine(self):
        """Initialize ONNX Runtime engine if model.onnx is available."""
        try:
            from inference.onnx_engine import InferenceEngine
            onnx_path = os.path.join("outputs", "model.onnx")
            conf = self.config.get("inference", {}).get("confidence_threshold", 0.5)
            iou = self.config.get("inference", {}).get("nms_threshold", 0.45)
            self.onnx_engine = InferenceEngine(
                model_path=onnx_path,
                imgsz=640,
                conf_thresh=conf,
                iou_thresh=iou,
                device="auto"
            )
        except Exception as e:
            logger.info(f"ONNX engine not available, using PyTorch fallback: {e}")
            self.onnx_engine = None

    def align_sensors(self, optical, thermal, height):
        """
        Aligns secondary sensors (thermal and depth) to match optical image dimensions and coordinates
        using homography transformation matrices.
        """
        h, w = optical.shape[:2]
        
        # Warps LWIR thermal
        if thermal is not None:
            aligned_thermal = cv2.warpPerspective(thermal, self.h_thermal_to_rgb, (w, h))
        else:
            aligned_thermal = None
            
        # Warps 3D Height Map
        if height is not None:
            aligned_height = cv2.warpPerspective(height, self.h_depth_to_rgb, (w, h))
        else:
            aligned_height = None
            
        return aligned_thermal, aligned_height

    def preprocess(self, optical, thermal, height):
        """
        Preprocesses and normalizes the multi-source image inputs.
        If secondary inputs are missing, fallback to zero tensors with warning metrics.
        """
        h, w = optical.shape[:2]
        fallback_active = False

        # 1. Optical Normalization [RGB -> Float 0..1]
        norm_optical = optical.astype(np.float32) / 255.0

        # 2. Thermal Normalization & Sensor Fallback Check
        if thermal is not None:
            # Map typical range [15°C .. 75°C] to [0 .. 1]
            norm_thermal = (thermal.astype(np.float32) - 15.0) / 60.0
            norm_thermal = np.clip(norm_thermal, 0.0, 1.0)
        else:
            logger.warning("Thermal sensor signal missing! Activating fallback degradation mode.")
            norm_thermal = np.zeros((h, w), dtype=np.float32)
            fallback_active = True

        # 3. Depth Normalization & Sensor Fallback Check
        if height is not None:
            # Map target height range [0 .. 500um] to [0 .. 1]
            norm_height = height.astype(np.float32) / 500.0
            norm_height = np.clip(norm_height, 0.0, 1.0)
        else:
            logger.warning("3D Depth profile signal missing! Activating fallback degradation mode.")
            norm_height = np.zeros((h, w), dtype=np.float32)
            fallback_active = True

        # 4. Construct 5-channel array: [R, G, B, Thermal, Height]
        if len(norm_thermal.shape) == 2:
            norm_thermal = np.expand_dims(norm_thermal, axis=-1)
        if len(norm_height.shape) == 2:
            norm_height = np.expand_dims(norm_height, axis=-1)

        fused_tensor = np.concatenate([norm_optical, norm_thermal, norm_height], axis=-1)
        
        # Transform to PyTorch Tensor shape: (Channels, Height, Width)
        tensor_data = torch.tensor(fused_tensor).permute(2, 0, 1).unsqueeze(0).to(self.device)
        return tensor_data, fallback_active

    def predict(self, optical, thermal=None, height=None):
        """
        Run aligned multi-source inputs through the forward pass.
        
        Uses ONNX Runtime engine if available (faster, production-ready),
        otherwise falls back to PyTorch SimpleFusionNetwork.
        
        Returns:
            dict containing class scores, segmentation masks, inference time, and status.
        """
        start_time = time.time()
        
        # Spatial registration warp
        aligned_thermal, aligned_height = self.align_sensors(optical, thermal, height)
        
        # Determine fallback status from sensor availability
        _, fallback_active = self.preprocess(optical, aligned_thermal, aligned_height)
        
        # Try ONNX engine first (YOLOv8 seg on RGB only - production path)
        if self.onnx_engine is not None and self.onnx_engine.is_loaded:
            onnx_result = self.onnx_engine.infer(optical)
            latency_ms = (time.time() - start_time) * 1000.0
            
            return {
                "class_probabilities": [0.0] * self.num_classes,
                "predicted_class_id": int(np.max(onnx_result["segmentation_mask"])),
                "segmentation_mask": onnx_result["segmentation_mask"],
                "detections": onnx_result.get("detections", []),
                "latency_ms": latency_ms,
                "fallback_active": fallback_active,
                "status": "Degraded Mode" if fallback_active else "Optimal",
                "engine": "ONNX Runtime"
            }
        
        # Fallback: PyTorch fusion network (5-channel input)
        input_tensor, fallback_active = self.preprocess(optical, aligned_thermal, aligned_height)
        orig_h, orig_w = optical.shape[:2]
        
        with torch.no_grad():
            cls_out, seg_out = self.model(input_tensor)
            
            # Postprocess outputs
            class_probs = torch.softmax(cls_out, dim=1).cpu().numpy()[0]
            predicted_class = int(np.argmax(class_probs))
            
            # Get class probability map for segmentation mask
            seg_probs = torch.softmax(seg_out, dim=1).cpu().numpy()[0]  # Shape (Classes, H, W)
            seg_mask = np.argmax(seg_probs, axis=0).astype(np.uint8)
            
            # Ensure mask matches original input dimensions
            if seg_mask.shape[0] != orig_h or seg_mask.shape[1] != orig_w:
                seg_mask = cv2.resize(seg_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            
        latency_ms = (time.time() - start_time) * 1000.0
        
        return {
            "class_probabilities": class_probs.tolist(),
            "predicted_class_id": predicted_class,
            "segmentation_mask": seg_mask,
            "detections": [],
            "latency_ms": latency_ms,
            "fallback_active": fallback_active,
            "status": "Degraded Mode" if fallback_active else "Optimal",
            "engine": "PyTorch (Fusion)"
        }
