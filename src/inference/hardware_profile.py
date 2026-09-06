"""Deterministic, fail-safe hardware profile selection for inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
import os
from typing import Any, Mapping


logger = logging.getLogger("SecureCoatingVision.HardwareProfile")
VALID_PROFILES = {"auto", "edge", "balanced", "performance"}
VALID_DEVICES = {"auto", "cpu", "cuda"}


@dataclass(frozen=True)
class HardwareCapabilities:
    cuda_available: bool
    gpu_name: str | None
    gpu_memory_gb: float | None
    onnx_providers: tuple[str, ...]


@dataclass(frozen=True)
class InferenceProfile:
    requested_profile: str
    active_profile: str
    requested_device: str
    device: str
    imgsz: int
    requested_precision: str
    engine_priority: tuple[str, ...]
    gpu_name: str | None
    gpu_memory_gb: float | None
    available_onnx_providers: tuple[str, ...]
    fallback_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["engine_priority"] = list(self.engine_priority)
        payload["available_onnx_providers"] = list(self.available_onnx_providers)
        return payload


def detect_hardware() -> HardwareCapabilities:
    """Probe optional runtimes without making startup depend on the probe."""
    cuda_available = False
    gpu_name = None
    gpu_memory_gb = None
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            gpu_name = str(torch.cuda.get_device_name(0))
            gpu_memory_gb = round(
                float(torch.cuda.get_device_properties(0).total_memory) / (1024**3), 2
            )
    except Exception as exc:  # pragma: no cover - host-dependent
        logger.info("PyTorch accelerator probe unavailable: %s", exc)

    providers: tuple[str, ...] = ()
    try:
        import onnxruntime as ort

        providers = tuple(str(item) for item in ort.get_available_providers())
    except Exception as exc:  # pragma: no cover - optional runtime
        logger.info("ONNX Runtime provider probe unavailable: %s", exc)

    return HardwareCapabilities(cuda_available, gpu_name, gpu_memory_gb, providers)


def _profile_settings(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    defaults = {
        "edge": {"imgsz": 512, "precision": "fp32", "engine_priority": ["onnx", "yolo"]},
        "balanced": {"imgsz": 640, "precision": "fp32", "engine_priority": ["onnx", "yolo"]},
        "performance": {"imgsz": 640, "precision": "fp16", "engine_priority": ["yolo", "onnx"]},
    }
    configured = config.get("hardware_profiles", {}).get(name, {}) or {}
    return {**defaults[name], **configured}


def resolve_inference_profile(
    config: Mapping[str, Any], capabilities: HardwareCapabilities | None = None
) -> InferenceProfile:
    """Resolve environment/config intent into one truthful active profile."""
    inference = config.get("inference", {}) or {}
    requested_profile = os.environ.get(
        "SECURECOATING_HARDWARE_PROFILE", inference.get("hardware_profile", "auto")
    ).strip().lower()
    if requested_profile not in VALID_PROFILES:
        raise ValueError(
            "SECURECOATING_HARDWARE_PROFILE must be auto, edge, balanced, or performance"
        )

    legacy_device = os.environ.get(
        "SECURECOATING_INFERENCE_DEVICE", inference.get("device", "auto")
    ).strip().lower()
    if legacy_device not in VALID_DEVICES:
        raise ValueError("SECURECOATING_INFERENCE_DEVICE must be auto, cpu, or cuda")

    caps = capabilities or detect_hardware()
    effective_request = requested_profile
    if legacy_device == "cpu":
        effective_request = "edge"
    elif legacy_device == "cuda" and requested_profile == "auto":
        effective_request = "performance"

    fallback_reason = None
    if effective_request == "auto":
        if caps.cuda_available and (caps.gpu_memory_gb or 0.0) >= 6.0:
            active_profile = "performance"
        elif caps.cuda_available:
            active_profile = "balanced"
        else:
            active_profile = "edge"
    elif effective_request in {"balanced", "performance"} and not caps.cuda_available:
        active_profile = "edge"
        fallback_reason = f"{effective_request} requested but CUDA is unavailable"
    else:
        active_profile = effective_request

    settings = _profile_settings(config, active_profile)
    engine_priority = tuple(str(item).lower() for item in settings["engine_priority"])
    if (
        active_profile == "performance"
        and "TensorrtExecutionProvider" in caps.onnx_providers
    ):
        engine_priority = ("onnx", "yolo")
    if sorted(engine_priority) != ["onnx", "yolo"]:
        raise ValueError(
            f"hardware_profiles.{active_profile}.engine_priority must contain onnx and yolo once"
        )

    resolved = InferenceProfile(
        requested_profile=requested_profile,
        active_profile=active_profile,
        requested_device=legacy_device,
        device="cpu" if active_profile == "edge" else "cuda",
        imgsz=int(settings["imgsz"]),
        requested_precision=str(settings["precision"]).lower(),
        engine_priority=engine_priority,
        gpu_name=caps.gpu_name,
        gpu_memory_gb=caps.gpu_memory_gb,
        available_onnx_providers=caps.onnx_providers,
        fallback_reason=fallback_reason,
    )
    logger.info(
        "Inference profile resolved: requested=%s active=%s device=%s priority=%s%s",
        resolved.requested_profile,
        resolved.active_profile,
        resolved.device,
        ">".join(resolved.engine_priority),
        f" fallback={resolved.fallback_reason}" if resolved.fallback_reason else "",
    )
    return resolved
