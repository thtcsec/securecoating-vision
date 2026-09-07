"""Record the local development/reproducibility compute environment.

Probe Torch/CUDA on this machine and write an immutable JSON artifact.
Does not run benchmarks or invent factory/cloud performance claims.
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "local_compute_environment.json"


def probe() -> dict:
    import torch

    cuda_available = bool(torch.cuda.is_available())
    gpu_name = None
    vram_gb = None
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        vram_gb = round(float(props.total_memory) / (1024**3), 2)

    return {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "evidence_scope": "local development/reproducibility environment",
        "gpu": gpu_name,
        "vram_gb": vram_gb,
        "cuda_available": cuda_available,
        "torch": torch.__version__,
        "torch_cuda": getattr(torch.version, "cuda", None),
        "python": platform.python_version(),
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "not_claimed": [
            "factory throughput",
            "TensorRT benchmark",
            "competition cloud benchmark",
            "end-to-end production latency",
        ],
        "notes": [
            "GPU-capable workloads may use this device when CUDA is available.",
            "Documented CoatingVision inference timing in "
            "reports/coatingvision_real_test_metrics.json is model-only CPU timing "
            "and is reported separately.",
            "This file is environment provenance, not a performance scoreboard.",
        ],
    }


def main() -> int:
    payload = probe()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(OUT.relative_to(ROOT).as_posix())
    print(json.dumps({k: payload[k] for k in ("gpu", "vram_gb", "torch", "cuda_available")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
