# Adaptive inference profiles

SecureCoating-Vision selects a runtime once at API startup. Selection affects
latency only; it never weakens the fail-closed inspection gate.

| Profile | Intended host | Priority | Size | Requested precision |
|---|---|---|---:|---|
| `edge` | CPU / limited memory | ONNX, then YOLO | 512 | FP32 |
| `balanced` | small CUDA GPU | ONNX CUDA, then YOLO | 640 | FP32 |
| `performance` | CUDA GPU with at least 6 GB VRAM | YOLO CUDA, then ONNX | 640 | FP16 |
| `auto` | unknown host | capability-selected profile | profile-specific | profile-specific |

The API reports requested and active profiles, active provider, actual model
input precision, image size, GPU identity, and any fallback reason under
`/health` and `/api/system/health-report`. An FP32 ONNX artifact is reported as
FP32 even when TensorRT or CUDA executes it.

## Run on CPU

```powershell
docker compose up --build -d
```

The CPU image contains CPU PyTorch and ONNX Runtime. `auto` therefore resolves
to `edge`; no GPU capability is claimed.

## Run on NVIDIA GPU

An NVIDIA driver and NVIDIA Container Toolkit are prerequisites.

```powershell
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d
```

The GPU overlay requests the GPU explicitly and uses CUDA PyTorch plus
`onnxruntime-gpu`. If CUDA is unavailable at runtime, startup records the reason
and resolves to the CPU-safe edge profile where the installed runtime permits.

## Demo wording

Show the runtime strip before an inspection: “The same safety contract adapts
to the host. This machine selected `<active profile>`, `<provider>`,
`<precision>`, and `<image size>`. If acceleration disappears, the API exposes
the fallback reason and the independent gate holds rather than inventing a
PASS.” Do not claim TensorRT, FP16, or factory throughput unless those exact
values are visible in live telemetry.
