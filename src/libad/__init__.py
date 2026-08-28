"""LIBAD validation-extension adapter for SecureCoating-Vision.

This package does not replace the RGB YOLOv8-seg/ONNX production-prototype path.
It adds an external multimodal evidence lane on aligned VIS + inline-compatible
X-rayL observations. DA-Core belongs to Sui et al. (2026). The local contribution
is the evidence-gated industrial decision layer.
"""

from libad.protocol import LIBAD_CITATION, OFFICIAL_SPLIT_SEEDS, load_project_identity

__all__ = ["LIBAD_CITATION", "OFFICIAL_SPLIT_SEEDS", "load_project_identity"]
