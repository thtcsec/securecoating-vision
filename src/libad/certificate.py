"""HMAC evidence certificate for a single LIBAD-gated inspection."""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import os
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from libad.protocol import load_project_identity, sha256_json


_CONFIGURED_FACTORY_SECRET = os.environ.get("SECURECOATING_FACTORY_SECRET", "").encode("utf-8")
_PROCESS_FACTORY_SECRET = _CONFIGURED_FACTORY_SECRET or secrets.token_bytes(32)


def _factory_secret() -> bytes:
    """Return one stable process key; production startup requires a configured key."""
    return _PROCESS_FACTORY_SECRET


@dataclass
class EvidenceCertificate:
    certificate_id: str
    roll_id: str
    batch_id: str
    part_id: str
    decision: str
    decision_reason: str
    vis_score: Optional[float]
    xray_score: Optional[float]
    fused_score: Optional[float]
    vis_state: str
    xray_state: str
    calibration_state: str
    model_hash: Optional[str]
    dataset_manifest_hash: Optional[str]
    commit_hash: str
    plc_state: str
    detector_attribution: str
    source_tree_dirty: Optional[bool] = None
    source_diff_sha256: Optional[str] = None
    issued_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    payload_hash_sha256: str = ""
    hmac_digital_signature: str = ""
    signature_algorithm: str = "HMAC-SHA256"

    def _payload(self) -> Dict[str, Any]:
        return {
            "certificate_id": self.certificate_id,
            "roll_id": self.roll_id,
            "batch_id": self.batch_id,
            "part_id": self.part_id,
            "decision": self.decision,
            "decision_reason": self.decision_reason,
            "vis_score": self.vis_score,
            "xray_score": self.xray_score,
            "fused_score": self.fused_score,
            "vis_state": self.vis_state,
            "xray_state": self.xray_state,
            "calibration_state": self.calibration_state,
            "model_hash": self.model_hash,
            "dataset_manifest_hash": self.dataset_manifest_hash,
            "commit_hash": self.commit_hash,
            "plc_state": self.plc_state,
            "detector_attribution": self.detector_attribution,
            "source_tree_dirty": self.source_tree_dirty,
            "source_diff_sha256": self.source_diff_sha256,
            "issued_at": self.issued_at,
            "signature_algorithm": self.signature_algorithm,
            "brand": load_project_identity()["brand"],
        }

    def sign(self, secret: Optional[bytes] = None) -> "EvidenceCertificate":
        raw = json.dumps(self._payload(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        key = secret if secret is not None else _factory_secret()
        self.payload_hash_sha256 = hashlib.sha256(raw).hexdigest()
        self.hmac_digital_signature = hmac.new(key, raw, hashlib.sha256).hexdigest()
        return self

    def verify(self, secret: Optional[bytes] = None) -> bool:
        raw = json.dumps(self._payload(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        key = secret if secret is not None else _factory_secret()
        expected_hash = hashlib.sha256(raw).hexdigest()
        expected_sig = hmac.new(key, raw, hashlib.sha256).hexdigest()
        return hmac.compare_digest(self.payload_hash_sha256, expected_hash) and hmac.compare_digest(
            self.hmac_digital_signature, expected_sig
        )

    def to_dict(self) -> Dict[str, Any]:
        if not self.hmac_digital_signature:
            self.sign()
        payload = copy.deepcopy(self._payload())
        payload.update(
            {
                "payload_hash_sha256": self.payload_hash_sha256,
                "hmac_digital_signature": self.hmac_digital_signature,
                "canonical_payload_sha256": sha256_json(self._payload()),
            }
        )
        return payload


def build_evidence_certificate(
    *,
    roll_id: str,
    batch_id: str,
    part_id: str,
    decision: str,
    decision_reason: str,
    vis_score: Optional[float],
    xray_score: Optional[float],
    fused_score: Optional[float],
    vis_state: str,
    xray_state: str,
    calibration_state: str,
    model_hash: Optional[str],
    dataset_manifest_hash: Optional[str],
    commit_hash: str,
    plc_state: str,
    detector_attribution: str,
    source_tree_dirty: Optional[bool] = None,
    source_diff_sha256: Optional[str] = None,
    secret: Optional[bytes] = None,
) -> EvidenceCertificate:
    cert = EvidenceCertificate(
        certificate_id=f"EVD_{part_id}",
        roll_id=roll_id,
        batch_id=batch_id,
        part_id=part_id,
        decision=decision,
        decision_reason=decision_reason,
        vis_score=vis_score,
        xray_score=xray_score,
        fused_score=fused_score,
        vis_state=vis_state,
        xray_state=xray_state,
        calibration_state=calibration_state,
        model_hash=model_hash,
        dataset_manifest_hash=dataset_manifest_hash,
        commit_hash=commit_hash,
        plc_state=plc_state,
        detector_attribution=detector_attribution,
        source_tree_dirty=source_tree_dirty,
        source_diff_sha256=source_diff_sha256,
    )
    return cert.sign(secret)
