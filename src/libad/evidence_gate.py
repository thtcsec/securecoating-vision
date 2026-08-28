"""Evidence-gated PASS / REJECT / HOLD for the LIBAD multimodal lane.

Detection scores never release production by themselves. Automatic PASS requires
both complementary modality evidence and the existing communication/safety
contracts. Uncertainty becomes HOLD, an operationally controlled state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


PASS = "PASS"
REJECT = "REJECT"
HOLD = "HOLD"

NORMAL = "NORMAL"
ANOMALY = "ANOMALY"
UNCERTAIN = "UNCERTAIN"
MISSING = "MISSING"


@dataclass
class EvidenceContracts:
    vis_available: bool = True
    xray_available: bool = True
    vis_stale: bool = False
    xray_stale: bool = False
    calibration_verified: bool = True
    identity_valid: bool = True
    traceability_ok: bool = True
    plc_ack_ok: bool = True
    plc_ack_stale: bool = False
    model_ready: bool = True
    require_both_modalities: bool = True


@dataclass
class GateDecision:
    action: str
    reason: str
    vis_state: str
    xray_state: str
    vis_score: Optional[float]
    xray_score: Optional[float]
    fused_score: Optional[float]
    contract_holds: List[str] = field(default_factory=list)
    details: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "action": self.action,
            "reason": self.reason,
            "vis_state": self.vis_state,
            "xray_state": self.xray_state,
            "vis_score": self.vis_score,
            "xray_score": self.xray_score,
            "fused_score": self.fused_score,
            "contract_holds": list(self.contract_holds),
            "details": dict(self.details),
        }


def classify_score(
    score: Optional[float],
    threshold: float = 1.0,
    uncertainty_band: float = 0.12,
    available: bool = True,
) -> str:
    """One-sided operating point around the normal reference.

    Scores at or below the reference threshold are normal. A gray band just
    above the threshold is HOLD rather than an automatic reject. Strong
    anomalies sit above threshold + margin.
    """
    if not available or score is None:
        return MISSING
    if score <= threshold:
        return NORMAL
    if score >= threshold + uncertainty_band:
        return ANOMALY
    return UNCERTAIN


def _contract_holds(contracts: EvidenceContracts) -> List[str]:
    holds: List[str] = []
    if not contracts.model_ready:
        holds.append("Model not ready; automatic release forbidden")
    if not contracts.calibration_verified:
        holds.append("Calibration is not verified")
    if not contracts.identity_valid:
        holds.append("Roll/batch/part identity is invalid")
    if not contracts.traceability_ok:
        holds.append("Traceability write failed")
    if not contracts.plc_ack_ok:
        holds.append("PLC ACK missing or mismatched")
    if contracts.plc_ack_stale:
        holds.append("PLC ACK is stale")
    if contracts.vis_stale:
        holds.append("VIS sensor frame is stale")
    if contracts.xray_stale:
        holds.append("X-rayL sensor frame is stale")
    if contracts.require_both_modalities:
        if not contracts.vis_available:
            holds.append("Required VIS modality is missing")
        if not contracts.xray_available:
            holds.append("Required X-rayL modality is missing")
    elif not contracts.vis_available and not contracts.xray_available:
        holds.append("No inspection modality is available")
    return holds


def decide_evidence_gate(
    vis_score: Optional[float],
    xray_score: Optional[float],
    contracts: Optional[EvidenceContracts] = None,
    threshold: float = 1.0,
    uncertainty_band: float = 0.12,
    strong_margin: float = 0.18,
) -> GateDecision:
    """Map modality scores and safety contracts onto PASS / REJECT / HOLD.

    Strong, one-sided evidence is complementary and may REJECT:
    - VIS strongly anomalous, X-ray weak -> REJECT, surface evidence
    - X-ray strongly anomalous, VIS normal -> REJECT, complementary X-ray evidence
    Near-threshold disagreement, missing evidence, or a failed contract -> HOLD.
    Both modalities confidently normal and contracts satisfied -> PASS.
    """
    contracts = contracts or EvidenceContracts()
    vis_state = classify_score(
        vis_score, threshold, uncertainty_band, available=contracts.vis_available
    )
    xray_state = classify_score(
        xray_score, threshold, uncertainty_band, available=contracts.xray_available
    )
    fused = None
    if vis_score is not None and xray_score is not None:
        fused = max(vis_score, xray_score)
    elif vis_score is not None:
        fused = vis_score
    elif xray_score is not None:
        fused = xray_score

    holds = _contract_holds(contracts)
    details = {
        "threshold": threshold,
        "uncertainty_band": uncertainty_band,
        "strong_margin": strong_margin,
        "rule": "evidence-gated selective decision; detection cannot self-release",
    }
    if holds:
        return GateDecision(
            action=HOLD,
            reason="; ".join(holds),
            vis_state=vis_state,
            xray_state=xray_state,
            vis_score=vis_score,
            xray_score=xray_score,
            fused_score=fused,
            contract_holds=holds,
            details=details,
        )

    vis_strong = vis_score is not None and vis_score >= threshold + strong_margin
    xray_strong = xray_score is not None and xray_score >= threshold + strong_margin

    if vis_state == UNCERTAIN or xray_state == UNCERTAIN:
        return GateDecision(
            HOLD,
            "Modality score is inside the uncertainty band; HOLD for manual QA",
            vis_state,
            xray_state,
            vis_score,
            xray_score,
            fused,
            details=details,
        )

    if vis_state == NORMAL and xray_state == NORMAL:
        return GateDecision(
            PASS,
            "Normal agreement: VIS and X-rayL are both below the release threshold",
            vis_state,
            xray_state,
            vis_score,
            xray_score,
            fused,
            details=details,
        )

    if vis_strong and xray_state in {NORMAL, ANOMALY, UNCERTAIN}:
        return GateDecision(
            REJECT,
            "REJECT — surface evidence: VIS anomaly with complementary or weaker X-rayL",
            vis_state,
            xray_state,
            vis_score,
            xray_score,
            fused,
            details=details,
        )

    if xray_strong and vis_state == NORMAL:
        return GateDecision(
            REJECT,
            "REJECT — complementary X-ray evidence: internal/density anomaly with near-normal VIS",
            vis_state,
            xray_state,
            vis_score,
            xray_score,
            fused,
            details=details,
        )

    if vis_state == ANOMALY and xray_state == ANOMALY:
        return GateDecision(
            REJECT,
            "REJECT — multimodal agreement on anomaly evidence",
            vis_state,
            xray_state,
            vis_score,
            xray_score,
            fused,
            details=details,
        )

    return GateDecision(
        HOLD,
        "HOLD — modality disagreement or insufficient margin for an automatic decision",
        vis_state,
        xray_state,
        vis_score,
        xray_score,
        fused,
        details=details,
    )
