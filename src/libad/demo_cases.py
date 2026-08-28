"""Four 90-second demo cases for the LIBAD evidence lane.

Case 1 — Normal agreement -> PASS
Case 2 — Surface-visible defect -> REJECT (surface evidence)
Case 3 — Internally visible anomaly -> REJECT (complementary X-ray)
Case 4 — Disagreement / missing evidence -> HOLD
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np

from libad.certificate import EvidenceCertificate, build_evidence_certificate
from libad.dataset import _draw_density_blob, _draw_scratch, _electrode_canvas
from libad.evidence_gate import EvidenceContracts, GateDecision, decide_evidence_gate
from libad.protocol import git_commit_sha, load_libad_config, load_project_identity
from libad.scorer import MemoryAnomalyScorer, SampleScore

DEMO_CASES = {
    1: {
        "name": "Normal agreement",
        "expected_action": "PASS",
        "story": "VIS normal, X-rayL normal, calibration and identity valid.",
    },
    2: {
        "name": "Surface-visible defect",
        "expected_action": "REJECT",
        "story": "VIS scratch/crack with weaker X-rayL; complementary surface evidence.",
    },
    3: {
        "name": "Internally visible anomaly",
        "expected_action": "REJECT",
        "story": "VIS near-normal, X-rayL anomaly score high; multimodal value, not a second pretty image.",
    },
    4: {
        "name": "Disagreement or missing evidence",
        "expected_action": "HOLD",
        "story": "Modality disagreement, stale sensor, or unverified calibration; manual QA required.",
    },
}


@dataclass
class DemoBank:
    scorer: MemoryAnomalyScorer
    vis: np.ndarray
    xray: np.ndarray


def _bgr(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def build_demo_bank(size: int = 64) -> DemoBank:
    rng = np.random.default_rng(7345)
    normals_vis = [_electrode_canvas(size, rng, mean=144.0) for _ in range(12)]
    normals_xray = [_electrode_canvas(size, rng, mean=126.0) for _ in range(12)]
    cfg = load_libad_config()
    scorer = MemoryAnomalyScorer(
        method=cfg["coreset"]["method"],
        coreset_ratio=0.5,
        density_weight=float(cfg["coreset"]["density_weight"]),
        knn=4,
        patch_size=16,
        stride=8,
        image_size=(size, size),
        seed=7345,
    )
    scorer.fit(vis_images=normals_vis, xray_images=normals_xray)
    best_index = 0
    best_score = float("inf")
    for index, (vis, xray) in enumerate(zip(normals_vis, normals_xray)):
        scored = scorer.score_sample(f"train-{index}", vis_a=vis, vis_b=vis, xray_l=xray)
        fused = scored.fused_score if scored.fused_score is not None else float("inf")
        if fused < best_score:
            best_score = fused
            best_index = index
    return DemoBank(scorer=scorer, vis=normals_vis[best_index], xray=normals_xray[best_index])


def _build_demo_frames(case_id: int, bank: DemoBank) -> Dict[str, np.ndarray]:
    vis = bank.vis.copy()
    xray = bank.xray.copy()
    rng = np.random.default_rng(42)
    if case_id == 2:
        vis = _draw_scratch(vis, rng)
    elif case_id == 3:
        xray = _draw_density_blob(xray, rng)
    elif case_id == 4:
        vis = _draw_scratch(vis, rng)
    return {"vis_a": vis, "vis_b": vis.copy(), "xray_l": xray}


def _demo_contracts(case_id: int) -> EvidenceContracts:
    if case_id == 4:
        return EvidenceContracts(calibration_verified=False, vis_stale=True)
    return EvidenceContracts()


def run_libad_demo_case(
    case_id: int,
    bank: Optional[DemoBank] = None,
    scorer: Optional[MemoryAnomalyScorer] = None,
    secret: Optional[bytes] = None,
) -> Dict[str, object]:
    if case_id not in DEMO_CASES:
        raise ValueError(f"Demo case must be 1-4, got {case_id}")
    spec = DEMO_CASES[case_id]
    bank = bank or build_demo_bank()
    if scorer is not None:
        bank = DemoBank(scorer=scorer, vis=bank.vis, xray=bank.xray)
    frames = _build_demo_frames(case_id, bank)
    score: SampleScore = bank.scorer.score_sample(
        sample_id=f"DEMO_CASE_{case_id}",
        vis_a=frames["vis_a"],
        vis_b=frames["vis_b"],
        xray_l=frames["xray_l"],
    )
    contracts = _demo_contracts(case_id)
    decision: GateDecision = decide_evidence_gate(
        vis_score=score.vis_score,
        xray_score=score.xray_score,
        contracts=contracts,
        threshold=1.0,
        uncertainty_band=0.12,
        strong_margin=0.18,
    )
    identity = load_project_identity()
    cfg = load_libad_config()
    roll_id = cfg["identity"]["roll_id"]
    batch_id = cfg["identity"]["batch_id"]
    part_id = f"PART_LIBAD_DEMO_{case_id:02d}"
    plc_state = "SIMULATED_HOLD" if decision.action == "HOLD" else f"SIMULATED_{decision.action}"
    certificate: EvidenceCertificate = build_evidence_certificate(
        roll_id=roll_id,
        batch_id=batch_id,
        part_id=part_id,
        decision=decision.action,
        decision_reason=decision.reason,
        vis_score=None if score.vis_score is None else round(score.vis_score, 6),
        xray_score=None if score.xray_score is None else round(score.xray_score, 6),
        fused_score=None if score.fused_score is None else round(score.fused_score, 6),
        vis_state=decision.vis_state,
        xray_state=decision.xray_state,
        calibration_state="VERIFIED" if contracts.calibration_verified else "UNVERIFIED",
        model_hash=None,
        dataset_manifest_hash=None,
        commit_hash=git_commit_sha(),
        plc_state=plc_state,
        detector_attribution=bank.scorer.provenance()["coreset_attribution"],
        secret=secret,
    )
    return {
        "case_id": case_id,
        "case_name": spec["name"],
        "story": spec["story"],
        "expected_action": spec["expected_action"],
        "decision": decision.to_dict(),
        "scores": {
            "vis": score.vis_score,
            "xray_l": score.xray_score,
            "fused": score.fused_score,
        },
        "identity": {
            "roll_id": roll_id,
            "batch_id": batch_id,
            "part_id": part_id,
        },
        "calibration_state": "VERIFIED" if contracts.calibration_verified else "UNVERIFIED",
        "plc_state": plc_state,
        "certificate": certificate.to_dict(),
        "frames": {
            "vis_bgr": _bgr(frames["vis_a"]),
            "xray_bgr": _bgr(frames["xray_l"]),
        },
        "title": identity["registered_title"],
        "tagline": identity["tagline"],
        "brand": identity["brand"],
        "detector_provenance": bank.scorer.provenance(),
    }


def run_all_demo_cases(secret: Optional[bytes] = None) -> List[Dict[str, object]]:
    bank = build_demo_bank()
    return [run_libad_demo_case(case_id, bank=bank, secret=secret) for case_id in (1, 2, 3, 4)]
