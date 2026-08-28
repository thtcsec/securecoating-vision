"""10 official-split evaluation harness for the LIBAD evidence lane."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from libad.dataset import LibadSample, LibadSplit, dataset_status, load_official_splits
from libad.evidence_gate import EvidenceContracts, decide_evidence_gate
from libad.metrics import academic_metrics, industrial_gate_metrics, summarize_splits
from libad.protocol import (
    LIBAD_CITATION,
    LIBAD_PAPER_RESULT_NOTE,
    OFFICIAL_SPLIT_SEEDS,
    git_commit_sha,
    git_source_provenance,
    hash_existing_files,
    load_libad_config,
)
from libad.scorer import MemoryAnomalyScorer, SampleScore


EXPERIMENT_MODALITIES = {
    "unimodal_vis": ("vis",),
    "unimodal_xray_l": ("xray_l",),
    "multimodal": ("vis", "xray_l"),
    "securecoating_gate": ("vis", "xray_l"),
}


def _scorer_from_config(seed: int, config: dict) -> MemoryAnomalyScorer:
    features = config["features"]
    coreset = config["coreset"]
    scoring = config["scoring"]
    return MemoryAnomalyScorer(
        method=coreset["method"],
        coreset_ratio=float(coreset["ratio"]),
        density_weight=float(coreset["density_weight"]),
        knn=int(coreset["knn"]),
        patch_size=int(features["patch_size"]),
        stride=int(features["stride"]),
        image_size=tuple(features["image_size"]),
        vis_scale=float(scoring["vis_scale"]),
        xray_scale=float(scoring["xray_scale"]),
        seed=int(seed),
    )


def _fit_on_split(scorer: MemoryAnomalyScorer, split: LibadSplit, modalities: Sequence[str]) -> None:
    vis = [sample.vis_a for sample in split.train] if "vis" in modalities else None
    vis_b = [sample.vis_b for sample in split.train] if "vis" in modalities else None
    vis_all = None
    if vis is not None:
        vis_all = list(vis) + list(vis_b or [])
    xray = [sample.xray_l for sample in split.train] if "xray_l" in modalities else None
    scorer.fit(vis_images=vis_all, xray_images=xray)


def _score_sample(scorer: MemoryAnomalyScorer, sample: LibadSample, modalities: Sequence[str]) -> SampleScore:
    return scorer.score_sample(
        sample_id=sample.sample_id,
        vis_a=sample.vis_a if "vis" in modalities else None,
        vis_b=sample.vis_b if "vis" in modalities else None,
        xray_l=sample.xray_l if "xray_l" in modalities else None,
        label=sample.label,
        defect_group=sample.defect_group,
    )


def _gate_contracts(modalities: Sequence[str]) -> EvidenceContracts:
    return EvidenceContracts(
        vis_available="vis" in modalities,
        xray_available="xray_l" in modalities,
        require_both_modalities=set(modalities) == {"vis", "xray_l"},
        calibration_verified=True,
        identity_valid=True,
        traceability_ok=True,
        plc_ack_ok=True,
        model_ready=True,
    )


def evaluate_split(
    split: LibadSplit,
    experiment: str,
    config: Optional[dict] = None,
) -> Dict[str, Any]:
    if experiment not in EXPERIMENT_MODALITIES:
        raise ValueError(f"Unknown experiment: {experiment}")
    cfg = config or load_libad_config()
    input_status = dataset_status(cfg)
    modalities = EXPERIMENT_MODALITIES[experiment]
    scorer = _scorer_from_config(split.seed, cfg)
    started = time.perf_counter()
    _fit_on_split(scorer, split, modalities)
    val_scores: List[float] = []
    for sample in split.val:
        scored = _score_sample(scorer, sample, modalities)
        value = scored.fused_score
        if "vis" in modalities and "xray_l" not in modalities:
            value = scored.vis_score
        elif "xray_l" in modalities and "vis" not in modalities:
            value = scored.xray_score
        if value is not None:
            val_scores.append(float(value))
    operating_threshold = float(np.percentile(val_scores, 80)) if val_scores else 1.0
    rows: List[Dict[str, Any]] = []
    labels: List[int] = []
    scores: List[float] = []
    decisions: List[str] = []
    for sample in split.test:
        scored = _score_sample(scorer, sample, modalities)
        if experiment == "securecoating_gate":
            gate = decide_evidence_gate(
                vis_score=scored.vis_score,
                xray_score=scored.xray_score,
                contracts=_gate_contracts(modalities),
                threshold=operating_threshold,
                uncertainty_band=float(cfg["gate"]["uncertainty_band"]),
                strong_margin=float(cfg["gate"]["strong_margin"]),
            )
            decision = gate.action
            score_value = scored.fused_score if scored.fused_score is not None else 0.0
            reason = gate.reason
        else:
            score_value = 0.0
            if "vis" in modalities and "xray_l" in modalities:
                score_value = scored.fused_score or 0.0
            elif "vis" in modalities:
                score_value = scored.vis_score or 0.0
            else:
                score_value = scored.xray_score or 0.0
            decision = "REJECT" if score_value >= operating_threshold else "PASS"
            reason = "score-threshold baseline without industrial HOLD"
        labels.append(int(sample.label))
        scores.append(float(score_value))
        decisions.append(decision)
        rows.append(
            {
                "sample_id": sample.sample_id,
                "seed": split.seed,
                "split_source": split.source,
                "label": int(sample.label),
                "defect_group": sample.defect_group,
                "vis_score": scored.vis_score,
                "xray_score": scored.xray_score,
                "fused_score": scored.fused_score,
                "decision": decision,
                "reason": reason,
            }
        )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    result: Dict[str, Any] = {
        "seed": split.seed,
        "experiment": experiment,
        "split_source": split.source,
        "comparable_to_paper": split.comparable_to_paper,
        "n_train": len(split.train),
        "n_val": len(split.val),
        "n_test": len(split.test),
        "academic": academic_metrics(labels, scores),
        "operating_threshold": operating_threshold,
        "latency_ms_split": elapsed_ms,
        "predictions": rows,
        "scorer_provenance": scorer.provenance(),
    }
    if experiment == "securecoating_gate":
        result["industrial"] = industrial_gate_metrics(labels, decisions)
    return result


def evaluate_official_splits(
    experiments: Optional[Sequence[str]] = None,
    seeds: Sequence[int] = OFFICIAL_SPLIT_SEEDS,
    allow_fixture: bool = True,
    config: Optional[dict] = None,
) -> Dict[str, Any]:
    cfg = config or load_libad_config()
    selected = list(experiments or EXPERIMENT_MODALITIES.keys())
    splits = load_official_splits(seeds=seeds, allow_fixture=allow_fixture, config=cfg)
    started = time.perf_counter()
    by_experiment: Dict[str, List[Dict[str, Any]]] = {name: [] for name in selected}
    raw_predictions: List[Dict[str, Any]] = []
    for split in splits:
        for experiment in selected:
            record = evaluate_split(split, experiment, config=cfg)
            by_experiment[experiment].append(record)
            raw_predictions.extend(record["predictions"])
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    summary = {}
    for experiment, records in by_experiment.items():
        payload: Dict[str, Any] = {
            "academic": summarize_splits(item["academic"] for item in records),
            "n_splits": len(records),
            "purpose": {
                "unimodal_vis": "Baseline surface inspection",
                "unimodal_xray_l": "Baseline internal/density inspection",
                "multimodal": "Complementary fusion value",
                "securecoating_gate": "PASS/REJECT/HOLD safety of acting on the scores",
            }.get(experiment, experiment),
        }
        if experiment == "securecoating_gate":
            payload["industrial"] = summarize_splits(
                item["industrial"] for item in records if "industrial" in item
            )
        summary[experiment] = payload
    structured_inputs = bool(splits) and all(
        split.source == "libad_structured_inputs" for split in splits
    )
    official_data = structured_inputs and bool(input_status["official_protocol_complete"])
    # This harness intentionally uses the repository's numpy patch descriptor.
    # Official data does not make its metrics comparable to the authors'
    # DINOv3/DA-Core implementation.
    comparable = False
    official_protocol_complete = (
        official_data and tuple(int(seed) for seed in seeds) == OFFICIAL_SPLIT_SEEDS
    )
    return {
        "benchmark": "LIBAD",
        "citation": LIBAD_CITATION,
        "paper_result_note": LIBAD_PAPER_RESULT_NOTE,
        "local_contribution": (
            "SecureCoating-Vision adds an evidence-gated industrial decision layer. "
            "It does not replace YOLO/ONNX surface localization and does not claim DA-Core."
        ),
        "official_split_seeds": list(seeds),
        "evidence_class": (
            "official_libad_local_adapter" if official_data
            else "unverified_libad_local_adapter" if structured_inputs
            else "protocol_fixture"
        ),
        "comparable_to_paper": comparable,
        "official_protocol_complete": official_protocol_complete,
        "paper_comparability_blockers": [
            "The local feature backbone is numpy_patch_descriptor, not the authors' official DINOv3 implementation",
            "No official external baseline runner/weight hash is recorded by this harness",
        ],
        "experiments": summary,
        "split_records": {
            experiment: [
                {k: v for k, v in record.items() if k != "predictions"}
                for record in records
            ]
            for experiment, records in by_experiment.items()
        },
        "latency_ms_total": elapsed_ms,
        "hashes": {
            "commit": git_commit_sha(),
            "source_provenance": git_source_provenance(),
            "adapter_implementation": hash_existing_files(
                [
                    "configs/libad.yaml",
                    "src/libad/features.py",
                    "src/libad/memory.py",
                    "src/libad/scorer.py",
                    "src/libad/evidence_gate.py",
                    "src/libad/dataset.py",
                ]
            ),
            "official_dataset_tree_sha256": (
                input_status["dataset_tree_sha256"] if official_data else None
            ),
            "official_splits_tree_sha256": (
                input_status["splits_tree_sha256"] if official_data else None
            ),
            "protocol_fixture_definition": (
                hash_existing_files(["src/libad/dataset.py", "configs/libad.yaml"])
                if not official_data else None
            ),
        },
        "predictions": raw_predictions,
        "feature_backbone": cfg["features"]["backbone"],
        "feature_backbone_note": cfg["features"]["backbone_note"],
    }
