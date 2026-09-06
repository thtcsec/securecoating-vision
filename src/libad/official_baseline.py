"""Harness helpers for the authors' evenrose/LIBAD DINOv3 + DA-Core runner.

This module does not reimplement DA-Core. It only launches the upstream
``run.py``, aggregates ``results/experiment_log.csv``, and records honest
comparability status.
"""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from libad.dataset import dataset_status
from libad.official_code import OFFICIAL_CODE_RELATIVE, official_code_root, official_code_status
from libad.protocol import (
    LIBAD_CITATION,
    LIBAD_PAPER_RESULT_NOTE,
    OFFICIAL_SPLIT_SEEDS,
    PROJECT_ROOT,
    load_libad_config,
)

PAPER_BACKBONE_VARIANT = "base"
DEFAULT_VRAM_BACKBONE_VARIANT = "tiny"
DINOV3_GATED_MODELS = (
    "facebook/dinov3-convnext-tiny-pretrain-lvd1689m",
    "facebook/dinov3-convnext-small-pretrain-lvd1689m",
    "facebook/dinov3-convnext-base-pretrain-lvd1689m",
)
MODALITY_KEYS = {
    "vis": "unimodal_vis",
    "xray_l": "unimodal_xray_l",
    "vis_xray_l": "multimodal",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_project_env(env_path: Optional[Path] = None) -> Dict[str, bool]:
    """Load KEY=VALUE pairs from the project .env without overriding existing env.

    Never logs secret values. Returns presence flags only.
    """
    path = env_path or (PROJECT_ROOT / ".env")
    loaded = {
        "env_file_present": path.is_file(),
        "hf_token_loaded": False,
        "hf_hub_token_loaded": False,
    }
    if not path.is_file():
        return loaded
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return loaded
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key or not value:
            continue
        if key in os.environ and os.environ.get(key):
            continue
        os.environ[key] = value
        if key == "HF_TOKEN":
            loaded["hf_token_loaded"] = True
        elif key == "HUGGING_FACE_HUB_TOKEN":
            loaded["hf_hub_token_loaded"] = True
    return loaded


def hf_auth_present() -> bool:
    load_project_env()
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return True
    try:
        from huggingface_hub import get_token

        return bool(get_token())
    except Exception:
        return False


def _probe_hf_model_access(repo_id: str) -> Dict[str, Any]:
    """Return access status for a gated/public HF repo without leaking tokens.

    Uses a small file download (not only model_info) because metadata can look
    readable while weight/processor files still return 403 for gated repos.
    """
    try:
        from huggingface_hub import hf_hub_download, model_info
        from huggingface_hub.errors import EntryNotFoundError
        from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError, HfHubHTTPError
    except Exception as exc:
        return {"repo_id": repo_id, "ok": False, "status": "hub_import_error", "detail": type(exc).__name__}

    try:
        model_info(repo_id, token=True)
        last_missing: Optional[Exception] = None
        for filename in ("processor_config.json", "config.json", "preprocessor_config.json"):
            try:
                hf_hub_download(repo_id, filename, token=True)
                return {"repo_id": repo_id, "ok": True, "status": "ok", "detail": "authorized"}
            except EntryNotFoundError as exc:
                last_missing = exc
                continue
        if last_missing is not None:
            # Repo metadata is reachable; none of the common tiny files exist.
            return {"repo_id": repo_id, "ok": True, "status": "ok", "detail": "authorized_metadata_only"}
        return {"repo_id": repo_id, "ok": True, "status": "ok", "detail": "authorized"}
    except GatedRepoError:
        return {"repo_id": repo_id, "ok": False, "status": "403_gated", "detail": "gated_or_unauthorized"}
    except RepositoryNotFoundError:
        return {"repo_id": repo_id, "ok": False, "status": "404", "detail": "not_found"}
    except HfHubHTTPError as exc:
        code = getattr(getattr(exc, "response", None), "status_code", None)
        return {"repo_id": repo_id, "ok": False, "status": str(code or "http_error"), "detail": "http_error"}
    except Exception as exc:
        return {"repo_id": repo_id, "ok": False, "status": "error", "detail": type(exc).__name__}


def dinov3_access_status(*, probe: bool = True) -> Dict[str, Any]:
    load_project_env()
    auth = hf_auth_present()
    probes: List[Dict[str, Any]] = []
    authorized = False
    if auth and probe:
        # Probe the default VRAM-adapted tiny weights first.
        tiny = _probe_hf_model_access(DINOV3_GATED_MODELS[0])
        probes.append(tiny)
        authorized = bool(tiny.get("ok"))
    if not auth:
        note = (
            "DINOv3 ConvNeXt weights on Hugging Face are gated. Accept the model "
            "terms for facebook/dinov3-convnext-* and put HF_TOKEN in .env (or the "
            "process environment / huggingface-cli login)."
        )
    elif not probe:
        note = (
            "HF auth detected; DINOv3 ConvNeXt model access not probed "
            "(offline-safe). Pass probe=True to verify gated repo authorization."
        )
    elif not authorized:
        note = (
            "HF_TOKEN detected, but this account is not yet authorized for the "
            "gated DINOv3 ConvNeXt repo. Open the model page, request/accept access, "
            "wait for approval if required, then re-run."
        )
    else:
        note = "HF auth detected and DINOv3 ConvNeXt-tiny access probe succeeded."
    return {
        "hf_auth_present": auth,
        "model_access_ok": bool(authorized),
        "gated_models": list(DINOV3_GATED_MODELS),
        "probes": probes,
        "ready": bool(auth and authorized),
        "note": note,
    }


def paper_config_match(config: Dict[str, Any]) -> bool:
    return (
        str(config.get("backbone_family", "")).lower() == "convnext"
        and str(config.get("dino_version", "")).lower() == "v3"
        and str(config.get("backbone_variant", "")).lower() == PAPER_BACKBONE_VARIANT
        and str(config.get("coreset_selection_method", "")).lower() == "density_fps"
        and abs(float(config.get("f_coreset", 0.0)) - 0.05) < 1e-9
        and abs(float(config.get("coreset_density_weight", 0.0)) - 0.7) < 1e-9
    )


def run_card(
    *,
    backbone_variant: str = DEFAULT_VRAM_BACKBONE_VARIANT,
    batch_size: int = 1,
    precision: str = "fp16",
    probe: bool = False,
) -> Dict[str, Any]:
    cfg = load_libad_config()
    status = dataset_status(cfg, verify_trees=False)
    code = official_code_status()
    access = dinov3_access_status(probe=probe)
    laptop = laptop_runner_defaults()
    config = {
        "profile": laptop["profile"],
        "backbone_family": "convnext",
        "dino_version": "v3",
        "backbone_variant": backbone_variant,
        "extractor_precision": precision,
        "coreset_selection_method": "density_fps",
        "coreset_density_weight": 0.7,
        "f_coreset": 0.05,
        "batch_size": batch_size,
        "test_batch_size": batch_size,
        "num_workers": 0,
        "resize_h": 640,
        "resize_w": 512,
        "coreset_device": laptop["coreset_device"],
        "distance_device": laptop["distance_device"],
        "distance_chunk_size": laptop["distance_chunk_size"],
        "density_chunk_size": laptop["density_chunk_size"],
        "modalities": list(laptop["modalities"]),
        "seeds": list(OFFICIAL_SPLIT_SEEDS),
        "pause_seconds": laptop["pause_seconds"],
        "save_bank": False,
    }
    blockers: List[str] = []
    if not code.get("present"):
        blockers.append("authors' evenrose/LIBAD checkout missing under third_party/evenrose-libad")
    if not status.get("official_protocol_complete"):
        blockers.append("official LIBAD mount incomplete or hashes not trusted")
    if not access.get("hf_auth_present"):
        blockers.append("Hugging Face auth missing for gated DINOv3 ConvNeXt weights")
    elif not access.get("model_access_ok"):
        if access.get("probes"):
            blockers.append("DINOv3 ConvNeXt gated access not authorized (403)")
        else:
            blockers.append("DINOv3 ConvNeXt gated access not verified (probe skipped)")
    if not paper_config_match(config):
        blockers.append(
            f"backbone_variant={backbone_variant!r} is not paper default "
            f"{PAPER_BACKBONE_VARIANT!r} (VRAM-adapted run on constrained GPUs)"
        )
    return {
        "generated_at": _utc_now(),
        "official_code": code,
        "dataset_status": {
            "official_protocol_complete": status.get("official_protocol_complete"),
            "dataset_tree_sha256": status.get("dataset_tree_sha256"),
            "splits_tree_sha256": status.get("splits_tree_sha256"),
            "comparability_blockers": status.get("comparability_blockers"),
        },
        "dinov3_access": access,
        "planned_config": config,
        "paper_config_match": paper_config_match(config),
        "comparable_to_paper_eligible": not blockers and paper_config_match(config),
        "paper_comparability_blockers": blockers,
        "attribution": LIBAD_CITATION["da_core_attribution"],
        "paper_result_note": LIBAD_PAPER_RESULT_NOTE,
        "cwd_required": OFFICIAL_CODE_RELATIVE,
        "output_report": "reports/libad/official_dinov3_dacore.json",
        "predictions_dir": "outputs/libad_official_baseline",
    }


def _mean_std(values: Sequence[float]) -> Dict[str, float]:
    clean = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not clean:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    if len(clean) == 1:
        return {"mean": clean[0], "std": 0.0, "n": 1}
    return {
        "mean": float(statistics.mean(clean)),
        "std": float(statistics.stdev(clean)),
        "n": len(clean),
    }


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def aggregate_experiment_log(
    csv_path: Path,
    *,
    seeds: Sequence[int],
    modalities: Sequence[str],
    backbone_variant: str,
) -> Dict[str, Any]:
    rows = _read_csv_rows(csv_path)
    seed_set = {int(s) for s in seeds}
    selected: List[Dict[str, Any]] = []
    for row in rows:
        try:
            seed = int(float(row.get("seed") or row.get("dataset_seed") or -1))
        except (TypeError, ValueError):
            continue
        modality = str(row.get("modality") or "").strip()
        variant = str(row.get("backbone_variant") or "").strip()
        method = str(row.get("coreset_selection_method") or "").strip()
        if seed not in seed_set or modality not in modalities:
            continue
        if variant and variant != backbone_variant:
            continue
        if method and method != "density_fps":
            continue
        selected.append(row)

    experiments: Dict[str, Any] = {}
    for modality in modalities:
        key = MODALITY_KEYS.get(modality, modality)
        modality_rows = [r for r in selected if r.get("modality") == modality]
        # Keep latest row per seed.
        latest: Dict[int, Dict[str, Any]] = {}
        for row in modality_rows:
            seed = int(float(row.get("seed") or row.get("dataset_seed") or -1))
            latest[seed] = row
        kept = [latest[s] for s in seeds if s in latest]
        academic = {}
        for metric_src, metric_dst in (
            ("image_auroc", "auroc"),
            ("image_aupr", "aupr"),
            ("image_best_f1", "f1_max"),
            ("image_fpr95", "fpr95"),
        ):
            vals = []
            for row in kept:
                raw = row.get(metric_src)
                if raw in (None, ""):
                    continue
                vals.append(float(raw))
            academic[metric_dst] = _mean_std(vals)
        experiments[key] = {
            "academic": academic,
            "n_splits": len(kept),
            "seeds_present": [int(float(r.get("seed") or r.get("dataset_seed"))) for r in kept],
            "purpose": {
                "vis": "Official authors' DINOv3 VIS baseline",
                "xray_l": "Official authors' DINOv3 X-rayL baseline",
                "vis_xray_l": "Official authors' DINOv3+DA-Core multimodal baseline",
            }.get(modality, modality),
        }
    return {
        "source_csv": str(csv_path.as_posix()),
        "n_matching_rows": len(selected),
        "experiments": experiments,
        "raw_rows": selected,
    }


def apply_laptop_process_guards() -> Dict[str, bool]:
    """Lower Windows process priority and set HF/CUDA env to reduce thrash.

    Does not read dataset trees or download models.
    """
    load_project_env()
    applied = {
        "below_normal_priority": False,
        "hf_hub_disable_xet": False,
        "cuda_alloc_conf": False,
    }
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    applied["hf_hub_disable_xet"] = os.environ.get("HF_HUB_DISABLE_XET") == "1"
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
    # Prefer expandable segments over aggressive caching that spikes RAM/VRAM.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
    applied["cuda_alloc_conf"] = True
    if os.name == "nt":
        try:
            import ctypes

            handle = ctypes.windll.kernel32.GetCurrentProcess()
            # BELOW_NORMAL_PRIORITY_CLASS
            applied["below_normal_priority"] = bool(
                ctypes.windll.kernel32.SetPriorityClass(handle, 0x00004000)
            )
        except OSError:
            applied["below_normal_priority"] = False
    return applied


def laptop_runner_defaults() -> Dict[str, Any]:
    """Safe defaults for RTX 4050-class 6 GB laptops under thermal/I/O pressure."""
    return {
        "profile": "laptop",
        "backbone_family": "convnext",
        "dino_version": "v3",
        "backbone_variant": DEFAULT_VRAM_BACKBONE_VARIANT,
        "extractor_precision": "fp16",
        "batch_size": 1,
        "num_workers": 0,
        "coreset_device": "cpu",
        "distance_device": "cuda",
        "distance_chunk_size": 8192,
        "density_chunk_size": 512,
        "modalities": ["vis_xray_l"],
        "pause_seconds": 8,
        "save_bank": False,
        "save_raw_scores": False,
        "save_pixel_maps": False,
        "note": (
            "Laptop profile: one modality (vis_xray_l), batch=1, workers=0, "
            "coreset on CPU, smaller distance chunks, no bank dumps, below-normal "
            "priority. Not paper ConvNeXt-base."
        ),
    }


def build_run_command(
    *,
    seed: int,
    modality: str,
    backbone_variant: str,
    batch_size: int,
    precision: str,
    root_path: Path,
    split_dir: Path,
    results_csv: Path,
    results_md: Path,
    coreset_device: str = "cpu",
    distance_device: str = "cuda",
    backbone_family: str = "convnext",
    dino_version: str = "v3",
    distance_chunk_size: int = 8192,
    density_chunk_size: int = 512,
) -> List[str]:
    py = sys.executable
    return [
        py,
        "run.py",
        "--root_path",
        str(root_path),
        "--split_dir",
        str(split_dir),
        "--modality",
        modality,
        "--coreset_selection_method",
        "density_fps",
        "--coreset_density_weight",
        "0.7",
        "--f_coreset",
        "0.05",
        "--dataset_seed",
        str(seed),
        "--backbone_family",
        backbone_family,
        "--dino_version",
        dino_version,
        "--backbone_variant",
        backbone_variant,
        "--extractor_precision",
        precision,
        "--batch_size",
        str(batch_size),
        "--test_batch_size",
        str(batch_size),
        "--num_workers",
        "0",
        "--coreset_device",
        coreset_device,
        "--distance_device",
        distance_device,
        "--distance_chunk_size",
        str(int(distance_chunk_size)),
        "--density_chunk_size",
        str(int(density_chunk_size)),
        "--results_csv",
        str(results_csv),
        "--results_md",
        str(results_md),
        # Explicitly omit --save_bank / --save_raw_scores / --save_pixel_maps
        # to avoid multi-GB disk dumps on laptops.
        "--notes",
        (
            f"securecoating_official_baseline seed={seed} modality={modality} "
            f"dino={dino_version} family={backbone_family} profile=laptop_safe"
        ),
    ]


def launch_official_run(
    cmd: Sequence[str],
    *,
    cwd: Path,
    env: Optional[Dict[str, str]] = None,
) -> subprocess.CompletedProcess:
    apply_laptop_process_guards()
    merged = os.environ.copy()
    if env:
        merged.update(env)
    merged.setdefault("HF_HUB_DISABLE_XET", "1")
    merged.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    merged.setdefault("TOKENIZERS_PARALLELISM", "false")
    return subprocess.run(
        list(cmd),
        cwd=str(cwd),
        env=merged,
        check=False,
        text=True,
    )


def build_report(
    *,
    config: Dict[str, Any],
    aggregation: Dict[str, Any],
    dataset_meta: Dict[str, Any],
    code_meta: Dict[str, Any],
    run_records: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    blockers: List[str] = []
    if not code_meta.get("present"):
        blockers.append("authors' evenrose/LIBAD checkout missing")
    if not dataset_meta.get("official_protocol_complete"):
        blockers.append("official LIBAD mount incomplete")
    if not paper_config_match(config):
        blockers.append(
            "run config differs from paper defaults "
            f"(backbone_variant={config.get('backbone_variant')!r}; paper uses {PAPER_BACKBONE_VARIANT!r})"
        )
    expected_modalities = list(config.get("modalities") or [])
    expected_seeds = list(config.get("seeds") or list(OFFICIAL_SPLIT_SEEDS))
    for modality in expected_modalities:
        key = MODALITY_KEYS.get(modality, modality)
        block = (aggregation.get("experiments") or {}).get(key) or {}
        if int(block.get("n_splits") or 0) < len(expected_seeds):
            blockers.append(f"incomplete {modality} coverage: {block.get('n_splits', 0)}/{len(expected_seeds)} seeds")

    comparable = len(blockers) == 0 and paper_config_match(config)
    evidence_class = (
        "official_libad_dinov3_dacore"
        if comparable
        else "official_libad_authors_runner_partial_or_adapted"
    )
    return {
        "benchmark": "LIBAD",
        "citation": LIBAD_CITATION,
        "paper_result_note": LIBAD_PAPER_RESULT_NOTE,
        "local_contribution": (
            "SecureCoating-Vision launches and records the authors' evenrose/LIBAD "
            "DINOv3+DA-Core runner. DA-Core remains attributed to Sui et al. The "
            "repository contribution is the evidence-gated PASS/REJECT/HOLD layer."
        ),
        "evidence_class": evidence_class,
        "comparable_to_paper": bool(comparable),
        "paper_comparability_blockers": blockers,
        "official_protocol_complete": all(
            int(((aggregation.get("experiments") or {}).get(MODALITY_KEYS.get(m, m)) or {}).get("n_splits") or 0)
            >= len(expected_seeds)
            for m in expected_modalities
        ),
        "feature_backbone": (
            f"dino{config.get('dino_version', 'v3')}_"
            f"{config.get('backbone_family', 'convnext')}_"
            f"{config.get('backbone_variant')}"
        ),
        "coreset_method": "density_fps",
        "coreset_attribution": LIBAD_CITATION["da_core_attribution"],
        "official_code": code_meta,
        "dataset": dataset_meta,
        "run_config": config,
        "official_split_seeds": expected_seeds,
        "experiments": aggregation.get("experiments") or {},
        "run_records": list(run_records),
        "recorded_at": _utc_now(),
        "source_csv": aggregation.get("source_csv"),
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
