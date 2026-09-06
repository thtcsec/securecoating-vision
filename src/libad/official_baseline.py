"""Harness helpers for the authors' evenrose/LIBAD DINOv3 + DA-Core runner.

This module does not reimplement DA-Core. It only launches the upstream
``run.py``, aggregates a harness-owned ledger (not a stale global CSV), and
records honest comparability status.

Protocol identities (must stay distinct):
- PAPER_SPEC (arXiv:2608.07958 text): frozen DINOv3 ViT-S/16 + DA-FPS + max-NN score.
- OFFICIAL_CODE_CORE: pinned evenrose/LIBAD defaults often resolve to ConvNeXt-base.
  That is *not* automatically PAPER_EXACT while paper↔code disagree.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from libad.dataset import dataset_status
from libad.official_code import OFFICIAL_CODE_RELATIVE, official_code_root, official_code_status
from libad.protocol import (
    LIBAD_CITATION,
    LIBAD_PAPER_RESULT_NOTE,
    OFFICIAL_SPLIT_SEEDS,
    PROJECT_ROOT,
    load_libad_config,
)

# Upstream evenrose/LIBAD CLI default path many READMEs reproduce (ConvNeXt-base).
OFFICIAL_CODE_BACKBONE_VARIANT = "base"
PAPER_BACKBONE_VARIANT = OFFICIAL_CODE_BACKBONE_VARIANT  # legacy alias; not PAPER_SPEC
# Textual paper methodology for DA-Core (ar5iv 2608.07958): DINOv3 ViT-S/16.
PAPER_SPEC_BACKBONE_FAMILY = "vit"
PAPER_SPEC_DINO_VERSION = "v3"
PAPER_SPEC_BACKBONE_VARIANT = "small"
PAPER_CODE_CONSISTENCY = "mismatch"
DEFAULT_VRAM_BACKBONE_VARIANT = "tiny"
LEDGER_FIELDNAMES = [
    "schema_version",
    "run_id",
    "config_sha256",
    "started_at_utc",
    "finished_at_utc",
    "seed",
    "modality",
    "dino_version",
    "backbone_family",
    "backbone_variant",
    "extractor_precision",
    "coreset_selection_method",
    "coreset_density_weight",
    "f_coreset",
    "returncode",
    "image_auroc",
    "image_aupr",
    "image_best_f1",
    "image_fpr95",
]
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


def new_run_id() -> str:
    return uuid.uuid4().hex


def config_fingerprint(config: Dict[str, Any]) -> str:
    payload = {
        "backbone_family": str(config.get("backbone_family", "")).lower(),
        "dino_version": str(config.get("dino_version", "")).lower(),
        "backbone_variant": str(config.get("backbone_variant", "")).lower(),
        "extractor_precision": str(config.get("extractor_precision", "")).lower(),
        "coreset_selection_method": str(config.get("coreset_selection_method", "")).lower(),
        "coreset_density_weight": float(config.get("coreset_density_weight", 0.0)),
        "f_coreset": float(config.get("f_coreset", 0.0)),
        "resize_h": int(config.get("resize_h") or 0),
        "resize_w": int(config.get("resize_w") or 0),
        "batch_size": int(config.get("batch_size") or 0),
        "modalities": list(config.get("modalities") or []),
        "seeds": list(config.get("seeds") or []),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def paper_spec_match(config: Dict[str, Any]) -> bool:
    """True only for the textual paper DA-Core backbone (DINOv3 ViT-S/16) + DA-FPS knobs."""
    return (
        str(config.get("backbone_family", "")).lower() == PAPER_SPEC_BACKBONE_FAMILY
        and str(config.get("dino_version", "")).lower() == PAPER_SPEC_DINO_VERSION
        and str(config.get("backbone_variant", "")).lower() == PAPER_SPEC_BACKBONE_VARIANT
        and str(config.get("coreset_selection_method", "")).lower() == "density_fps"
        and abs(float(config.get("f_coreset", 0.0)) - 0.05) < 1e-9
        and abs(float(config.get("coreset_density_weight", 0.0)) - 0.7) < 1e-9
    )


def official_code_core_match(config: Dict[str, Any]) -> bool:
    """True for the common pinned-upstream ConvNeXt-base core knobs (not PAPER_EXACT)."""
    return (
        str(config.get("backbone_family", "")).lower() == "convnext"
        and str(config.get("dino_version", "")).lower() == "v3"
        and str(config.get("backbone_variant", "")).lower() == OFFICIAL_CODE_BACKBONE_VARIANT
        and str(config.get("coreset_selection_method", "")).lower() == "density_fps"
        and abs(float(config.get("f_coreset", 0.0)) - 0.05) < 1e-9
        and abs(float(config.get("coreset_density_weight", 0.0)) - 0.7) < 1e-9
    )


def paper_config_match(config: Dict[str, Any]) -> bool:
    """Deprecated alias: upstream ConvNeXt-base core match (NOT textual paper ViT-S/16)."""
    return official_code_core_match(config)


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
    if not paper_spec_match(config):
        blockers.append(
            "config is not textual PAPER_SPEC (DINOv3 ViT-S/16); "
            f"got {config.get('backbone_family')}/{config.get('dino_version')}/"
            f"{config.get('backbone_variant')}"
        )
    if not official_code_core_match(config):
        blockers.append(
            f"backbone_variant={backbone_variant!r} is not official-code core "
            f"{OFFICIAL_CODE_BACKBONE_VARIANT!r} (VRAM-adapted / adapted reproduction)"
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
        "paper_code_consistency": PAPER_CODE_CONSISTENCY,
        "paper_spec_match": paper_spec_match(config),
        "official_code_core_match": official_code_core_match(config),
        "paper_config_match": official_code_core_match(config),
        "comparable_to_paper_eligible": not blockers and paper_spec_match(config),
        "paper_comparability_blockers": blockers,
        "attribution": LIBAD_CITATION["da_core_attribution"],
        "paper_result_note": LIBAD_PAPER_RESULT_NOTE,
        "cwd_required": OFFICIAL_CODE_RELATIVE,
        "output_report": "reports/libad/official_dinov3_dacore.json",
        "predictions_dir": "outputs/libad_official_baseline",
    }


def _mean_std(values: Sequence[float]) -> Dict[str, Any]:
    clean = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not clean:
        # Use nulls, not NaN — keep reports strict-JSON serializable.
        return {"mean": None, "std": None, "n": 0}
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


def _norm_token(value: Any) -> str:
    return str(value or "").strip().lower()


def _float_close(raw: Any, expected: float) -> bool:
    try:
        return abs(float(raw) - expected) < 1e-9
    except (TypeError, ValueError):
        return False


def _row_config_matches(
    row: Dict[str, Any],
    *,
    dino_version: str,
    backbone_family: str,
    backbone_variant: str,
    f_coreset: float,
    coreset_density_weight: float,
) -> bool:
    """Reject rows that disagree with requested config when fields are present.

    Missing scientific fields are rejected: silent defaults would reintroduce
    stale-CSV mixing across DINOv2/v3 and ConvNeXt/ViT runs.
    """
    required = {
        "dino_version": _norm_token(dino_version),
        "backbone_family": _norm_token(backbone_family),
        "backbone_variant": _norm_token(backbone_variant),
        "coreset_selection_method": "density_fps",
    }
    for key, expected in required.items():
        raw = row.get(key)
        if raw in (None, ""):
            return False
        if _norm_token(raw) != expected:
            return False
    if row.get("f_coreset") in (None, "") or not _float_close(row.get("f_coreset"), f_coreset):
        return False
    if row.get("coreset_density_weight") in (None, "") or not _float_close(
        row.get("coreset_density_weight"), coreset_density_weight
    ):
        return False
    return True


def _cell_key(seed: int, modality: str) -> Tuple[int, str]:
    return (int(seed), str(modality).strip())


def append_harness_ledger_row(ledger_path: Path, row: Dict[str, Any]) -> None:
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    exists = ledger_path.is_file()
    with ledger_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LEDGER_FIELDNAMES, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in LEDGER_FIELDNAMES})


def harvest_upstream_metrics(
    csv_path: Path,
    *,
    seed: int,
    modality: str,
    dino_version: str,
    backbone_family: str,
    backbone_variant: str,
    f_coreset: float = 0.05,
    coreset_density_weight: float = 0.7,
) -> Dict[str, Any]:
    """Take the last matching upstream row for one cell; empty dict if none."""
    matches: List[Dict[str, Any]] = []
    for row in _read_csv_rows(csv_path):
        try:
            row_seed = int(float(row.get("seed") or row.get("dataset_seed") or -1))
        except (TypeError, ValueError):
            continue
        if row_seed != int(seed):
            continue
        if str(row.get("modality") or "").strip() != modality:
            continue
        if not _row_config_matches(
            row,
            dino_version=dino_version,
            backbone_family=backbone_family,
            backbone_variant=backbone_variant,
            f_coreset=f_coreset,
            coreset_density_weight=coreset_density_weight,
        ):
            # Upstream CSV often omits config columns; fall back to variant+method only
            # for *harvest after a live process*, then stamp full config into our ledger.
            variant = str(row.get("backbone_variant") or "").strip()
            method = str(row.get("coreset_selection_method") or "").strip()
            has_strict = any(
                row.get(k) not in (None, "")
                for k in ("dino_version", "backbone_family", "f_coreset", "coreset_density_weight")
            )
            if has_strict:
                continue
            if variant and variant != backbone_variant:
                continue
            if method and method != "density_fps":
                continue
        matches.append(row)
    if not matches:
        return {}
    last = matches[-1]
    out: Dict[str, Any] = {}
    for key in ("image_auroc", "image_aupr", "image_best_f1", "image_fpr95"):
        if last.get(key) not in (None, ""):
            out[key] = last.get(key)
    return out


def _experiment_purpose(
    modality: str,
    *,
    dino_version: str,
    backbone_family: str,
    backbone_variant: str,
) -> str:
    dino = str(dino_version or "v3").lower().lstrip("v")
    family = str(backbone_family or "convnext").lower()
    variant = str(backbone_variant or "base").lower()
    label = f"DINO v{dino} {family}-{variant}"
    if modality == "vis":
        return f"Authors' runner {label} VIS baseline"
    if modality == "xray_l":
        return f"Authors' runner {label} X-rayL baseline"
    if modality == "vis_xray_l":
        return f"Authors' runner {label} + DA-Core multimodal baseline"
    return f"Authors' runner {label} ({modality})"


def aggregate_experiment_log(
    csv_path: Path,
    *,
    seeds: Sequence[int],
    modalities: Sequence[str],
    backbone_variant: str,
    dino_version: str = "v3",
    backbone_family: str = "convnext",
    f_coreset: float = 0.05,
    coreset_density_weight: float = 0.7,
    run_id: Optional[str] = None,
    accepted_cells: Optional[Sequence[Tuple[int, str]]] = None,
) -> Dict[str, Any]:
    """Aggregate metrics only from this invocation's accepted cells / run_id.

    Never trusts a global CSV alone: without ``run_id`` or ``accepted_cells``,
    no metric rows are kept (prevents stale-run inflation).
    """
    accepted: Optional[Set[Tuple[int, str]]] = None
    if accepted_cells is not None:
        accepted = {_cell_key(s, m) for s, m in accepted_cells}
    rows = _read_csv_rows(csv_path)
    seed_set = {int(s) for s in seeds}
    selected: List[Dict[str, Any]] = []
    for row in rows:
        try:
            seed = int(float(row.get("seed") or row.get("dataset_seed") or -1))
        except (TypeError, ValueError):
            continue
        modality = str(row.get("modality") or "").strip()
        if seed not in seed_set or modality not in modalities:
            continue
        if run_id is not None:
            if str(row.get("run_id") or "").strip() != run_id:
                continue
        elif accepted is not None:
            if _cell_key(seed, modality) not in accepted:
                continue
        else:
            # No provenance scope → refuse to harvest (stale-CSV guard).
            continue
        if not _row_config_matches(
            row,
            dino_version=dino_version,
            backbone_family=backbone_family,
            backbone_variant=backbone_variant,
            f_coreset=f_coreset,
            coreset_density_weight=coreset_density_weight,
        ):
            continue
        selected.append(row)

    experiments: Dict[str, Any] = {}
    for modality in modalities:
        key = MODALITY_KEYS.get(modality, modality)
        modality_rows = [r for r in selected if r.get("modality") == modality]
        # Prefer finished_at_utc when present; else last matching row in file order.
        latest: Dict[int, Dict[str, Any]] = {}
        for row in modality_rows:
            seed = int(float(row.get("seed") or row.get("dataset_seed") or -1))
            prev = latest.get(seed)
            if prev is None:
                latest[seed] = row
                continue
            prev_ts = str(prev.get("finished_at_utc") or "")
            cur_ts = str(row.get("finished_at_utc") or "")
            if cur_ts and cur_ts >= prev_ts:
                latest[seed] = row
            elif not prev_ts:
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
            "purpose": _experiment_purpose(
                modality,
                dino_version=dino_version,
                backbone_family=backbone_family,
                backbone_variant=backbone_variant,
            ),
        }
    return {
        "source_csv": str(csv_path.as_posix()),
        "n_matching_rows": len(selected),
        "run_id": run_id,
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
    run_id: str = "",
    config_sha256: str = "",
) -> List[str]:
    py = sys.executable
    note = (
        f"securecoating_official_baseline seed={seed} modality={modality} "
        f"dino={dino_version} family={backbone_family} profile=laptop_safe"
    )
    if run_id:
        note += f" run_id={run_id}"
    if config_sha256:
        note += f" config_sha256={config_sha256[:16]}"
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
        note,
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
    run_id: Optional[str] = None,
    config_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    blockers: List[str] = []
    records = list(run_records)
    expected_modalities = list(config.get("modalities") or [])
    # Empty seeds must NOT silently expand to all official seeds (false OK).
    raw_seeds = config.get("seeds")
    expected_seeds = list(raw_seeds) if raw_seeds is not None else list(OFFICIAL_SPLIT_SEEDS)

    code_ok = bool(code_meta.get("present"))
    dataset_ok = bool(dataset_meta.get("official_protocol_complete"))
    if not code_ok:
        blockers.append("authors' evenrose/LIBAD checkout missing")
    if not dataset_ok:
        blockers.append("official LIBAD mount incomplete")

    paper_exact_cfg = paper_spec_match(config)
    official_core_cfg = official_code_core_match(config)
    if not paper_exact_cfg:
        blockers.append(
            "config is not textual PAPER_SPEC (DINOv3 ViT-S/16 + DA-FPS λ=0.7, f=0.05); "
            f"paper↔code consistency={PAPER_CODE_CONSISTENCY}"
        )
    if not official_core_cfg:
        blockers.append(
            "config differs from official-code core defaults "
            f"(ConvNeXt-{OFFICIAL_CODE_BACKBONE_VARIANT}); "
            f"got {config.get('backbone_family')}/{config.get('dino_version')}/"
            f"{config.get('backbone_variant')}"
        )

    expected_cells = len(expected_seeds) * len(expected_modalities)
    processes_ok = (
        bool(records)
        and expected_cells > 0
        and len(records) >= expected_cells
        and all(int(r.get("returncode") or 0) == 0 for r in records)
    )
    if not processes_ok:
        blockers.append(
            "official runner processes incomplete or failed "
            f"(records={len(records)}, expected_cells={expected_cells})"
        )

    coverage_ok = (
        bool(expected_modalities)
        and bool(expected_seeds)
        and all(
            int(
                ((aggregation.get("experiments") or {}).get(MODALITY_KEYS.get(m, m)) or {}).get(
                    "n_splits"
                )
                or 0
            )
            >= len(expected_seeds)
            for m in expected_modalities
        )
    )
    if not coverage_ok:
        for modality in expected_modalities:
            key = MODALITY_KEYS.get(modality, modality)
            block = (aggregation.get("experiments") or {}).get(key) or {}
            blockers.append(
                f"incomplete {modality} coverage: {block.get('n_splits', 0)}/{len(expected_seeds)} seeds"
            )
        if not expected_modalities or not expected_seeds:
            blockers.append("empty modalities/seeds — refusing false-complete coverage")

    if run_id is None and not aggregation.get("run_id"):
        blockers.append("missing harness run_id provenance scope")

    has_run_scope = bool(run_id or aggregation.get("run_id"))
    # Claim gates use booleans directly — blockers remain the human-readable audit trail.
    comparable = (
        code_ok and dataset_ok and processes_ok and coverage_ok and paper_exact_cfg and has_run_scope
    )
    official_repro = (
        code_ok and dataset_ok and processes_ok and coverage_ok and official_core_cfg and has_run_scope
    )

    if comparable:
        claim_class = "PAPER_EXACT"
        evidence_class = "official_libad_paper_spec_dinov3_vit_s"
    elif official_repro:
        claim_class = "OFFICIAL_CODE_REPRODUCTION"
        evidence_class = "official_libad_official_code_reproduction"
    elif not processes_ok or not coverage_ok:
        claim_class = "INVALID"
        evidence_class = "official_libad_authors_runner_partial_or_adapted"
    else:
        claim_class = "ADAPTED_REPRODUCTION"
        evidence_class = "official_libad_authors_runner_partial_or_adapted"

    dino_version = str(config.get("dino_version", "v3"))
    backbone_family = str(config.get("backbone_family", "convnext"))
    backbone_variant = str(config.get("backbone_variant", "base"))
    failed = any(int(r.get("returncode") or 0) != 0 for r in records)
    status = "OK" if processes_ok and coverage_ok else ("FAILED" if failed or not records else "PARTIAL")
    runner_label = _experiment_purpose(
        "vis_xray_l",
        dino_version=dino_version,
        backbone_family=backbone_family,
        backbone_variant=backbone_variant,
    )
    return {
        "benchmark": "LIBAD",
        "citation": LIBAD_CITATION,
        "paper_result_note": LIBAD_PAPER_RESULT_NOTE,
        "local_contribution": (
            "SecureCoating-Vision launches and records the authors' evenrose/LIBAD "
            f"runner ({runner_label}). DA-Core remains attributed to Sui et al. The "
            "repository contribution is the evidence-gated PASS/REJECT/HOLD layer."
        ),
        "claim_class": claim_class,
        "paper_code_consistency": PAPER_CODE_CONSISTENCY,
        "paper_spec_match": paper_exact_cfg,
        "official_code_core_match": official_core_cfg,
        "official_code_reproduction": bool(official_repro),
        "evidence_class": evidence_class,
        "comparable_to_paper": bool(comparable),
        "paper_comparability_blockers": blockers,
        "status": status,
        "status_note": (
            None
            if status == "OK"
            else (
                "Retained for transparency: authors' runner did not complete a "
                "trusted result. Do not treat incomplete metrics as paper-comparable."
                if status == "FAILED"
                else "Partial authors' runner coverage; not paper-comparable."
            )
        ),
        "dataset_protocol_complete": dataset_ok,
        "experiment_coverage_complete": bool(coverage_ok),
        # Legacy key: kept as experiment coverage for older readers; prefer the two fields above.
        "official_protocol_complete": bool(coverage_ok),
        "feature_backbone": (
            f"dino{dino_version}_"
            f"{backbone_family}_"
            f"{backbone_variant}"
        ),
        "coreset_method": "density_fps",
        "coreset_attribution": LIBAD_CITATION["da_core_attribution"],
        "official_code": code_meta,
        "dataset": dataset_meta,
        "run_config": config,
        "run_id": run_id or aggregation.get("run_id"),
        "config_sha256": config_sha256 or config.get("config_sha256"),
        "official_split_seeds": expected_seeds,
        "experiments": aggregation.get("experiments") or {},
        "run_records": records,
        "recorded_at": _utc_now(),
        "source_csv": aggregation.get("source_csv"),
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
