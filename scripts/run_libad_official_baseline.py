"""Launch the authors' evenrose/LIBAD DINOv3 + DA-Core runner on official splits.

Default profile is **laptop-safe** for RTX 4050-class 6 GB machines:
  - one modality (vis_xray_l) unless overridden
  - batch_size=1, num_workers=0
  - coreset on CPU, smaller distance chunks
  - no bank/score/pixel dumps
  - below-normal process priority + HF transfer throttles
  - pause between seed/modality cells

DA-Core and DINOv3 belong to Sui et al. / Meta. comparable_to_paper stays false
unless paper-default ConvNeXt-base completes all requested seeds/modalities.

Usage (lightweight first):
  .venv\\Scripts\\python.exe scripts/run_libad_official_baseline.py --run-card
  .venv\\Scripts\\python.exe scripts/run_libad_official_baseline.py --smoke

Full 10-seed multimodal (still laptop profile; do not leave unattended on a hot laptop):
  .venv\\Scripts\\python.exe scripts/run_libad_official_baseline.py --allow-heavy --seeds ...

Paper-config is opt-in and refused without --allow-heavy.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from libad.dataset import dataset_status  # noqa: E402
from libad.official_baseline import (  # noqa: E402
    DEFAULT_VRAM_BACKBONE_VARIANT,
    PAPER_BACKBONE_VARIANT,
    aggregate_experiment_log,
    apply_laptop_process_guards,
    build_report,
    build_run_command,
    dinov3_access_status,
    laptop_runner_defaults,
    launch_official_run,
    load_project_env,
    run_card,
    write_json,
)
from libad.official_code import official_code_root, official_code_status  # noqa: E402
from libad.protocol import OFFICIAL_SPLIT_SEEDS, load_libad_config  # noqa: E402


def _parse_seeds(raw: str) -> List[int]:
    if not raw.strip():
        return list(OFFICIAL_SPLIT_SEEDS)
    seeds = [int(item.strip()) for item in raw.split(",") if item.strip()]
    unknown = [seed for seed in seeds if seed not in OFFICIAL_SPLIT_SEEDS]
    if unknown:
        raise SystemExit(f"Non-official seeds refused: {unknown}")
    return seeds


def _parse_modalities(raw: str) -> List[str]:
    allowed = {"vis", "xray_l", "vis_xray_l"}
    items = [item.strip() for item in raw.split(",") if item.strip()]
    bad = [item for item in items if item not in allowed]
    if bad:
        raise SystemExit(f"Unsupported modalities: {bad}")
    return items or list(laptop_runner_defaults()["modalities"])


def main(argv: Optional[Sequence[str]] = None) -> int:
    load_project_env()
    laptop = laptop_runner_defaults()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-card", action="store_true", help="Print readiness card and exit")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="One seed (347), multimodal only. Intended first run on this laptop.",
    )
    parser.add_argument("--seeds", default="", help="Comma-separated official seeds")
    parser.add_argument(
        "--modalities",
        default="vis_xray_l",
        help="Default laptop-safe: vis_xray_l only. Pass vis,xray_l,vis_xray_l for full ablation.",
    )
    parser.add_argument(
        "--backbone-variant",
        default=DEFAULT_VRAM_BACKBONE_VARIANT,
        choices=["tiny", "small", "base"],
        help="Backbone variant. Default tiny (ConvNeXt) for 6 GB VRAM.",
    )
    parser.add_argument(
        "--backbone-family",
        default="convnext",
        choices=["convnext", "vit"],
        help="Upstream backbone_family. Paper path uses convnext; DINOv2 interim uses vit.",
    )
    parser.add_argument(
        "--dino-version",
        default="v3",
        choices=["v1", "v2", "v3"],
        help="Upstream dino_version. Paper-comparable path requires v3.",
    )
    parser.add_argument(
        "--paper-config",
        action="store_true",
        help=f"Force paper-default backbone_variant={PAPER_BACKBONE_VARIANT} (requires --allow-heavy)",
    )
    parser.add_argument("--batch-size", type=int, default=int(laptop["batch_size"]))
    parser.add_argument("--precision", default=str(laptop["extractor_precision"]), choices=["fp16", "bf16", "fp32"])
    parser.add_argument(
        "--coreset-device",
        default=str(laptop["coreset_device"]),
        choices=["cuda", "cpu"],
        help="Laptop default: cpu (less VRAM spike while building coreset).",
    )
    parser.add_argument(
        "--distance-device",
        default=str(laptop["distance_device"]),
        choices=["cuda", "cpu"],
    )
    parser.add_argument(
        "--distance-chunk-size",
        type=int,
        default=int(laptop["distance_chunk_size"]),
        help="Smaller chunks reduce RAM/VRAM spikes (laptop default 8192).",
    )
    parser.add_argument(
        "--density-chunk-size",
        type=int,
        default=int(laptop["density_chunk_size"]),
        help="Laptop default 512.",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=float(laptop["pause_seconds"]),
        help="Cool-down pause between seed/modality cells.",
    )
    parser.add_argument(
        "--allow-heavy",
        action="store_true",
        help="Required for >1 seed, multi-modality sweeps, or --paper-config on this laptop profile.",
    )
    parser.add_argument(
        "--out",
        default="reports/libad/official_dinov3_dacore.json",
        help="Checked-in summary report path",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Rebuild report from existing experiment_log.csv without launching runs",
    )
    args = parser.parse_args(argv)

    backbone = PAPER_BACKBONE_VARIANT if args.paper_config else args.backbone_variant
    paper_path = args.dino_version == "v3" and args.backbone_family == "convnext"
    if args.run_card:
        guards = apply_laptop_process_guards()
        card = run_card(
            backbone_variant=backbone,
            batch_size=args.batch_size,
            precision=args.precision,
            probe=False,
        )
        card["laptop_guards"] = guards
        card["laptop_defaults"] = laptop
        print(json.dumps(card, indent=2))
        ready = bool(card["dataset_status"]["official_protocol_complete"])
        return 0 if ready else 2

    seeds = [347] if args.smoke else _parse_seeds(args.seeds)
    modalities = ["vis_xray_l"] if args.smoke else _parse_modalities(args.modalities)
    cell_count = len(seeds) * len(modalities)
    heavy = (
        args.paper_config
        or cell_count > 1
        or len(modalities) > 1
        or backbone == PAPER_BACKBONE_VARIANT
    )
    if heavy and not args.allow_heavy and not args.aggregate_only and not args.smoke:
        print(
            "REFUSED: laptop-safe profile blocks multi-cell / paper-base runs without --allow-heavy.\n"
            "Use --smoke first (1 cell). Example full multimodal 10-seed:\n"
            "  .venv\\Scripts\\python.exe scripts/run_libad_official_baseline.py "
            "--allow-heavy --modalities vis_xray_l\n"
            "Keep Task Manager open; stop if disk/GPU thermals spike.",
            file=sys.stderr,
        )
        return 4
    if args.paper_config and not args.allow_heavy:
        print("REFUSED: --paper-config requires --allow-heavy on this laptop.", file=sys.stderr)
        return 4

    cfg = load_libad_config()
    ds = dataset_status(cfg, verify_trees=False)
    code = official_code_status()
    access = dinov3_access_status(probe=False)
    code_root = official_code_root()
    if not code.get("present"):
        print("ERROR: third_party/evenrose-libad missing. Run scripts/fetch_libad_official_code.py", file=sys.stderr)
        return 2
    if not ds.get("official_protocol_complete"):
        print("ERROR: official LIBAD mount is not protocol-complete/hash-verified.", file=sys.stderr)
        print(json.dumps(ds.get("comparability_blockers"), indent=2), file=sys.stderr)
        return 2
    if not args.aggregate_only and paper_path:
        # Probe only when about to run v3/convnext — one metadata call, not a dataset scan.
        access = dinov3_access_status(probe=True)
        if not access.get("ready"):
            print("ERROR: Hugging Face gated DINOv3 ConvNeXt access is required for the v3/convnext path.", file=sys.stderr)
            print(access["note"], file=sys.stderr)
            print(
                "Token may be in .env but DINOv3 model access still needs Accept on HF model page.\n"
                "Fallback interim (not paper-comparable):\n"
                "  .venv\\Scripts\\python.exe scripts/run_libad_official_baseline.py --smoke "
                "--dino-version v2 --backbone-family vit --backbone-variant small "
                "--out reports/libad/official_dinov2_dacore_interim.json",
                file=sys.stderr,
            )
            write_json(
                PROJECT_ROOT / "reports/libad/official_dinov3_run_card.json",
                run_card(backbone_variant=backbone, probe=True),
            )
            return 3

    results_csv = code_root / "results" / "securecoating_experiment_log.csv"
    results_md = code_root / "results" / "securecoating_experiment_log.md"
    root_path = (PROJECT_ROOT / cfg["paths"]["dataset_root"]).resolve()
    split_dir = (PROJECT_ROOT / cfg["paths"]["splits_root"]).resolve()
    pred_dir = PROJECT_ROOT / "outputs" / "libad_official_baseline"
    pred_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "profile": "laptop",
        "backbone_family": args.backbone_family,
        "dino_version": args.dino_version,
        "backbone_variant": backbone,
        "extractor_precision": args.precision,
        "coreset_selection_method": "density_fps",
        "coreset_density_weight": 0.7,
        "f_coreset": 0.05,
        "batch_size": args.batch_size,
        "test_batch_size": args.batch_size,
        "num_workers": 0,
        "resize_h": 640,
        "resize_w": 512,
        "coreset_device": args.coreset_device,
        "distance_device": args.distance_device,
        "distance_chunk_size": args.distance_chunk_size,
        "density_chunk_size": args.density_chunk_size,
        "pause_seconds": args.pause_seconds,
        "save_bank": False,
        "save_raw_scores": False,
        "save_pixel_maps": False,
        "modalities": modalities,
        "seeds": seeds,
    }
    if not paper_path:
        config["interim_note"] = (
            "DINOv2/ViT interim stronger baseline only. Not paper-comparable; "
            "paper table uses DINOv3 ConvNeXt + DA-Core."
        )

    run_records = []
    if not args.aggregate_only:
        guards = apply_laptop_process_guards()
        print(
            f"Laptop-safe official runner: cells={cell_count} seeds={seeds} "
            f"modalities={modalities} backbone={args.backbone_family}/{backbone} "
            f"dino={args.dino_version} coreset={args.coreset_device} "
            f"distance={args.distance_device} pause={args.pause_seconds}s "
            f"guards={guards}",
            flush=True,
        )
        for index, seed in enumerate(seeds):
            for modality in modalities:
                cmd = build_run_command(
                    seed=seed,
                    modality=modality,
                    backbone_variant=backbone,
                    batch_size=args.batch_size,
                    precision=args.precision,
                    root_path=root_path,
                    split_dir=split_dir,
                    results_csv=results_csv,
                    results_md=results_md,
                    coreset_device=args.coreset_device,
                    distance_device=args.distance_device,
                    backbone_family=args.backbone_family,
                    dino_version=args.dino_version,
                    distance_chunk_size=args.distance_chunk_size,
                    density_chunk_size=args.density_chunk_size,
                )
                print(f"\n=== seed={seed} modality={modality} ===", flush=True)
                print(" ".join(cmd), flush=True)
                completed = launch_official_run(cmd, cwd=code_root)
                record = {
                    "seed": seed,
                    "modality": modality,
                    "returncode": completed.returncode,
                    "command": cmd,
                }
                run_records.append(record)
                if completed.returncode != 0:
                    print(
                        f"ERROR: official run failed seed={seed} modality={modality} "
                        f"rc={completed.returncode}",
                        file=sys.stderr,
                    )
                remaining = cell_count - (len(run_records))
                if remaining > 0 and args.pause_seconds > 0:
                    print(f"Cool-down {args.pause_seconds}s ({remaining} cells left)...", flush=True)
                    time.sleep(float(args.pause_seconds))

    aggregation = aggregate_experiment_log(
        results_csv,
        seeds=seeds,
        modalities=modalities,
        backbone_variant=backbone,
    )
    report = build_report(
        config=config,
        aggregation=aggregation,
        dataset_meta={
            "official_protocol_complete": ds.get("official_protocol_complete"),
            "dataset_tree_sha256": ds.get("dataset_tree_sha256"),
            "splits_tree_sha256": ds.get("splits_tree_sha256"),
            "mounted_sample_count": ds.get("mounted_sample_count"),
            "license": ds.get("license"),
        },
        code_meta=code,
        run_records=run_records,
    )
    out_path = PROJECT_ROOT / args.out
    write_json(out_path, report)
    write_json(pred_dir / "last_run_records.json", {"records": run_records, "config": config})
    write_json(
        PROJECT_ROOT / "reports/libad/official_dinov3_run_card.json",
        run_card(backbone_variant=backbone, probe=False),
    )
    print(json.dumps({
        "wrote": str(out_path.as_posix()),
        "profile": "laptop",
        "evidence_class": report["evidence_class"],
        "comparable_to_paper": report["comparable_to_paper"],
        "paper_comparability_blockers": report["paper_comparability_blockers"],
        "experiments": report["experiments"],
    }, indent=2))
    if any(r.get("returncode", 1) != 0 for r in run_records):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
