# SecureCoating-Vision — Analysis Report

Judge-facing technical summary for Track 4. Metrics below are copied from authoritative
checked-in artifacts only. No new measurements are invented here.

## 1. Executive Summary

SecureCoating-Vision is an evidence-gated multimodal inspection prototype for battery
electrode manufacturing quality decisions. The implemented contract separates model output
from release authority: detections may inform, but only PASS / REJECT / HOLD may leave the
gate under explicit evidence and communication contracts.

Authoritative RGB detection evidence is a public CoatingVision image-disjoint evaluation.
LIBAD multimodal numbers in this archive are historical, non-paper-comparable validation
artifacts. Electrochemical / material performance prediction is future validation, not a
current claim.

## 2. Evidence Classes and Evaluation Boundaries

| Class | Artifact | Bound |
|---|---|---|
| Public real optical RGB | `reports/coatingvision_real_test_metrics.json` | Image-disjoint public split; **not** factory roll-disjoint |
| Historical LIBAD local adapter | `reports/libad/official_local_adapter.json` | Official inputs + local numpy descriptor; `comparable_to_paper: false` |
| Historical authors' DINOv2 interim | `reports/libad/official_dinov2_dacore_interim.json` | 1-seed smoke; not DINOv3/DA-Core paper reproduction |
| Protocol fixtures | `reports/libad_demo/*`, `reports/libad/libad_benchmark.json` | Deterministic CI demos; not plant captures |
| Software verification | `reports/test_manifest.json` | Repository contract tests only |
| Final pack provenance | `reports/submission_manifest.json` | Hash map for the packed ZIP |

Simulated / injected thermal and profilometry adapters validate registration and fail-closed
behavior. They are not claimed as plant-instrument measurements.

## 3. RGB CoatingVision Evaluation

Source: CoatingVision public real optical dataset
(Figshare DOI `10.6084/m9.figshare.29260121.v1`, CC BY 4.0).

- Evidence class: `public_real_optical_image_split`
- Split: fixed **image-disjoint** test split, seed **71**, **88** test images
- Factory roll-disjoint: **false**
- Precision: **0.645**
- Recall: **0.642**
- mAP50: **0.634** (raw report `0.633617`)
- mAP50-95: **0.354** (raw report `0.354307`)
- Model-only CPU inference: **~21.6 ms/image** (`speed_ms_per_image.inference` = 21.632 ms)
- Weights SHA-256: `f72a8f2bd64add6f9ae6c42684bb09e0705b1ed70cdc9123bb82923ef76d5719`
- Dataset tree SHA-256: `d1db7823f8fd5789d4c9518523f408537d837d5e6bbfc0e017d4aa9e4784f48d`
- Manifest SHA-256: `99329ef3363891951fd95b57c24ae591755137f372675d040dc07d7aacda8d75`
- Config / weights paths: `configs/coatingvision_real_test.yaml`, `outputs/best.pt`

This lane does **not** claim factory roll-disjoint validation or line throughput.

## 4. LIBAD Multimodal Validation

### A. Historical local numpy adapter

- Artifact: `reports/libad/official_local_adapter.json`
- Evidence generation: `LEGACY_HISTORICAL_EVIDENCE`
- Methodology note: historical evidence retained for continuity; **not regenerated from current HEAD**
- Mean multimodal AUROC / FPR95 over 10 official splits: **0.700 / 0.839**
- `comparable_to_paper`: **false**
- Prediction artifact row count (across lanes): **19,680 evaluation rows** — these are
  evaluation rows across lanes, **not** 19,680 independent multimodal samples

### B. Historical authors' DINOv2 interim smoke

- Artifact: `reports/libad/official_dinov2_dacore_interim.json`
- 1-seed interim smoke AUROC / FPR95: **0.856 / 0.716**
- Not DINOv3 / DA-Core paper reproduction
- `comparable_to_paper`: **false**

### C. Paper DA-Core reference (Sui et al.)

Sui et al. report that the best LIBAD setting still has FPR95 of 54.3% at AUROC 86.7%, AUPR 95.7%, and F1-max 90.6%. The authors conclude that this false-positive rate is too high for direct deployment and call out modality disagreement and closed-loop process control as open problems. LIBAD addresses how to detect; SecureCoating-Vision addresses when a detection is safe enough to act on.

DA-Core belongs to Sui et al. SecureCoating-Vision does **not** claim DA-Core as an original
contribution and does **not** claim paper-exact LIBAD reproduction.

## 5. Evidence-Gated PASS / REJECT / HOLD Evaluation

Operational contribution: an evidence gate that prevents model confidence alone from
self-releasing product. PASS requires agreement and healthy contracts; REJECT requires
supporting evidence plus completed communication contracts; HOLD catches disagreement,
missing/stale evidence, calibration faults, or unconfirmed PLC/ACK paths.

This is a software fail-closed contract on real optical and multimodal validation evidence.
It is **not** factory safety qualification, physical PLC/HIL qualification, or a
safety-rated E-stop claim.

## 6. Software Verification

Authoritative software verification is recorded in `reports/test_manifest.json` and `reports/pytest_*`.

- Collected / passed / failed / skipped: **282 / 282 / 0 / 0**
- Python: **3.11.9**
- Snapshot commit: `afc95b0149b8` (working_tree_dirty=False)
- Duration: 79.34s

This count is read from the manifest at report generation time and is not hard-coded elsewhere as a mutable claim.

## 7. Reproducibility and Artifact Provenance

Final competition release provenance is `reports/submission_manifest.json` (artifact SHA-256 map for
the packed ZIP). Protocol-fixture certificates record **fixture generation** provenance only
(`provenance_scope=protocol_fixture_generation`) and must not be read as the packed-release
HEAD. Model hashes are declared in `configs/model.yaml` and verified against `outputs/`.

## 8. Limitations

- No factory roll-disjoint validation of the RGB headline metric
- No physical PLC / HIL qualification
- No safety-rated E-stop claim
- Thermal / profilometry remain simulated or injected adapters
- No experimentally validated electrochemical / material performance prediction
- No paper-comparable DINOv3 / DA-Core claim

## 9. Next Validation Gates

1. Independent roll-disjoint factory optical validation
2. PLC / HIL, plant calibration, and safety testing
3. Authors' DINOv3 / DA-Core validation only if paper-comparable LIBAD benchmarking is required
4. Downstream material / electrochemical performance prediction as a separate future evidence class
