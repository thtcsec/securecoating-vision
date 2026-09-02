# Development Evaluation Status

This report supersedes the previous performance narrative, which contained values that did not match the tracked JSON artifact and latency claims without an accepted benchmark manifest.

`reports/evaluation_results.json` records a 50-image development run with 30 ground-truth instances at IoU 0.50 and confidence 0.25. The artifact reports box precision 0.5000, box recall 0.6667, box F1 0.5714, and mean CPU inference latency 100.90 ms. It also records zero mask matches under that script's matching policy.

These numbers are **not independent model-performance evidence**: the tracked `data/evaluation` images overlap development validation data, and the current evaluation script rejects that overlap. The separate Ultralytics JSON lacks the full model/dataset/commit provenance required for a release claim.

No accepted artifact currently proves GPU latency, end-to-end latency, probability of detection, escape rate, false rejection rate, factory yield, PLC actuation latency, or regulatory compliance.

The LIBAD adapter can emit AUROC/AUPR/F1-max/FPR95 plus Automatic Decision Coverage, HOLD Rate, Escape Rate, and Selective Risk. Checked-in fixture runs stay `evidence_class=protocol_fixture` and `comparable_to_paper=false`. Official local-adapter numbers, when recorded, live in `reports/libad/official_local_adapter.json` after a hash-verified mount; they still use `numpy_patch_descriptor` and remain `comparable_to_paper=false`.


To create a publishable report, use an immutable roll-disjoint dataset, record model/dataset/commit hashes, retain raw predictions and matching policy, calculate confidence intervals, and verify the generated artifact independently.
