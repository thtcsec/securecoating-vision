# Evidence Dossier Status

The former dossier was withdrawn because it mixed development artifacts with unsupported claims, including roll-disjoint evaluation, zero escapes, factory savings, hardware timing, and standards-aligned uncertainty.

## Evidence that remains usable

- `reports/coatingvision_real_test_metrics.json` is a historical external-dataset JSON artifact. Before publication, verify its dataset split, model hash, source license, exact command, and absence of leakage.
- `reports/evaluation_results.json` is a development artifact only. Its dataset overlaps development validation data and it is not a generalization estimate.
- Repository software tests support specific fail-closed and tamper-detection contracts; they do not prove physical PLC or factory behavior.

## Required before making quantitative claims

1. Immutable roll/group-disjoint train, validation, and test manifests.
2. Model, dataset, source-commit, environment, and command hashes.
3. Raw predictions, class-aware one-to-one matching policy, negatives, and confidence intervals.
4. Target-hardware latency measurements including acquisition, inference, PLC ACK, and actuator feedback.
5. Vendor PLC HIL and plant safety approval.

Until those gates are complete, describe the system as a research/demo prototype only.
