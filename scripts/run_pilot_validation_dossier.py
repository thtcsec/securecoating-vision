"""Generate a pilot dossier only from explicit, integrity-checked evidence."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


REQUIRED_FIELDS = {"study_id", "methodology", "source_artifacts", "rolls", "metrics"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_evidence(path: Path) -> dict:
    evidence = json.loads(path.read_text(encoding="utf-8"))
    missing = REQUIRED_FIELDS.difference(evidence)
    if missing:
        raise ValueError(f"Evidence manifest is missing fields: {sorted(missing)}")
    if not evidence["source_artifacts"]:
        raise ValueError("At least one source artifact is required")
    for artifact in evidence["source_artifacts"]:
        artifact_path = (path.parent / artifact["path"]).resolve()
        if not artifact_path.is_file():
            raise FileNotFoundError(artifact_path)
        actual_hash = sha256_file(artifact_path)
        expected_hash = artifact.get("sha256")
        if not expected_hash:
            raise ValueError(f"Missing SHA-256 for evidence artifact {artifact_path}")
        if actual_hash.lower() != expected_hash.lower():
            raise ValueError(f"Evidence hash mismatch for {artifact_path}")
    return evidence


def generate_dossier(evidence_path: Path, output_path: Path) -> None:
    evidence = load_evidence(evidence_path)
    lines = [
        "# SecureCoating-Vision Pilot Validation Dossier",
        "",
        f"- Study ID: `{evidence['study_id']}`",
        f"- Generated: {datetime.now(timezone.utc).isoformat()}",
        f"- Evidence manifest: `{evidence_path}`",
        "",
        "## Methodology",
        "",
        str(evidence["methodology"]),
        "",
        "## Source artifacts",
        "",
    ]
    for artifact in evidence["source_artifacts"]:
        lines.append(f"- `{artifact['path']}` — SHA-256 `{artifact['sha256']}`")

    lines.extend(["", "## Roll evidence", ""])
    if not evidence["rolls"]:
        lines.append("No roll evidence supplied.")
    for roll in evidence["rolls"]:
        lines.append(f"### {roll.get('roll_id', 'UNIDENTIFIED ROLL')}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(roll, indent=2, ensure_ascii=False, sort_keys=True))
        lines.append("```")

    lines.extend(["", "## Reported metrics", ""])
    lines.append("Metrics below are copied verbatim from the hashed evidence manifest; this generator does not calculate or invent values.")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(evidence["metrics"], indent=2, ensure_ascii=False, sort_keys=True))
    lines.append("```")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/pilot_validation_dossier.md"))
    args = parser.parse_args()
    generate_dossier(args.evidence.resolve(), args.output.resolve())
