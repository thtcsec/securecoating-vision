"""Surgical Application Form claim fixes — touch only known section cells."""

from __future__ import annotations

from pathlib import Path

from docx import Document

ROOT = Path(__file__).resolve().parents[1]
DOCX = ROOT / "Al + Materials Competition Application Form.docx"

REPLACEMENTS = (
    (
        "certificates use canonical HMAC-SHA256 payloads",
        "certificates use HMAC-SHA256 authenticated traceability tags",
    ),
    (
        "and can be represented in a signed certificate snapshot.",
        "and can be represented in an HMAC-SHA256 authenticated certificate snapshot.",
    ),
    (
        "Next Validation Steps: Obtain an independent roll-disjoint dataset, hash-verify "
        "official LIBAD, complete PLC/HIL and safety testing, validate factory calibration, "
        "and publish only measurements supported by immutable evidence.",
        "Next Validation Steps: Reproduce the official-LIBAD validation under the current "
        "clean release and preserve current-source provenance; obtain an independent "
        "roll-disjoint factory dataset; complete PLC/HIL and plant calibration; and keep "
        "any future performance-risk proxy explicitly simulation-validated and separate "
        "from electrochemical cell-performance claims.",
    ),
)

TEAM = (
    "7、Team Member Information\n\n"
    "Team Leader — Trịnh Hoàng Tú\n"
    "Team Member — N/A"
)

ADVISOR_LINE = (
    "Advisor / 指导老师: Prof. Kris Singh — Visiting Professor, Tsinghua University; "
    "Founder & CEO, SRII. Advisory scope: innovation, industrialization, and "
    "final-defense guidance."
)


def _set_cell_text(cell, text: str) -> None:
    paragraphs = cell.paragraphs
    if not paragraphs:
        return
    for paragraph in list(paragraphs[1:]):
        parent = paragraph._element.getparent()
        if parent is not None:
            parent.remove(paragraph._element)
    paragraphs[0].clear()
    lines = text.split("\n")
    paragraphs[0].add_run(lines[0] if lines else "")
    for line in lines[1:]:
        cell.add_paragraph(line)


def _strip_advisor_block(text: str) -> str:
    keep = []
    for line in text.splitlines():
        s = line.strip()
        if (
            s.startswith("Advisor")
            or s.startswith("指导老师")
            or s.startswith("Advisory scope")
            or "kris@thesrii.org" in s
        ):
            continue
        keep.append(line)
    return "\n".join(keep).rstrip()


def fix(path: Path = DOCX) -> dict:
    doc = Document(str(path))
    if len(doc.tables) < 2:
        raise SystemExit("expected cover + body tables")
    body = doc.tables[1]
    stats = {"replacements": 0, "section6": False, "section7": False}

    # Only edit the first cell of each body section row (BTC template).
    for row in body.rows:
        cell = row.cells[0]
        text = cell.text or ""
        updated = text
        for old, new in REPLACEMENTS:
            if old in updated:
                updated = updated.replace(old, new)
                stats["replacements"] += 1

        if updated.strip().startswith("6、Additional Information"):
            updated = _strip_advisor_block(updated)
            if ADVISOR_LINE not in updated:
                updated = updated.rstrip() + "\n\n" + ADVISOR_LINE
            _set_cell_text(cell, updated)
            stats["section6"] = True
            continue

        if updated.strip().startswith("7、Team Member Information"):
            _set_cell_text(cell, TEAM)
            stats["section7"] = True
            continue

        if updated != text:
            _set_cell_text(cell, updated)

    # Cover table identity checks (do not rewrite title).
    title = doc.tables[0].rows[0].cells[1].text.strip()
    if "Zero-Trust Edge-Cloud" not in title:
        raise SystemExit(f"refusing to save: cover title drifted: {title[:80]!r}")

    doc.save(str(path))
    return stats


def main() -> None:
    import json

    print(json.dumps(fix(), indent=2))


if __name__ == "__main__":
    main()
