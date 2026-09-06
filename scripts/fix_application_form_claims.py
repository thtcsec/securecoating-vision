"""Apply final Application Form claim/advisor placement rules."""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[1]
DOCX = ROOT / "Al + Materials Competition Application Form.docx"

OLD = (
    "Reproducible Evaluation Boundaries: The evaluation engine validates artifact hashes, "
    "requires an immutable roll-disjoint manifest, uses one instance-matching policy for "
    "box and mask metrics, and records model, dataset, manifest, and source-commit provenance."
)
NEW = (
    "Reproducible Evaluation Boundaries: The evaluation engine validates immutable dataset "
    "manifests, records split provenance, and enforces consistent metric semantics. "
    "Independent roll-disjoint evidence remains a required gate for factory qualification."
)

# Section 7 keeps the BTC team-member template for contestants only.
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


def _replace_in_cell(cell, old: str, new: str) -> bool:
    text = cell.text or ""
    if old not in text:
        return False
    updated = text.replace(old, new)
    paragraphs = cell.paragraphs
    if not paragraphs:
        return False
    for paragraph in paragraphs:
        for run in paragraph.runs:
            run.text = ""
    paragraphs[0].clear()
    paragraphs[0].add_run(updated)
    for paragraph in list(paragraphs[1:]):
        parent = paragraph._element.getparent()
        if parent is not None:
            parent.remove(paragraph._element)
    return True


def _strip_advisor_lines(text: str) -> str:
    lines = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("Advisor") or stripped.startswith("指导老师"):
            continue
        lines.append(line)
    return "\n".join(lines).rstrip()


def fix(path: Path = DOCX) -> dict:
    doc = Document(str(path))
    replaced = 0
    team_filled = 0
    advisor_placed = 0
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                raw = cell.text or ""
                if OLD in raw and _replace_in_cell(cell, OLD, NEW):
                    replaced += 1
                    raw = cell.text or ""

                text = raw.strip()
                if text.startswith("7、Team Member Information"):
                    _set_cell_text(cell, TEAM)
                    team_filled += 1
                    continue

                if text.startswith("6、Additional Information"):
                    body = _strip_advisor_lines(text)
                    if ADVISOR_LINE not in body:
                        body = body.rstrip() + "\n\n" + ADVISOR_LINE
                    _set_cell_text(cell, body)
                    advisor_placed += 1

    body = doc.element.body
    removed_paras = 0
    for child in list(body):
        if child.tag != qn("w:p"):
            continue
        text = "".join(node.text or "" for node in child.iter() if node.text)
        has_drawing = any(True for _ in child.iter(qn("w:drawing"))) or any(
            True for _ in child.iter(qn("w:pict"))
        )
        if text.strip() or has_drawing:
            continue
        ppr = child.find(qn("w:pPr"))
        if ppr is not None and ppr.find(qn("w:sectPr")) is not None:
            continue
        body.remove(child)
        removed_paras += 1

    doc.save(str(path))
    return {
        "roll_disjoint_sentence_replaced": replaced,
        "team_member_rows_filled": team_filled,
        "advisor_in_section_6": advisor_placed,
        "empty_paragraphs_removed": removed_paras,
    }


def main() -> None:
    import json

    print(json.dumps(fix(), indent=2))


if __name__ == "__main__":
    main()
