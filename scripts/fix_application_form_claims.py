"""One-shot clarity fixes for the competition application DOCX."""

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

TEAM = (
    "7、Team Member Information\n\n"
    "Team Leader: Trịnh Hoàng Tú\n"
    "Team Members: N/A (individual submission)\n"
    "Affiliation: Ho Chi Minh City University of Foreign Languages – Information "
    "Technology (HUFLIT)\n"
    "Advisor: Prof. Kris Singh · Visiting Professor, Tsinghua University · "
    "Founder & CEO, SRII\n"
    "Role: Sole author of the SecureCoating-Vision research prototype, evidence pipeline, "
    "and Track 4 final-defense materials. Advisor provides academic guidance only and is "
    "not a co-author of the detector metrics."
)


def _replace_in_cell(cell, old: str, new: str) -> bool:
    text = cell.text or ""
    if old not in text:
        return False
    # Prefer paragraph-level rewrite when the whole cell is one logical block.
    updated = text.replace(old, new)
    paragraphs = cell.paragraphs
    if not paragraphs:
        return False
    # Clear existing runs then write updated text into the first paragraph.
    for paragraph in paragraphs:
        for run in paragraph.runs:
            run.text = ""
        if paragraph._element.find(qn("w:hyperlink")) is not None:
            continue
    paragraphs[0].clear()
    paragraphs[0].add_run(updated)
    # Remove leftover empty paragraphs after the first to avoid blank page padding.
    for paragraph in list(paragraphs[1:]):
        parent = paragraph._element.getparent()
        if parent is not None:
            parent.remove(paragraph._element)
    return True


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


def fix(path: Path = DOCX) -> dict:
    doc = Document(str(path))
    replaced = 0
    team_filled = 0
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                if OLD in (cell.text or ""):
                    if _replace_in_cell(cell, OLD, NEW):
                        replaced += 1
                text = (cell.text or "").strip()
                if text.startswith("7、Team Member Information") and "Advisor:" not in text:
                    _set_cell_text(cell, TEAM)
                    team_filled += 1
                elif text == "7、Team Member Information" or (
                    text.startswith("7、Team Member Information") and len(text) < 80
                ):
                    _set_cell_text(cell, TEAM)
                    team_filled += 1
    # Drop trailing empty paragraphs that push a near-blank final page.
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
        # Keep sectPr-bearing final paragraph.
        if child.find(qn("w:pPr")) is not None and child.find(qn("w:pPr")).find(qn("w:sectPr")) is not None:
            continue
        body.remove(child)
        removed_paras += 1

    doc.save(str(path))
    return {
        "roll_disjoint_sentence_replaced": replaced,
        "team_member_rows_filled": team_filled,
        "empty_paragraphs_removed": removed_paras,
    }


def main() -> None:
    import json

    print(json.dumps(fix(), indent=2))


if __name__ == "__main__":
    main()
