"""Polish Application Form DOCX pagination without changing scientific wording."""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION_START
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[1]
DOCX = ROOT / "Al + Materials Competition Application Form.docx"


def _collapse_leading_empty_paragraphs(doc: Document, keep: int = 1) -> int:
    """Reduce empty paragraphs after the title block so cover content stays on page 1."""
    removed = 0
    empties: list = []
    for paragraph in doc.paragraphs:
        text = (paragraph.text or "").strip()
        xml = paragraph._element.xml
        if "w:br" in xml and "page" in xml:
            continue
        if text:
            if empties:
                # Keep at most `keep` empty paragraphs before this content.
                for node in empties[keep:]:
                    parent = node.getparent()
                    if parent is not None:
                        parent.remove(node)
                        removed += 1
                empties = []
            continue
        empties.append(paragraph._element)
    return removed


def _merge_extra_sections(doc: Document) -> int:
    """Keep a single continuous section so template content does not start on a blank page."""
    body = doc.element.body
    sects = [child for child in body if child.tag == qn("w:sectPr")]
    # python-docx keeps final sectPr on body; earlier section breaks live in paragraphs.
    removed = 0
    for paragraph in list(doc.paragraphs):
        ppr = paragraph._element.find(qn("w:pPr"))
        if ppr is None:
            continue
        sect = ppr.find(qn("w:sectPr"))
        if sect is not None:
            ppr.remove(sect)
            removed += 1
    # Ensure remaining section is continuous (no forced new page).
    if doc.sections:
        doc.sections[0].start_type = WD_SECTION_START.CONTINUOUS
    return removed


def _trim_trailing_empty_table_rows(doc: Document) -> int:
    removed = 0
    for table in doc.tables:
        # Walk from bottom; drop fully empty trailing rows.
        while len(table.rows) > 1:
            last = table.rows[-1]
            texts = [
                (cell.text or "").strip()
                for cell in last.cells
            ]
            if any(texts):
                break
            last._tr.getparent().remove(last._tr)
            removed += 1
    return removed


def polish(path: Path = DOCX) -> dict:
    doc = Document(str(path))
    stats = {
        "empty_paragraphs_removed": _collapse_leading_empty_paragraphs(doc, keep=1),
        "section_breaks_removed": _merge_extra_sections(doc),
        "empty_table_rows_removed": _trim_trailing_empty_table_rows(doc),
        "tables": len(doc.tables),
        "sections_after": len(doc.sections),
    }
    doc.save(str(path))
    return stats


def main() -> None:
    import json

    print(json.dumps(polish(), indent=2))


if __name__ == "__main__":
    main()
