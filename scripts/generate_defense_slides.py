"""Generate the 6-minute final-defense deck from the registered identity.

Optional extra (not an API runtime dependency):
    .venv\\Scripts\\python.exe -m pip install python-pptx
    .venv\\Scripts\\python.exe scripts/generate_defense_slides.py
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "SecureCoating-Vision_Final_Defense_6min.pptx"
LEGACY_DECK = ROOT / "SecureCoating-Vision_Final_Defense_6min_Coating_Surface_Evidence.pptx"
EXTERNAL_DEMO = ROOT / "reports/external_coatingvision_demo/coatingvision_model_output.png"

BG = RGBColor(0x0A, 0x0D, 0x14)
CARD = RGBColor(0x15, 0x1B, 0x28)
ACCENT = RGBColor(0x00, 0xE5, 0xFF)
GREEN = RGBColor(0x00, 0xE6, 0x76)
YELLOW = RGBColor(0xFF, 0xD6, 0x00)
RED = RGBColor(0xFF, 0x17, 0x44)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
MUTED = RGBColor(0x8C, 0x9B, 0xAE)
LINE = RGBColor(0x22, 0x2C, 0x3E)

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)
TITLE = (
    "SecureCoating Vision: A High-Throughput and Zero-Trust Edge-Cloud "
    "Pipeline for Inline Battery Electrode Defect Inspection and Traceable "
    "Quality Decisions"
)
TAGLINE = "Evidence-Gated Multimodal Inspection for Battery Electrode Manufacturing"


def _set_run(run, text, size_pt, color, bold=False):
    run.text = text
    run.font.size = Pt(size_pt)
    run.font.color.rgb = color
    run.font.bold = bold
    run.font.name = "Calibri"


def _fill(shape, color):
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()


def _box(slide, left, top, width, height, fill=CARD):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    _fill(shape, fill)
    shape.line.color.rgb = LINE
    shape.line.width = Emu(6350)
    return shape


def add_textbox(slide, left, top, width, height, lines, align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    for i, item in enumerate(lines):
        paragraph = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        paragraph.alignment = align
        paragraph.space_after = Pt(item.get("space_after", 6))
        run = paragraph.add_run()
        _set_run(run, item["text"], item.get("size", 16), item.get("color", WHITE), item.get("bold", False))
    return box


def footer(slide, number):
    add_textbox(
        slide,
        Inches(0.45),
        Inches(7.12),
        Inches(8.5),
        Inches(0.28),
        [{"text": "Team 71  ·  Trịnh Hoàng Tú  ·  HUFLIT  ·  Track 4", "size": 11, "color": MUTED}],
    )
    add_textbox(
        slide,
        Inches(10.6),
        Inches(7.12),
        Inches(2.2),
        Inches(0.28),
        [{"text": f"TEAM 71  /  {number}", "size": 11, "color": MUTED, "bold": True}],
        align=PP_ALIGN.RIGHT,
    )


def paint_bg(slide):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, SLIDE_H)
    _fill(shape, BG)
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(0.12), SLIDE_H)
    _fill(bar, ACCENT)
    bottom = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(7.05), SLIDE_W, Inches(0.45))
    _fill(bottom, RGBColor(0x08, 0x0B, 0x12))


def kicker(slide, code, timing):
    add_textbox(
        slide,
        Inches(0.45),
        Inches(0.22),
        Inches(12.4),
        Inches(0.32),
        [{"text": f"{code}    {timing}", "size": 12, "color": ACCENT, "bold": True}],
    )


def extract_legacy_image(name: str) -> bytes | None:
    if not LEGACY_DECK.is_file():
        return None
    with zipfile.ZipFile(LEGACY_DECK) as archive:
        target = f"ppt/media/{name}"
        if target in archive.namelist():
            return archive.read(target)
    return None


def add_picture_bytes(slide, payload: bytes, left, top, width):
    slide.shapes.add_picture(io.BytesIO(payload), left, top, width=width)


def build() -> Path:
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H
    blank = prs.slide_layouts[6]
    coating = extract_legacy_image("image3.png")

    s = prs.slides.add_slide(blank)
    paint_bg(s)
    add_textbox(s, Inches(0.45), Inches(0.28), Inches(12.4), Inches(0.3), [
        {"text": "GLOBAL AI + MATERIALS INNOVATION COMPETITION 2026", "size": 13, "color": ACCENT, "bold": True}
    ])
    add_textbox(s, Inches(0.45), Inches(0.7), Inches(12.4), Inches(0.35), [
        {"text": "SecureCoating-Vision", "size": 28, "color": WHITE, "bold": True}
    ])
    add_textbox(s, Inches(0.45), Inches(1.15), Inches(12.4), Inches(1.55), [
        {"text": TITLE, "size": 22, "color": WHITE, "bold": True, "space_after": 0}
    ])
    add_textbox(s, Inches(0.45), Inches(2.85), Inches(12.4), Inches(0.45), [
        {"text": TAGLINE, "size": 20, "color": ACCENT, "bold": True}
    ])
    cards = (
        ("Team", "71  ·  Trịnh Hoàng Tú"),
        ("Affiliation", "HUFLIT"),
        ("Track", "4  ·  AI + Materials Testing and Characterization"),
        ("Advisor", "Prof. Kris Singh  ·  SRII / Tsinghua (visiting)"),
    )
    for i, (label, value) in enumerate(cards):
        left = Inches(0.45 + (i % 2) * 6.3)
        top = Inches(3.55 + (i // 2) * 1.15)
        _box(s, left, top, Inches(6.05), Inches(1.0))
        add_textbox(s, left + Inches(0.2), top + Inches(0.12), Inches(5.6), Inches(0.28), [
            {"text": label.upper(), "size": 11, "color": MUTED, "bold": True}
        ])
        add_textbox(s, left + Inches(0.2), top + Inches(0.42), Inches(5.6), Inches(0.45), [
            {"text": value, "size": 16, "color": WHITE, "bold": True}
        ])
    add_textbox(s, Inches(0.45), Inches(6.0), Inches(12.4), Inches(0.7), [
        {"text": "Research prototype. Not production-qualified. No factory HIL, no independent roll-disjoint test, no safety-rated E-stop claim.", "size": 14, "color": YELLOW}
    ])
    footer(s, 1)

    s = prs.slides.add_slide(blank)
    paint_bg(s)
    kicker(s, "02", "0:35–1:15")
    add_textbox(s, Inches(0.45), Inches(0.5), Inches(7.4), Inches(1.1), [
        {"text": "The defect is tiny. An automatic PASS is not.", "size": 28, "color": WHITE, "bold": True}
    ])
    add_textbox(s, Inches(0.45), Inches(1.7), Inches(7.4), Inches(2.4), [
        {"text": "Electrode coating defects become scrap, rework, or untraceable cell risk.", "size": 18, "color": WHITE, "space_after": 10},
        {"text": "A detector that ranks anomalies well can still be unsafe to act on if sensors disagree, calibration is unverified, or the PLC does not acknowledge.", "size": 16, "color": MUTED, "space_after": 10},
        {"text": "Academic detection answers how to detect. A factory needs when a detection is safe enough to act on.", "size": 16, "color": ACCENT, "bold": True},
    ])
    _box(s, Inches(0.45), Inches(4.3), Inches(7.4), Inches(2.4))
    add_textbox(s, Inches(0.65), Inches(4.45), Inches(7.0), Inches(2.1), [
        {"text": "WE DO NOT CLAIM", "size": 12, "color": RED, "bold": True},
        {"text": "99.4% mAP  ·  0.02 ppm escape  ·  −84% scrap  ·  production P99.9 SLA", "size": 15, "color": WHITE, "bold": True, "space_after": 8},
        {"text": "Those numbers need independent roll-disjoint data, HIL, and a safety owner. This deck does not invent them.", "size": 14, "color": MUTED},
    ])
    if coating:
        _box(s, Inches(8.15), Inches(0.55), Inches(4.7), Inches(6.2))
        add_picture_bytes(s, coating, Inches(8.3), Inches(0.95), Inches(4.4))
        add_textbox(s, Inches(8.3), Inches(5.85), Inches(4.4), Inches(0.7), [
            {"text": "Real coating surface (held-out CoatingVision sample). RGB-only. Not a multimodal plant capture.", "size": 12, "color": MUTED}
        ])
    footer(s, 2)

    s = prs.slides.add_slide(blank)
    paint_bg(s)
    kicker(s, "03", "1:15–2:00")
    add_textbox(s, Inches(0.45), Inches(0.5), Inches(12.4), Inches(0.7), [
        {"text": "Two evidence lanes. One fail-closed contract.", "size": 28, "color": WHITE, "bold": True}
    ])
    _box(s, Inches(0.45), Inches(1.4), Inches(6.05), Inches(5.3))
    add_textbox(s, Inches(0.65), Inches(1.55), Inches(5.65), Inches(4.9), [
        {"text": "LANE A  ·  IMPLEMENTED", "size": 12, "color": GREEN, "bold": True},
        {"text": "RGB YOLOv8-seg / ONNX", "size": 22, "color": WHITE, "bold": True},
        {"text": "FastAPI  ·  SQLite PENDING→final  ·  OPC UA / Modbus command+ACK  ·  HMAC certificate", "size": 15, "color": MUTED, "space_after": 12},
        {"text": "Thermal and profilometry are simulated adapters. They are not plant-instrument measurements.", "size": 15, "color": YELLOW},
    ])
    _box(s, Inches(6.8), Inches(1.4), Inches(6.05), Inches(5.3))
    add_textbox(s, Inches(7.0), Inches(1.55), Inches(5.65), Inches(4.9), [
        {"text": "LANE B  ·  VALIDATION EXTENSION", "size": 12, "color": ACCENT, "bold": True},
        {"text": "LIBAD VIS + X-rayL adapter", "size": 22, "color": WHITE, "bold": True},
        {"text": "Official 4.84 GB dataset and 10 splits are not in this runtime. Checked-in cases are protocol fixtures, not paper-comparable.", "size": 15, "color": MUTED, "space_after": 12},
        {"text": "DA-Core is Sui et al. SecureCoating-Vision adds the evidence gate, not a new detector claim.", "size": 15, "color": ACCENT},
    ])
    footer(s, 3)

    s = prs.slides.add_slide(blank)
    paint_bg(s)
    kicker(s, "04", "2:00–2:45")
    add_textbox(s, Inches(0.45), Inches(0.5), Inches(12.4), Inches(0.7), [
        {"text": "Model output cannot self-release the line.", "size": 28, "color": WHITE, "bold": True}
    ])
    for i, (name, color, body) in enumerate((
        ("PASS", GREEN, "OPTIMAL inference, trained model, verified calibration, healthy traceability, PLC ACK."),
        ("REJECT", RED, "Evidence supports a defect and the communication contract completes."),
        ("HOLD", YELLOW, "Disagreement, stale/missing sensor, timeout, unverified calibration, DB fault, or unconfirmed PLC."),
    )):
        left = Inches(0.45 + i * 4.2)
        _box(s, left, Inches(1.4), Inches(4.0), Inches(2.4))
        add_textbox(s, left + Inches(0.2), Inches(1.55), Inches(3.6), Inches(0.45), [
            {"text": name, "size": 26, "color": color, "bold": True}
        ])
        add_textbox(s, left + Inches(0.2), Inches(2.15), Inches(3.6), Inches(1.4), [
            {"text": body, "size": 15, "color": WHITE}
        ])
    _box(s, Inches(0.45), Inches(4.05), Inches(12.4), Inches(2.65))
    add_textbox(s, Inches(0.7), Inches(4.2), Inches(12.0), Inches(2.3), [
        {"text": "INDUSTRIAL MEANING", "size": 12, "color": ACCENT, "bold": True},
        {"text": "Uncertainty becomes a controlled operational state, not a hidden false-positive rate.", "size": 20, "color": WHITE, "bold": True, "space_after": 10},
        {"text": "Software E-stop latches local interlock and requests PLC channels. It is not a safety-rated hardwired stop.", "size": 15, "color": MUTED},
    ])
    footer(s, 4)

    s = prs.slides.add_slide(blank)
    paint_bg(s)
    kicker(s, "05", "2:45–3:35")
    add_textbox(s, Inches(0.45), Inches(0.5), Inches(12.4), Inches(0.7), [
        {"text": "Production looks like a line terminal, not a demo console.", "size": 26, "color": WHITE, "bold": True}
    ])
    for i, (title, body) in enumerate((
        ("OPERATE", "Line disposition, roll identity, quality counts, PLC registers, recent inspections. One snapshot timestamp."),
        ("DIAGNOSE", "Readiness, sensors, model, transport. No invented Cpk, optical-budget SLA, or equipment health from density."),
        ("TRACEABILITY", "Roll ledger, certificate HMAC, control-audit log. Missing data renders as NO DATA, not 0%."),
    )):
        top = Inches(1.3 + i * 1.35)
        _box(s, Inches(0.45), top, Inches(8.2), Inches(1.22))
        add_textbox(s, Inches(0.65), top + Inches(0.12), Inches(7.8), Inches(0.32), [
            {"text": title, "size": 16, "color": ACCENT, "bold": True}
        ])
        add_textbox(s, Inches(0.65), top + Inches(0.48), Inches(7.8), Inches(0.6), [
            {"text": body, "size": 14, "color": WHITE}
        ])
    _box(s, Inches(8.85), Inches(1.3), Inches(4.0), Inches(5.35))
    add_textbox(s, Inches(9.05), Inches(1.45), Inches(3.6), Inches(5.0), [
        {"text": "NOT IN PRODUCTION", "size": 12, "color": RED, "bold": True},
        {"text": "Recipe sliders", "size": 16, "color": WHITE, "bold": True, "space_after": 4},
        {"text": "Defect injection", "size": 16, "color": WHITE, "bold": True, "space_after": 4},
        {"text": "7-stage simulator", "size": 16, "color": WHITE, "bold": True, "space_after": 4},
        {"text": "LIBAD 90s demo", "size": 16, "color": WHITE, "bold": True, "space_after": 4},
        {"text": "Send offset to PLC", "size": 16, "color": WHITE, "bold": True, "space_after": 12},
        {"text": "Those exist only in SECURECOATING_DASHBOARD_SANDBOX=true.", "size": 13, "color": MUTED},
        {"text": "Control = confirm-audit only.", "size": 14, "color": ACCENT, "bold": True},
    ])
    footer(s, 5)

    s = prs.slides.add_slide(blank)
    paint_bg(s)
    kicker(s, "06", "3:35–4:20")
    add_textbox(s, Inches(0.45), Inches(0.5), Inches(12.4), Inches(0.7), [
        {"text": "Trace first. Command second. ACK or HOLD.", "size": 28, "color": WHITE, "bold": True}
    ])
    steps = [
        ("1", "Inspect", "RGB path + fail-safe deadline"),
        ("2", "PENDING", "SQLite claims (batch, part)"),
        ("3", "PLC", "One command owner + ACK sequence"),
        ("4", "Finalize", "PASS/REJECT, or HOLD if ACK fails"),
        ("5", "Certificate", "HMAC-SHA256 over canonical payload"),
        ("6", "Audit", "Operator confirm phrase is durable"),
    ]
    for i, (number, title, body) in enumerate(steps):
        left = Inches(0.45 + (i % 3) * 4.2)
        top = Inches(1.4 + (i // 3) * 2.5)
        _box(s, left, top, Inches(4.0), Inches(2.25))
        add_textbox(s, left + Inches(0.2), top + Inches(0.18), Inches(3.6), Inches(0.35), [
            {"text": number, "size": 14, "color": ACCENT, "bold": True}
        ])
        add_textbox(s, left + Inches(0.2), top + Inches(0.55), Inches(3.6), Inches(0.45), [
            {"text": title, "size": 22, "color": WHITE, "bold": True}
        ])
        add_textbox(s, left + Inches(0.2), top + Inches(1.15), Inches(3.6), Inches(0.8), [
            {"text": body, "size": 15, "color": MUTED}
        ])
    footer(s, 6)

    s = prs.slides.add_slide(blank)
    paint_bg(s)
    kicker(s, "07", "4:20–5:10")
    add_textbox(s, Inches(0.45), Inches(0.5), Inches(12.4), Inches(0.9), [
        {"text": "LIBAD’s own FPR is too high for direct release. We do not hide it.", "size": 24, "color": WHITE, "bold": True}
    ])
    metrics = (
        ("AUROC", "86.7%", "Sui et al., best reported setting"),
        ("AUPR", "95.7%", "paper number, not our runtime"),
        ("F1-max", "90.6%", "paper number, not our runtime"),
        ("FPR95", "54.3%", "too high for automatic PASS"),
    )
    for i, (key, value, note) in enumerate(metrics):
        left = Inches(0.45 + i * 3.15)
        _box(s, left, Inches(1.55), Inches(3.0), Inches(2.15))
        add_textbox(s, left + Inches(0.15), Inches(1.68), Inches(2.7), Inches(0.3), [
            {"text": key, "size": 13, "color": MUTED, "bold": True}
        ])
        add_textbox(s, left + Inches(0.15), Inches(2.05), Inches(2.7), Inches(0.55), [
            {"text": value, "size": 28, "color": YELLOW if key == "FPR95" else WHITE, "bold": True}
        ])
        add_textbox(s, left + Inches(0.15), Inches(2.7), Inches(2.7), Inches(0.7), [
            {"text": note, "size": 12, "color": MUTED}
        ])
    _box(s, Inches(0.45), Inches(3.95), Inches(12.4), Inches(2.75))
    add_textbox(s, Inches(0.7), Inches(4.15), Inches(12.0), Inches(2.4), [
        {"text": "SECURECOATING METRICS ON OFFICIAL SPLITS ARE NOT YET GENERATED", "size": 13, "color": ACCENT, "bold": True},
        {"text": "Automatic Decision Coverage  ·  HOLD Rate  ·  Escape Rate  ·  Selective Risk", "size": 18, "color": WHITE, "bold": True, "space_after": 10},
        {"text": "Checked-in demo/benchmark files are protocol fixtures. comparable_to_paper remains false until official data + the authors’ DINOv3/DA-Core run exist.", "size": 15, "color": MUTED},
    ])
    footer(s, 7)

    s = prs.slides.add_slide(blank)
    paint_bg(s)
    kicker(s, "08", "5:10–6:00")
    add_textbox(s, Inches(0.45), Inches(0.5), Inches(12.4), Inches(0.55), [
        {"text": "Four staged cases. One closing identity screen.", "size": 26, "color": WHITE, "bold": True}
    ])
    cases = [
        ("1", GREEN, "PASS", "Normal VIS + X-ray agreement"),
        ("2", RED, "REJECT", "Surface defect, surface evidence"),
        ("3", RED, "REJECT", "Internal X-ray complementary evidence"),
        ("4", YELLOW, "HOLD", "Near-threshold disagreement → manual QA"),
    ]
    for i, (number, color, action, body) in enumerate(cases):
        left = Inches(0.45 + (i % 2) * 6.3)
        top = Inches(1.15 + (i // 2) * 1.35)
        _box(s, left, top, Inches(6.05), Inches(1.22))
        add_textbox(s, left + Inches(0.2), top + Inches(0.12), Inches(5.6), Inches(0.3), [
            {"text": f"CASE {number}  ·  {action}", "size": 14, "color": color, "bold": True}
        ])
        add_textbox(s, left + Inches(0.2), top + Inches(0.5), Inches(5.6), Inches(0.55), [
            {"text": body, "size": 16, "color": WHITE}
        ])
    add_textbox(s, Inches(0.45), Inches(4.0), Inches(7.6), Inches(2.6), [
        {"text": "CLOSE", "size": 12, "color": ACCENT, "bold": True},
        {"text": "LIBAD detects. SecureCoating-Vision decides when detection is safe enough to act on.", "size": 18, "color": WHITE, "bold": True, "space_after": 10},
        {"text": "Ask: a supervised pilot with HIL, factory calibration, and roll-disjoint data — not a production-ready stamp.", "size": 14, "color": MUTED},
    ])
    if EXTERNAL_DEMO.is_file():
        _box(s, Inches(8.2), Inches(3.95), Inches(4.65), Inches(2.7))
        s.shapes.add_picture(str(EXTERNAL_DEMO), Inches(8.35), Inches(4.1), width=Inches(4.35))
    footer(s, 8)

    prs.save(OUTPUT)
    return OUTPUT


if __name__ == "__main__":
    print(build())
