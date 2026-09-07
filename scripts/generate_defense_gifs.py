"""Build looping defense GIFs from checked-in honest artifacts.

These are presentation loops, not factory video and not paper-comparable LIBAD
DINOv3 results. They reuse existing CoatingVision overlays and protocol-fixture
demo stills so judges see the 6-minute story without a live dashboard.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "defense_gifs"
RGB_SIZE = (960, 540)
LIBAD_SIZE = (720, 480)
CAPTION_H = 64
# Shared beat with the Minute-6 MP4 composer so left/right never drift.
FRAME_DURATION_MS = 3500
BG = (10, 13, 20)
ACCENT = (0, 229, 255)
WHITE = (255, 255, 255)
MUTED = (140, 155, 174)
YELLOW = (255, 214, 0)
RED = (255, 23, 68)
GREEN = (0, 230, 118)


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("C:/Windows/Fonts/calibri.ttf", "C:/Windows/Fonts/arial.ttf"):
        path = Path(name)
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _fit(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    canvas = Image.new("RGB", size, BG)
    work = image.convert("RGB")
    work.thumbnail(size, Image.Resampling.LANCZOS)
    x = (size[0] - work.width) // 2
    y = (size[1] - work.height) // 2
    canvas.paste(work, (x, y))
    return canvas


def _captioned(
    image: Image.Image,
    kicker: str,
    title: str,
    note: str,
    title_color: tuple[int, int, int] = WHITE,
) -> Image.Image:
    width, height = image.size
    canvas = Image.new("RGB", (width, height + CAPTION_H), BG)
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, height, width, height + CAPTION_H), fill=(8, 11, 18))
    draw.rectangle((0, height, 8, height + CAPTION_H), fill=ACCENT)
    draw.text((18, height + 6), kicker, font=_font(14), fill=ACCENT)
    draw.text((18, height + 24), title, font=_font(22), fill=title_color)
    bbox = draw.textbbox((0, 0), note, font=_font(14))
    note_w = bbox[2] - bbox[0]
    draw.text((width - note_w - 16, height + 38), note, font=_font(14), fill=MUTED)
    return canvas


def _load(relative: str) -> Image.Image:
    path = ROOT / relative
    if not path.is_file():
        raise FileNotFoundError(relative)
    return Image.open(path).convert("RGB")


def _save_gif(path: Path, frames: Sequence[Image.Image], duration_ms: int) -> Path:
    if len(frames) < 2:
        raise ValueError(f"{path.name} needs at least two frames")
    path.parent.mkdir(parents=True, exist_ok=True)
    quantized = [frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=128) for frame in frames]
    quantized[0].save(
        path,
        save_all=True,
        append_images=quantized[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
        optimize=True,
    )
    return path


def build_rgb_hold_replay() -> Path:
    frames = [
        _captioned(
            _fit(_load("reports/coatingvision_visual_evidence/image_1548_optical_raw.png"), RGB_SIZE),
            "01  ACQUIRED OPTICAL",
            "CoatingVision JPEG  ·  DOI 10.6084/m9.figshare.29260121.v1",
            "real RGB  ·  not factory roll-disjoint",
        ),
        _captioned(
            _fit(_load("reports/coatingvision_visual_evidence/image_1548_contrast_clahe.png"), RGB_SIZE),
            "02  INSPECTABLE SURFACE",
            "CLAHE contrast on the same capture",
            "presentation view  ·  not model input",
        ),
        _captioned(
            _fit(_load("reports/coatingvision_real_demo/coatingvision_model_output.png"), RGB_SIZE),
            "03  YOLO26n DETECT",
            "surface_crack  ·  confidence 0.53  ·  f72a8f2b…",
            "PyTorch/ONNX same two-class map",
            title_color=RED,
        ),
        _captioned(
            _fit(_load("reports/coatingvision_visual_evidence/image_1548_yolo_candidate_zoom.png"), RGB_SIZE),
            "04  HOLD",
            "Detection exists. Mock PLC + unverified calibration block release.",
            "not a factory PASS",
            title_color=YELLOW,
        ),
    ]
    return _save_gif(OUT / "rgb_hold_replay.gif", frames, FRAME_DURATION_MS)


def build_libad_gate() -> Path:
    manifest = json.loads((ROOT / "reports/libad_demo/demo_manifest.json").read_text(encoding="utf-8"))
    expected = ("PASS", "REJECT", "REJECT", "HOLD")
    colors = {"PASS": GREEN, "REJECT": RED, "HOLD": YELLOW}
    frames = []
    for index, action in enumerate(expected, 1):
        item = manifest[index - 1]
        if item["decision"]["action"] != action:
            raise ValueError(f"Demo case {index} is {item['decision']['action']}, expected {action}")
        story = str(item.get("story") or item["decision"]["reason"])
        frames.append(
            _captioned(
                _fit(_load(f"reports/libad_demo/case_0{index}.png"), LIBAD_SIZE),
                f"LIBAD FIXTURE  {index}/4  ·  {action}",
                story[:78],
                "protocol fixture  ·  comparable_to_paper=false",
                title_color=colors[action],
            )
        )
    return _save_gif(OUT / "libad_gate.gif", frames, FRAME_DURATION_MS)


def build_all() -> list[Path]:
    return [build_rgb_hold_replay(), build_libad_gate()]


def main() -> int:
    written = build_all()
    for path in written:
        print(f"Wrote {path.relative_to(ROOT)} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
