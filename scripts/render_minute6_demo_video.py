"""Compose the Minute-6 defense demo MP4 from the two defense GIFs.

Left (RGB HOLD) and right (LIBAD gate) advance on ONE shared stage index so
the panels cannot drift. Each stage holds ~3.5 s. Captions stay honest.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageSequence

ROOT = Path(__file__).resolve().parents[1]
RGB = ROOT / "reports" / "defense_gifs" / "rgb_hold_replay.gif"
LIBAD = ROOT / "reports" / "defense_gifs" / "libad_gate.gif"
OUT_DIR = ROOT / "reports" / "demo_video"
OUT = OUT_DIR / "SecureCoatingVision_Minute6_Demo.mp4"
FONT = Path(r"C:\Windows\Fonts\arial.ttf")
FONT_BOLD = Path(r"C:\Windows\Fonts\arialbd.ttf")

STAGE_COUNT = 4
STAGE_HOLD_S = 3.5
LOOPS = 4  # 4 stages × 3.5 s × 4 loops = 56 s
FPS = 10
CANVAS_W, CANVAS_H = 1920, 1080
PANEL_W, PANEL_H = 920, 560
MARGIN_X = 30
GAP = 20
TOP = 150
BOTTOM = 100
BG = (8, 10, 16)
CHROME = (12, 16, 24)
ACCENT = (0, 229, 255)
WHITE = (255, 255, 255)
MUTED = (180, 190, 205)
STAGE_LABELS = (
    "1/4  acquire · fixture PASS",
    "2/4  contrast · fixture REJECT",
    "3/4  detect · fixture REJECT",
    "4/4  HOLD · fixture HOLD",
)


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    path = FONT_BOLD if bold and FONT_BOLD.is_file() else FONT
    if path.is_file():
        return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _load_gif_frames(path: Path) -> list[Image.Image]:
    gif = Image.open(path)
    frames = [frame.convert("RGB") for frame in ImageSequence.Iterator(gif)]
    if len(frames) != STAGE_COUNT:
        raise SystemExit(f"{path.name}: expected {STAGE_COUNT} frames, got {len(frames)}")
    return frames


def _fit(panel: Image.Image, width: int, height: int) -> Image.Image:
    fitted = Image.new("RGB", (width, height), BG)
    copy = panel.copy()
    copy.thumbnail((width, height), Image.Resampling.LANCZOS)
    fitted.paste(copy, ((width - copy.width) // 2, (height - copy.height) // 2))
    return fitted


def main() -> None:
    if not RGB.is_file() or not LIBAD.is_file():
        raise SystemExit("defense GIFs missing; run scripts/generate_defense_gifs.py")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    left_src = _load_gif_frames(RGB)
    right_src = _load_gif_frames(LIBAD)

    font_title = _font(42, bold=True)
    font_sub = _font(26)
    font_stage = _font(28, bold=True)
    font_foot = _font(24)
    font_lane = _font(22, bold=True)

    frames_per_stage = max(1, int(round(STAGE_HOLD_S * FPS)))
    total_frames = STAGE_COUNT * frames_per_stage * LOOPS
    duration_s = total_frames / FPS

    writer = cv2.VideoWriter(
        str(OUT),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (CANVAS_W, CANVAS_H),
    )
    if not writer.isOpened():
        raise SystemExit("failed to open VideoWriter")

    left_x = MARGIN_X
    right_x = MARGIN_X + PANEL_W + GAP
    for i in range(total_frames):
        stage = (i // frames_per_stage) % STAGE_COUNT
        left = _fit(left_src[stage], PANEL_W, PANEL_H)
        right = _fit(right_src[stage], PANEL_W, PANEL_H)

        canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, CANVAS_W, TOP), fill=CHROME)
        draw.rectangle((0, CANVAS_H - BOTTOM, CANVAS_W, CANVAS_H), fill=CHROME)
        draw.rectangle((0, 0, 10, CANVAS_H), fill=ACCENT)

        canvas.paste(left, (left_x, TOP + 10))
        canvas.paste(right, (right_x, TOP + 10))

        draw.text((40, 22), "SecureCoating-Vision  ·  Minute-6 demo", fill=WHITE, font=font_title)
        draw.text(
            (40, 72),
            "Left: CoatingVision RGB → HOLD   |   Right: LIBAD protocol fixture  PASS / REJECT / REJECT / HOLD",
            fill=MUTED,
            font=font_sub,
        )
        draw.text((40, 108), f"Shared stage  {STAGE_LABELS[stage]}", fill=ACCENT, font=font_stage)

        draw.text((left_x, TOP + PANEL_H + 18), "RGB lane", fill=ACCENT, font=font_lane)
        draw.text((right_x, TOP + PANEL_H + 18), "LIBAD gate", fill=ACCENT, font=font_lane)

        draw.text(
            (40, CANVAS_H - 68),
            "The model finds defects. The evidence gate controls when the line may act.",
            fill=WHITE,
            font=font_foot,
        )
        draw.text(
            (40, CANVAS_H - 38),
            "HOLD · mock PLC · unverified calibration · comparable_to_paper=false · protocol fixture",
            fill=MUTED,
            font=font_sub,
        )
        writer.write(cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR))

    writer.release()

    h264 = OUT_DIR / "SecureCoatingVision_Minute6_Demo_h264.mp4"
    try:
        if shutil.which("ffmpeg"):
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(OUT),
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-crf",
                    "20",
                    "-preset",
                    "medium",
                    str(h264),
                ],
                check=True,
            )
            OUT.unlink(missing_ok=True)
            h264.rename(OUT)
    except Exception:
        if h264.is_file():
            h264.unlink(missing_ok=True)

    print(OUT)
    print(
        f"bytes={OUT.stat().st_size} duration_s={duration_s:.1f} fps={FPS} "
        f"shared_stage_hold_s={STAGE_HOLD_S} stages={STAGE_COUNT} loops={LOOPS}"
    )


if __name__ == "__main__":
    main()
