"""Compose the Minute-6 defense demo MP4 from the two looping GIFs.

Respects each GIF frame duration so captions stay readable (not a 10 fps
timelapse chop of stills that were meant to hold ~3 s each).
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

# Target wall-clock length; frames are held for their native GIF durations and
# the shorter side loops until this budget is reached.
DURATION_S = 56
FPS = 10
CANVAS_W, CANVAS_H = 1920, 1080
PANEL_W, PANEL_H = 960, 604
TOP = 120
BOTTOM = 80


def _load_gif(path: Path) -> tuple[list[Image.Image], list[int]]:
    gif = Image.open(path)
    frames: list[Image.Image] = []
    durations_ms: list[int] = []
    for frame in ImageSequence.Iterator(gif):
        frames.append(frame.convert("RGB"))
        durations_ms.append(max(int(frame.info.get("duration") or 3000), 500))
    if not frames:
        raise SystemExit(f"no frames in {path}")
    return frames, durations_ms


def _fit(panel: Image.Image, width: int, height: int) -> Image.Image:
    fitted = Image.new("RGB", (width, height), (0, 0, 0))
    copy = panel.copy()
    copy.thumbnail((width, height), Image.Resampling.LANCZOS)
    fitted.paste(copy, ((width - copy.width) // 2, (height - copy.height) // 2))
    return fitted


def _expand(frames: list[Image.Image], durations_ms: list[int], fps: int) -> list[Image.Image]:
    expanded: list[Image.Image] = []
    for frame, duration_ms in zip(frames, durations_ms):
        hold = max(1, int(round((duration_ms / 1000.0) * fps)))
        expanded.extend([frame] * hold)
    return expanded


def main() -> None:
    if not RGB.is_file() or not LIBAD.is_file():
        raise SystemExit("defense GIFs missing; run scripts/generate_defense_gifs.py")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    left_src, left_dur = _load_gif(RGB)
    right_src, right_dur = _load_gif(LIBAD)
    left_frames = _expand(left_src, left_dur, FPS)
    right_frames = _expand(right_src, right_dur, FPS)

    font_title = ImageFont.truetype(str(FONT), 36)
    font_mid = ImageFont.truetype(str(FONT), 24)
    font_foot = ImageFont.truetype(str(FONT), 22)

    writer = cv2.VideoWriter(
        str(OUT),
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (CANVAS_W, CANVAS_H),
    )
    if not writer.isOpened():
        raise SystemExit("failed to open VideoWriter")

    total = DURATION_S * FPS
    for i in range(total):
        left = _fit(left_frames[i % len(left_frames)], PANEL_W, PANEL_H)
        right = _fit(right_frames[i % len(right_frames)], PANEL_W, PANEL_H)
        canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), (0, 0, 0))
        canvas.paste(left, (0, TOP))
        canvas.paste(right, (PANEL_W, TOP))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 0, CANVAS_W, TOP), fill=(0, 0, 0))
        draw.rectangle((0, CANVAS_H - BOTTOM, CANVAS_W, CANVAS_H), fill=(0, 0, 0))
        draw.text(
            (40, 28),
            "SecureCoating-Vision | Minute-6 demo (no voiceover)",
            fill=(255, 255, 255),
            font=font_title,
        )
        draw.text(
            (40, 72),
            "Left: CoatingVision RGB -> HOLD  |  Right: LIBAD fixture PASS / REJECT / REJECT / HOLD",
            fill=(207, 207, 207),
            font=font_mid,
        )
        draw.text(
            (40, CANVAS_H - 52),
            "The model finds defects. The evidence gate controls when the line may act. "
            "(comparable_to_paper: false)",
            fill=(255, 255, 255),
            font=font_foot,
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
        f"bytes={OUT.stat().st_size} duration_s={DURATION_S} fps={FPS} "
        f"rgb_hold_ms={left_dur} libad_hold_ms={right_dur}"
    )


if __name__ == "__main__":
    main()
