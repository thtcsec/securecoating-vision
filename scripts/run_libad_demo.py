"""Generate the four 90-second LIBAD evidence-lane demo artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from libad.demo_cases import run_all_demo_cases  # noqa: E402


def _panel(vis: np.ndarray, xray: np.ndarray, title: str, action: str) -> np.ndarray:
    vis_r = cv2.resize(vis, (256, 256))
    xray_r = cv2.resize(xray, (256, 256))
    xray_color = cv2.applyColorMap(cv2.cvtColor(xray_r, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_BONE)
    canvas = np.zeros((320, 532, 3), dtype=np.uint8)
    canvas[40:296, 10:266] = vis_r
    canvas[40:296, 276:532] = xray_color
    color = {"PASS": (40, 180, 70), "REJECT": (40, 40, 220), "HOLD": (20, 180, 220)}[action]
    cv2.putText(canvas, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(canvas, action, (400, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    cv2.putText(canvas, "VIS", (12, 312), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(canvas, "X-rayL", (276, 312), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="reports/libad_demo")
    args = parser.parse_args()
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    results = run_all_demo_cases()
    serializable = []
    for item in results:
        case_id = item["case_id"]
        panel = _panel(
            item["frames"]["vis_bgr"],
            item["frames"]["xray_bgr"],
            f"Case {case_id}: {item['case_name']}",
            item["decision"]["action"],
        )
        image_path = out_dir / f"case_{case_id:02d}.png"
        cv2.imwrite(str(image_path), panel)
        payload = {k: v for k, v in item.items() if k != "frames"}
        payload["overlay"] = str(image_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        serializable.append(payload)
        (out_dir / f"case_{case_id:02d}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    final_screen = {
        "brand": results[0]["brand"],
        "title": results[0]["title"],
        "tagline": results[0]["tagline"],
        "evidence_class": "protocol_fixture",
        "release_provenance_ref": "reports/submission_manifest.json",
        "cases": [
            {
                "case_id": item["case_id"],
                "name": item["case_name"],
                "decision": item["decision"]["action"],
                "reason": item["decision"]["reason"],
                "identity": item["identity"],
                "scores": item["scores"],
                "calibration_state": item["calibration_state"],
                "plc_state": item["plc_state"],
                "certificate_signature": item["certificate"]["hmac_digital_signature"],
                "commit_hash": item["certificate"]["commit_hash"],
                "provenance_scope": item["certificate"].get("provenance_scope"),
                "fixture_generation_provenance": item["fixture_generation_provenance"],
            }
            for item in results
        ],
        "closing_screen": results[3]["certificate"] if results else {},
    }
    (out_dir / "final_screen.json").write_text(json.dumps(final_screen, indent=2), encoding="utf-8")
    (out_dir / "demo_manifest.json").write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    print(json.dumps({item["case_id"]: item["decision"]["action"] for item in results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
