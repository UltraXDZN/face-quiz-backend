"""
overlay.py
==========
Draws bounding boxes, landmarks, stats HUD, and alert banner onto the video frame.
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np

from detectors import Detection


MODEL_COLORS = {

    "YOLOv8n-face": (59, 130, 246),
    "Haar Cascade": (251, 191, 36),
    "InsightFace SCRFD": (74, 222, 128),
}
DEFAULT_COLOR = (200, 200, 200)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
THICKNESS = 1


def _get_color(model_name: str):
    return MODEL_COLORS.get(model_name, DEFAULT_COLOR)


def draw_overlay(
    frame: np.ndarray,
    detections: List[Detection],
    model_name: str,
    fps: float,
    inf_ms: float,
    controls_hint: str = "1-3: Switch model",
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    color = _get_color(model_name)
    n_faces = len(detections)

    for det in detections:
        x1, y1, x2, y2 = det.bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        label = f"{det.conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, FONT, FONT_SCALE, THICKNESS)
        lx, ly = x1, max(y1 - 6, th + 4)
        cv2.rectangle(out, (lx, ly - th - 4), (lx + tw + 6, ly + 2), color, -1)
        cv2.putText(out, label, (lx + 3, ly - 2), FONT, FONT_SCALE, (20, 20, 20), THICKNESS, cv2.LINE_AA)

        if det.landmarks:
            for lx_, ly_ in det.landmarks:
                cv2.circle(out, (lx_, ly_), 3, color, -1)

    _draw_alert_banner(out, n_faces, w, color)
    _draw_hud(out, model_name, fps, inf_ms, n_faces, h, color)
    _draw_controls(out, w, h, controls_hint)

    return out


def _draw_alert_banner(out, n_faces, w, color):
    banner_h = 38
    alpha = 0.55

    if n_faces > 0:
        msg = f"  FACE DETECTED  ({n_faces} face{'s' if n_faces > 1 else ''})"
        text_color = color
    else:
        msg = "  No face in frame"
        text_color = (100, 100, 100)

    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, out)

    (tw, th), _ = cv2.getTextSize(msg, FONT, 0.75, 2)
    tx = (w - tw) // 2
    ty = banner_h // 2 + th // 2 - 1
    cv2.putText(out, msg, (tx, ty), FONT, 0.75, text_color, 2, cv2.LINE_AA)
    cv2.line(out, (0, banner_h), (w, banner_h), color, 1)


def _draw_hud(out, model_name, fps, inf_ms, n_faces, h, color):
    lines = [
        f"Model : {model_name}",
        f"FPS   : {fps:.1f}",
        f"Infer : {inf_ms:.1f} ms",
        f"Faces : {n_faces}",
    ]
    pad = 10
    line_h = 22
    box_h = len(lines) * line_h + pad * 2
    box_w = 240
    bx, by = pad, h - box_h - pad

    overlay = out.copy()
    cv2.rectangle(overlay, (bx, by), (bx + box_w, by + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)
    cv2.rectangle(out, (bx, by), (bx + box_w, by + box_h), color, 1)

    for i, line in enumerate(lines):
        y = by + pad + (i + 1) * line_h - 4
        cv2.putText(out, line, (bx + 8, y), FONT, FONT_SCALE, (220, 220, 220), THICKNESS, cv2.LINE_AA)


def _draw_controls(out, w, h, controls_hint):
    hints = [controls_hint, "s: Screenshot", "q: Quit"]
    font_scale = 0.42
    pad = 8
    line_h = 18
    box_h = len(hints) * line_h + pad * 2
    box_w = 160
    bx = w - box_w - pad
    by = h - box_h - pad

    overlay = out.copy()
    cv2.rectangle(overlay, (bx, by), (bx + box_w, by + box_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, out, 0.55, 0, out)

    for i, hint in enumerate(hints):
        y = by + pad + (i + 1) * line_h - 4
        cv2.putText(out, hint, (bx + 6, y), FONT, font_scale, (160, 160, 160), 1, cv2.LINE_AA)
