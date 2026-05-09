from __future__ import annotations

import functools
from typing import List, Optional

import cv2
import numpy as np

from detectors import Detection


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_COLORS = {
    "YOLOv8n-face":       (59,  130, 246),
    "Haar Cascade":       (251, 191,  36),
    "InsightFace SCRFD":  (74,  222, 128),
}
DEFAULT_COLOR = (200, 200, 200)

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
THICKNESS  = 1


def _get_color(model_name: str) -> tuple:
    return MODEL_COLORS.get(model_name, DEFAULT_COLOR)


# ---------------------------------------------------------------------------
# Text-size cache
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=64)
def _cached_text_size(label: str, font_scale: float, thickness: int):
    """Cache cv2.getTextSize results for repeated label strings."""
    (tw, th), baseline = cv2.getTextSize(label, FONT, font_scale, thickness)
    return tw, th, baseline


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def draw_overlay(
    frame: np.ndarray,
    detections: List[Detection],
    model_name: str,
    fps: float,
    inf_ms: float,
    controls_hint: str = "1-3: Switch model",
) -> np.ndarray:
    """
    Compose all overlay elements onto a copy of *frame* and return it.

    Key change: a single overlay buffer is allocated once and all
    semi-transparent rectangles are drawn into it before the single
    addWeighted call that blends everything, cutting per-frame blend
    operations from 2 down to 1.
    """
    out = frame.copy()
    h, w = out.shape[:2]
    color = _get_color(model_name)
    n_faces = len(detections)

    # --- bounding boxes + landmarks (no blending needed) -------------------
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        label = f"{det.conf:.2f}"
        tw, th, _ = _cached_text_size(label, FONT_SCALE, THICKNESS)
        lx, ly = x1, max(y1 - 6, th + 4)
        cv2.rectangle(out, (lx, ly - th - 4), (lx + tw + 6, ly + 2), color, -1)
        cv2.putText(
            out, label, (lx + 3, ly - 2),
            FONT, FONT_SCALE, (20, 20, 20), THICKNESS, cv2.LINE_AA,
        )

        if det.landmarks:
            for lx_, ly_ in det.landmarks:
                cv2.circle(out, (lx_, ly_), 3, color, -1)

    # --- single composite blend for all semi-transparent panels ------------
    _draw_blended_panels(out, model_name, fps, inf_ms, n_faces, w, h, color, controls_hint)

    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _draw_blended_panels(
    out: np.ndarray,
    model_name: str,
    fps: float,
    inf_ms: float,
    n_faces: int,
    w: int,
    h: int,
    color: tuple,
    controls_hint: str,
) -> None:
    """
    Draw ALL semi-transparent rectangles into one overlay copy, then blend
    once.  Previously each panel did its own copy+blend (O(W×H) per panel);
    now it's a single O(W×H) operation regardless of panel count.
    """
    overlay = out.copy()   # ← the ONLY extra frame allocation in this path

    # Banner
    BANNER_H = 38
    cv2.rectangle(overlay, (0, 0), (w, BANNER_H), (0, 0, 0), -1)

    # HUD box
    HUD_PAD   = 10
    HUD_LINE  = 22
    HUD_LINES = 4
    HUD_H     = HUD_LINES * HUD_LINE + HUD_PAD * 2
    HUD_W     = 240
    hx, hy    = HUD_PAD, h - HUD_H - HUD_PAD
    cv2.rectangle(overlay, (hx, hy), (hx + HUD_W, hy + HUD_H), (0, 0, 0), -1)

    # Controls box
    CTRL_HINTS  = [controls_hint, "s: Screenshot", "q: Quit"]
    CTRL_LINE   = 18
    CTRL_PAD    = 8
    CTRL_H      = len(CTRL_HINTS) * CTRL_LINE + CTRL_PAD * 2
    CTRL_W      = 160
    cx          = w - CTRL_W - CTRL_PAD
    cy          = h - CTRL_H - CTRL_PAD
    cv2.rectangle(overlay, (cx, cy), (cx + CTRL_W, cy + CTRL_H), (0, 0, 0), -1)

    # Blend all dark panels at once (one call instead of three)
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)

    # --- Banner text + separator ----------------------------------------
    if n_faces > 0:
        msg        = f"  FACE DETECTED  ({n_faces} face{'s' if n_faces > 1 else ''})"
        text_color = color
    else:
        msg        = "  No face in frame"
        text_color = (100, 100, 100)

    tw, th, _ = _cached_text_size(msg, 0.75, 2)
    tx = (w - tw) // 2
    ty = BANNER_H // 2 + th // 2 - 1
    cv2.putText(out, msg, (tx, ty), FONT, 0.75, text_color, 2, cv2.LINE_AA)
    cv2.line(out, (0, BANNER_H), (w, BANNER_H), color, 1)

    # --- HUD text -------------------------------------------------------
    cv2.rectangle(out, (hx, hy), (hx + HUD_W, hy + HUD_H), color, 1)
    hud_lines = [
        f"Model : {model_name}",
        f"FPS   : {fps:.1f}",
        f"Infer : {inf_ms:.1f} ms",
        f"Faces : {n_faces}",
    ]
    for i, line in enumerate(hud_lines):
        y = hy + HUD_PAD + (i + 1) * HUD_LINE - 4
        cv2.putText(
            out, line, (hx + 8, y),
            FONT, FONT_SCALE, (220, 220, 220), THICKNESS, cv2.LINE_AA,
        )

    # --- Controls text --------------------------------------------------
    for i, hint in enumerate(CTRL_HINTS):
        y = cy + CTRL_PAD + (i + 1) * CTRL_LINE - 4
        cv2.putText(
            out, hint, (cx + 6, y),
            FONT, 0.42, (160, 160, 160), 1, cv2.LINE_AA,
        )
