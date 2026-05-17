"""
overlay.py  (optimised v3)
==========================
Draws bounding boxes, landmarks, stats HUD, alert banner, and liveness badges.

v3 keeps the old bool/None liveness API, but also supports richer
LivenessResult-like objects with state/score/track_id fields.
"""

from __future__ import annotations

import functools
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from detectors import Detection


MODEL_COLORS: Dict[str, Tuple[int, int, int]] = {
    "YOLOv8n-face":      (59, 130, 246),
    "Haar Cascade":      (251, 191, 36),
    "InsightFace SCRFD": (74, 222, 128),
}
DEFAULT_COLOR = (200, 200, 200)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
THICKNESS = 1

_LIVE_COLOR = (60, 200, 60)
_SPOOF_COLOR = (40, 40, 220)
_PENDING_COLOR = (140, 140, 140)
_UNCERTAIN_COLOR = (50, 170, 230)


def _get_color(model_name: str) -> Tuple[int, int, int]:
    return MODEL_COLORS.get(model_name, DEFAULT_COLOR)


@functools.lru_cache(maxsize=256)
def _cached_text_size(label: str, font_scale: float, thickness: int):
    (tw, th), baseline = cv2.getTextSize(label, FONT, font_scale, thickness)
    return tw, th, baseline


def _bbox_of(det: Union[Detection, Sequence[float]]) -> Tuple[int, int, int, int]:
    if hasattr(det, "bbox"):
        return det.bbox
    x1, y1, x2, y2 = det[:4]
    return int(x1), int(y1), int(x2), int(y2)


def _conf_of(det: Union[Detection, Sequence[float]]) -> float:
    if hasattr(det, "conf"):
        return float(det.conf)
    return float(det[4]) if len(det) > 4 else 0.0


def _landmarks_of(det: Union[Detection, Sequence[float]]):
    return getattr(det, "landmarks", None)


def draw_overlay(
    frame: np.ndarray,
    detections: List[Union[Detection, Sequence[float]]],
    model_name: str,
    fps: float,
    inf_ms: float,
    controls_hint: str = "1-3: Switch model",
    liveness_results: Optional[Dict[int, object]] = None,
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    color = _get_color(model_name)
    n_faces = len(detections)

    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = _bbox_of(det)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        conf_label = f"{_conf_of(det):.2f}"
        tw, th, _ = _cached_text_size(conf_label, FONT_SCALE, THICKNESS)
        lbx = x1
        lby = max(y1 - 6, th + 4)
        cv2.rectangle(out, (lbx, lby - th - 4), (lbx + tw + 6, lby + 2), color, -1)
        cv2.putText(
            out, conf_label, (lbx + 3, lby - 2),
            FONT, FONT_SCALE, (20, 20, 20), THICKNESS, cv2.LINE_AA,
        )

        landmarks = _landmarks_of(det)
        if landmarks:
            for lx_, ly_ in landmarks:
                cv2.circle(out, (int(lx_), int(ly_)), 3, color, -1)

        if liveness_results is not None:
            _draw_liveness_badge(out, x1, lby, th, liveness_results.get(idx))

    _draw_blended_panels(out, model_name, fps, inf_ms, n_faces, w, h, color, controls_hint)
    return out


def _decode_liveness_state(live_state: object) -> Tuple[str, Tuple[int, int, int]]:
    if live_state is True:
        return "LIVE", _LIVE_COLOR
    if live_state is False:
        return "SPOOF", _SPOOF_COLOR
    if live_state is None:
        return "...", _PENDING_COLOR

    state = str(getattr(live_state, "state", "PENDING")).upper()
    score = getattr(live_state, "score", None)
    track_id = getattr(live_state, "track_id", None)

    if state == "LIVE":
        color = _LIVE_COLOR
        label = "LIVE"
    elif state == "SPOOF":
        color = _SPOOF_COLOR
        label = "SPOOF"
    elif state == "UNCERTAIN":
        color = _UNCERTAIN_COLOR
        label = "CHECK"
    else:
        color = _PENDING_COLOR
        label = "..."

    if score is not None and state in {"LIVE", "SPOOF", "UNCERTAIN"}:
        label = f"{label} {float(score):.2f}"
    if track_id is not None and state in {"LIVE", "SPOOF", "UNCERTAIN"}:
        label = f"T{int(track_id)} {label}"

    return label, color


def _draw_liveness_badge(out: np.ndarray, x1: int, lby: int, conf_th: int, live_state: object) -> None:
    badge_text, badge_color = _decode_liveness_state(live_state)

    btw, bth, _ = _cached_text_size(badge_text, 0.45, 1)
    bx = x1
    by = max(lby - conf_th - 10, bth + 4)
    cv2.rectangle(out, (bx, by - bth - 4), (bx + btw + 6, by + 2), badge_color, -1)
    cv2.putText(
        out, badge_text, (bx + 3, by - 2),
        FONT, 0.45, (20, 20, 20), 1, cv2.LINE_AA,
    )


def _draw_blended_panels(
    out: np.ndarray,
    model_name: str,
    fps: float,
    inf_ms: float,
    n_faces: int,
    w: int,
    h: int,
    color: Tuple[int, int, int],
    controls_hint: str,
) -> None:
    overlay = out.copy()

    BANNER_H = 38
    cv2.rectangle(overlay, (0, 0), (w, BANNER_H), (0, 0, 0), -1)

    HUD_PAD = 10
    HUD_LINE = 22
    HUD_LINES = 4
    HUD_H = HUD_LINES * HUD_LINE + HUD_PAD * 2
    HUD_W = 240
    hx, hy = HUD_PAD, h - HUD_H - HUD_PAD
    cv2.rectangle(overlay, (hx, hy), (hx + HUD_W, hy + HUD_H), (0, 0, 0), -1)

    CTRL_HINTS = [controls_hint, "s: Screenshot", "q: Quit"]
    CTRL_LINE = 18
    CTRL_PAD = 8
    CTRL_H = len(CTRL_HINTS) * CTRL_LINE + CTRL_PAD * 2
    CTRL_W = 190
    cx = w - CTRL_W - CTRL_PAD
    cy = h - CTRL_H - CTRL_PAD
    cv2.rectangle(overlay, (cx, cy), (cx + CTRL_W, cy + CTRL_H), (0, 0, 0), -1)

    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)

    if n_faces > 0:
        msg = f"  FACE DETECTED  ({n_faces} face{'s' if n_faces > 1 else ''})"
        text_color = color
    else:
        msg = "  No face in frame"
        text_color = (100, 100, 100)

    tw, th, _ = _cached_text_size(msg, 0.75, 2)
    tx = (w - tw) // 2
    ty = BANNER_H // 2 + th // 2 - 1
    cv2.putText(out, msg, (tx, ty), FONT, 0.75, text_color, 2, cv2.LINE_AA)
    cv2.line(out, (0, BANNER_H), (w, BANNER_H), color, 1)

    cv2.rectangle(out, (hx, hy), (hx + HUD_W, hy + HUD_H), color, 1)
    for i, line in enumerate([
        f"Model : {model_name}",
        f"FPS   : {fps:.1f}",
        f"Infer : {inf_ms:.1f} ms",
        f"Faces : {n_faces}",
    ]):
        y = hy + HUD_PAD + (i + 1) * HUD_LINE - 4
        cv2.putText(out, line, (hx + 8, y), FONT, FONT_SCALE, (220, 220, 220), THICKNESS, cv2.LINE_AA)

    for i, hint in enumerate(CTRL_HINTS):
        y = cy + CTRL_PAD + (i + 1) * CTRL_LINE - 4
        cv2.putText(out, hint, (cx + 6, y), FONT, 0.42, (160, 160, 160), 1, cv2.LINE_AA)
