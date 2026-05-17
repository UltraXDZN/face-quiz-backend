"""
liveness.py  (v4 — passive FAS + heuristic fallback)
=====================================================
Passive liveness / face anti-spoofing for non-interactive monitoring.

Main idea
---------
1. If an ONNX anti-spoofing model is provided, it becomes the primary signal.
2. The old motion/texture heuristic remains available and is still used as:
   - fallback when no model is configured
   - weak supporting signal when a model is configured
3. Decisions are smoothed per face track so the badge does not flicker.

Public API compatibility
------------------------
Existing usage still works:
    checker = LivenessChecker(...)
    results = checker.update(frame, detections, frame_idx)

By default update() returns Dict[int, Optional[bool]] just like before.
For richer overlay/debug output:
    checker.update(..., return_details=True)
"""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from detectors import Detection


# ---------------------------------------------------------------------------
# Tuneable heuristic defaults
# ---------------------------------------------------------------------------

_PATCH_SIZE = (64, 64)
_ROI_MARGIN = 0.08

_MOTION_BUF_LEN = 12
_MOTION_MIN_FRAMES = 4
_MOTION_LIVE_THR = 1.8
_MOTION_MAX = 8.0

_SPEC_PERCENTILE = 95
_SPEC_MIN_COV = 0.002
_SPEC_MAX_COV = 0.20

_GRAD_BINS = 18
_GRAD_LIVE_ENTROPY = 2.5

_W_MOTION = 0.65
_W_SPEC = 0.20
_W_GRAD = 0.15

_SCORE_THRESHOLD = 0.42


# ---------------------------------------------------------------------------
# Small data containers
# ---------------------------------------------------------------------------

class LivenessResult:
    __slots__ = (
        "track_id",
        "state",
        "score",
        "raw_score",
        "model_score",
        "heuristic_score",
        "motion",
        "spec",
        "grad",
        "samples",
        "method",
    )

    def __init__(
        self,
        track_id: int,
        state: str,
        score: Optional[float],
        raw_score: Optional[float],
        model_score: Optional[float],
        heuristic_score: Optional[float],
        motion: Optional[float],
        spec: Optional[float],
        grad: Optional[float],
        samples: int,
        method: str,
    ):
        self.track_id = track_id
        self.state = state
        self.score = score
        self.raw_score = raw_score
        self.model_score = model_score
        self.heuristic_score = heuristic_score
        self.motion = motion
        self.spec = spec
        self.grad = grad
        self.samples = samples
        self.method = method

    @property
    def is_live(self) -> Optional[bool]:
        if self.state == "LIVE":
            return True
        if self.state == "SPOOF":
            return False
        return None

    def as_bool(self) -> Optional[bool]:
        return self.is_live

    def as_dict(self) -> Dict[str, Optional[Union[int, float, str]]]:
        return {
            "track_id": self.track_id,
            "state": self.state,
            "score": self.score,
            "raw_score": self.raw_score,
            "model_score": self.model_score,
            "heuristic_score": self.heuristic_score,
            "motion": self.motion,
            "spec": self.spec,
            "grad": self.grad,
            "samples": self.samples,
            "method": self.method,
        }

    def __bool__(self) -> bool:
        return self.is_live is True


class _TrackState:
    __slots__ = (
        "track_id",
        "bbox",
        "last_seen_frame",
        "last_eval_frame",
        "cached",
        "motion_buf",
        "ema_score",
        "samples",
    )

    def __init__(self, track_id: int, bbox: Tuple[int, int, int, int], frame_idx: int):
        self.track_id = track_id
        self.bbox = bbox
        self.last_seen_frame = frame_idx
        self.last_eval_frame = -10**9
        self.cached: Optional[LivenessResult] = None
        self.motion_buf = deque(maxlen=_MOTION_BUF_LEN)
        self.ema_score: Optional[float] = None
        self.samples = 0


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _bbox_of(det: Union[Detection, Sequence[float]]) -> Tuple[int, int, int, int]:
    if hasattr(det, "bbox"):
        x1, y1, x2, y2 = det.bbox
    else:
        x1, y1, x2, y2 = det[:4]
    return int(x1), int(y1), int(x2), int(y2)


def _clip_bbox(
    bbox: Tuple[int, int, int, int],
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    return x1, y1, x2, y2


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / max(area_a + area_b - inter, 1e-6)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32).reshape(-1)
    x = x - np.max(x)
    e = np.exp(x)
    return e / max(float(np.sum(e)), 1e-9)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-float(x)))


# ---------------------------------------------------------------------------
# Crop extraction
# ---------------------------------------------------------------------------

def _extract_roi(frame: np.ndarray, det: Union[Detection, Sequence[float]]) -> Optional[np.ndarray]:
    fh, fw = frame.shape[:2]
    x1, y1, x2, y2 = _bbox_of(det)
    mx = int((x2 - x1) * _ROI_MARGIN)
    my = int((y2 - y1) * _ROI_MARGIN)
    cx1 = max(0, x1 - mx)
    cy1 = max(0, y1 - my)
    cx2 = min(fw, x2 + mx)
    cy2 = min(fh, y2 + my)
    if cx2 - cx1 < 10 or cy2 - cy1 < 10:
        return None
    return cv2.resize(frame[cy1:cy2, cx1:cx2], _PATCH_SIZE, interpolation=cv2.INTER_AREA)


def _extract_scaled_face_crop(
    frame: np.ndarray,
    det: Union[Detection, Sequence[float]],
    output_size: Tuple[int, int],
    scale: float,
) -> Optional[np.ndarray]:
    fh, fw = frame.shape[:2]
    x1, y1, x2, y2 = _bbox_of(det)
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    cx = x1 + bw * 0.5
    cy = y1 + bh * 0.5
    side = max(bw, bh) * max(1.0, scale)

    nx1 = int(round(cx - side * 0.5))
    ny1 = int(round(cy - side * 0.5))
    nx2 = int(round(cx + side * 0.5))
    ny2 = int(round(cy + side * 0.5))
    nx1, ny1, nx2, ny2 = _clip_bbox((nx1, ny1, nx2, ny2), fw, fh)

    if nx2 - nx1 < 10 or ny2 - ny1 < 10:
        return None

    crop = frame[ny1:ny2, nx1:nx2]
    return cv2.resize(crop, output_size, interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Heuristic cues, preserved from v3
# ---------------------------------------------------------------------------

def _score_motion(buf: deque) -> float:
    if len(buf) < _MOTION_MIN_FRAMES:
        return 0.35

    diffs = [
        float(np.mean(np.abs(buf[i].astype(np.float32) - buf[i - 1].astype(np.float32))))
        for i in range(1, len(buf))
    ]
    mad = float(np.mean(diffs))

    if mad < _MOTION_LIVE_THR:
        return (mad / _MOTION_LIVE_THR) * 0.5
    return min(1.0, 0.5 + (mad - _MOTION_LIVE_THR) / (_MOTION_MAX - _MOTION_LIVE_THR) * 0.5)


def _score_specular(gray: np.ndarray) -> float:
    thr = float(np.percentile(gray, _SPEC_PERCENTILE))
    mask = gray >= thr
    coverage = mask.sum() / gray.size

    if coverage < _SPEC_MIN_COV:
        return 0.15
    if coverage > _SPEC_MAX_COV:
        return 0.15

    ys, xs = np.where(mask)
    if len(xs) < 3:
        return 0.2

    std_x = float(np.std(xs))
    std_y = float(np.std(ys))
    spread = (std_x + std_y) / 2.0
    return min(1.0, spread / 20.0)


def _score_gradient_entropy(gray: np.ndarray) -> float:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)

    thresh = float(np.percentile(mag, 60))
    strong = mag > thresh
    if strong.sum() < 10:
        return 0.5

    angles = np.degrees(np.arctan2(gy[strong], gx[strong])) % 180.0
    hist, _ = np.histogram(angles, bins=_GRAD_BINS, range=(0, 180))

    hist = hist.astype(np.float32) + 1e-6
    p = hist / hist.sum()
    entropy = float(-np.sum(p * np.log2(p)))
    max_entropy = np.log2(_GRAD_BINS)

    raw = entropy / max_entropy
    if entropy < _GRAD_LIVE_ENTROPY:
        return raw * 0.5
    return min(1.0, raw)


def _compute_score(
    bgr: np.ndarray,
    gray: np.ndarray,
    motion_buf: deque,
) -> Tuple[float, Dict[str, float]]:
    s_motion = _score_motion(motion_buf)
    s_spec = _score_specular(gray)
    s_grad = _score_gradient_entropy(gray)
    total = _W_MOTION * s_motion + _W_SPEC * s_spec + _W_GRAD * s_grad
    return total, {"motion": s_motion, "spec": s_spec, "grad": s_grad}


# ---------------------------------------------------------------------------
# ONNX anti-spoofing backend
# ---------------------------------------------------------------------------

class AntiSpoofONNX:
    """Tiny wrapper around an RGB face anti-spoofing ONNX model."""

    def __init__(
        self,
        model_path: Union[str, Path],
        input_size: int = 128,
        live_index: int = 1,
        crop_scale: float = 2.7,
        input_scale: float = 1.0 / 255.0,
        channel_order: str = "rgb",
        providers: Optional[List[str]] = None,
    ):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError("pip install onnxruntime") from exc

        self.model_path = str(model_path)
        self.live_index = int(live_index)
        self.crop_scale = float(crop_scale)
        self.input_scale = float(input_scale)
        self.channel_order = channel_order.lower().strip()

        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = 1
        sess_opts.inter_op_num_threads = 1
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        if providers is None:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(self.model_path, sess_options=sess_opts, providers=providers)
        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        self.input_shape = list(model_input.shape)
        self.input_type = model_input.type
        self.layout, self.height, self.width = self._parse_input_shape(input_size)

    def _parse_input_shape(self, fallback_size: int) -> Tuple[str, int, int]:
        shape = self.input_shape
        if len(shape) != 4:
            return "NCHW", fallback_size, fallback_size

        dims = [d if isinstance(d, int) else None for d in shape]
        if dims[1] in (1, 3):
            h = dims[2] or fallback_size
            w = dims[3] or fallback_size
            return "NCHW", int(h), int(w)

        if dims[3] in (1, 3):
            h = dims[1] or fallback_size
            w = dims[2] or fallback_size
            return "NHWC", int(h), int(w)

        return "NCHW", fallback_size, fallback_size

    def predict(self, frame: np.ndarray, det: Union[Detection, Sequence[float]]) -> Optional[float]:
        crop = _extract_scaled_face_crop(
            frame=frame,
            det=det,
            output_size=(self.width, self.height),
            scale=self.crop_scale,
        )
        if crop is None:
            return None

        if self.channel_order == "rgb":
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        x = crop.astype(np.float32) * self.input_scale

        if self.layout == "NCHW":
            x = np.transpose(x, (2, 0, 1))[None, ...]
        else:
            x = x[None, ...]

        if "float16" in self.input_type:
            x = x.astype(np.float16)
        else:
            x = x.astype(np.float32)

        outputs = self.session.run(None, {self.input_name: x})
        if not outputs:
            return None
        return self._output_to_live_score(outputs[0])

    def _output_to_live_score(self, output: np.ndarray) -> Optional[float]:
        arr = np.asarray(output).reshape(-1).astype(np.float32)
        if arr.size == 0:
            return None

        if arr.size == 1:
            v = float(arr[0])
            return v if 0.0 <= v <= 1.0 else _sigmoid(v)

        if np.all(arr >= 0.0) and abs(float(np.sum(arr)) - 1.0) <= 0.08:
            probs = arr / max(float(np.sum(arr)), 1e-9)
        else:
            probs = _softmax(arr)

        idx = self.live_index
        if idx < 0 or idx >= probs.size:
            idx = int(np.argmax(probs))
        return float(probs[idx])


# ---------------------------------------------------------------------------
# Public checker
# ---------------------------------------------------------------------------

class LivenessChecker:
    """
    Stateful passive liveness checker.

    Without onnx_model_path:
        Uses the old motion/texture heuristic and returns LIVE/SPOOF/None.

    With onnx_model_path:
        Uses ONNX anti-spoofing as primary score + weak heuristic support,
        then smooths the result per face track.
    """

    def __init__(
        self,
        cooldown_frames: int = 12,
        score_threshold: float = _SCORE_THRESHOLD,
        min_roi_px: int = 48,
        debug: bool = False,
        onnx_model_path: Optional[Union[str, Path]] = None,
        onnx_input_size: int = 128,
        onnx_live_index: int = 1,
        onnx_crop_scale: float = 2.7,
        onnx_input_scale: float = 1.0 / 255.0,
        ema_alpha: float = 0.45,
        min_samples: int = 3,
        uncertain_margin: float = 0.07,
        model_weight: float = 0.78,
        heuristic_weight: float = 0.22,
        track_iou_threshold: float = 0.25,
        max_lost_frames: int = 90,
    ):
        self._cooldown = max(1, int(cooldown_frames))
        self._threshold = float(score_threshold)
        self._min_roi = int(min_roi_px)
        self._debug = bool(debug)

        self._ema_alpha = float(np.clip(ema_alpha, 0.01, 1.0))
        self._min_samples = max(1, int(min_samples))
        self._uncertain_margin = max(0.0, float(uncertain_margin))
        self._model_weight = max(0.0, float(model_weight))
        self._heuristic_weight = max(0.0, float(heuristic_weight))
        total_w = self._model_weight + self._heuristic_weight
        if total_w <= 0.0:
            self._model_weight, self._heuristic_weight = 1.0, 0.0
        else:
            self._model_weight /= total_w
            self._heuristic_weight /= total_w

        self._track_iou_threshold = float(track_iou_threshold)
        self._max_lost_frames = int(max_lost_frames)

        self._tracks: Dict[int, _TrackState] = {}
        self._next_track_id = 0

        self._antispoof: Optional[AntiSpoofONNX] = None
        if onnx_model_path:
            self._antispoof = AntiSpoofONNX(
                model_path=onnx_model_path,
                input_size=onnx_input_size,
                live_index=onnx_live_index,
                crop_scale=onnx_crop_scale,
                input_scale=onnx_input_scale,
            )

    @property
    def backend_name(self) -> str:
        return "onnx+heuristic" if self._antispoof is not None else "heuristic"

    def update(
        self,
        frame: np.ndarray,
        detections: List[Union[Detection, Sequence[float]]],
        frame_idx: int,
        return_details: bool = False,
    ) -> Dict[int, Union[Optional[bool], LivenessResult]]:
        assignments = self._assign_tracks(detections, frame_idx)
        active_track_ids = set(assignments.values())
        self._prune_tracks(frame_idx, active_track_ids)

        results: Dict[int, Union[Optional[bool], LivenessResult]] = {}

        for det_idx, det in enumerate(detections):
            track_id = assignments[det_idx]
            track = self._tracks[track_id]
            x1, y1, x2, y2 = _bbox_of(det)
            too_small = (x2 - x1) < self._min_roi or (y2 - y1) < self._min_roi

            roi = _extract_roi(frame, det)
            if roi is not None and not too_small:
                track.motion_buf.append(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))

            if too_small:
                result = self._make_result(track, "PENDING", None, None, None, None, None, None, None, "too-small")
                track.cached = result
                results[det_idx] = result if return_details else result.as_bool()
                continue

            if frame_idx - track.last_eval_frame < self._cooldown and track.cached is not None:
                results[det_idx] = track.cached if return_details else track.cached.as_bool()
                continue

            result = self._evaluate(frame, det, track, frame_idx)
            track.cached = result
            track.last_eval_frame = frame_idx
            results[det_idx] = result if return_details else result.as_bool()

        return results

    def _assign_tracks(
        self,
        detections: List[Union[Detection, Sequence[float]]],
        frame_idx: int,
    ) -> Dict[int, int]:
        assignments: Dict[int, int] = {}
        used_tracks = set()

        for det_idx, det in enumerate(detections):
            bbox = _bbox_of(det)
            best_track_id = None
            best_iou = 0.0

            for track_id, track in self._tracks.items():
                if track_id in used_tracks:
                    continue
                score = _iou(bbox, track.bbox)
                if score > best_iou:
                    best_iou = score
                    best_track_id = track_id

            if best_track_id is None or best_iou < self._track_iou_threshold:
                best_track_id = self._next_track_id
                self._next_track_id += 1
                self._tracks[best_track_id] = _TrackState(best_track_id, bbox, frame_idx)

            track = self._tracks[best_track_id]
            track.bbox = bbox
            track.last_seen_frame = frame_idx
            used_tracks.add(best_track_id)
            assignments[det_idx] = best_track_id

        return assignments

    def _prune_tracks(self, frame_idx: int, active_track_ids: set) -> None:
        stale = [
            track_id for track_id, track in self._tracks.items()
            if track_id not in active_track_ids and frame_idx - track.last_seen_frame > self._max_lost_frames
        ]
        for track_id in stale:
            del self._tracks[track_id]

    def _evaluate(
        self,
        frame: np.ndarray,
        det: Union[Detection, Sequence[float]],
        track: _TrackState,
        frame_idx: int,
    ) -> LivenessResult:
        roi = _extract_roi(frame, det)
        if roi is None:
            return self._make_result(track, "PENDING", None, None, None, None, None, None, None, "no-roi")

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        heuristic_score, cues = _compute_score(roi, gray, track.motion_buf)

        model_score = None
        method = "heuristic"

        if self._antispoof is not None:
            try:
                model_score = self._antispoof.predict(frame, det)
            except Exception as exc:
                if self._debug:
                    print(f"[liveness] ONNX inference failed: {exc}")
                model_score = None

        if model_score is not None:
            raw_score = self._model_weight * model_score + self._heuristic_weight * heuristic_score
            method = "onnx+heuristic"
        else:
            raw_score = heuristic_score

        if track.ema_score is None:
            track.ema_score = raw_score
        else:
            a = self._ema_alpha
            track.ema_score = a * raw_score + (1.0 - a) * track.ema_score
        track.samples += 1

        state = self._state_from_score(track.ema_score, track.samples)

        result = self._make_result(
            track=track,
            state=state,
            score=track.ema_score,
            raw_score=raw_score,
            model_score=model_score,
            heuristic_score=heuristic_score,
            motion=cues.get("motion"),
            spec=cues.get("spec"),
            grad=cues.get("grad"),
            method=method,
        )

        if self._debug:
            ms = -1.0 if model_score is None else model_score
            print(
                f"[liveness] track={track.track_id} {state} "
                f"score={result.score:.3f} raw={raw_score:.3f} model={ms:.3f} "
                f"heur={heuristic_score:.3f} motion={cues['motion']:.3f} "
                f"spec={cues['spec']:.3f} grad={cues['grad']:.3f} "
                f"samples={track.samples} backend={method}"
            )

        return result

    def _state_from_score(self, score: float, samples: int) -> str:
        if samples < self._min_samples:
            return "PENDING"
        if score >= self._threshold + self._uncertain_margin:
            return "LIVE"
        if score <= self._threshold - self._uncertain_margin:
            return "SPOOF"
        return "UNCERTAIN"

    def _make_result(
        self,
        track: _TrackState,
        state: str,
        score: Optional[float],
        raw_score: Optional[float],
        model_score: Optional[float],
        heuristic_score: Optional[float],
        motion: Optional[float],
        spec: Optional[float],
        grad: Optional[float],
        method: str,
    ) -> LivenessResult:
        def r(v):
            return None if v is None else round(float(v), 3)

        return LivenessResult(
            track_id=track.track_id,
            state=state,
            score=r(score),
            raw_score=r(raw_score),
            model_score=r(model_score),
            heuristic_score=r(heuristic_score),
            motion=r(motion),
            spec=r(spec),
            grad=r(grad),
            samples=track.samples,
            method=method,
        )

    def reset(self) -> None:
        self._tracks.clear()
        self._next_track_id = 0

    def get_debug_scores(
        self,
        frame: np.ndarray,
        det: Union[Detection, Sequence[float]],
        slot_idx: int = 0,
    ) -> Dict[str, float]:
        roi = _extract_roi(frame, det)
        if roi is None:
            return {}

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        track = self._tracks.get(slot_idx)
        buf = track.motion_buf if track is not None else deque(maxlen=_MOTION_BUF_LEN)
        heuristic_total, cue_scores = _compute_score(roi, gray, buf)

        model_score = None
        if self._antispoof is not None:
            try:
                model_score = self._antispoof.predict(frame, det)
            except Exception:
                model_score = None

        if model_score is not None:
            total = self._model_weight * model_score + self._heuristic_weight * heuristic_total
        else:
            total = heuristic_total

        cue_scores["heuristic"] = heuristic_total
        cue_scores["model"] = -1.0 if model_score is None else model_score
        cue_scores["total"] = total
        return {k: round(float(v), 3) for k, v in cue_scores.items()}
