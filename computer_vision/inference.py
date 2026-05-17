"""
playground.py  (optimised v3)
==============================
SCRFD face detection playground with passive liveness / anti-spoofing.

Existing features preserved:
* threaded camera capture
* SCRFD detection
* optional inference downscale
* optional tracking on skipped frames
* periodic frame saving into human/empty/multiple folders
* optional video recording
* multi-face warning
* screenshots
* overlay/HUD

New liveness path:
* optional ONNX anti-spoof model as primary passive FAS signal
* old heuristic liveness remains fallback/support signal
* per-track smoothing to avoid flickering LIVE/SPOOF decisions
"""

from __future__ import annotations

import argparse
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2

from detectors import Detection, InsightFaceSCRFDDetector
from liveness import LivenessChecker
from overlay import draw_overlay

MODEL_NAME = "InsightFace SCRFD"
CONTROL_HINT = "s  Screenshot\nq  Quit"


# ---------------------------------------------------------------------------
# FrameGrabber
# ---------------------------------------------------------------------------

class FrameGrabber:
    """Runs cap.read() in a background thread; exposes the latest frame."""

    def __init__(self, source):
        self._cap = cv2.VideoCapture(source)
        self._lock = threading.Lock()
        self._frame = None
        self._ok = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def configure(self, width: int, height: int) -> None:
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def open(self) -> bool:
        if not self._cap.isOpened():
            return False
        self._thread.start()
        return True

    def read(self):
        with self._lock:
            return self._ok, self._frame

    def release(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._cap.release()

    def _run(self) -> None:
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            with self._lock:
                self._ok = ok
                self._frame = frame
            if not ok:
                break


# ---------------------------------------------------------------------------
# Tracker helpers
# ---------------------------------------------------------------------------

def _make_tracker():
    legacy = getattr(cv2, "legacy", None)
    if legacy is None:
        return None
    for attr in ("TrackerMOSSE_create", "TrackerKCF_create", "TrackerCSRT_create"):
        ctor = getattr(legacy, attr, None)
        if ctor is not None:
            return ctor()
    return None


class MultiTracker:
    """Thin wrapper: re-initialise from SCRFD boxes, update on skipped frames."""

    def __init__(self):
        self._trackers: list = []

    def reset(self, frame, detections: List[Detection]) -> None:
        self._trackers = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0:
                continue
            tracker = _make_tracker()
            if tracker is None:
                continue
            tracker.init(frame, (int(x1), int(y1), int(bw), int(bh)))
            self._trackers.append(tracker)

    def update(self, frame) -> List[Detection]:
        results, alive = [], []
        for tracker in self._trackers:
            ok, bbox = tracker.update(frame)
            if not ok:
                continue
            x, y, bw, bh = (int(v) for v in bbox)
            results.append(Detection((x, y, x + bw, y + bh), 0.0, None))
            alive.append(tracker)
        self._trackers = alive
        return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SCRFD face detection playground")
    p.add_argument("--source", default=0, help="Video source: 0=webcam or path to video file")
    p.add_argument("--save", action="store_true", help="Record output to <run>/recording.mp4")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--conf", type=float, default=0.5, help="SCRFD confidence threshold")
    p.add_argument(
        "--infer-size", type=int, default=0,
        help="Downscale inference input to this width (0=full res). E.g. 320 or 480.",
    )
    p.add_argument(
        "--skip", type=int, default=1,
        help="Run SCRFD every N frames; use tracker on the rest. 1=every frame.",
    )
    p.add_argument(
        "--store-every", type=int, default=30,
        help="Save one inference frame every N frames (0=disable).",
    )
    p.add_argument("--detect-multiple", action="store_true", help="Enable warning when multiple faces are detected.")
    p.add_argument("--multi-face-min", type=int, default=2, help="Minimum faces to trigger multi-face warning.")

    p.add_argument("--no-liveness", action="store_true", help="Disable liveness check entirely.")
    p.add_argument(
        "--liveness-cooldown", type=int, default=12,
        help="Frames between full liveness re-evaluations per face track.",
    )
    p.add_argument(
        "--liveness-threshold", type=float, default=None,
        help="Final smoothed liveness threshold. Default: 0.62 with ONNX, 0.42 heuristic-only.",
    )
    p.add_argument("--debug-liveness", action="store_true", help="Print per-cue liveness scores.")
    p.add_argument("--antispoof-model", type=str, default=None, help="Path to ONNX anti-spoofing model.")
    p.add_argument("--liveness-input-size", type=int, default=128, help="Fallback ONNX input size for dynamic models.")
    p.add_argument(
        "--liveness-live-index", type=int, default=1,
        help="Output class index treated as LIVE/REAL. Silent-Face uses 1.",
    )
    p.add_argument(
        "--liveness-crop-scale", type=float, default=2.7,
        help="Face crop expansion for ONNX anti-spoofing. MiniFAS-style models often use ~2.7.",
    )
    p.add_argument(
        "--liveness-input-scale", type=float, default=1.0 / 255.0,
        help="Pixel multiplier before ONNX inference. Default converts 0-255 to 0-1.",
    )
    p.add_argument(
        "--liveness-min-samples", type=int, default=3,
        help="Number of evaluated samples before LIVE/SPOOF is emitted.",
    )
    p.add_argument(
        "--liveness-ema-alpha", type=float, default=0.45,
        help="EMA smoothing alpha for per-track liveness score.",
    )
    p.add_argument(
        "--liveness-uncertain-margin", type=float, default=0.07,
        help="Dead-zone around threshold where status is CHECK/uncertain.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_source(src):
    try:
        return int(src)
    except (ValueError, TypeError):
        return src


def create_run_folder() -> Path:
    ts = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    run_dir = Path("output") / "runs" / ts
    for label in ("human", "empty", "multiple"):
        (run_dir / "scrfd" / label).mkdir(parents=True, exist_ok=True)
    print(f"[+] Run folder: {run_dir}")
    return run_dir


def save_inference_image(
    run_dir: Path,
    frame,
    detections: List[Detection],
    frame_idx: int,
    detect_multiple: bool = False,
    multi_face_min: int = 2,
) -> None:
    face_count = len(detections) if detections else 0
    if detect_multiple and face_count >= multi_face_min:
        label = "multiple"
    elif face_count > 0:
        label = "human"
    else:
        label = "empty"
    ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    filename = f"{ts}_scrfd_{frame_idx:06d}.jpg"
    cv2.imwrite(
        str(run_dir / "scrfd" / label / filename),
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, 85],
    )


def draw_multi_face_warning(vis, face_count: int):
    text = f"MULTIPLE FACES DETECTED: {face_count}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.85
    thickness = 2
    margin = 12

    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = 20, 45

    cv2.rectangle(
        vis,
        (x - margin, y - th - margin),
        (x + tw + margin, y + baseline + margin),
        (0, 0, 255),
        -1,
    )
    cv2.putText(vis, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return vis


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    args.skip = max(1, args.skip)
    args.multi_face_min = max(2, args.multi_face_min)

    Path("output").mkdir(exist_ok=True)
    run_dir = create_run_folder()

    print("[+] Loading InsightFace SCRFD ...")
    try:
        detector = InsightFaceSCRFDDetector(conf_threshold=args.conf)
        print("[+] SCRFD ready.")
    except Exception as exc:
        print(f"[!] Failed to load SCRFD: {exc}")
        return

    liveness_checker: Optional[LivenessChecker] = None
    if not args.no_liveness:
        threshold = args.liveness_threshold
        if threshold is None:
            threshold = 0.62 if args.antispoof_model else 0.42

        try:
            liveness_checker = LivenessChecker(
                cooldown_frames=args.liveness_cooldown,
                score_threshold=threshold,
                debug=args.debug_liveness,
                onnx_model_path=args.antispoof_model,
                onnx_input_size=args.liveness_input_size,
                onnx_live_index=args.liveness_live_index,
                onnx_crop_scale=args.liveness_crop_scale,
                onnx_input_scale=args.liveness_input_scale,
                ema_alpha=args.liveness_ema_alpha,
                min_samples=args.liveness_min_samples,
                uncertain_margin=args.liveness_uncertain_margin,
            )
            debug_tag = " [DEBUG]" if args.debug_liveness else ""
            print(
                f"[+] Liveness enabled: backend={liveness_checker.backend_name}, "
                f"cooldown={args.liveness_cooldown}, threshold={threshold:.2f}, "
                f"min_samples={args.liveness_min_samples}{debug_tag}."
            )
        except Exception as exc:
            print(f"[!] Failed to initialise ONNX liveness; falling back to heuristic-only. Reason: {exc}")
            liveness_checker = LivenessChecker(
                cooldown_frames=args.liveness_cooldown,
                score_threshold=0.42 if args.liveness_threshold is None else args.liveness_threshold,
                debug=args.debug_liveness,
            )
            print("[+] Liveness enabled: backend=heuristic fallback.")
    else:
        print("[+] Liveness disabled (--no-liveness).")

    source = normalize_source(args.source)
    grabber = FrameGrabber(source)
    grabber.configure(args.width, args.height)

    if not grabber.open():
        print(f"[!] Cannot open video source: {source}")
        return

    print("[+] Waiting for first frame ...")
    first_frame = None
    deadline = time.monotonic() + 5.0

    while time.monotonic() < deadline:
        ok, frame = grabber.read()
        if ok and frame is not None:
            first_frame = frame
            break
        time.sleep(0.02)

    if first_frame is None:
        print("[!] Timed out waiting for first frame.")
        grabber.release()
        return

    fh, fw = first_frame.shape[:2]
    print(f"[+] Camera: {fw}x{fh} — stream live.")

    if args.detect_multiple:
        print(f"[+] Multi-face detection enabled. Trigger: >= {args.multi_face_min} faces.")

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        rec_path = run_dir / "recording.mp4"
        writer = cv2.VideoWriter(str(rec_path), fourcc, 30, (fw, fh))
        print(f"[+] Recording -> {rec_path}")

    if args.infer_size > 0 and args.infer_size < fw:
        scale = args.infer_size / fw
        infer_w = args.infer_size
        infer_h = int(fh * scale)
        print(f"[+] Inference input scaled to {infer_w}x{infer_h}")
    else:
        scale = 1.0
        infer_w = fw
        infer_h = fh

    tracker = MultiTracker()
    fps_buf = deque(maxlen=30)
    prev_time = time.monotonic()
    screenshot_idx = 0
    frame_idx = 0
    detections: List[Detection] = []
    inf_ms = 0.0
    multi_face_active = False
    liveness_results: Dict[int, object] = {}

    print(f"\n[Controls]\n{CONTROL_HINT}\n")

    try:
        while True:
            ok, frame = grabber.read()
            if not ok or frame is None:
                print("[!] Frame grab failed — end of stream or camera error.")
                break

            frame_idx += 1
            run_scrfd = args.skip <= 1 or frame_idx % args.skip == 1

            if run_scrfd:
                infer_frame = (
                    cv2.resize(frame, (infer_w, infer_h), interpolation=cv2.INTER_LINEAR)
                    if scale < 1.0
                    else frame
                )

                t0 = time.perf_counter()
                try:
                    raw_dets = detector.detect(infer_frame)
                except Exception as exc:
                    print(f"[!] Inference error: {exc}")
                    raw_dets = []
                inf_ms = (time.perf_counter() - t0) * 1000

                if scale < 1.0 and raw_dets:
                    inv = 1.0 / scale
                    detections = [det.scaled(inv, inv).clipped(fw, fh) for det in raw_dets]
                else:
                    detections = [det.clipped(fw, fh) for det in raw_dets]

                if args.skip > 1:
                    tracker.reset(frame, detections)
            else:
                detections = tracker.update(frame)

            face_count = len(detections)

            if liveness_checker is not None and face_count > 0:
                liveness_results = liveness_checker.update(frame, detections, frame_idx, return_details=True)
            elif face_count == 0:
                liveness_results = {}

            multiple_faces = args.detect_multiple and face_count >= args.multi_face_min
            if args.detect_multiple and multiple_faces != multi_face_active:
                multi_face_active = multiple_faces
                if multi_face_active:
                    print(f"[!] Multiple faces detected: {face_count} faces at frame {frame_idx}")
                else:
                    print(f"[+] Multiple-face condition cleared at frame {frame_idx}")

            if args.store_every > 0 and frame_idx % args.store_every == 0:
                save_inference_image(
                    run_dir,
                    frame,
                    detections,
                    frame_idx,
                    detect_multiple=args.detect_multiple,
                    multi_face_min=args.multi_face_min,
                )

            now = time.monotonic()
            fps_buf.append(1.0 / max(now - prev_time, 1e-6))
            prev_time = now
            fps = sum(fps_buf) / len(fps_buf)

            vis = draw_overlay(
                frame,
                detections,
                MODEL_NAME,
                fps,
                inf_ms,
                controls_hint="s: Screenshot  q: Quit",
                liveness_results=liveness_results if liveness_checker else None,
            )

            if multiple_faces:
                vis = draw_multi_face_warning(vis, face_count)

            cv2.imshow("SCRFD Face Detection", vis)

            if writer:
                writer.write(vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                path = run_dir / f"screenshot_{screenshot_idx:04d}.jpg"
                cv2.imwrite(str(path), vis, [cv2.IMWRITE_JPEG_QUALITY, 95])
                print(f"[+] Screenshot saved: {path}")
                screenshot_idx += 1
    finally:
        grabber.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print("[+] Done.")


if __name__ == "__main__":
    main()
