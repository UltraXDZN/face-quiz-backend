"""
playground.py  (optimised)
==========================
"""

from __future__ import annotations

import argparse
import queue
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Deque, List, Optional

import cv2

from detectors import Detection, InsightFaceSCRFDDetector
from overlay import draw_overlay


MODEL_NAME   = "InsightFace SCRFD"
CONTROL_HINT = "s  Screenshot\nq  Quit"

# Sentinel used to stop the save worker thread
_STOP_SENTINEL = None


# ---------------------------------------------------------------------------
# Background image-save worker
# ---------------------------------------------------------------------------

class ImageSaveWorker:
    """
    Encodes and writes JPEG frames in a dedicated daemon thread so the main
    loop is never blocked by disk I/O or JPEG compression.

    Queue items: (path, frame, quality) tuples.
    The worker exits when it receives _STOP_SENTINEL.
    """

    def __init__(self, maxsize: int = 8):
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, path: Path, frame, quality: int = 85) -> None:
        """Non-blocking; drops frame silently if the queue is full."""
        try:
            self._q.put_nowait((path, frame, quality))
        except queue.Full:
            pass   # prefer dropping a frame over blocking the capture loop

    def stop(self, timeout: float = 2.0) -> None:
        self._q.put(_STOP_SENTINEL)
        self._thread.join(timeout=timeout)

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is _STOP_SENTINEL:
                break
            path, frame, quality = item
            try:
                cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            except Exception as exc:
                print(f"[!] Save error ({path}): {exc}")


# ---------------------------------------------------------------------------
# Frame grabber
# ---------------------------------------------------------------------------

class FrameGrabber:
    """Runs cap.read() in a background thread; exposes the latest frame."""

    def __init__(self, source):
        self._cap   = cv2.VideoCapture(source)
        self._lock  = threading.Lock()
        self._frame: Optional[object] = None
        self._ok    = False
        self._stop  = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def configure(self, width: int, height: int) -> None:
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    def open(self) -> bool:
        if not self._cap.isOpened():
            return False
        self._thread.start()
        return True

    def read(self):
        with self._lock:
            return self._ok, None if self._frame is None else self._frame.copy()

    def release(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._cap.release()

    def _run(self) -> None:
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            with self._lock:
                self._ok    = ok
                self._frame = frame
            if not ok:
                break


# ---------------------------------------------------------------------------
# Tracker helpers
# ---------------------------------------------------------------------------

def _find_tracker_ctor():
    """
    Probe once at startup rather than on every MultiTracker.reset() call.
    Returns a callable or None.
    """
    legacy = getattr(cv2, "legacy", None)
    if legacy is None:
        return None
    for attr in ("TrackerMOSSE_create", "TrackerKCF_create", "TrackerCSRT_create"):
        ctor = getattr(legacy, attr, None)
        if ctor is not None:
            return ctor
    return None


# Resolved once at module import time.
_TRACKER_CTOR = _find_tracker_ctor()


class MultiTracker:
    """
    Thin wrapper: re-initialise from SCRFD boxes, update on skipped frames.

    Returns List[Detection] (not raw lists) so the rest of the pipeline
    always works with a single concrete type.
    """

    def __init__(self):
        self._trackers = []

    def reset(self, frame, detections: List[Detection]) -> None:
        self._trackers = []

        if _TRACKER_CTOR is None:
            return

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            bw, bh = x2 - x1, y2 - y1
            if bw <= 0 or bh <= 0:
                continue
            tracker = _TRACKER_CTOR()
            tracker.init(frame, (x1, y1, bw, bh))
            self._trackers.append(tracker)

    def update(self, frame) -> List[Detection]:
        """Return List[Detection] with conf=0.0 (tracker-estimated)."""
        results: List[Detection] = []
        alive = []

        for tracker in self._trackers:
            ok, bbox = tracker.update(frame)
            if not ok:
                continue
            x, y, bw, bh = (int(v) for v in bbox)
            results.append(Detection(bbox=(x, y, x + bw, y + bh), conf=0.0))
            alive.append(tracker)

        self._trackers = alive
        return results


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="SCRFD face detection playground")

    p.add_argument("--source",  default=0,
                   help="Video source: 0=webcam or path to video file")
    p.add_argument("--save",    action="store_true",
                   help="Record output to <run>/recording.mp4")
    p.add_argument("--width",   type=int,   default=1280)
    p.add_argument("--height",  type=int,   default=720)
    p.add_argument("--conf",    type=float, default=0.5,
                   help="SCRFD confidence threshold")
    p.add_argument("--infer-size", type=int, default=0,
                   help="Downscale inference input to this width (0 = full res). "
                        "E.g. 320 or 480 for a large speed boost.")
    p.add_argument("--skip",    type=int,   default=1,
                   help="Run SCRFD every N frames; use tracker on the rest. "
                        "1 = every frame (no skip).")
    p.add_argument("--store-every", type=int, default=30,
                   help="Save one inference frame every N frames (0 = disable).")
    p.add_argument("--detect-multiple", action="store_true",
                   help="Enable warning when multiple faces are detected.")
    p.add_argument("--multi-face-min",  type=int, default=2,
                   help="Minimum faces to trigger multi-face warning.")

    return p.parse_args()


def normalize_source(src):
    try:
        return int(src)
    except (ValueError, TypeError):
        return src


def create_run_folder() -> Path:
    ts      = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    run_dir = Path("output") / "runs" / ts
    for label in ("human", "empty", "multiple"):
        (run_dir / "scrfd" / label).mkdir(parents=True, exist_ok=True)
    print(f"[+] Run folder: {run_dir}")
    return run_dir


# ---------------------------------------------------------------------------
# Face-count helpers (unchanged logic, consolidated here)
# ---------------------------------------------------------------------------

def count_faces(detections: List[Detection]) -> int:
    return len(detections)


def has_multiple_faces(detections: List[Detection], min_faces: int) -> bool:
    return len(detections) >= min_faces


# ---------------------------------------------------------------------------
# Frame saving (delegates to the background worker)
# ---------------------------------------------------------------------------

def enqueue_inference_image(
    worker: ImageSaveWorker,
    run_dir: Path,
    frame,
    detections: List[Detection],
    frame_idx: int,
    detect_multiple: bool = False,
    multi_face_min: int = 2,
) -> None:
    face_count = count_faces(detections)

    if detect_multiple and face_count >= multi_face_min:
        label = "multiple"
    elif face_count > 0:
        label = "human"
    else:
        label = "empty"

    ts       = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    filename = f"{ts}_scrfd_{frame_idx:06d}.jpg"
    path     = run_dir / "scrfd" / label / filename

    worker.submit(path, frame, quality=85)


# ---------------------------------------------------------------------------
# Multi-face warning banner
# ---------------------------------------------------------------------------

def draw_multi_face_warning(vis, face_count: int):
    text       = f"MULTIPLE FACES DETECTED: {face_count}"
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.85
    thickness  = 2
    margin     = 12

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

def main():
    args = parse_args()
    args.skip           = max(1, args.skip)
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

    source  = normalize_source(args.source)
    grabber = FrameGrabber(source)
    grabber.configure(args.width, args.height)

    if not grabber.open():
        print(f"[!] Cannot open video source: {source}")
        return

    # Start the background save worker
    save_worker = ImageSaveWorker(maxsize=8)

    print("[+] Waiting for first frame ...")
    first_frame = None
    deadline    = time.time() + 5.0

    while time.time() < deadline:
        ok, frame = grabber.read()
        if ok and frame is not None:
            first_frame = frame
            break
        time.sleep(0.02)

    if first_frame is None:
        print("[!] Timed out waiting for first frame. Check camera/source.")
        grabber.release()
        save_worker.stop()
        return

    fh, fw = first_frame.shape[:2]
    print(f"[+] Camera: {fw}x{fh} — stream live.")

    if args.detect_multiple:
        print(f"[+] Multi-face detection enabled. Trigger: >= {args.multi_face_min} faces.")

    writer = None
    if args.save:
        fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
        rec_path = run_dir / "recording.mp4"
        writer   = cv2.VideoWriter(str(rec_path), fourcc, 30, (fw, fh))
        print(f"[+] Recording -> {rec_path}")

    if args.infer_size > 0 and args.infer_size < fw:
        scale   = args.infer_size / fw
        infer_w = args.infer_size
        infer_h = int(fh * scale)
        inv_scale = 1.0 / scale
        print(f"[+] Inference input scaled to {infer_w}x{infer_h}")
    else:
        scale     = 1.0
        infer_w   = fw
        infer_h   = fh
        inv_scale = 1.0

    tracker = MultiTracker()

    # ---- O(1) FPS buffer using deque with a fixed max length ---------------
    fps_buf: Deque[float] = deque(maxlen=30)
    prev_time     = time.perf_counter()
    screenshot_idx = 0
    frame_idx      = 0
    detections: List[Detection] = []
    inf_ms         = 0.0
    multi_face_active = False

    print(f"\n[Controls]\n{CONTROL_HINT}\n")

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

            # Scale detections back to display resolution — build Detection
            # objects directly; no intermediate list of raw floats.
            if scale < 1.0 and raw_dets:
                detections = [
                    Detection(
                        bbox=(
                            int(d.bbox[0] * inv_scale),
                            int(d.bbox[1] * inv_scale),
                            int(d.bbox[2] * inv_scale),
                            int(d.bbox[3] * inv_scale),
                        ),
                        conf=d.conf,
                        landmarks=(
                            [(int(x * inv_scale), int(y * inv_scale)) for x, y in d.landmarks]
                            if d.landmarks else None
                        ),
                    )
                    for d in raw_dets
                ]
            else:
                detections = raw_dets

            if args.skip > 1:
                tracker.reset(frame, detections)

        else:
            detections = tracker.update(frame)

        face_count      = count_faces(detections)
        multiple_faces  = (
            args.detect_multiple
            and has_multiple_faces(detections, args.multi_face_min)
        )

        if args.detect_multiple and multiple_faces != multi_face_active:
            multi_face_active = multiple_faces
            if multi_face_active:
                print(f"[!] Multiple faces detected: {face_count} faces at frame {frame_idx}")
            else:
                print(f"[+] Multiple-face condition cleared at frame {frame_idx}")

        # Offload save to background thread — no longer blocks the loop.
        if args.store_every > 0 and frame_idx % args.store_every == 0:
            enqueue_inference_image(
                save_worker,
                run_dir,
                frame,
                detections,
                frame_idx,
                detect_multiple=args.detect_multiple,
                multi_face_min=args.multi_face_min,
            )

        now = time.perf_counter()
        fps_buf.append(1.0 / max(now - prev_time, 1e-6))   # O(1) append
        prev_time = now
        fps = sum(fps_buf) / len(fps_buf)

        vis = draw_overlay(
            frame,
            detections,
            MODEL_NAME,
            fps,
            inf_ms,
            controls_hint="s: Screenshot  q: Quit",
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

    # ------- Cleanup -------------------------------------------------------
    grabber.release()
    save_worker.stop()

    if writer:
        writer.release()

    cv2.destroyAllWindows()
    print("[+] Done.")


if __name__ == "__main__":
    main()
