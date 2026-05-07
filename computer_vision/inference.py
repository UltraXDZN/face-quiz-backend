import argparse
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2

from detectors import InsightFaceSCRFDDetector
from overlay import draw_overlay

MODEL_NAME = "InsightFace SCRFD"
CONTROL_HINT = "s  Screenshot\nq  Quit"


class FrameGrabber:
    """Runs cap.read() in a background thread; exposes the latest frame."""

    def __init__(self, source):
        self._cap = cv2.VideoCapture(source)
        self._lock = threading.Lock()
        self._frame = None
        self._ok = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def configure(self, width, height):
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def open(self):
        if not self._cap.isOpened():
            return False
        self._thread.start()
        return True

    def read(self):
        with self._lock:
            return self._ok, None if self._frame is None else self._frame.copy()

    def release(self):
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._cap.release()

    def _run(self):
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            with self._lock:
                self._ok = ok
                self._frame = frame
            if not ok:
                break


def _make_tracker():
    """Return the fastest available OpenCV tracker."""
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
        self._trackers = []

    def reset(self, frame, detections):
        self._trackers = []

        for det in detections:
            x1, y1, x2, y2 = (int(v) for v in det[:4])
            w, h = x2 - x1, y2 - y1

            if w <= 0 or h <= 0:
                continue

            tracker = _make_tracker()
            if tracker is None:
                continue

            tracker.init(frame, (x1, y1, w, h))
            self._trackers.append(tracker)

    def update(self, frame):
        """Return list of [x1, y1, x2, y2, 0.0] from trackers."""
        results = []
        alive = []

        for tracker in self._trackers:
            ok, bbox = tracker.update(frame)
            if not ok:
                continue

            x, y, w, h = (int(v) for v in bbox)
            results.append([x, y, x + w, y + h, 0.0])
            alive.append(tracker)

        self._trackers = alive
        return results


def parse_args():
    p = argparse.ArgumentParser(description="SCRFD face detection playground")

    p.add_argument(
        "--source",
        default=0,
        help="Video source: 0=webcam or path to video file",
    )
    p.add_argument(
        "--save",
        action="store_true",
        help="Record output to <run>/recording.mp4",
    )
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="SCRFD confidence threshold",
    )
    p.add_argument(
        "--infer-size",
        type=int,
        default=0,
        help=(
            "Downscale inference input to this width (0 = full res). "
            "E.g. 320 or 480 for a large speed boost."
        ),
    )
    p.add_argument(
        "--skip",
        type=int,
        default=1,
        help=(
            "Run SCRFD every N frames; use tracker on the rest. "
            "1 = every frame (no skip)."
        ),
    )
    p.add_argument(
        "--store-every",
        type=int,
        default=30,
        help="Save one inference frame every N frames (0 = disable).",
    )
    p.add_argument(
        "--detect-multiple",
        action="store_true",
        help="Enable warning when multiple faces are detected in the frame.",
    )
    p.add_argument(
        "--multi-face-min",
        type=int,
        default=2,
        help="Minimum number of faces required to trigger multi-face warning.",
    )

    return p.parse_args()


def normalize_source(src):
    try:
        return int(src)
    except (ValueError, TypeError):
        return src


def create_run_folder():
    ts = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    run_dir = Path("output") / "runs" / ts

    for label in ("human", "empty", "multiple"):
        (run_dir / "scrfd" / label).mkdir(parents=True, exist_ok=True)

    print(f"[+] Run folder: {run_dir}")
    return run_dir


def count_faces(detections):
    if detections is None:
        return 0
    return len(detections)


def has_multiple_faces(detections, min_faces):
    return count_faces(detections) >= min_faces


def save_inference_image(
    run_dir,
    frame,
    detections,
    frame_idx,
    detect_multiple=False,
    multi_face_min=2,
):
    face_count = count_faces(detections)

    if detect_multiple and face_count >= multi_face_min:
        label = "multiple"
    elif face_count > 0:
        label = "human"
    else:
        label = "empty"

    ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    filename = f"{ts}_scrfd_{frame_idx:06d}.jpg"
    path = run_dir / "scrfd" / label / filename

    cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])


def draw_multi_face_warning(vis, face_count):
    text = f"MULTIPLE FACES DETECTED: {face_count}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.85
    thickness = 2
    margin = 12

    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    x = 20
    y = 45

    cv2.rectangle(
        vis,
        (x - margin, y - th - margin),
        (x + tw + margin, y + baseline + margin),
        (0, 0, 255),
        -1,
    )
    cv2.putText(
        vis,
        text,
        (x, y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

    return vis


def main():
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

    source = normalize_source(args.source)
    grabber = FrameGrabber(source)
    grabber.configure(args.width, args.height)

    if not grabber.open():
        print(f"[!] Cannot open video source: {source}")
        return

    print("[+] Waiting for first frame ...")
    first_frame = None
    deadline = time.time() + 5.0

    while time.time() < deadline:
        ok, frame = grabber.read()
        if ok and frame is not None:
            first_frame = frame
            break
        time.sleep(0.02)

    if first_frame is None:
        print("[!] Timed out waiting for first frame. Check camera/source.")
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
        infer_w, infer_h = fw, fh

    tracker = MultiTracker()

    fps_buf: list[float] = []
    prev_time = time.perf_counter()
    screenshot_idx = 0
    frame_idx = 0
    detections: list = []
    inf_ms = 0.0
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

            if scale < 1.0 and len(raw_dets) > 0:
                inv = 1.0 / scale
                detections = [
                    [det[0] * inv, det[1] * inv, det[2] * inv, det[3] * inv, *det[4:]]
                    for det in raw_dets
                ]
            else:
                detections = raw_dets

            if args.skip > 1:
                tracker.reset(frame, detections)
        else:
            detections = tracker.update(frame)

        face_count = count_faces(detections)
        multiple_faces = (
            args.detect_multiple
            and has_multiple_faces(detections, args.multi_face_min)
        )

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

        now = time.perf_counter()
        fps_buf.append(1.0 / max(now - prev_time, 1e-6))

        if len(fps_buf) > 30:
            fps_buf.pop(0)

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

    grabber.release()

    if writer:
        writer.release()

    cv2.destroyAllWindows()
    print("[+] Done.")


if __name__ == "__main__":
    main()
