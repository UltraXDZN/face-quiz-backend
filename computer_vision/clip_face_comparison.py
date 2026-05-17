"""
Anchor Mode — InsightFace SCRFD + CLIP
=================================================

Pipeline
--------
  Main thread:
    InsightFace SCRFD detects face bbox + 5 landmarks.
    The largest valid face is aligned with InsightFace norm_crop.
    The aligned face crop is pushed to the inference thread.

  Inference thread:
    CLIP embeds the aligned face crop.
    AnchorVerifier compares it against the locked anchor centroid.

Why this version exists
-----------------------
  No MediaPipe.
  No Haar cascades.
  Face detection and alignment are handled by InsightFace SCRFD.
"""

import queue
import threading
import time
import warnings
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import torch
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from transformers import CLIPProcessor, CLIPModel

from insightface.app import FaceAnalysis
from insightface.utils import face_align


# Model
MODEL_NAME = "openai/clip-vit-base-patch32"
DIM = 512
USE_INT8 = True

# InsightFace / SCRFD
SCRFD_NAME = "buffalo_l"
SCRFD_PROVIDERS = ["CPUExecutionProvider"]
SCRFD_DET_SIZE = (640, 640)
SCRFD_DET_THRESH = 0.55
MIN_FACE_AREA = 70 * 70
REQUIRE_SINGLE_FACE = True
ALIGNED_FACE_SIZE = 224

# Timing
UPDATE_INTERVAL_MS = 30
INFERENCE_INTERVAL = 0.40

# Anchor
ANCHOR_BUILD_FRAMES = 15

# Thresholds for L2 distance on normalized CLIP embeddings
THRESHOLD_SAME = 0.38
THRESHOLD_DIFF = 0.52

# Voting
VOTE_WINDOW = 6
VOTE_QUORUM = 0.75

# Display
DISPLAY_W, DISPLAY_H = 680, 480

# Queues
_frame_q = queue.Queue(maxsize=1)
_result_q = queue.Queue(maxsize=8)


warnings.filterwarnings(
    "ignore",
    message=".*torch.ao.quantization is deprecated.*",
    category=DeprecationWarning,
)


def normalize(v):
    v = np.asarray(v, dtype=np.float32).flatten()
    return v / (np.linalg.norm(v) + 1e-9)


def ts():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


class CLIPEmbedder:
    def __init__(self):
        self.processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        model = CLIPModel.from_pretrained(MODEL_NAME).eval()

        if USE_INT8:
            self.model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
        else:
            self.model = model

    def embed(self, face_bgr):
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=face_rgb, return_tensors="pt")

        with torch.no_grad():
            out = self.model.vision_model(pixel_values=inputs["pixel_values"])
            emb = self.model.visual_projection(out.pooler_output)

        return normalize(emb.squeeze(0).cpu().numpy())


class InsightFaceDetector:
    def __init__(self):
        self.app = FaceAnalysis(
            name=SCRFD_NAME,
            allowed_modules=["detection"],
            providers=SCRFD_PROVIDERS,
        )
        self.app.prepare(
            ctx_id=-1,
            det_size=SCRFD_DET_SIZE,
            det_thresh=SCRFD_DET_THRESH,
        )

    def detect(self, frame_bgr):
        faces = self.app.get(frame_bgr)
        valid = []

        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame_bgr.shape[1] - 1, x2)
            y2 = min(frame_bgr.shape[0] - 1, y2)

            area = max(0, x2 - x1) * max(0, y2 - y1)
            if area < MIN_FACE_AREA:
                continue

            if getattr(face, "kps", None) is None:
                continue

            valid.append({
                "face": face,
                "box": (x1, y1, x2, y2),
                "area": area,
                "score": float(getattr(face, "det_score", 0.0)),
            })

        if not valid:
            return None, None, [], "NO_FACE"

        valid.sort(key=lambda item: item["area"], reverse=True)
        all_boxes = [item["box"] for item in valid]

        if REQUIRE_SINGLE_FACE and len(valid) > 1:
            return None, None, all_boxes, "MULTI_FACE"

        best = valid[0]
        kps = best["face"].kps.astype(np.float32)

        aligned = face_align.norm_crop(
            frame_bgr,
            kps,
            image_size=ALIGNED_FACE_SIZE,
        )

        return aligned, best["box"], all_boxes, "OK"


class AnchorVerifier:
    def __init__(self):
        self.reset()

    def reset(self):
        self._anchor_embs = []
        self._anchor = None
        self._vote_buf = deque(maxlen=VOTE_WINDOW)
        self._state = "SAME"
        self.building = True

    @property
    def anchor_locked(self):
        return self._anchor is not None

    def anchor_progress(self):
        return len(self._anchor_embs)

    def feed(self, emb):
        if not self.anchor_locked:
            self._anchor_embs.append(emb)
            if len(self._anchor_embs) >= ANCHOR_BUILD_FRAMES:
                self._anchor = normalize(np.mean(self._anchor_embs, axis=0))
                self.building = False
            return "BUILDING", 0.0

        dist = float(np.linalg.norm(emb - self._anchor))

        if dist < THRESHOLD_SAME:
            self._vote_buf.append(True)
        elif dist > THRESHOLD_DIFF:
            self._vote_buf.append(False)

        min_votes = max(1, int(VOTE_WINDOW * VOTE_QUORUM))
        if len(self._vote_buf) >= min_votes:
            ratio = sum(self._vote_buf) / len(self._vote_buf)
            if ratio >= VOTE_QUORUM:
                self._state = "SAME"
            elif ratio <= (1.0 - VOTE_QUORUM):
                self._state = "ALERT"

        return self._state, dist


def _inference_worker(embedder, stop_event):
    while not stop_event.is_set():
        try:
            face_bgr = _frame_q.get(timeout=0.5)
        except queue.Empty:
            continue

        try:
            emb = embedder.embed(face_bgr)

            if _result_q.full():
                try:
                    _result_q.get_nowait()
                except queue.Empty:
                    pass

            _result_q.put_nowait(emb)
        except Exception:
            pass


class AnchorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Anchor Verification SCRFD")
        self.root.geometry("980x660")
        self.root.configure(bg="#06080f")

        self.verifier = AnchorVerifier()
        self.detector = None
        self.cap = cv2.VideoCapture(0)

        self._stop_evt = threading.Event()
        self._inf_thread = None

        self._verdict = "BUILDING"
        self._dist = 0.0
        self._last_push = 0.0
        self._face_status = "LOADING"
        self._last_boxes = []

        self.photo_ref = None

        self._build_ui()
        self.log("Loading CLIP + InsightFace SCRFD...")
        self.root.after(120, self._load_models)
        self.root.after(UPDATE_INTERVAL_MS, self._update_loop)

    def _build_ui(self):
        hdr = tk.Frame(self.root, bg="#06080f")
        hdr.pack(fill="x", padx=16, pady=10)

        tk.Label(
            hdr,
            fg="#f97316",
            bg="#06080f",
            font=("Courier New", 24, "bold"),
        ).pack(side="left")

        tk.Label(
            hdr,
            text="anchor verification  //  InsightFace SCRFD + CLIP INT8",
            fg="#334155",
            bg="#06080f",
            font=("Courier New", 10),
        ).pack(side="left", padx=10)

        self.status_var = tk.StringVar(value="loading...")
        tk.Label(
            hdr,
            textvariable=self.status_var,
            fg="#475569",
            bg="#06080f",
            font=("Courier New", 10),
        ).pack(side="right")

        body = tk.Frame(self.root, bg="#06080f")
        body.pack(fill="both", expand=True, padx=16, pady=4)

        left = tk.Frame(body, bg="#0d111a")
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))

        self.video_label = tk.Label(left, bg="#06080f")
        self.video_label.pack(fill="both", expand=True, padx=8, pady=8)

        self.verdict_var = tk.StringVar(value="")
        self.verdict_label = tk.Label(
            left,
            textvariable=self.verdict_var,
            font=("Courier New", 22, "bold"),
            bg="#0d111a",
            fg="#f8fafc",
            pady=6,
        )
        self.verdict_label.pack(fill="x")

        right = tk.Frame(body, bg="#0d111a", width=250)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)
        p = dict(padx=14)

        tk.Label(
            right,
            text="Controls",
            fg="#94a3b8",
            bg="#0d111a",
            font=("Courier New", 9, "bold"),
        ).pack(anchor="w", pady=(14, 4), **p)

        tk.Button(
            right,
            text="RESET ANCHOR",
            command=self._reset_anchor,
            bg="#f97316",
            fg="#06080f",
            relief="flat",
            font=("Courier New", 11, "bold"),
            pady=8,
        ).pack(fill="x", pady=3, **p)

        tk.Label(
            right,
            text="Anchor build",
            fg="#475569",
            bg="#0d111a",
            font=("Courier New", 9),
        ).pack(anchor="w", pady=(14, 2), **p)

        self.progress_cv = tk.Canvas(
            right,
            height=16,
            bg="#1e293b",
            bd=0,
            highlightthickness=0,
        )
        self.progress_cv.pack(fill="x", pady=(0, 6), **p)

        tk.Label(
            right,
            text="Distance to anchor",
            fg="#475569",
            bg="#0d111a",
            font=("Courier New", 9),
        ).pack(anchor="w", pady=(6, 2), **p)

        self.dist_cv = tk.Canvas(
            right,
            height=16,
            bg="#1e293b",
            bd=0,
            highlightthickness=0,
        )
        self.dist_cv.pack(fill="x", pady=(0, 4), **p)

        self.dist_lbl = tk.Label(
            right,
            text="—",
            fg="#475569",
            bg="#0d111a",
            font=("Courier New", 10),
        )
        self.dist_lbl.pack(anchor="w", **p)

        info = (
            f"SAME  < {THRESHOLD_SAME:.2f}\n"
            f"DEAD   {THRESHOLD_SAME:.2f} - {THRESHOLD_DIFF:.2f}\n"
            f"DIFF  > {THRESHOLD_DIFF:.2f}\n"
            f"Vote : {VOTE_WINDOW} frames  quorum {int(VOTE_QUORUM * 100)}%\n"
            f"Anchor: {ANCHOR_BUILD_FRAMES} frames\n"
            f"Detector: InsightFace SCRFD\n"
            f"Align: 5-point norm_crop\n"
            f"Single face: {REQUIRE_SINGLE_FACE}"
        )
        tk.Label(
            right,
            text=info,
            fg="#334155",
            bg="#0d111a",
            font=("Courier New", 8),
            justify="left",
        ).pack(anchor="w", pady=(12, 0), **p)

        tk.Label(
            right,
            text="Perf",
            fg="#475569",
            bg="#0d111a",
            font=("Courier New", 9),
        ).pack(anchor="w", pady=(14, 2), **p)

        self.perf_lbl = tk.Label(
            right,
            text="— ms",
            fg="#334155",
            bg="#0d111a",
            font=("Courier New", 10),
        )
        self.perf_lbl.pack(anchor="w", **p)

        tk.Label(
            right,
            text="Face status",
            fg="#475569",
            bg="#0d111a",
            font=("Courier New", 9),
        ).pack(anchor="w", pady=(10, 2), **p)

        self.face_status_lbl = tk.Label(
            right,
            text="—",
            fg="#334155",
            bg="#0d111a",
            font=("Courier New", 10),
        )
        self.face_status_lbl.pack(anchor="w", **p)

        tk.Label(
            right,
            text="Log",
            fg="#94a3b8",
            bg="#0d111a",
            font=("Courier New", 9, "bold"),
        ).pack(anchor="w", pady=(16, 4), **p)

        self.log_box = tk.Text(
            right,
            bg="#06080f",
            fg="#475569",
            relief="flat",
            font=("Courier New", 8),
            height=10,
            wrap="word",
        )
        self.log_box.pack(fill="both", expand=True, padx=14, pady=(0, 12))

    def log(self, msg):
        self.log_box.insert("1.0", f"[{ts()}] {msg}\n")
        self.log_box.see("1.0")

    def _bar(self, canvas, frac, color):
        canvas.update_idletasks()
        w = canvas.winfo_width() or 200
        h = canvas.winfo_height() or 16
        canvas.delete("all")
        canvas.create_rectangle(
            0,
            0,
            int(w * min(1.0, max(0.0, frac))),
            h,
            fill=color,
            outline="",
        )

    def _load_models(self):
        try:
            self.detector = InsightFaceDetector()
            embedder = CLIPEmbedder()

            self.log("InsightFace SCRFD ready.")
            self.log("CLIP loaded.")

            self._inf_thread = threading.Thread(
                target=_inference_worker,
                args=(embedder, self._stop_evt),
                daemon=True,
                name="inference",
            )
            self._inf_thread.start()

            self.status_var.set("ready — waiting for first face")
            self.log("Ready. Anchor will be built from first valid face seen.")
        except Exception as exc:
            self.status_var.set("load failed")
            self.log(f"ERROR: {exc}")
            messagebox.showerror("Load error", str(exc))

    def _reset_anchor(self):
        self.verifier.reset()
        self._verdict = "BUILDING"
        self._dist = 0.0
        self._last_push = 0.0

        for q in (_frame_q, _result_q):
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

        self.verdict_var.set("")
        self.verdict_label.configure(bg="#0d111a", fg="#f8fafc")
        self.status_var.set("anchor reset — waiting for first face")
        self.log("Anchor reset.")

    def _update_loop(self):
        ok, frame = self.cap.read()

        if ok and self.detector is not None:
            frame = cv2.flip(frame, 1)
            display = frame.copy()

            t_det = time.perf_counter()
            face_bgr, box, all_boxes, face_status = self.detector.detect(frame)
            det_ms = (time.perf_counter() - t_det) * 1000

            self._face_status = face_status
            self._last_boxes = all_boxes
            self.perf_lbl.configure(text=f"det {det_ms:.1f} ms")
            self.face_status_lbl.configure(text=face_status)

            self._draw_boxes(display, all_boxes, box, face_status)

            if face_bgr is not None:
                now = time.time()
                if now - self._last_push >= INFERENCE_INTERVAL:
                    try:
                        _frame_q.get_nowait()
                    except queue.Empty:
                        pass

                    try:
                        _frame_q.put_nowait(face_bgr.copy())
                        self._last_push = now
                    except queue.Full:
                        pass

                got = False
                while not _result_q.empty():
                    try:
                        emb = _result_q.get_nowait()
                        self._verdict, self._dist = self.verifier.feed(emb)
                        got = True
                    except queue.Empty:
                        break

                if got:
                    self._update_sidebar()

                self._draw_verdict(display, box)
            else:
                if face_status == "MULTI_FACE":
                    cv2.putText(
                        display,
                        "MULTIPLE FACES - HOLDING STATE",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (249, 115, 22),
                        2,
                    )
                    self.status_var.set("multiple faces — paused")
                else:
                    cv2.putText(
                        display,
                        "NO FACE",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.85,
                        (60, 60, 60),
                        2,
                    )
                    self.status_var.set("no face")

            self._render(display)

        self.root.after(UPDATE_INTERVAL_MS, self._update_loop)

    def _draw_boxes(self, display, all_boxes, active_box, face_status):
        for b in all_boxes:
            x1, y1, x2, y2 = b
            if active_box is not None and b == active_box:
                continue
            cv2.rectangle(display, (x1, y1), (x2, y2), (100, 116, 139), 1)

        if active_box is None:
            return

        x1, y1, x2, y2 = active_box
        color = {
            "BUILDING": (200, 140, 30),
            "SAME": (34, 197, 94),
            "ALERT": (239, 68, 68),
        }.get(self._verdict, (80, 80, 80))

        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

    def _draw_verdict(self, display, box):
        if box is None:
            return

        x1, y1, _, _ = box
        color = {
            "BUILDING": (200, 140, 30),
            "SAME": (34, 197, 94),
            "ALERT": (239, 68, 68),
        }.get(self._verdict, (80, 80, 80))

        if self._verdict == "BUILDING":
            n = self.verifier.anchor_progress()
            text = f"Building anchor {n}/{ANCHOR_BUILD_FRAMES}"
        elif self._verdict == "SAME":
            text = f"OK d={self._dist:.3f}"
        else:
            text = f"ALERT d={self._dist:.3f}"

        cv2.putText(
            display,
            text,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            color,
            2,
        )

    def _update_sidebar(self):
        v = self._verdict
        d = self._dist

        frac = min(1.0, self.verifier.anchor_progress() / ANCHOR_BUILD_FRAMES)
        self._bar(
            self.progress_cv,
            frac,
            "#f97316" if not self.verifier.anchor_locked else "#22c55e",
        )

        max_d = THRESHOLD_DIFF * 1.8
        dcolor = (
            "#22c55e"
            if d < THRESHOLD_SAME
            else "#f97316"
            if d < THRESHOLD_DIFF
            else "#ef4444"
        )
        self._bar(self.dist_cv, d / max_d, dcolor)
        self.dist_lbl.configure(text=f"{d:.4f}")

        if v == "BUILDING":
            n = self.verifier.anchor_progress()
            self.verdict_var.set(f"Building anchor... {n}/{ANCHOR_BUILD_FRAMES}")
            self.verdict_label.configure(fg="#f97316", bg="#0d111a")
            self.status_var.set("building anchor")
        elif v == "SAME":
            self.verdict_var.set("OK — same person")
            self.verdict_label.configure(fg="#22c55e", bg="#071a0f")
            self.status_var.set("SAME")
        else:
            self.verdict_var.set("ALERT — different person")
            self.verdict_label.configure(fg="#ef4444", bg="#1a0707")
            self.status_var.set("ALERT")
            self.log(f"ALERT d={d:.4f}")

    def _render(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize((DISPLAY_W, DISPLAY_H), Image.NEAREST)
        self.photo_ref = ImageTk.PhotoImage(img)
        self.video_label.configure(image=self.photo_ref)

    def on_close(self):
        self._stop_evt.set()
        if self.cap:
            self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = AnchorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
