"""
============================

Fix for 2 fps problem
----------------------
v6 ran detection + ArcFace embedding synchronously in the main thread.
ArcFace ONNX on CPU takes ~80-150 ms per frame → 2-4 fps display.

Solution: split into two separate ONNX sessions:
  Main thread       SCRFD detection only     ~15-25 ms  → ~33 fps display
  Inference thread  ArcFace embedding only   ~50-100 ms → ~8-12 fps embed

Two FaceAnalysis instances, each with allowed_modules restricted:
  _detector  → ["detection"]   (SCRFD, fast)
  _embedder  → ["recognition"] (ArcFace, slow, background thread)

The aligned face crop (norm_crop 112×112) is produced in the main thread
from SCRFD landmarks and pushed to the inference thread via the lock-free
deque(maxlen=1) + Event slot pattern from v5.

Architecture
------------
  Main thread
    ├─ cap.read()
    ├─ SCRFD detect → bbox + 5 landmarks
    ├─ norm_crop(112×112) from landmarks
    ├─ push crop to _frame_slot (deque maxlen=1, drop-stale)
    ├─ consume _result_slot if ready → AnchorVerifier.feed()
    └─ render display frame + diagnostic overlay

  Inference thread (daemon)
    ├─ wait on _frame_ready Event
    ├─ pop crop from _frame_slot
    ├─ ArcFace.get_feat() → 512-dim embedding
    └─ write to _result_slot[0], set _result_ready

IPC pattern (from v5, unchanged)
    _frame_slot   deque(maxlen=1)   drop-stale, GIL-safe append/pop
    _result_slot  list[1]           atomic GIL-safe assignment
    _frame_ready  threading.Event   producer signals consumer
    _result_ready threading.Event   consumer signals main thread
"""

import math
import threading
import time
import traceback
import warnings
from collections import deque
from datetime import datetime
from enum import IntEnum
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from insightface.utils import face_align


# ── Config ─────────────────────────────────────────────────────────────────
SCRFD_NAME           = "buffalo_l"
SCRFD_PROVIDERS      = ["CPUExecutionProvider"]
SCRFD_DET_SIZE       = (224, 224)
SCRFD_DET_THRESH     = 0.55
MIN_FACE_AREA        = 60 * 60
REQUIRE_SINGLE_FACE  = True
ALIGNED_SIZE         = 112          # ArcFace canonical input

UPDATE_INTERVAL_MS   = 30           # display ~33 fps
INFERENCE_INTERVAL   = 0.20         # max embed rate (s); ArcFace ~80ms so ~8 fps

ANCHOR_BUILD_FRAMES  = 20
# ArcFace cosine distance empirical ranges:
#   same person  → 0.05 – 0.30
#   diff person  → 0.45 – 0.80
THRESHOLD_SAME       = 0.28
THRESHOLD_DIFF       = 0.45

VOTE_WINDOW          = 7
VOTE_QUORUM          = 0.72

DISPLAY_W, DISPLAY_H = 680, 480
DIAG_HISTORY         = 80
DIAG_H, DIAG_W       = 110, 300

# Pre-computed
_VOTE_MIN  = max(1, int(VOTE_WINDOW * VOTE_QUORUM))
_INV_ABF   = 1.0 / ANCHOR_BUILD_FRAMES
_DIST_SCALE = 1.0 / (THRESHOLD_DIFF * 1.8)

# ── IPC slots ───────────────────────────────────────────────────────────────
_frame_slot:   deque = deque(maxlen=1)
_frame_ready:  threading.Event = threading.Event()
_result_slot:  list = [None]
_result_ready: threading.Event = threading.Event()


# ── Verdict ─────────────────────────────────────────────────────────────────
class V(IntEnum):
    BUILDING = 0
    SAME     = 1
    ALERT    = 2

_V_BGR    = ((30, 140, 200), (94, 197, 34), (68, 68, 239))
_V_FG_HEX = ("#f97316", "#22c55e", "#ef4444")
_V_BG_HEX = ("#0d111a", "#071a0f", "#1a0707")


class Mode(IntEnum):
    LIVE     = 0
    CAL_SAME = 1
    CAL_DIFF = 2
    CAL_DONE = 3


# ═══════════════════════════════════════════════════════════════════════════
#  Math
# ═══════════════════════════════════════════════════════════════════════════

def _norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).ravel()
    v /= math.sqrt(float(np.dot(v, v))) + 1e-9
    return v

def _cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.dot(a, b))

def ts() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


# ═══════════════════════════════════════════════════════════════════════════
#  SCRFD detector — main thread only, detection module only
# ═══════════════════════════════════════════════════════════════════════════

class SCRFDDetector:
    def __init__(self):
        self._app = FaceAnalysis(
            name=SCRFD_NAME,
            allowed_modules=["detection"],
            providers=SCRFD_PROVIDERS,
        )
        self._app.prepare(ctx_id=-1, det_size=SCRFD_DET_SIZE,
                          det_thresh=SCRFD_DET_THRESH)

    def detect(self, frame_bgr: np.ndarray):
        """
        Returns (aligned_112, box, all_boxes, status).
        aligned_112 : norm_crop 112×112 BGR ready for ArcFace, or None
        box         : (x1,y1,x2,y2) primary face in frame coords, or None
        all_boxes   : all detected boxes (for drawing)
        status      : "OK" | "NO_FACE" | "MULTI_FACE" | "NO_KPS"
        """
        faces  = self._app.get(frame_bgr)
        fh, fw = frame_bgr.shape[:2]
        valid  = []

        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(fw - 1, x2), min(fh - 1, y2)
            area   = max(0, x2 - x1) * max(0, y2 - y1)
            if area < MIN_FACE_AREA or getattr(face, "kps", None) is None:
                continue
            valid.append((face, (x1, y1, x2, y2), area))

        if not valid:
            return None, None, [], "NO_FACE"

        valid.sort(key=lambda t: t[2], reverse=True)
        all_boxes = [t[1] for t in valid]

        if REQUIRE_SINGLE_FACE and len(valid) > 1:
            return None, None, all_boxes, "MULTI_FACE"

        best_face, best_box, _ = valid[0]
        aligned = face_align.norm_crop(
            frame_bgr,
            best_face.kps.astype(np.float32),
            image_size=ALIGNED_SIZE,
        )
        return aligned, best_box, all_boxes, "OK"


# ═══════════════════════════════════════════════════════════════════════════
#  ArcFace embedder — inference thread only, recognition module only
# ═══════════════════════════════════════════════════════════════════════════

class ArcFaceEmbedder:
    def __init__(self):
        # IMPORTANT:
        # FaceAnalysis cannot be used with allowed_modules=["recognition"] only.
        # Its constructor asserts that a detection model exists, so it can fail
        # with a blank AssertionError right after finding w600k_r50.onnx.
        # Load the ArcFace ONNX model directly instead.
        rec_path = Path.home() / ".insightface" / "models" / SCRFD_NAME / "w600k_r50.onnx"
        if not rec_path.exists():
            raise FileNotFoundError(
                f"ArcFace model not found: {rec_path}. "
                f"Run the detector once or delete ~/.insightface/models/{SCRFD_NAME} "
                "so InsightFace can re-download the model pack."
            )

        self._rec = get_model(str(rec_path), providers=SCRFD_PROVIDERS)
        if self._rec is None or not hasattr(self._rec, "get_feat"):
            raise RuntimeError(f"Invalid ArcFace recognition model: {rec_path}")

        self._rec.prepare(ctx_id=-1)

    def embed(self, aligned_bgr: np.ndarray) -> np.ndarray:
        """aligned 112×112 BGR → unit-norm float32 (512,)"""
        if aligned_bgr is None:
            raise ValueError("ArcFace received None crop.")
        if aligned_bgr.shape[:2] != (ALIGNED_SIZE, ALIGNED_SIZE):
            aligned_bgr = cv2.resize(aligned_bgr, (ALIGNED_SIZE, ALIGNED_SIZE))

        feat = self._rec.get_feat(aligned_bgr)
        return _norm(np.asarray(feat, dtype=np.float32).ravel())


# ═══════════════════════════════════════════════════════════════════════════
#  Inference worker — daemon thread
# ═══════════════════════════════════════════════════════════════════════════

def _inference_worker(embedder: ArcFaceEmbedder, stop: threading.Event):
    while not stop.is_set():
        if not _frame_ready.wait(timeout=0.5):
            continue
        _frame_ready.clear()

        try:
            crop = _frame_slot[-1]
        except IndexError:
            continue

        try:
            emb = embedder.embed(crop)
            _result_slot[0] = emb
            _result_ready.set()
        except Exception:
            print("\n[ArcFace worker error]")
            traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════════════
#  Anchor verifier — O(1) circular vote buffer (unchanged from v5/v6)
# ═══════════════════════════════════════════════════════════════════════════

class AnchorVerifier:
    __slots__ = ("_acc", "_acc_ptr", "_anchor",
                 "_votes", "_vote_ptr", "_vote_sum", "_vote_count",
                 "_state", "building")

    def __init__(self):
        self.reset()

    def reset(self):
        self._acc       = np.zeros((ANCHOR_BUILD_FRAMES, 512), dtype=np.float32)
        self._acc_ptr   = 0
        self._anchor    = None
        self._votes     = np.zeros(VOTE_WINDOW, dtype=np.int8)
        self._vote_ptr  = 0
        self._vote_sum  = 0
        self._vote_count = 0
        self._state     = V.SAME
        self.building   = True

    @property
    def anchor_locked(self) -> bool:
        return self._anchor is not None

    @property
    def anchor_progress(self) -> int:
        return self._acc_ptr

    def vote_state(self) -> np.ndarray:
        out = np.full(VOTE_WINDOW, -1, dtype=np.int8)
        n   = self._vote_count
        for i in range(n):
            idx    = (self._vote_ptr - n + i) % VOTE_WINDOW
            out[i] = int(self._votes[idx])
        return out

    def feed(self, emb: np.ndarray):
        if not self.anchor_locked:
            self._acc[self._acc_ptr] = emb
            self._acc_ptr += 1
            if self._acc_ptr >= ANCHOR_BUILD_FRAMES:
                self._anchor = _norm(self._acc.mean(axis=0))
                self.building = False
            return V.BUILDING, 0.0

        dist = _cosine_dist(emb, self._anchor)

        if dist < THRESHOLD_SAME:
            new_vote = 1
        elif dist > THRESHOLD_DIFF:
            new_vote = 0
        else:
            return self._state, dist   # dead-band

        old = int(self._votes[self._vote_ptr])
        self._votes[self._vote_ptr] = new_vote
        self._vote_ptr   = (self._vote_ptr + 1) % VOTE_WINDOW
        self._vote_sum  += new_vote - old
        self._vote_count = min(self._vote_count + 1, VOTE_WINDOW)

        if self._vote_count >= _VOTE_MIN:
            r = self._vote_sum / self._vote_count
            if r >= VOTE_QUORUM:
                self._state = V.SAME
            elif r <= 1.0 - VOTE_QUORUM:
                self._state = V.ALERT

        return self._state, dist


# ═══════════════════════════════════════════════════════════════════════════
#  Canvas bar widget
# ═══════════════════════════════════════════════════════════════════════════

class BarWidget:
    __slots__ = ("canvas", "_rect", "_last_color")

    def __init__(self, parent, height=14, **pack_kw):
        self.canvas = tk.Canvas(parent, height=height, bg="#1e293b",
                                bd=0, highlightthickness=0)
        self.canvas.pack(**pack_kw)
        self._rect       = self.canvas.create_rectangle(0, 0, 0, height,
                                                        fill="#22c55e", outline="")
        self._last_color = "#22c55e"

    def update(self, frac: float, color: str):
        self.canvas.update_idletasks()
        w = self.canvas.winfo_width() or 200
        h = self.canvas.winfo_height() or 14
        if color != self._last_color:
            self.canvas.itemconfigure(self._rect, fill=color)
            self._last_color = color
        self.canvas.coords(self._rect, 0, 0,
                           int(w * max(0.0, min(1.0, frac))), h)


# ═══════════════════════════════════════════════════════════════════════════
#  Diagnostic overlay
# ═══════════════════════════════════════════════════════════════════════════

_PLOT_MARGIN = 8

def _draw_diag(frame: np.ndarray, history: deque, votes: np.ndarray,
               verdict: V, dist: float) -> None:
    fh, fw   = frame.shape[:2]
    ph, pw   = DIAG_H, DIAG_W
    x0, y0   = 8, fh - ph - 8
    inner_h  = ph - 2 * _PLOT_MARGIN
    inner_w  = pw - 2 * _PLOT_MARGIN
    px0, py0 = x0 + _PLOT_MARGIN, y0 + _PLOT_MARGIN
    y_max    = THRESHOLD_DIFF * 1.5

    # Semi-transparent background
    roi = frame[y0:y0+ph, x0:x0+pw]
    cv2.addWeighted(roi, 0.3, np.zeros_like(roi), 0.7, 0, roi)
    frame[y0:y0+ph, x0:x0+pw] = roi

    def _y(d):
        return py0 + inner_h - int(min(d, y_max) / y_max * inner_h)

    y_same = _y(THRESHOLD_SAME)
    y_diff = _y(THRESHOLD_DIFF)

    # Coloured zones
    cv2.rectangle(frame, (px0, y_same), (px0+inner_w, py0+inner_h), (20, 60, 20), -1)
    cv2.rectangle(frame, (px0, y_diff), (px0+inner_w, y_same),       (20, 50, 70), -1)
    cv2.rectangle(frame, (px0, py0),    (px0+inner_w, y_diff),       (40, 15, 50), -1)

    # Threshold lines
    cv2.line(frame, (px0, y_same), (px0+inner_w, y_same), (34, 197, 94), 1)
    cv2.line(frame, (px0, y_diff), (px0+inner_w, y_diff), (68, 68, 239), 1)

    # Rolling curve
    pts  = list(history)
    n    = len(pts)
    if n > 1:
        step = inner_w / (DIAG_HISTORY - 1)
        for i in range(1, n):
            col = (_V_BGR[V.SAME]  if pts[i] < THRESHOLD_SAME else
                   _V_BGR[V.ALERT] if pts[i] > THRESHOLD_DIFF else
                   _V_BGR[V.BUILDING])
            cv2.line(frame,
                     (px0 + int((i-1)*step), _y(pts[i-1])),
                     (px0 + int(i*step),     _y(pts[i])),
                     col, 2)

    # Current dot
    if pts:
        cv2.circle(frame, (px0+inner_w, _y(pts[-1])), 4, _V_BGR[verdict], -1)

    # Vote squares
    sq, gap = 10, 3
    vx, vy  = px0, y0 + ph - sq - 4
    for v in votes:
        col = ((80,80,80) if v < 0 else
               (34,197,94) if v == 1 else (68,68,239))
        cv2.rectangle(frame, (vx, vy), (vx+sq, vy+sq), col, -1)
        vx += sq + gap

    # Labels
    cv2.putText(frame, f"d={dist:.3f}", (px0, y0+13),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180,180,180), 1)
    cv2.putText(frame, f"S<{THRESHOLD_SAME}", (px0+inner_w-55, y_same-2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, (34,197,94), 1)
    cv2.putText(frame, f"D>{THRESHOLD_DIFF}", (px0+inner_w-55, y_diff-2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, (68,68,239), 1)


# ═══════════════════════════════════════════════════════════════════════════
#  Application
# ═══════════════════════════════════════════════════════════════════════════

class AnchorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Perceptryx — ArcFace Anchor v7")
        self.root.geometry("980x680")
        self.root.configure(bg="#06080f")

        self.detector:  SCRFDDetector | None  = None
        self.verifier   = AnchorVerifier()
        self.cap        = cv2.VideoCapture(0)

        self._stop_evt  = threading.Event()
        self._inf_th:   threading.Thread | None = None

        self._verdict   = V.BUILDING
        self._prev_v    = V.BUILDING
        self._dist      = 0.0
        self._last_push = 0.0
        self._show_diag = True
        self._mode      = Mode.LIVE

        self._dist_hist: deque = deque([0.0]*DIAG_HISTORY, maxlen=DIAG_HISTORY)

        # Calibration
        self._cal_same:  list[float] = []
        self._cal_diff:  list[float] = []
        self._cal_prev:  np.ndarray | None = None
        _CAL_N = 40
        self._CAL_N = _CAL_N

        self.photo_ref = None
        self._build_ui()
        self.log("Loading SCRFD detector + ArcFace embedder...")
        self.root.after(150, self._load_models)
        self.root.after(UPDATE_INTERVAL_MS, self._update_loop)

    # ── UI ─────────────────────────────────────────────────────────────────

    def _build_ui(self):
        hdr = tk.Frame(self.root, bg="#06080f")
        hdr.pack(fill="x", padx=16, pady=10)
        tk.Label(hdr, text="PERCEPTRYX", fg="#f97316", bg="#06080f",
                 font=("Courier New", 24, "bold")).pack(side="left")
        tk.Label(hdr, text="SCRFD detect  //  ArcFace embed  //  v7",
                 fg="#334155", bg="#06080f",
                 font=("Courier New", 10)).pack(side="left", padx=10)
        self.status_var = tk.StringVar(value="loading...")
        tk.Label(hdr, textvariable=self.status_var, fg="#475569",
                 bg="#06080f", font=("Courier New", 10)).pack(side="right")

        body = tk.Frame(self.root, bg="#06080f")
        body.pack(fill="both", expand=True, padx=16, pady=4)

        left = tk.Frame(body, bg="#0d111a")
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))
        self.video_label = tk.Label(left, bg="#06080f")
        self.video_label.pack(fill="both", expand=True, padx=8, pady=8)
        self.verdict_var = tk.StringVar(value="")
        self.verdict_label = tk.Label(
            left, textvariable=self.verdict_var,
            font=("Courier New", 22, "bold"),
            bg="#0d111a", fg="#f8fafc", pady=6)
        self.verdict_label.pack(fill="x")

        right = tk.Frame(body, bg="#0d111a", width=255)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)
        p = dict(padx=14)

        tk.Label(right, text="Controls", fg="#94a3b8", bg="#0d111a",
                 font=("Courier New", 9, "bold")).pack(anchor="w", pady=(14,4), **p)
        tk.Button(right, text="RESET ANCHOR", command=self._reset_anchor,
                  bg="#f97316", fg="#06080f", relief="flat",
                  font=("Courier New", 10, "bold"), pady=7
                  ).pack(fill="x", pady=2, **p)
        self.diag_btn = tk.Button(right, text="DIAG: ON",
                                  command=self._toggle_diag,
                                  bg="#334155", fg="#f8fafc", relief="flat",
                                  font=("Courier New", 10, "bold"), pady=7)
        self.diag_btn.pack(fill="x", pady=2, **p)
        self.cal_btn = tk.Button(right, text="CALIBRATE",
                                 command=self._start_cal,
                                 bg="#7c3aed", fg="#f8fafc", relief="flat",
                                 font=("Courier New", 10, "bold"), pady=7)
        self.cal_btn.pack(fill="x", pady=2, **p)
        self.cal_next_btn = tk.Button(right, text="NEXT PERSON →",
                                      command=self._cal_next,
                                      bg="#1e293b", fg="#475569", relief="flat",
                                      font=("Courier New", 9, "bold"), pady=6,
                                      state="disabled")
        self.cal_next_btn.pack(fill="x", pady=2, **p)

        tk.Label(right, text="Anchor build", fg="#475569", bg="#0d111a",
                 font=("Courier New", 9)).pack(anchor="w", pady=(12,2), **p)
        self._anchor_bar = BarWidget(right, fill="x", pady=(0,4), padx=14)

        tk.Label(right, text="Distance", fg="#475569", bg="#0d111a",
                 font=("Courier New", 9)).pack(anchor="w", pady=(4,2), **p)
        self._dist_bar = BarWidget(right, fill="x", pady=(0,2), padx=14)
        self.dist_lbl = tk.Label(right, text="—", fg="#475569", bg="#0d111a",
                                 font=("Courier New", 10))
        self.dist_lbl.pack(anchor="w", **p)

        self.cal_info_lbl = tk.Label(right, text="", fg="#7c3aed", bg="#0d111a",
                                     font=("Courier New", 8), justify="left",
                                     wraplength=220)
        self.cal_info_lbl.pack(anchor="w", pady=(4,0), **p)

        info = (f"Model  buffalo_l  ArcFace w600k_r50\n"
                f"SAME  < {THRESHOLD_SAME:.2f}  (cosine)\n"
                f"DEAD   {THRESHOLD_SAME:.2f} – {THRESHOLD_DIFF:.2f}\n"
                f"DIFF  > {THRESHOLD_DIFF:.2f}\n"
                f"Vote  {VOTE_WINDOW} frames  {int(VOTE_QUORUM*100)}%\n"
                f"Anchor {ANCHOR_BUILD_FRAMES} frames\n"
                f"Det   SCRFD @ {SCRFD_DET_SIZE[0]}px  (main)\n"
                f"Emb   ArcFace 112px  (thread)")
        tk.Label(right, text=info, fg="#334155", bg="#0d111a",
                 font=("Courier New", 8), justify="left"
                 ).pack(anchor="w", pady=(10,0), **p)

        tk.Label(right, text="Perf", fg="#475569", bg="#0d111a",
                 font=("Courier New", 9)).pack(anchor="w", pady=(10,2), **p)
        self.perf_lbl = tk.Label(right, text="—", fg="#334155", bg="#0d111a",
                                 font=("Courier New", 10))
        self.perf_lbl.pack(anchor="w", **p)

        tk.Label(right, text="Status", fg="#475569", bg="#0d111a",
                 font=("Courier New", 9)).pack(anchor="w", pady=(6,2), **p)
        self.face_st_lbl = tk.Label(right, text="—", fg="#334155", bg="#0d111a",
                                    font=("Courier New", 10))
        self.face_st_lbl.pack(anchor="w", **p)

        tk.Label(right, text="Log", fg="#94a3b8", bg="#0d111a",
                 font=("Courier New", 9, "bold")).pack(anchor="w", pady=(10,4), **p)
        self.log_box = tk.Text(right, bg="#06080f", fg="#475569", relief="flat",
                               font=("Courier New", 8), height=8, wrap="word")
        self.log_box.pack(fill="both", expand=True, padx=14, pady=(0,10))

    # ── helpers ────────────────────────────────────────────────────────────

    def log(self, msg: str):
        line = f"[{ts()}] {msg}"
        print(line, flush=True)
        self.log_box.insert("1.0", line + "\n")
        if int(self.log_box.index("end-1c").split(".")[0]) > 120:
            self.log_box.delete("80.0", "end")

    def _toggle_diag(self):
        self._show_diag = not self._show_diag
        self.diag_btn.configure(text=f"DIAG: {'ON' if self._show_diag else 'OFF'}")

    # ── model loading ──────────────────────────────────────────────────────

    def _load_models(self):
        try:
            self.detector = SCRFDDetector()
            self.log(f"SCRFD ready @ {SCRFD_DET_SIZE[0]}px.")

            embedder = ArcFaceEmbedder()
            self.log("ArcFace ready.")

            self._inf_th = threading.Thread(
                target=_inference_worker,
                args=(embedder, self._stop_evt),
                daemon=True, name="arcface",
            )
            self._inf_th.start()

            self.status_var.set("ready — waiting for first face")
            self.log("Ready. First face → anchor.")
        except Exception as exc:
            self.status_var.set("load failed")
            self.log(f"ERROR: {type(exc).__name__}: {exc!r}")
            traceback.print_exc()
            messagebox.showerror("Load error", f"{type(exc).__name__}: {exc}")

    # ── anchor ─────────────────────────────────────────────────────────────

    def _reset_anchor(self):
        self.verifier.reset()
        self._verdict  = V.BUILDING
        self._prev_v   = V.BUILDING
        self._dist     = 0.0
        self._last_push = 0.0
        self._dist_hist = deque([0.0]*DIAG_HISTORY, maxlen=DIAG_HISTORY)
        _frame_ready.clear(); _result_ready.clear()
        _frame_slot.clear();  _result_slot[0] = None
        self.verdict_var.set("")
        self.verdict_label.configure(bg="#0d111a", fg="#f8fafc")
        self.status_var.set("anchor reset")
        self.log("Anchor reset.")

    # ── calibration ────────────────────────────────────────────────────────

    def _start_cal(self):
        self._mode = Mode.CAL_SAME
        self._cal_same.clear(); self._cal_diff.clear()
        self._cal_prev = None
        self.cal_btn.configure(text="STOP", command=self._stop_cal, bg="#ef4444")
        self.cal_next_btn.configure(state="disabled", bg="#1e293b", fg="#475569")
        self.cal_info_lbl.configure(text="Phase 1: same person — stay in frame")
        self.status_var.set("CAL phase 1: same person")
        self.log("Calibration start — phase 1")

    def _cal_next(self):
        if self._mode != Mode.CAL_SAME:
            return
        self._mode = Mode.CAL_DIFF
        self._cal_prev = None
        self.cal_next_btn.configure(state="disabled", bg="#1e293b", fg="#475569")
        self.cal_info_lbl.configure(text="Phase 2: different person — swap now")
        self.status_var.set("CAL phase 2: different person")
        self.log("Phase 2 — different person")

    def _stop_cal(self):
        self._mode = Mode.CAL_DONE
        self.cal_btn.configure(text="CALIBRATE", command=self._start_cal,
                               bg="#7c3aed")
        self._cal_report()

    def _cal_feed(self, emb: np.ndarray):
        prev = self._cal_prev
        if prev is not None:
            d = _cosine_dist(emb, prev)
            if self._mode == Mode.CAL_SAME:
                self._cal_same.append(d)
                n = len(self._cal_same)
                self.cal_info_lbl.configure(
                    text=f"Phase 1: {n}/{self._CAL_N} samples")
                if n >= self._CAL_N:
                    self.cal_next_btn.configure(
                        state="normal", bg="#22c55e", fg="#06080f")
                    self.log("Phase 1 done — press NEXT PERSON")
            elif self._mode == Mode.CAL_DIFF:
                self._cal_diff.append(d)
                n = len(self._cal_diff)
                self.cal_info_lbl.configure(
                    text=f"Phase 2: {n}/{self._CAL_N} samples")
                if n >= self._CAL_N:
                    self._stop_cal()
        self._cal_prev = emb

    def _cal_report(self):
        lines = ["── Calibration ──"]
        rec_same = rec_diff = None
        if self._cal_same:
            a = np.array(self._cal_same, dtype=np.float32)
            mu, sd = float(a.mean()), float(a.std())
            rec_same = round(mu + 2*sd, 3)
            lines.append(f"SAME μ={mu:.3f} σ={sd:.3f} → T_SAME={rec_same}")
        if self._cal_diff:
            a = np.array(self._cal_diff, dtype=np.float32)
            mu, sd = float(a.mean()), float(a.std())
            rec_diff = round(mu - 2*sd, 3)
            lines.append(f"DIFF μ={mu:.3f} σ={sd:.3f} → T_DIFF={rec_diff}")
        if rec_same and rec_diff:
            gap = rec_diff - rec_same
            lines.append(f"Gap = {gap:.3f} {'✓' if gap > 0.05 else '⚠ overlap!'}")
        for l in lines:
            self.log(l)
        self.cal_info_lbl.configure(text="\n".join(lines))

    # ── main loop ──────────────────────────────────────────────────────────

    def _update_loop(self):
        ok, frame = self.cap.read()

        if ok and self.detector is not None:
            frame = cv2.flip(frame, 1)

            t0 = time.perf_counter()
            aligned, box, all_boxes, status = self.detector.detect(frame)
            det_ms = (time.perf_counter() - t0) * 1000

            self.perf_lbl.configure(text=f"det {det_ms:.1f} ms")
            self.face_st_lbl.configure(text=status)

            # Draw secondary boxes
            for bx in all_boxes:
                if bx != box:
                    cv2.rectangle(frame, bx[:2], bx[2:], (50, 50, 50), 1)

            if aligned is not None:
                now = time.monotonic()
                if now - self._last_push >= INFERENCE_INTERVAL:
                    _frame_slot.append(aligned)
                    _frame_ready.set()
                    self._last_push = now

                # Consume latest embedding result
                if _result_ready.is_set():
                    _result_ready.clear()
                    emb = _result_slot[0]
                    if emb is not None:
                        if self._mode in (Mode.CAL_SAME, Mode.CAL_DIFF):
                            self._cal_feed(emb)
                        else:
                            v, d = self.verifier.feed(emb)
                            self._verdict = v
                            self._dist    = d
                            self._dist_hist.append(d)
                            self._refresh_sidebar()

                # Draw box + text
                color = _V_BGR[self._verdict]
                cv2.rectangle(frame, box[:2], box[2:], color, 2)

                if self._verdict == V.BUILDING:
                    n   = self.verifier.anchor_progress
                    txt = f"Building  {n}/{ANCHOR_BUILD_FRAMES}"
                elif self._verdict == V.SAME:
                    txt = f"OK  d={self._dist:.3f}"
                else:
                    txt = f"ALERT  d={self._dist:.3f}"

                cv2.putText(frame, txt, (box[0], max(box[1]-10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.68, color, 2)

            elif status == "MULTI_FACE":
                cv2.putText(frame, "MULTIPLE FACES — HOLDING", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (22, 140, 200), 2)
            else:
                cv2.putText(frame, "NO FACE", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.80, (55, 55, 55), 2)

            if self._show_diag and self._mode in (Mode.LIVE, Mode.CAL_DONE):
                _draw_diag(frame, self._dist_hist, self.verifier.vote_state(),
                           self._verdict, self._dist)

            self._render(frame)

        self.root.after(UPDATE_INTERVAL_MS, self._update_loop)

    # ── sidebar ────────────────────────────────────────────────────────────

    def _refresh_sidebar(self):
        v, d = self._verdict, self._dist

        self._anchor_bar.update(
            self.verifier.anchor_progress * _INV_ABF,
            "#f97316" if not self.verifier.anchor_locked else "#22c55e")

        dcolor = ("#22c55e" if d < THRESHOLD_SAME else
                  "#f97316" if d < THRESHOLD_DIFF else "#ef4444")
        self._dist_bar.update(d * _DIST_SCALE, dcolor)
        self.dist_lbl.configure(text=f"{d:.4f}")

        if v != self._prev_v:
            self.verdict_label.configure(fg=_V_FG_HEX[v], bg=_V_BG_HEX[v])
            self._prev_v = v

        if v == V.BUILDING:
            n = self.verifier.anchor_progress
            self.verdict_var.set(f"Building anchor  {n}/{ANCHOR_BUILD_FRAMES}")
            self.status_var.set("building anchor")
        elif v == V.SAME:
            self.verdict_var.set("OK  —  same person")
            self.status_var.set("SAME")
        else:
            self.verdict_var.set("ALERT  —  different person")
            self.status_var.set("ALERT")
            self.log(f"ALERT  d={d:.4f}")

    # ── render ─────────────────────────────────────────────────────────────

    def _render(self, frame_bgr: np.ndarray):
        small = cv2.resize(frame_bgr, (DISPLAY_W, DISPLAY_H),
                           interpolation=cv2.INTER_NEAREST)
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        img   = Image.frombuffer("RGB", (DISPLAY_W, DISPLAY_H),
                                 rgb.tobytes(), "raw", "RGB", 0, 1)
        self.photo_ref = ImageTk.PhotoImage(img)
        self.video_label.configure(image=self.photo_ref)

    # ── cleanup ────────────────────────────────────────────────────────────

    def on_close(self):
        self._stop_evt.set()
        _frame_ready.set()
        if self._inf_th:
            self._inf_th.join(timeout=2.0)
        if self.cap:
            self.cap.release()
        self.root.destroy()


# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    root = tk.Tk()
    app  = AnchorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
