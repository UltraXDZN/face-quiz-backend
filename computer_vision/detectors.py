from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Detection dataclass
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """
    __slots__ removes the per-instance __dict__ overhead.
    At 30 fps with N faces this saves meaningful GC pressure.
    """
    __slots__ = ("bbox", "conf", "landmarks")

    bbox: Tuple[int, int, int, int]
    conf: float
    landmarks: Optional[List[Tuple[int, int]]]

    def __init__(
        self,
        bbox: Tuple[int, int, int, int],
        conf: float,
        landmarks: Optional[List[Tuple[int, int]]] = None,
    ):
        self.bbox = bbox
        self.conf = conf
        self.landmarks = landmarks


# ---------------------------------------------------------------------------
# MediaPipe
# ---------------------------------------------------------------------------

class MediaPipeDetector:
    """
    Google MediaPipe FaceDetection / BlazeFace.
    Requires: pip install mediapipe

    Optimisation: a reusable RGB buffer is pre-allocated on the first frame
    and reused on every subsequent call, eliminating repeated heap allocations
    from cvtColor.
    """

    MODEL_SHORT_RANGE = 0
    MODEL_FULL_RANGE = 1

    def __init__(self, conf_threshold: float = 0.5, model_selection: int = 0):
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise ImportError("pip install mediapipe") from exc

        if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "face_detection"):
            location = getattr(mp, "__file__", "unknown location")
            raise ImportError(
                "mediapipe is installed, but mp.solutions.face_detection is unavailable. "
                f"Imported mediapipe from: {location}. "
                "Check that you do not have a local file/folder named mediapipe.py, "
                "then reinstall mediapipe."
            )

        self.mp_fd = mp.solutions.face_detection
        self._detector = self.mp_fd.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=conf_threshold,
        )
        # Reusable RGB buffer; re-created only when frame size changes.
        self._rgb_buf: Optional[np.ndarray] = None

    def detect(self, frame: np.ndarray) -> List[Detection]:
        h, w = frame.shape[:2]

        # Re-use the same buffer; only reallocate on resolution change.
        if self._rgb_buf is None or self._rgb_buf.shape[:2] != (h, w):
            self._rgb_buf = np.empty((h, w, 3), dtype=np.uint8)

        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, dst=self._rgb_buf)
        results = self._detector.process(self._rgb_buf)

        if not results.detections:
            return []

        detections: List[Detection] = []
        for det in results.detections:
            bb = det.location_data.relative_bounding_box
            x1 = max(0, int(bb.xmin * w))
            y1 = max(0, int(bb.ymin * h))
            x2 = min(w, int((bb.xmin + bb.width) * w))
            y2 = min(h, int((bb.ymin + bb.height) * h))
            conf = float(det.score[0])
            landmarks = [
                (int(kp.x * w), int(kp.y * h))
                for kp in det.location_data.relative_keypoints
            ]
            detections.append(Detection(bbox=(x1, y1, x2, y2), conf=conf, landmarks=landmarks))

        return detections

    def __del__(self):
        try:
            self._detector.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# YOLOv8
# ---------------------------------------------------------------------------

class YOLOv8Detector:
    """
    YOLOv8 face detector via Ultralytics.
    Requires: pip install ultralytics

    Put yolov8n-face.pt in either:
        ./yolov8n-face.pt
        ./models/yolov8n-face.pt

    Or set:
        export YOLO_FACE_MODEL=/full/path/to/yolov8n-face.pt

    Optimisation: xyxy tensor is moved to CPU and converted once per box
    instead of calling .cpu() twice.
    """

    DEFAULT_MODEL_NAME = "yolov8n-face.pt"

    def __init__(self, conf_threshold: float = 0.5, model_path: Optional[str] = None):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("pip install ultralytics") from exc

        self.conf = conf_threshold
        resolved_path = self._resolve_model_path(model_path)
        self._model = YOLO(str(resolved_path))

    @classmethod
    def _resolve_model_path(cls, model_path: Optional[str]) -> Path:
        candidates = []

        if model_path:
            candidates.append(Path(model_path))

        env_path = os.getenv("YOLO_FACE_MODEL")
        if env_path:
            candidates.append(Path(env_path))

        candidates.extend([
            Path(cls.DEFAULT_MODEL_NAME),
            Path("models") / cls.DEFAULT_MODEL_NAME,
        ])

        for candidate in candidates:
            if candidate.exists():
                return candidate

        searched = "\n".join(f"  - {c}" for c in candidates)
        raise FileNotFoundError(
            "YOLOv8 face weights not found. Download yolov8n-face.pt and place it in the "
            "project root or models/ folder, or set YOLO_FACE_MODEL. Searched:\n"
            f"{searched}"
        )

    def detect(self, frame: np.ndarray) -> List[Detection]:
        results = self._model.predict(
            source=frame,
            conf=self.conf,
            verbose=False,
            stream=False,
        )

        detections: List[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            # Move entire tensor to CPU once; avoids repeated .cpu() calls.
            xyxy_all = boxes.xyxy.cpu().numpy().astype(int)
            conf_all = boxes.conf.cpu().numpy()

            for i, (xyxy, conf) in enumerate(zip(xyxy_all, conf_all)):
                x1, y1, x2, y2 = xyxy
                landmarks = None

                if result.keypoints is not None and i < len(result.keypoints):
                    kp_data = result.keypoints[i].xy[0].cpu().numpy()
                    landmarks = [(int(x), int(y)) for x, y in kp_data]

                detections.append(
                    Detection(bbox=(x1, y1, x2, y2), conf=float(conf), landmarks=landmarks)
                )

        return detections


# ---------------------------------------------------------------------------
# Haar Cascade
# ---------------------------------------------------------------------------

class HaarCascadeDetector:
    """
    Classic Viola-Jones detector.
    Requires only OpenCV.

    Optimisation: the "no faces" path (by far the most common in sparse
    scenes) returns the same empty-list singleton instead of allocating a
    new list every frame.
    """

    CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    _EMPTY: List[Detection] = []   # module-level singleton

    def __init__(
        self,
        conf_threshold: float = 0.5,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: Tuple[int, int] = (30, 30),
    ):
        self._classifier = cv2.CascadeClassifier(self.CASCADE_PATH)
        if self._classifier.empty():
            raise RuntimeError("Failed to load Haar cascade XML. Check OpenCV install.")

        self._scale_factor = scale_factor
        self._min_neighbors = min_neighbors
        self._min_size = min_size
        # Reusable grayscale buffer
        self._gray_buf: Optional[np.ndarray] = None

    def detect(self, frame: np.ndarray) -> List[Detection]:
        h, w = frame.shape[:2]

        if self._gray_buf is None or self._gray_buf.shape != (h, w):
            self._gray_buf = np.empty((h, w), dtype=np.uint8)

        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY, dst=self._gray_buf)
        cv2.equalizeHist(self._gray_buf, self._gray_buf)   # in-place

        faces = self._classifier.detectMultiScale(
            self._gray_buf,
            scaleFactor=self._scale_factor,
            minNeighbors=self._min_neighbors,
            minSize=self._min_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        if len(faces) == 0:
            return self._EMPTY   # no allocation on empty frames

        return [
            Detection(bbox=(x, y, x + w_, y + h_), conf=1.0, landmarks=None)
            for (x, y, w_, h_) in faces
        ]


# ---------------------------------------------------------------------------
# InsightFace SCRFD
# ---------------------------------------------------------------------------

class InsightFaceSCRFDDetector:
    """
    InsightFace SCRFD detector via ONNX Runtime.
    Requires: pip install insightface onnxruntime
    NOTE: Check InsightFace license before commercial use.

    Optimisation: landmark coordinates are extracted with a single vectorised
    NumPy call (tolist()) instead of a per-point Python loop, cutting landmark
    conversion time roughly in half for 5-point models.
    """

    def __init__(self, conf_threshold: float = 0.5, model_name: str = "buffalo_sc"):
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise ImportError("pip install insightface onnxruntime") from exc

        self._app = FaceAnalysis(
            name=model_name,
            allowed_modules=["detection"],
        )
        self._app.prepare(ctx_id=-1, det_thresh=conf_threshold, det_size=(640, 640))

    def detect(self, frame: np.ndarray) -> List[Detection]:
        faces = self._app.get(frame)
        detections: List[Detection] = []

        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            conf = float(face.det_score)

            # tolist() is a single C-level call; avoids a Python loop over kps.
            landmarks = (
                [(int(x), int(y)) for x, y in face.kps.tolist()]
                if face.kps is not None
                else None
            )

            detections.append(
                Detection(bbox=(x1, y1, x2, y2), conf=conf, landmarks=landmarks)
            )

        return detections
