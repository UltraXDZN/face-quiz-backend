"""
detectors.py  (optimised v3)
============================
Unified interface for supported face detectors.

Each detector exposes:
    detect(frame: np.ndarray) -> List[Detection]

Detection fields:
    bbox      : (x1, y1, x2, y2) in pixel coords
    conf      : float 0-1
    landmarks : list of (x, y) tuples or None

v3 additions
------------
* Detection stays object-based, but now also behaves like the old xyxy list
  where needed: det[0], det[:4], len(det), iter(det).
* Detection.scaled() makes inference-downscale restoration clean.
* Existing detector classes and public API are preserved.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np


BBox = Tuple[int, int, int, int]
Landmarks = Optional[List[Tuple[int, int]]]


class Detection:
    """
    Slot-based detection object.

    It also supports list-like access for legacy code:
        det[:4] -> (x1, y1, x2, y2)
        det[4]  -> conf
        list(det) -> [x1, y1, x2, y2, conf]
    """

    __slots__ = ("bbox", "conf", "landmarks")

    def __init__(
        self,
        bbox: BBox,
        conf: float,
        landmarks: Landmarks = None,
    ):
        x1, y1, x2, y2 = bbox
        self.bbox = (int(x1), int(y1), int(x2), int(y2))
        self.conf = float(conf)
        self.landmarks = landmarks

    def as_tuple(self) -> Tuple[int, int, int, int, float]:
        x1, y1, x2, y2 = self.bbox
        return x1, y1, x2, y2, self.conf

    def as_list(self) -> List[Union[int, float]]:
        return list(self.as_tuple())

    def scaled(self, sx: float, sy: Optional[float] = None) -> "Detection":
        if sy is None:
            sy = sx
        x1, y1, x2, y2 = self.bbox
        landmarks = None
        if self.landmarks is not None:
            landmarks = [(int(x * sx), int(y * sy)) for x, y in self.landmarks]
        return Detection(
            bbox=(int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)),
            conf=self.conf,
            landmarks=landmarks,
        )

    def clipped(self, width: int, height: int) -> "Detection":
        x1, y1, x2, y2 = self.bbox
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))
        return Detection((x1, y1, x2, y2), self.conf, self.landmarks)

    def __iter__(self) -> Iterator[Union[int, float]]:
        return iter(self.as_tuple())

    def __len__(self) -> int:
        return 5

    def __getitem__(self, item):
        return self.as_tuple()[item]

    def __repr__(self) -> str:
        return f"Detection(bbox={self.bbox}, conf={self.conf:.3f}, landmarks={self.landmarks is not None})"


# ---------------------------------------------------------------------------
# MediaPipe
# ---------------------------------------------------------------------------

class MediaPipeDetector:
    """Google MediaPipe FaceDetection / BlazeFace."""

    MODEL_SHORT_RANGE = 0
    MODEL_FULL_RANGE  = 1

    def __init__(self, conf_threshold: float = 0.5, model_selection: int = 0):
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise ImportError("pip install mediapipe") from exc

        if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "face_detection"):
            location = getattr(mp, "__file__", "unknown location")
            raise ImportError(
                "mediapipe is installed but mp.solutions.face_detection is unavailable. "
                f"Imported mediapipe from: {location}. "
                "Ensure no local file/folder shadows the package, then reinstall."
            )

        self.mp_fd      = mp.solutions.face_detection
        self._detector  = self.mp_fd.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=conf_threshold,
        )
        self._rgb_buf: Optional[np.ndarray] = None

    def detect(self, frame: np.ndarray) -> List[Detection]:
        h, w = frame.shape[:2]

        if self._rgb_buf is None or self._rgb_buf.shape[:2] != (h, w):
            self._rgb_buf = np.empty((h, w, 3), dtype=np.uint8)

        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, dst=self._rgb_buf)
        results = self._detector.process(self._rgb_buf)

        if results.detections is None or len(results.detections) == 0:
            return []

        detections: List[Detection] = []
        for det in results.detections:
            bb = det.location_data.relative_bounding_box
            x1 = max(0, int(bb.xmin * w))
            y1 = max(0, int(bb.ymin * h))
            x2 = min(w, int((bb.xmin + bb.width)  * w))
            y2 = min(h, int((bb.ymin + bb.height) * h))
            conf = float(det.score[0])
            landmarks = [
                (int(kp.x * w), int(kp.y * h))
                for kp in det.location_data.relative_keypoints
            ]
            detections.append(Detection((x1, y1, x2, y2), conf, landmarks))

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
    """YOLOv8 face detector via Ultralytics."""

    DEFAULT_MODEL_NAME = "yolov8n-face.pt"

    def __init__(self, conf_threshold: float = 0.5, model_path: Optional[str] = None):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("pip install ultralytics") from exc

        self.conf   = conf_threshold
        self._model = YOLO(str(self._resolve_model_path(model_path)))

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
        for c in candidates:
            if c.exists():
                return c
        searched = "\n".join(f"  - {c}" for c in candidates)
        raise FileNotFoundError(
            "YOLOv8 face weights not found. Download yolov8n-face.pt and place it in the "
            f"project root or models/ folder, or set YOLO_FACE_MODEL.\nSearched:\n{searched}"
        )

    def detect(self, frame: np.ndarray) -> List[Detection]:
        results = self._model.predict(source=frame, conf=self.conf, verbose=False, stream=False)

        detections: List[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            xyxy_all = boxes.xyxy.cpu().numpy().astype(int)
            conf_all = boxes.conf.cpu().numpy()

            for i, (xyxy, conf) in enumerate(zip(xyxy_all, conf_all)):
                x1, y1, x2, y2 = xyxy
                landmarks = None

                if result.keypoints is not None and i < len(result.keypoints):
                    kp_data = result.keypoints[i].xy[0].cpu().numpy()
                    landmarks = [(int(x), int(y)) for x, y in kp_data]

                detections.append(Detection((x1, y1, x2, y2), float(conf), landmarks))

        return detections


# ---------------------------------------------------------------------------
# Haar Cascade
# ---------------------------------------------------------------------------

class HaarCascadeDetector:
    """Classic Viola-Jones detector, requires only OpenCV."""

    CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    _EMPTY: List[Detection] = []

    def __init__(
        self,
        conf_threshold: float = 0.5,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: Tuple[int, int] = (30, 30),
        equalize_hist: bool = True,
    ):
        self._classifier = cv2.CascadeClassifier(self.CASCADE_PATH)
        if self._classifier.empty():
            raise RuntimeError("Failed to load Haar cascade XML. Check OpenCV install.")

        self._scale_factor = scale_factor
        self._min_neighbors = min_neighbors
        self._min_size = min_size
        self._equalize = equalize_hist
        self._gray_buf: Optional[np.ndarray] = None

    def detect(self, frame: np.ndarray) -> List[Detection]:
        h, w = frame.shape[:2]

        if self._gray_buf is None or self._gray_buf.shape != (h, w):
            self._gray_buf = np.empty((h, w), dtype=np.uint8)

        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY, dst=self._gray_buf)
        if self._equalize:
            cv2.equalizeHist(self._gray_buf, self._gray_buf)

        faces = self._classifier.detectMultiScale(
            self._gray_buf,
            scaleFactor=self._scale_factor,
            minNeighbors=self._min_neighbors,
            minSize=self._min_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        if len(faces) == 0:
            return self._EMPTY

        return [
            Detection((x, y, x + w_, y + h_), 1.0, None)
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
    """

    def __init__(self, conf_threshold: float = 0.5, model_name: str = "buffalo_sc"):
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:
            raise ImportError("pip install insightface onnxruntime") from exc

        self._app = FaceAnalysis(name=model_name, allowed_modules=["detection"])
        self._app.prepare(ctx_id=-1, det_thresh=conf_threshold, det_size=(640, 640))

    def detect(self, frame: np.ndarray) -> List[Detection]:
        faces = self._app.get(frame)
        detections: List[Detection] = []

        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int).tolist()
            conf = float(face.det_score)
            landmarks = (
                [(int(x), int(y)) for x, y in face.kps.tolist()]
                if face.kps is not None
                else None
            )
            detections.append(Detection((x1, y1, x2, y2), conf, landmarks))

        return detections
