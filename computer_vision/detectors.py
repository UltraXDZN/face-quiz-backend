"""
detectors.py
============
Unified interface for supported face detectors.
Each detector exposes:

    detect(frame: np.ndarray) -> List[Detection]

Detection fields:
    bbox      : (x1, y1, x2, y2) in pixel coords
    conf      : float 0-1
    landmarks : list of (x, y) tuples or None
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]
    conf: float
    landmarks: Optional[List[Tuple[int, int]]] = field(default=None)


class MediaPipeDetector:
    """
    Google MediaPipe FaceDetection / BlazeFace.
    Requires: pip install mediapipe
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
                "Check that you do not have a local file/folder named mediapipe.py, then reinstall mediapipe."
            )

        self.mp_fd = mp.solutions.face_detection
        self._detector = self.mp_fd.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=conf_threshold,
        )

    def detect(self, frame: np.ndarray) -> List[Detection]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        results = self._detector.process(rgb)

        detections: List[Detection] = []
        if not results.detections:
            return detections

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


class YOLOv8Detector:
    """
    YOLOv8 face detector via Ultralytics.
    Requires: pip install ultralytics

    Put yolov8n-face.pt in either:
        ./yolov8n-face.pt
        ./models/yolov8n-face.pt

    Or set:
        export YOLO_FACE_MODEL=/full/path/to/yolov8n-face.pt
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

        searched = "\n".join(f"  - {candidate}" for candidate in candidates)
        raise FileNotFoundError(
            "YOLOv8 face weights not found. Download yolov8n-face.pt and place it in the project root "
            "or models/ folder, or set YOLO_FACE_MODEL. Searched:\n"
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

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu())
                landmarks = None

                if result.keypoints is not None and i < len(result.keypoints):
                    kp_data = result.keypoints[i].xy[0].cpu().numpy()
                    landmarks = [(int(x), int(y)) for x, y in kp_data]

                detections.append(Detection(bbox=(x1, y1, x2, y2), conf=conf, landmarks=landmarks))

        return detections


class HaarCascadeDetector:
    """
    Classic Viola-Jones detector.
    Requires only OpenCV.
    """

    CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

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

    def detect(self, frame: np.ndarray) -> List[Detection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self._classifier.detectMultiScale(
            gray,
            scaleFactor=self._scale_factor,
            minNeighbors=self._min_neighbors,
            minSize=self._min_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        return [
            Detection(bbox=(x, y, x + w, y + h), conf=1.0, landmarks=None)
            for (x, y, w, h) in faces
        ]


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
            landmarks = None

            if face.kps is not None:
                landmarks = [(int(x), int(y)) for x, y in face.kps]

            detections.append(Detection(bbox=(x1, y1, x2, y2), conf=conf, landmarks=landmarks))

        return detections
