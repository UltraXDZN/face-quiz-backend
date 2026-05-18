import os
from threading import Lock
from typing import Tuple

import cv2
import numpy as np

from computer_vision.detectors import InsightFaceSCRFDDetector

_detector = None
_detector_lock = Lock()


def _load_detector() -> InsightFaceSCRFDDetector:
    global _detector
    if _detector is not None:
        return _detector

    with _detector_lock:
        if _detector is not None:
            return _detector

        conf_threshold = float(os.getenv("SCRFD_CONF_THRESHOLD", "0.5"))
        _detector = InsightFaceSCRFDDetector(conf_threshold=conf_threshold)
        return _detector


def detect_faces(image_bgr: np.ndarray, threshold: float = 0.5) -> Tuple[int, np.ndarray]:
    detector = _load_detector()
    detections = detector.detect(image_bgr)

    if not detections:
        return 0, np.empty((0, 5), dtype=np.float32)

    bboxes = np.array(
        [[det.bbox[0], det.bbox[1], det.bbox[2], det.bbox[3], det.conf] for det in detections],
        dtype=np.float32,
    )
    return len(detections), bboxes
