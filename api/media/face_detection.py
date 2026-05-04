import os
from threading import Lock
from typing import Tuple

import cv2
import numpy as np
from insightface.model_zoo import get_model
from insightface.utils.storage import ensure_available

_model = None
_model_lock = Lock()


def _ensure_model_dir(root: str, name: str) -> str:
    """Download model into ~/.insightface/models/<name> if missing."""
    return ensure_available('models', name, root=root)


def _load_model():
    global _model
    if _model is not None:
        return _model

    with _model_lock:
        if _model is not None:
            return _model

        model_path = os.getenv("SCRFD_MODEL_PATH")
        model_root = os.getenv("SCRFD_MODEL_ROOT", "~/.insightface")
        providers = ["CPUExecutionProvider"]

        if model_path and os.path.exists(model_path):
            model = get_model(model_path, providers=providers)
        else:
            root = os.path.expanduser(model_root)
            _ensure_model_dir(root, "buffalo_sc")
            default_model_path = os.path.join(root, "models", "buffalo_sc", "det_500m.onnx")
            model = get_model(default_model_path, providers=providers)

        det_size = int(os.getenv("SCRFD_DET_SIZE", "640"))
        model.prepare(ctx_id=0, det_size=(det_size, det_size))
        _model = model
        return _model


def detect_faces(image_bgr: np.ndarray, threshold: float = 0.5) -> Tuple[int, np.ndarray]:
    model = _load_model()
    # SCRFD/RetinaFace detect uses internal det_thresh, not a threshold arg.
    det_size = int(os.getenv("SCRFD_DET_SIZE", "640"))
    # Optional: update threshold if model exposes det_thresh
    if hasattr(model, 'det_thresh'):
        model.det_thresh = float(threshold)
    bboxes, _ = model.detect(image_bgr, input_size=(det_size, det_size))
    if bboxes is None:
        return 0, np.empty((0, 5))
    return len(bboxes), bboxes
