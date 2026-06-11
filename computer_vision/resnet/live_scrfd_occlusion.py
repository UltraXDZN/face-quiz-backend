from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

from src.model import Model


BBox = Tuple[int, int, int, int]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="weights/resnet18/best_resnet18.pth")
    parser.add_argument("--source", default="0")
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--det-thresh", type=float, default=0.5)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--padding", type=float, default=0.08)
    parser.add_argument("--detect-every", type=int, default=3)
    parser.add_argument("--classify-every", type=int, default=3)
    parser.add_argument("--max-cache-age", type=int, default=15)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--input-size", type=int, default=224)
    return parser.parse_args()


class FastPreprocessor:
    def __init__(self, input_size: int = 224):
        self.input_size = input_size
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def __call__(self, crop_bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(
            rgb,
            (self.input_size, self.input_size),
            interpolation=cv2.INTER_AREA,
        )

        x = resized.astype(np.float32) / 255.0
        x = (x - self.mean) / self.std
        x = np.transpose(x, (2, 0, 1))
        x = np.ascontiguousarray(x)

        return torch.from_numpy(x).unsqueeze(0)


def load_occlusion_model(weights_path: str, device: torch.device):
    model = Model("resnet18", 2, False).to(device)

    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    if isinstance(checkpoint, dict) and "epoch" in checkpoint:
        print("Model epoch:", checkpoint["epoch"])

    return model


def load_scrfd(det_thresh: float):
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="buffalo_sc", allowed_modules=["detection"])
    app.prepare(ctx_id=-1, det_thresh=det_thresh, det_size=(640, 640))

    return app


def bbox_area(bbox: np.ndarray) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def pick_best_face(faces) -> Optional[object]:
    if not faces:
        return None

    return max(
        faces,
        key=lambda face: float(face.det_score) * bbox_area(face.bbox.astype(int)),
    )


def clamp_bbox(bbox: BBox, width: int, height: int) -> BBox:
    x1, y1, x2, y2 = bbox

    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))

    return x1, y1, x2, y2


def expand_bbox(bbox: BBox, width: int, height: int, padding: float) -> BBox:
    x1, y1, x2, y2 = bbox

    bw = x2 - x1
    bh = y2 - y1

    pad_x = int(bw * padding)
    pad_y = int(bh * padding)

    return clamp_bbox(
        (
            x1 - pad_x,
            y1 - pad_y,
            x2 + pad_x,
            y2 + pad_y,
        ),
        width,
        height,
    )


def crop_face(frame: np.ndarray, bbox: BBox, padding: float) -> Optional[np.ndarray]:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = expand_bbox(bbox, w, h, padding)

    if x2 <= x1 or y2 <= y1:
        return None

    return frame[y1:y2, x1:x2]


def predict_occlusion(
    model,
    preprocessor: FastPreprocessor,
    crop_bgr: np.ndarray,
    device: torch.device,
) -> tuple[float, float]:
    x = preprocessor(crop_bgr).to(device)

    with torch.inference_mode():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    non_occluded_prob = float(probs[0].item())
    occluded_prob = float(probs[1].item())

    return non_occluded_prob, occluded_prob


def draw_box(frame, bbox: BBox, label: str, color: tuple[int, int, int]):
    x1, y1, x2, y2 = bbox

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    thickness = 2

    (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
    y_label = max(35, y1 - 10)

    cv2.rectangle(
        frame,
        (x1, y_label - th - 10),
        (x1 + tw + 10, y_label + 6),
        color,
        -1,
    )

    cv2.putText(
        frame,
        label,
        (x1 + 5, y_label - 3),
        font,
        scale,
        (20, 20, 20),
        thickness,
        cv2.LINE_AA,
    )


def draw_hud(
    frame,
    fps: float,
    scrfd_conf: Optional[float],
    non_occ: Optional[float],
    occ: Optional[float],
    detect_every: int,
    classify_every: int,
):
    lines = [
        f"FPS: {fps:.1f}",
        f"detect every: {detect_every}",
        f"classify every: {classify_every}",
    ]

    if scrfd_conf is not None:
        lines.append(f"SCRFD: {scrfd_conf:.2f}")

    if non_occ is not None and occ is not None:
        lines.append(f"non-occ: {non_occ:.2f}")
        lines.append(f"occ:     {occ:.2f}")

    x, y = 20, 35

    for line in lines:
        cv2.putText(
            frame,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (230, 230, 230),
            2,
            cv2.LINE_AA,
        )
        y += 24


def open_capture(source, width: int, height: int):
    cap = cv2.VideoCapture(source)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    except Exception:
        pass

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    return cap


def main():
    args = parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    source = args.source
    try:
        source = int(source)
    except ValueError:
        pass

    torch.set_num_threads(max(1, args.threads))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Torch threads:", torch.get_num_threads())

    print("[+] Loading occlusion model...")
    model = load_occlusion_model(str(weights_path), device)
    preprocessor = FastPreprocessor(args.input_size)
    print("[+] Occlusion model ready.")

    print("[+] Loading SCRFD...")
    scrfd = load_scrfd(args.det_thresh)
    print("[+] SCRFD ready.")

    cap = open_capture(source, args.width, args.height)

    prev_time = time.perf_counter()
    fps = 0.0
    frame_idx = 0

    last_bbox: Optional[BBox] = None
    last_bbox_age = args.max_cache_age + 1
    last_scrfd_conf: Optional[float] = None

    last_non_occ_prob: Optional[float] = None
    last_occ_prob: Optional[float] = None
    last_label = "NO FACE"
    last_color = (60, 60, 220)

    current_crop = None
    crop_idx = 0

    print("[Controls] q = quit, s = save current face crop")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_idx += 1

        now = time.perf_counter()
        instant_fps = 1.0 / max(now - prev_time, 1e-6)
        fps = instant_fps if fps == 0.0 else 0.9 * fps + 0.1 * instant_fps
        prev_time = now

        h, w = frame.shape[:2]

        should_detect = frame_idx % max(1, args.detect_every) == 0 or last_bbox is None
        should_classify = frame_idx % max(1, args.classify_every) == 0

        if should_detect:
            faces = scrfd.get(frame)
            face = pick_best_face(faces)

            if face is None:
                last_bbox_age += 1

                if last_bbox_age > args.max_cache_age:
                    last_bbox = None
                    last_scrfd_conf = None
                    last_non_occ_prob = None
                    last_occ_prob = None
                    last_label = "NO FACE"
                    last_color = (60, 60, 220)
            else:
                last_bbox = clamp_bbox(tuple(face.bbox.astype(int)), w, h)
                last_bbox_age = 0
                last_scrfd_conf = float(face.det_score)
        else:
            last_bbox_age += 1

        if last_bbox is not None and last_bbox_age <= args.max_cache_age:
            current_crop = crop_face(frame, last_bbox, args.padding)

            if current_crop is None:
                last_label = "BAD CROP"
                last_color = (60, 60, 220)
            else:
                if should_classify or last_occ_prob is None:
                    last_non_occ_prob, last_occ_prob = predict_occlusion(
                        model=model,
                        preprocessor=preprocessor,
                        crop_bgr=current_crop,
                        device=device,
                    )

                    if last_occ_prob >= args.threshold:
                        last_label = f"OCCLUDED {last_occ_prob:.2f}"
                        last_color = (0, 191, 255)
                    else:
                        last_label = f"NON-OCCLUDED {last_non_occ_prob:.2f}"
                        last_color = (74, 222, 128)

                draw_box(frame, last_bbox, last_label, last_color)
        else:
            current_crop = None
            cv2.putText(
                frame,
                "NO FACE",
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (60, 60, 220),
                2,
                cv2.LINE_AA,
            )

        draw_hud(
            frame=frame,
            fps=fps,
            scrfd_conf=last_scrfd_conf,
            non_occ=last_non_occ_prob,
            occ=last_occ_prob,
            detect_every=args.detect_every,
            classify_every=args.classify_every,
        )

        cv2.imshow("SCRFD + occlusion classifier FAST", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("s") and current_crop is not None:
            out = f"debug_crop_{crop_idx:04d}.jpg"
            cv2.imwrite(out, current_crop)
            print("[+] Saved crop:", out)
            crop_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()