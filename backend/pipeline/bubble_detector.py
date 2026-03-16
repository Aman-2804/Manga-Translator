"""Detect speech bubbles in manga pages using a pretrained YOLOv8 model.

Model: ogkalu/comic-speech-bubble-detector-yolov8m (Hugging Face)
"""

import logging
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

BoundingBox = Tuple[int, int, int, int]  # (x, y, w, h)

_HF_REPO = "ogkalu/comic-speech-bubble-detector-yolov8m"
_HF_FILENAME = "comic-speech-bubble-detector.pt"
_CONF_THRESHOLD = 0.45
_MIN_AREA_RATIO = 0.01
_NMS_IOU_THRESH = 0.5

# ── Thread-safe lazy singleton for the YOLO model ────────────────────────────

_model = None
_model_lock = threading.Lock()


def _get_model():
    """Download (once) and return the cached YOLO model instance."""
    global _model

    if _model is not None:
        return _model

    with _model_lock:
        if _model is not None:
            return _model

        logger.info("Downloading YOLO weights from Hugging Face (%s)…", _HF_REPO)
        try:
            from huggingface_hub import hf_hub_download
            weights_path = hf_hub_download(
                repo_id=_HF_REPO,
                filename=_HF_FILENAME,
            )
        except Exception as exc:
            logger.error("Failed to download model weights: %s", exc)
            raise RuntimeError(
                f"Could not download bubble-detector weights from {_HF_REPO}: {exc}"
            ) from exc

        logger.info("Loading YOLO model from %s", weights_path)
        try:
            from ultralytics import YOLO
            _model = YOLO(weights_path)
        except Exception as exc:
            logger.error("Failed to load YOLO model: %s", exc)
            raise RuntimeError(f"Could not load YOLO model: {exc}") from exc

        logger.info("YOLO bubble-detector model ready")

    return _model


def _xyxy_to_xywh(x1: float, y1: float, x2: float, y2: float) -> list[int]:
    """Convert (x1, y1, x2, y2) corners to [x, y, w, h]."""
    x, y = int(round(x1)), int(round(y1))
    w, h = int(round(x2 - x1)), int(round(y2 - y1))
    return [x, y, w, h]


def _rect_contour(x: int, y: int, w: int, h: int) -> np.ndarray:
    """Build a rectangular contour (4 corner points) for a bounding box."""
    return np.array([
        [[x, y]],
        [[x + w, y]],
        [[x + w, y + h]],
        [[x, y + h]],
    ], dtype=np.int32)


# ── Class-based API (used by main.py pipeline) ──────────────────────────────


class BubbleDetector:
    """Locates speech bubble regions using YOLOv8 (PIL-based API for the pipeline)."""

    @staticmethod
    def detect(page: Image.Image) -> List[BoundingBox]:
        """Find speech bubble bounding boxes in a manga page.

        Args:
            page: PIL Image of a single manga page.

        Returns:
            List of (x, y, width, height) tuples for each detected bubble.
        """
        model = _get_model()
        results = model(page, conf=_CONF_THRESHOLD, verbose=False)

        bubbles: list[BoundingBox] = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x, y = int(round(x1)), int(round(y1))
                w, h = int(round(x2 - x1)), int(round(y2 - y1))
                bubbles.append((x, y, w, h))

        return bubbles


# ── Standalone function (file-path based) ────────────────────────────────────


def _save_debug_image(
    image: np.ndarray,
    bubbles: list[dict],
    dest: str,
) -> None:
    """Draw detected bubbles on a copy and save as a debug image."""
    debug = image.copy()

    for b in bubbles:
        x, y, w, h = b["bbox"]
        conf = b["confidence"]

        cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cx, cy = b["center"]
        cv2.circle(debug, (cx, cy), 4, (0, 0, 255), -1)

        label = f"{conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(debug, (x, y - th - 6), (x + tw + 4, y), (0, 255, 0), -1)
        cv2.putText(
            debug, label, (x + 2, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
        )

    cv2.imwrite(dest, debug)
    logger.debug("Debug image saved → %s", dest)


def detect_bubbles(image_path: str, text_mask: Optional[np.ndarray] = None) -> list[dict]:
    """Detect speech bubbles using segmentation mask (primary) or YOLOv8 (fallback).

    When text_mask is provided from segmentation, derives text boxes from the
    mask (Whalefishin-style). Otherwise uses YOLO.

    Args:
        image_path: Path to a manga page image (PNG / JPG / WEBP).
        text_mask: Optional binary mask (255=text) from segmentation.

    Returns:
        List of dicts sorted top-to-bottom, left-to-right.  Each dict has:
            - ``bbox``       – [x, y, w, h]
            - ``confidence`` – float (0–1)
            - ``center``     – (cx, cy)
            - ``area``       – float
            - ``contour``    – rectangular contour as numpy array

    Raises:
        FileNotFoundError: If *image_path* does not exist.
        RuntimeError:      If the image cannot be decoded or the model fails.
    """
    path = Path(image_path).resolve()

    if not path.is_file():
        logger.error("Image not found: %s", path)
        raise FileNotFoundError(f"Image file does not exist: {path}")

    logger.info("Loading image: %s", path.name)
    img = cv2.imread(str(path))
    if img is None:
        logger.error("Cannot decode image: %s", path.name)
        raise RuntimeError(f"Cannot decode image (corrupt or unsupported): {path.name}")

    img_h, img_w = img.shape[:2]
    img_area = img_h * img_w
    min_area = img_area * _MIN_AREA_RATIO

    # Always use YOLO for bubble detection — it's trained to find actual speech bubble
    # shapes and gives proper bounding boxes for text placement and font sizing.
    # The text_mask is used only for inpainting (precise text erasure), not detection.
    model = _get_model()
    results = model(str(path), conf=_CONF_THRESHOLD, verbose=False)

    raw_boxes: list[list[int]] = []
    raw_confs: list[float] = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            bbox = _xyxy_to_xywh(x1, y1, x2, y2)
            x, y, w, h = bbox
            area = float(w * h)

            if area < min_area:
                logger.debug("Skipping tiny detection: area=%.0f < min=%.0f", area, min_area)
                continue

            raw_boxes.append(bbox)
            raw_confs.append(conf)

    keep = list(range(len(raw_boxes)))
    if raw_boxes:
        nms_rects = [[x, y, w, h] for x, y, w, h in raw_boxes]
        indices = cv2.dnn.NMSBoxes(nms_rects, raw_confs, _CONF_THRESHOLD, _NMS_IOU_THRESH)
        keep = [int(i) for i in (indices.flatten() if len(indices) else [])]

    bubbles: list[dict] = []
    for i in keep:
        bbox = raw_boxes[i]
        conf = raw_confs[i]
        x, y, w, h = bbox
        cx, cy = x + w // 2, y + h // 2

        bubbles.append({
            "bbox": bbox,
            "confidence": conf,
            "center": (cx, cy),
            "area": float(w * h),
            "contour": _rect_contour(x, y, w, h),
        })

        logger.info(
            "Bubble detected: bbox=[%d,%d,%d,%d]  conf=%.2f",
            x, y, w, h, conf,
        )

    bubbles.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))

    debug_path = str(path.parent / "debug_bubbles.png")
    _save_debug_image(img, bubbles, debug_path)

    logger.info(
        "detect_bubbles complete: %d bubble(s) in %s",
        len(bubbles),
        path.name,
    )
    return bubbles


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    sample = sys.argv[1] if len(sys.argv) > 1 else "sample_page.png"

    try:
        results = detect_bubbles(sample)
        print(f"\nDetected {len(results)} bubble(s):")
        for i, b in enumerate(results, 1):
            print(
                f"  #{i}  bbox={b['bbox']}  center={b['center']}  "
                f"conf={b['confidence']:.2f}  area={b['area']:.0f}"
            )
    except (FileNotFoundError, RuntimeError) as err:
        print(f"\nError: {err}", file=sys.stderr)
        sys.exit(1)
