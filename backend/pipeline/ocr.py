"""Japanese OCR for manga speech bubbles using manga-ocr.

Provides both a class-based API (`MangaOCR`) for the FastAPI pipeline and
standalone functions (`extract_text_from_bubble`, `extract_all_bubbles`) for
direct use with file paths and bubble dicts.
"""

import logging
import re
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

BoundingBox = Tuple[int, int, int, int]

# ── Thread-safe lazy singleton for the manga-ocr model ───────────────────────

_model_instance = None
_model_lock = threading.Lock()
_model_init_error: Optional[Exception] = None


def _get_model():
    """Return the shared MangaOcr instance, creating it on first call.

    Thread-safe: only the first caller pays the initialisation cost; all
    subsequent callers receive the cached instance immediately.

    Raises:
        RuntimeError: If the model fails to load (missing weights,
                      CUDA errors, etc.).
    """
    global _model_instance, _model_init_error

    if _model_instance is not None:
        return _model_instance

    with _model_lock:
        if _model_instance is not None:
            return _model_instance

        if _model_init_error is not None:
            raise RuntimeError(
                "manga-ocr model previously failed to initialise"
            ) from _model_init_error

        logger.info("Initialising manga-ocr model (first call — this may take a moment)…")
        try:
            from manga_ocr import MangaOcr
            _model_instance = MangaOcr()
            logger.info("manga-ocr model ready")
        except Exception as exc:
            _model_init_error = exc
            logger.error("Failed to initialise manga-ocr: %s", exc)
            raise RuntimeError(
                f"Could not load manga-ocr model: {exc}"
            ) from exc

    return _model_instance


# ── Class-based API (used by main.py pipeline) ──────────────────────────────


class MangaOCR:
    """Extracts Japanese text from detected speech bubble regions.

    Thin wrapper kept for backward-compatibility with the FastAPI pipeline
    in ``main.py``.
    """

    @classmethod
    def _get_model(cls):
        return _get_model()

    @classmethod
    def recognize(cls, page: Image.Image, bubbles: List[BoundingBox]) -> List[str]:
        """Run OCR on each speech bubble region.

        Args:
            page:    PIL Image of the full manga page.
            bubbles: List of (x, y, w, h) bounding boxes.

        Returns:
            List of recognised Japanese text strings, one per bubble.
        """
        model = cls._get_model()
        texts: list[str] = []

        for x, y, w, h in bubbles:
            crop = page.crop((x, y, x + w, y + h))
            text = model(crop)
            texts.append(text)

        return texts


# ── Text filtering ────────────────────────────────────────────────────────────

def _is_meaningful(text: str) -> bool:
    """Return True if text is non-empty and not just whitespace."""
    return bool(text and not text.isspace())


# ── Standalone functions (file-path / dict based) ────────────────────────────

BBOX_PADDING = 5
_MIN_CROP_DIM = 20
_UPSCALE_THRESH = 64
_CONF_FILTER_THRESH = 30  # Skip bubbles with Tesseract avg confidence below this


def _preprocess_crop(crop: np.ndarray) -> np.ndarray:
    """Enhance a bubble crop for better OCR accuracy.

    Steps: upscale small crops, CLAHE contrast enhancement, light blur.
    """
    h, w = crop.shape[:2]
    if h < _UPSCALE_THRESH or w < _UPSCALE_THRESH:
        crop = cv2.resize(crop, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)

    return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)


def _tesseract_confidence(crop: np.ndarray) -> Optional[float]:
    """Run Tesseract on crop, return mean confidence. None if Tesseract unavailable."""
    try:
        import pytesseract
        data = pytesseract.image_to_data(
            crop, lang="jpn_vert", config="--psm 5", output_type="dict"
        )
        confs = [c for c in data["conf"] if c != -1]
        return float(np.mean(confs)) if confs else -1.0
    except Exception:
        return None


def extract_text_from_bubble(
    image: np.ndarray,
    bbox: list,
    conf_filter_thresh: Optional[float] = None,
) -> str:
    """Run manga-ocr on a single bubble region.

    Applies preprocessing (upscale, CLAHE, denoise) before OCR for
    better accuracy on small or low-contrast text.

    Args:
        image: Full page image as a NumPy array (BGR, as returned by
               ``cv2.imread``).
        bbox:  Bounding box ``[x, y, w, h]`` of the bubble.

    Returns:
        Recognised Japanese string, or ``""`` if nothing was detected.
    """
    model = _get_model()

    h_img, w_img = image.shape[:2]
    x, y, w, h = bbox

    if w < _MIN_CROP_DIM or h < _MIN_CROP_DIM:
        logger.debug("Crop too small (%dx%d) for bbox %s — skipping", w, h, bbox)
        return ""

    x1 = max(x - BBOX_PADDING, 0)
    y1 = max(y - BBOX_PADDING, 0)
    x2 = min(x + w + BBOX_PADDING, w_img)
    y2 = min(y + h + BBOX_PADDING, h_img)

    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        logger.warning("Empty crop for bbox %s — skipping", bbox)
        return ""

    if conf_filter_thresh is not None:
        conf = _tesseract_confidence(crop)
        if conf is not None and conf >= 0 and conf < conf_filter_thresh:
            logger.debug("Bbox %s: Tesseract conf %.1f < %.1f — skipping", bbox, conf, conf_filter_thresh)
            return ""

    processed = _preprocess_crop(crop)
    pil_crop = Image.fromarray(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))

    try:
        text: str = model(pil_crop)
    except Exception as exc:
        logger.error("OCR failed on bbox %s: %s", bbox, exc)
        return ""

    return text.strip() if text else ""


def extract_all_bubbles(
    image_path: str,
    bubbles: list[dict],
    conf_filter_thresh: Optional[float] = _CONF_FILTER_THRESH,
) -> list[dict]:
    """Run OCR on every bubble in a page and attach the recognised text.

    For each bubble dict (as produced by
    :func:`~pipeline.bubble_detector.detect_bubbles`), the key
    ``"japanese_text"`` is added with the recognised string.  Bubbles whose
    text is empty or whitespace-only are dropped from the returned list.

    Args:
        image_path: Path to the manga page image on disk.
        bubbles:    List of bubble dicts, each containing at least a
                    ``"bbox"`` key with value ``[x, y, w, h]``.

    Returns:
        Filtered list of bubble dicts augmented with ``"japanese_text"``.

    Raises:
        FileNotFoundError: If *image_path* does not exist.
        RuntimeError:      If the image cannot be decoded.
    """
    path = Path(image_path).resolve()

    if not path.is_file():
        logger.error("Image not found: %s", path)
        raise FileNotFoundError(f"Image file does not exist: {path}")

    image = cv2.imread(str(path))
    if image is None:
        logger.error("Cannot decode image: %s", path.name)
        raise RuntimeError(f"Cannot decode image (corrupt or unsupported): {path.name}")

    logger.info("Running OCR on %d bubble(s) in %s", len(bubbles), path.name)

    results: list[dict] = []

    for idx, bubble in enumerate(bubbles):
        bbox = bubble["bbox"]
        text = extract_text_from_bubble(image, bbox, conf_filter_thresh=conf_filter_thresh)

        if not _is_meaningful(text):
            logger.debug("Bubble #%d: junk/empty text \"%s\" — skipped", idx, text)
            continue

        bubble_copy = dict(bubble)
        bubble_copy["japanese_text"] = text
        results.append(bubble_copy)

        logger.info("Bubble #%d: \"%s\"", idx, text)

    logger.info(
        "OCR complete: %d / %d bubble(s) yielded text",
        len(results),
        len(bubbles),
    )
    return results


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    img_path = sys.argv[1] if len(sys.argv) > 1 else "sample_page.png"

    sample_bubbles: list[dict] = [
        {"bbox": [100, 100, 200, 80], "center": (200, 140), "area": 16000.0},
        {"bbox": [400, 300, 180, 90], "center": (490, 345), "area": 16200.0},
    ]

    try:
        enriched = extract_all_bubbles(img_path, sample_bubbles)
        print(f"\nExtracted text from {len(enriched)} bubble(s):")
        for i, b in enumerate(enriched, 1):
            print(f"  #{i}  bbox={b['bbox']}  text=\"{b['japanese_text']}\"")
    except (FileNotFoundError, RuntimeError) as err:
        print(f"\nError: {err}", file=sys.stderr)
        sys.exit(1)
