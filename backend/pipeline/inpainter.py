"""Erase Japanese text from speech bubbles by inpainting."""

import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

BoundingBox = Tuple[int, int, int, int]

_INPAINT_RADIUS = 20
_DARK_THRESH = 120
_TEXT_AREA_MAX_RATIO = 0.05
_EDGE_MARGIN_PX = 4
_DILATE_PX = 8
_SHADOW_THRESH = 230
_FEATHER_KERNEL = 3


def _is_edge_contour(contour: np.ndarray, roi_w: int, roi_h: int) -> bool:
    """Return True if *contour* touches or nearly touches the crop edges."""
    for pt in contour.reshape(-1, 2):
        px, py = pt
        if (px <= _EDGE_MARGIN_PX or px >= roi_w - _EDGE_MARGIN_PX
                or py <= _EDGE_MARGIN_PX or py >= roi_h - _EDGE_MARGIN_PX):
            return True
    return False


def _filter_text_contours(
    contours: list, roi_w: int, roi_h: int,
) -> list:
    """Keep only small interior contours (text strokes), discarding borders."""
    bbox_area = roi_h * roi_w
    text_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 2:
            continue
        if area > bbox_area * _TEXT_AREA_MAX_RATIO:
            continue
        if _is_edge_contour(c, roi_w, roi_h):
            continue
        text_contours.append(c)
    return text_contours


def _find_text_contours_in_roi(
    gray_roi: np.ndarray,
) -> list:
    """Find dark contours that look like text strokes (not bubble border).

    Uses both fixed thresholding and adaptive thresholding, then merges
    the results for better coverage of varying brightness.
    """
    roi_h, roi_w = gray_roi.shape[:2]

    dark_fixed = (gray_roi < _DARK_THRESH).astype(np.uint8) * 255
    contours_fixed, _ = cv2.findContours(dark_fixed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fixed = _filter_text_contours(contours_fixed, roi_w, roi_h)

    if fixed:
        return fixed

    dark_adaptive = cv2.adaptiveThreshold(
        gray_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 10,
    )
    contours_adaptive, _ = cv2.findContours(dark_adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return _filter_text_contours(contours_adaptive, roi_w, roi_h)


def _cleanup_shadows(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Force-white shadows, feather mask edges, then blend for clean result."""
    result = image.copy()

    feathered = cv2.GaussianBlur(
        mask, (_FEATHER_KERNEL * 2 + 1, _FEATHER_KERNEL * 2 + 1), 0,
    )
    feather_zone = feathered > 0

    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if result.ndim == 3 else result
    shadow_pixels = feather_zone & (gray < _SHADOW_THRESH)
    result[shadow_pixels] = 255

    blurred = cv2.GaussianBlur(result, (3, 3), 0)
    if result.ndim == 3:
        mask_3c = np.stack([feather_zone] * 3, axis=-1)
        result[mask_3c] = blurred[mask_3c]
    else:
        result[feather_zone] = blurred[feather_zone]

    return result


# ── Class-based API (used by main.py pipeline) ──────────────────────────────


class Inpainter:
    """Removes original text from speech bubbles using inpainting."""

    @staticmethod
    def inpaint(page: Image.Image, bubbles: List[BoundingBox]) -> Image.Image:
        """Erase text inside each bubble region.

        Args:
            page:    PIL Image of the manga page.
            bubbles: List of (x, y, w, h) bounding boxes.

        Returns:
            PIL Image with Japanese text removed from bubbles.
        """
        img = np.array(page)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        full_mask = np.zeros(gray.shape, dtype=np.uint8)

        for x, y, w, h in bubbles:
            roi = gray[y:y + h, x:x + w]
            text_contours = _find_text_contours_in_roi(roi)
            roi_mask = np.zeros(roi.shape, dtype=np.uint8)
            cv2.drawContours(roi_mask, text_contours, -1, 255, thickness=cv2.FILLED)
            kernel = np.ones((_DILATE_PX, _DILATE_PX), np.uint8)
            roi_mask = cv2.dilate(roi_mask, kernel, iterations=1)
            full_mask[y:y + h, x:x + w] = cv2.bitwise_or(
                full_mask[y:y + h, x:x + w], roi_mask,
            )

        result = cv2.inpaint(img, full_mask, _INPAINT_RADIUS, cv2.INPAINT_NS)
        result = _cleanup_shadows(result, full_mask)
        return Image.fromarray(result)


# ── Standalone functions (file-path / dict based) ────────────────────────────


def create_bubble_mask(image: np.ndarray, bubble: dict) -> np.ndarray:
    """Create a binary mask targeting **only text pixels** inside a bubble.

    Strategy:
        1. Crop the grayscale image to the bubble's bounding box.
        2. Threshold to find all dark pixels (< 80 — manga text is black).
        3. Find contours of those dark regions.
        4. Exclude the bubble border — any contour that touches the crop
           edges or is larger than 5 % of the crop area.
        5. Draw the remaining small interior contours (text strokes) into
           a mask.
        6. Dilate by 6 px to cover anti-aliased edges.

    The result is a mask where **only the Japanese characters** are white;
    the bubble border, white interior, and surrounding artwork are black.

    Args:
        image:  Full page image as a BGR numpy array.
        bubble: Bubble dict with a ``"bbox"`` key ([x, y, w, h]).

    Returns:
        Single-channel uint8 mask (same size as *image*).
        White (255) = text to erase, black (0) = keep.
    """
    img_h, img_w = image.shape[:2]

    bx, by, bw, bh = bubble["bbox"]
    x1, y1 = max(bx, 0), max(by, 0)
    x2, y2 = min(bx + bw, img_w), min(by + bh, img_h)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    roi = gray[y1:y2, x1:x2]

    text_contours = _find_text_contours_in_roi(roi)

    roi_mask = np.zeros(roi.shape, dtype=np.uint8)
    cv2.drawContours(roi_mask, text_contours, -1, 255, thickness=cv2.FILLED)

    kernel = np.ones((_DILATE_PX, _DILATE_PX), np.uint8)
    roi_mask = cv2.dilate(roi_mask, kernel, iterations=1)

    full_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = roi_mask

    return full_mask


def inpaint_bubble(image: np.ndarray, bubble: dict) -> np.ndarray:
    """Inpaint (erase) the contents of a single bubble.

    Generates a mask with :func:`create_bubble_mask` then fills the
    masked region using the Telea inpainting algorithm.

    Args:
        image:  Full page image as a BGR numpy array.
        bubble: Bubble dict with ``"contour"`` and/or ``"bbox"``.

    Returns:
        Copy of *image* with the bubble interior inpainted.
    """
    mask = create_bubble_mask(image, bubble)
    result = cv2.inpaint(image, mask, _INPAINT_RADIUS, cv2.INPAINT_NS)
    result = _cleanup_shadows(result, mask)
    return result


def inpaint_all_bubbles(image_path: str, bubbles: list[dict]) -> np.ndarray:
    """Sequentially inpaint every bubble in a manga page.

    Each inpainting pass builds on the result of the previous one so
    overlapping regions are handled correctly.  A debug copy is saved
    as ``debug_inpainted.png`` next to the source image.

    Args:
        image_path: Path to the manga page image on disk.
        bubbles:    List of bubble dicts (from the detector / OCR stages).

    Returns:
        Final inpainted image as a BGR numpy array.

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

    logger.info("Inpainting %d bubble(s) in %s", len(bubbles), path.name)

    for idx, bubble in enumerate(bubbles):
        image = inpaint_bubble(image, bubble)
        logger.info(
            "Bubble #%d inpainted  (bbox=%s)",
            idx, bubble.get("bbox", "?"),
        )

    debug_path = str(path.parent / "debug_inpainted.png")
    cv2.imwrite(debug_path, image)
    logger.info("Debug inpainted image saved → %s", debug_path)

    logger.info("Inpainting complete: %d bubble(s) processed", len(bubbles))
    return image


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    img_path = sys.argv[1] if len(sys.argv) > 1 else "sample_page.png"

    sample_bubbles = [
        {"bbox": [100, 100, 200, 80], "contour": None},
    ]

    try:
        result = inpaint_all_bubbles(img_path, sample_bubbles)
        print(f"\nInpainting done — output shape: {result.shape}")
    except (FileNotFoundError, RuntimeError) as err:
        print(f"\nError: {err}", file=sys.stderr)
        sys.exit(1)
