"""Erase Japanese text from speech bubbles by inpainting."""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

BoundingBox = Tuple[int, int, int, int]

_INPAINT_RADIUS = 20
_DARK_THRESH = 120
_TEXT_AREA_MAX_RATIO = 0.05
_EDGE_MARGIN_PX = 4
_DILATE_PX = 12
_SHADOW_THRESH = 230
_FEATHER_KERNEL = 3
_FLOOD_FILL_TOLERANCE = 15
_WHITE_INTERIOR_THRESH = 235


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


def _find_light_contours_in_roi(gray_roi: np.ndarray) -> list:
    """Find bright contours that look like white text on a dark background."""
    roi_h, roi_w = gray_roi.shape[:2]
    bbox_area = roi_h * roi_w

    light_mask = (gray_roi > 200).astype(np.uint8) * 255
    contours, _ = cv2.findContours(light_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    text_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 2:
            continue
        if area > bbox_area * _TEXT_AREA_MAX_RATIO:
            continue
        text_contours.append(c)
    return text_contours


def _force_white_interior(image: np.ndarray, bubbles: list) -> np.ndarray:
    """Flood-fill from the center of each bubble to force the interior clean white.

    After inpainting there can be residual halftone dots or faint text strokes.
    This finds the connected white region from the bubble center and paints it
    pure white, eliminating any leftover artefacts without touching the art
    outside the bubble.
    """
    result = image.copy()
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if result.ndim == 3 else result.copy()

    for b in bubbles:
        if b.get("skip"):
            continue
        bx, by, bw, bh = b.get("bbox", (0, 0, 0, 0))
        x1, y1 = max(bx, 0), max(by, 0)
        x2, y2 = min(bx + bw, result.shape[1]), min(by + bh, result.shape[0])
        if x2 <= x1 or y2 <= y1:
            continue

        # Try multiple seed points (center, then upper/lower halves)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        seeds = [
            (cx, cy),
            (cx, y1 + (y2 - y1) // 3),
            (cx, y1 + 2 * (y2 - y1) // 3),
        ]

        is_dark = b.get("dark_bubble", False)
        fill_color = (0, 0, 0) if is_dark else (255, 255, 255)
        seed_check = lambda v: v < 80 if is_dark else v >= _WHITE_INTERIOR_THRESH

        for sx, sy in seeds:
            if not seed_check(gray[sy, sx]):
                continue
            roi = result[y1:y2, x1:x2].copy()
            flood_mask = np.zeros((y2 - y1 + 2, x2 - x1 + 2), np.uint8)
            tol = (_FLOOD_FILL_TOLERANCE,) * 3
            cv2.floodFill(roi, flood_mask, (sx - x1, sy - y1), fill_color,
                          loDiff=tol, upDiff=tol)
            result[y1:y2, x1:x2] = roi
            break  # one successful fill per bubble is enough

    return result


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


def inpaint_all_bubbles(
    image_path: str,
    bubbles: list[dict],
    text_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Inpaint text regions. Uses segmentation mask when available for cleaner results.

    Args:
        image_path: Path to the manga page image on disk.
        bubbles:    List of bubble dicts (from the detector / OCR stages).
        text_mask:  Optional binary mask (255=text) from segmentation.

    Returns:
        Final inpainted image as a BGR numpy array.
    """
    path = Path(image_path).resolve()

    if not path.is_file():
        logger.error("Image not found: %s", path)
        raise FileNotFoundError(f"Image file does not exist: {path}")

    image = cv2.imread(str(path))
    if image is None:
        logger.error("Cannot decode image: %s", path.name)
        raise RuntimeError(f"Cannot decode image (corrupt or unsupported): {path.name}")

    to_inpaint = [b for b in bubbles if not b.get("skip")]

    if text_mask is not None and text_mask.shape[:2] == image.shape[:2]:
        bubble_mask = np.zeros_like(text_mask)
        for b in to_inpaint:
            bx, by, bw, bh = b.get("bbox", (0, 0, 0, 0))
            x1, y1 = max(bx, 0), max(by, 0)
            x2, y2 = min(bx + bw, image.shape[1]), min(by + bh, image.shape[0])
            bubble_mask[y1:y2, x1:x2] = 255
        mask_uint8 = (text_mask.astype(np.uint8) & bubble_mask)
        logger.info("Inpainting with segmentation mask in %s", path.name)
        if mask_uint8.max() > 0:
            image = cv2.inpaint(image, mask_uint8, _INPAINT_RADIUS, cv2.INPAINT_NS)
            image = _cleanup_shadows(image, mask_uint8)
    else:
        logger.info("Inpainting %d bubble(s) in %s", len(bubbles), path.name)
        for idx, bubble in enumerate(to_inpaint):
            is_dark = bubble.get("dark_bubble", False)
            if is_dark:
                # White text on black: find bright contours, inpaint with black neighbors
                bx, by, bw, bh = bubble["bbox"]
                x1, y1 = max(bx, 0), max(by, 0)
                x2, y2 = min(bx + bw, image.shape[1]), min(by + bh, image.shape[0])
                gray_roi = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[y1:y2, x1:x2]
                light_contours = _find_light_contours_in_roi(gray_roi)
                roi_mask = np.zeros(gray_roi.shape, dtype=np.uint8)
                cv2.drawContours(roi_mask, light_contours, -1, 255, thickness=cv2.FILLED)
                kernel = np.ones((_DILATE_PX, _DILATE_PX), np.uint8)
                roi_mask = cv2.dilate(roi_mask, kernel, iterations=1)
                full_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                full_mask[y1:y2, x1:x2] = roi_mask
                if full_mask.max() > 0:
                    image = cv2.inpaint(image, full_mask, _INPAINT_RADIUS, cv2.INPAINT_NS)
            else:
                image = inpaint_bubble(image, bubble)
            logger.info("Bubble #%d inpainted  (dark=%s bbox=%s)", idx, is_dark, bubble.get("bbox", "?"))

    # Final pass: flood-fill bubble interiors to remove any residual halftone/text
    image = _force_white_interior(image, to_inpaint)

    debug_path = str(path.parent / "debug_inpainted.png")
    cv2.imwrite(debug_path, image)
    logger.info("Debug inpainted image saved → %s", debug_path)

    logger.info("Inpainting complete: %d bubble(s) processed", len(to_inpaint))
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
