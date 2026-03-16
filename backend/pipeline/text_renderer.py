"""Render translated text into speech bubble regions."""

import logging
import textwrap
import unicodedata
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

BoundingBox = Tuple[int, int, int, int]

_FONTS_DIR = Path(__file__).resolve().parent.parent.parent / "fonts"

_FONT_CASCADE_LATIN = [
    _FONTS_DIR / "Bangers-Regular.ttf",
    _FONTS_DIR / "ComicNeue-Bold.ttf",
    _FONTS_DIR / "Outfit-ExtraBold.ttf",
    _FONTS_DIR / "Outfit-Bold.ttf",
]

# Per-script font cascades: variable fonts at Black (900) first, then static Bold
_VARIABLE_FONT_WEIGHT = 900  # Black weight for bold manga text

_SCRIPT_FONT_CASCADES = {
    "HANGUL": [
        (_FONTS_DIR / "NotoSansKR[wght].ttf", True),   # variable font
        _FONTS_DIR / "NotoSansKR-Black.ttf",
        _FONTS_DIR / "NotoSansKR-ExtraBold.ttf",
        _FONTS_DIR / "NotoSansKR-Bold.ttf",
    ],
    "CJK": [
        (_FONTS_DIR / "NotoSansSC[wght].ttf", True),
        (_FONTS_DIR / "NotoSansCJK[wght].ttf", True),
        _FONTS_DIR / "NotoSansSC-Black.ttf",
        _FONTS_DIR / "NotoSansSC-ExtraBold.ttf",
        _FONTS_DIR / "NotoSansSC-Bold.ttf",
    ],
    "ARABIC": [
        (_FONTS_DIR / "NotoSansArabic[wght].ttf", True),
        _FONTS_DIR / "NotoSansArabic-Black.ttf",
        _FONTS_DIR / "NotoSansArabic-ExtraBold.ttf",
        _FONTS_DIR / "NotoSansArabic-Bold.ttf",
    ],
    "DEVANAGARI": [
        (_FONTS_DIR / "NotoSansDevanagari[wght].ttf", True),
        _FONTS_DIR / "NotoSansDevanagari-Black.ttf",
        _FONTS_DIR / "NotoSansDevanagari-ExtraBold.ttf",
        _FONTS_DIR / "NotoSansDevanagari-Bold.ttf",
    ],
}

_SAFE_W_RATIO = 0.76
_SAFE_H_RATIO = 0.76
_MIN_FONT_SIZE = 8
_MAX_FONT_START = 72
_FONT_STEP = 1
_MAX_DIM_RATIO = 0.76
_WHITE_THRESH = 200
_TALL_ASPECT = 1.3
_TALL_GAP_MULT = 0.5
_NORMAL_GAP_MULT = 1.0

_resolved_latin_font = ""


def _detect_script(text: str) -> str:
    """Detect the dominant non-Latin script in the text.

    Returns one of: 'HANGUL', 'CJK', 'ARABIC', 'DEVANAGARI', or 'LATIN'.
    """
    counts = {"HANGUL": 0, "CJK": 0, "ARABIC": 0, "DEVANAGARI": 0, "LATIN": 0}

    for ch in text:
        if ch.isspace() or unicodedata.category(ch).startswith("P"):
            continue
        try:
            name = unicodedata.name(ch, "")
        except ValueError:
            continue

        if "HANGUL" in name:
            counts["HANGUL"] += 1
        elif "CJK" in name or "IDEOGRAPH" in name:
            counts["CJK"] += 1
        elif "ARABIC" in name:
            counts["ARABIC"] += 1
        elif "DEVANAGARI" in name:
            counts["DEVANAGARI"] += 1
        else:
            counts["LATIN"] += 1

    best = max(counts, key=counts.get)
    if counts[best] == 0:
        return "LATIN"
    return best


def _resolve_latin_font() -> str:
    """Walk the Latin font cascade and return the first valid path."""
    global _resolved_latin_font
    if _resolved_latin_font:
        return _resolved_latin_font
    for p in _FONT_CASCADE_LATIN:
        if p.is_file():
            try:
                ImageFont.truetype(str(p), 12)
                _resolved_latin_font = str(p)
                logger.info("Latin font: %s", p.name)
                return _resolved_latin_font
            except (OSError, IOError):
                continue
    logger.warning("No Latin fonts found — using PIL default")
    _resolved_latin_font = ""
    return _resolved_latin_font


def _font_for_text(text: str) -> Tuple[str, bool]:
    """Pick the boldest available font for the given text based on script detection.

    Returns:
        (font_path, is_variable) - is_variable True if font supports weight axis.
    """
    script = _detect_script(text)
    if script != "LATIN" and script in _SCRIPT_FONT_CASCADES:
        for entry in _SCRIPT_FONT_CASCADES[script]:
            if isinstance(entry, tuple):
                path, is_var = entry
            else:
                path, is_var = entry, False
            if path.is_file():
                logger.debug("Script=%s → font=%s (variable=%s)", script, path.name, is_var)
                return (str(path), is_var)
        logger.warning(
            "No fonts for script %s found (tried %s)",
            script,
            [e[0].name if isinstance(e, tuple) else e.name for e in _SCRIPT_FONT_CASCADES[script]],
        )
    return (_resolve_latin_font(), False)


def _load_font(
    font_path: str, size: int, is_variable: bool = False
) -> ImageFont.FreeTypeFont:
    """Load a TrueType font, optionally at Black weight for variable fonts."""
    if font_path:
        try:
            font = ImageFont.truetype(font_path, size)
            if is_variable:
                try:
                    axes = font.get_variation_axes()
                    if axes:
                        values = []
                        for a in axes:
                            name = (a.get("name") or a.get("tag") or b"").decode("utf-8", errors="ignore").lower()
                            default = a.get("default", 400)
                            values.append(_VARIABLE_FONT_WEIGHT if "weight" in name or "wght" in name else default)
                        if values:
                            font.set_variation_by_axes(values)
                except (OSError, TypeError, AttributeError, IndexError, UnicodeDecodeError):
                    pass
            return font
        except (OSError, IOError):
            pass
    resolved = _resolve_latin_font()
    if resolved:
        try:
            return ImageFont.truetype(resolved, size)
        except (OSError, IOError):
            pass
    return ImageFont.load_default()


def _stroke_for_size(font_size: int) -> int:
    """Scale stroke width proportionally to font size for bold manga-style text."""
    return max(2, font_size // 8)


def _measure_lines(
    font: ImageFont.FreeTypeFont, lines: List[str], gap: int,
) -> Tuple[int, int, List[Tuple[int, int]]]:
    """Return (max_line_width, total_height, [(lw, lh), ...])."""
    metrics = []
    max_w = 0
    for line in lines:
        lb = font.getbbox(line)
        lw, lh = lb[2] - lb[0], lb[3] - lb[1]
        metrics.append((lw, lh))
        if lw > max_w:
            max_w = lw
    total_h = sum(h for _, h in metrics) + max(0, len(lines) - 1) * gap
    return max_w, total_h, metrics


def _analyze_white_interior(
    gray_crop: np.ndarray,
) -> Tuple[np.ndarray, int, int, int, int, int, int]:
    """Analyze the white interior of a bubble crop.

    Returns:
        (white_mask, centroid_x, centroid_y, top_y, bottom_y, left_x, right_x)
        All coordinates are relative to the crop.
    """
    white_mask = (gray_crop >= _WHITE_THRESH).astype(np.uint8)
    ys, xs = np.where(white_mask)

    if len(ys) == 0:
        h, w = gray_crop.shape[:2]
        return white_mask, w // 2, h // 2, 0, h - 1, 0, w - 1

    cx = int(np.mean(xs))
    cy = int(np.mean(ys))
    top_y = int(np.min(ys))
    bot_y = int(np.max(ys))
    left_x = int(np.min(xs))
    right_x = int(np.max(xs))

    return white_mask, cx, cy, top_y, bot_y, left_x, right_x


def _get_white_hspan_at_row(
    white_mask: np.ndarray, row: int, half_h: int,
) -> Tuple[int, int]:
    """Find the leftmost and rightmost white pixel in a horizontal band.

    Scans a band of rows [row - half_h, row + half_h] to avoid
    single-pixel noise.
    """
    h, w = white_mask.shape[:2]
    r0 = max(0, row - half_h)
    r1 = min(h, row + half_h + 1)
    band = white_mask[r0:r1, :]

    col_hits = np.any(band, axis=0)
    cols = np.where(col_hits)[0]
    if len(cols) == 0:
        return 0, w - 1
    return int(cols[0]), int(cols[-1])


_SHOUT_CHARS = set("！！!?？")
_MIN_MARGIN_RATIO = 0.05


def _should_uppercase(text: str, bubble: dict) -> bool:
    """Heuristic: uppercase if the source had shout punctuation or is short + energetic."""
    jp = bubble.get("japanese_text", "")
    if any(c in _SHOUT_CHARS for c in jp):
        return True
    if text.endswith("!") or text.endswith("!!") or text.endswith("!!!"):
        return True
    return False


def _dynamic_line_gap(font_size: int, num_lines: int) -> int:
    """More lines = tighter gap to fit in the bubble."""
    base = font_size // 4
    if num_lines >= 4:
        return max(1, base // 2)
    if num_lines >= 3:
        return max(1, int(base * 0.7))
    return base


def fit_text_in_bubble(
    text: str,
    bbox: list,
    font_path: str = "",
) -> Tuple[ImageFont.FreeTypeFont, List[str]]:
    """Find the largest font size + wrapping that fits inside the bbox.

    Uses 80 % of bbox width/height as the safe zone.  Two-pass wrapping:
    first tries normal wrapping, then allows hyphenation for long words.
    """
    _, _, bw, bh = bbox
    safe_w = int(bw * _SAFE_W_RATIO)
    safe_h = int(bh * _SAFE_H_RATIO)

    if font_path:
        fpath, is_variable = font_path, False
    else:
        fpath, is_variable = _font_for_text(text)
    smaller_dim = min(bw, bh)
    hard_cap = int(smaller_dim * _MAX_DIM_RATIO)
    start_size = min(_MAX_FONT_START, hard_cap, max(bh // 2, _MIN_FONT_SIZE))

    best_font = _load_font(fpath, _MIN_FONT_SIZE, is_variable)
    best_lines = [text]

    for size in range(start_size, _MIN_FONT_SIZE - 1, -_FONT_STEP):
        font = _load_font(fpath, size, is_variable)

        sample = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnop"
        sample_bb = font.getbbox(sample)
        avg_char_w = max((sample_bb[2] - sample_bb[0]) / len(sample), 1)
        chars_per_line = max(int(safe_w / avg_char_w), 1)

        lines = textwrap.wrap(text, width=chars_per_line, break_long_words=False)
        if not lines:
            lines = textwrap.wrap(text, width=chars_per_line, break_long_words=True, break_on_hyphens=True)
        if not lines:
            lines = [text]

        line_gap = _dynamic_line_gap(size, len(lines))
        rendered_w, rendered_h, _ = _measure_lines(font, lines, line_gap)

        if rendered_w > safe_w or rendered_h > safe_h:
            continue

        best_font = font
        best_lines = lines
        break

    return best_font, best_lines


def render_text_on_bubble(image: np.ndarray, bubble: dict) -> np.ndarray:
    """Draw the English translation centered on the bubble's white interior.

    Instead of centering on the bbox, this finds the actual white area
    inside the bubble and centers text there. For each line it also
    finds the horizontal white span at that row so text stays inside
    irregular/spiky bubble shapes.

    Args:
        image:  Full page image as a BGR numpy array.
        bubble: Bubble dict containing ``"bbox"`` and ``"english_text"``.

    Returns:
        Image with the translated text rendered.
    """
    en_text = bubble.get("english_text", "")
    if not en_text or not en_text.strip():
        return image

    if _should_uppercase(en_text, bubble):
        en_text = en_text.upper()

    bbox = bubble["bbox"]
    bx, by, bw, bh = bbox

    is_dark = bubble.get("dark_bubble", False)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_crop = gray[by:by + bh, bx:bx + bw]

    if is_dark:
        # Dark narrator box: center on the full bbox (whole interior is black)
        cx_local, cy_local = bw // 2, bh // 2
        interior_w, interior_h = bw, bh
        left_x, top_y, right_x, bot_y = 0, 0, bw, bh
    else:
        # White speech bubble: find the actual white oval interior
        _, cx_local, cy_local, top_y, bot_y, left_x, right_x = _analyze_white_interior(gray_crop)
        interior_w = max(right_x - left_x, bw // 2)
        interior_h = max(bot_y - top_y, bh // 2)

    is_tall = interior_h > interior_w * _TALL_ASPECT

    font, lines = fit_text_in_bubble(en_text, [0, 0, interior_w, interior_h])
    font_size = font.size if hasattr(font, "size") else _MIN_FONT_SIZE
    line_gap = _dynamic_line_gap(font_size, len(lines))
    if is_tall:
        line_gap = max(1, int(line_gap * _TALL_GAP_MULT))

    margin_x = int(interior_w * _MIN_MARGIN_RATIO)
    margin_y = int(interior_h * _MIN_MARGIN_RATIO)

    _, total_h, line_metrics = _measure_lines(font, lines, line_gap)

    start_y_local = cy_local - total_h // 2
    start_y_local = max(top_y + margin_y, start_y_local)
    start_y_local = min(bot_y - total_h - margin_y, start_y_local)

    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    cursor_y_local = start_y_local

    # Dark box → white text with black stroke; light bubble → black text with white stroke
    text_fill = "white" if is_dark else "black"
    stroke_fill = "black" if is_dark else "white"

    for line, (lw, lh) in zip(lines, line_metrics):
        tx = bx + cx_local - lw // 2
        tx = max(tx, bx + left_x + margin_x)
        tx = min(tx, bx + right_x - lw - margin_x)
        ty = by + cursor_y_local

        sw = _stroke_for_size(font_size)
        draw.text(
            (tx, ty), line, font=font,
            fill=text_fill, stroke_width=sw, stroke_fill=stroke_fill,
        )
        cursor_y_local += lh + line_gap

    logger.info(
        "Bubble bbox=%s  dark=%s  interior=(%d,%d,%dx%d)  centroid=(%d,%d)  font_size=%d  lines=%d  text=%r",
        bbox, is_dark, left_x, top_y, interior_w, interior_h, cx_local, cy_local,
        font_size, len(lines), en_text,
    )

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def render_all_bubbles(image: np.ndarray, bubbles: list[dict]) -> np.ndarray:
    """Render English text into every bubble that has a translation.

    Args:
        image:   Full page image as a BGR numpy array (already inpainted).
        bubbles: List of bubble dicts, each optionally containing
                 ``"english_text"``.

    Returns:
        Final image with all translated text rendered.
    """
    logger.info("Rendering text into %d bubble(s)", len(bubbles))
    result = image.copy()

    rendered = 0
    for bubble in bubbles:
        if bubble.get("english_text", "").strip():
            result = render_text_on_bubble(result, bubble)
            rendered += 1

    logger.info("Rendering complete: %d bubble(s) rendered", rendered)
    return result


# ── Class-based API (used by main.py pipeline) ──────────────────────────────


class TextRenderer:
    """Draws translated text onto cleaned manga page bubbles."""

    @staticmethod
    def render(
        page: Image.Image,
        bubbles: List[BoundingBox],
        texts: List[str],
    ) -> Image.Image:
        """Render English text into each speech bubble.

        Args:
            page:    PIL Image with inpainted (text-removed) bubbles.
            bubbles: List of (x, y, w, h) bounding boxes.
            texts:   List of English strings to render, one per bubble.

        Returns:
            PIL Image with English text rendered into bubbles.
        """
        img = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)

        bubble_dicts = [
            {"bbox": [x, y, w, h], "english_text": t}
            for (x, y, w, h), t in zip(bubbles, texts)
        ]

        result = render_all_bubbles(img, bubble_dicts)
        return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    img_path = sys.argv[1] if len(sys.argv) > 1 else "sample_page.png"

    sample_bubbles = [
        {
            "bbox": [380, 32, 258, 399],
            "english_text": "Hmm, back then they were like barbarians.",
        },
        {
            "bbox": [15, 206, 220, 434],
            "english_text": "I was often assaulted too.",
        },
    ]

    try:
        image = cv2.imread(img_path)
        if image is None:
            raise RuntimeError(f"Cannot read {img_path}")
        result = render_all_bubbles(image, sample_bubbles)
        out = str(Path(img_path).parent / "debug_rendered.png")
        cv2.imwrite(out, result)
        print(f"Rendered → {out}")
    except Exception as err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)
