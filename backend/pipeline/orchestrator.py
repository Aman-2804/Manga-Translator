"""End-to-end manga translation pipeline orchestrator.

Accepts either a folder of page images or a PDF, translates every page,
and outputs a single translated PDF.
"""

import logging
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
from PIL import Image

from pipeline.bubble_detector import detect_bubbles
from pipeline.inpainter import inpaint_all_bubbles
from pipeline.text_segmentation import segment_page_safe
from pipeline.lang_detect import detect_source_language
from pipeline.ocr import extract_all_bubbles
from pipeline.text_renderer import render_all_bubbles
from pipeline.translator import translate_page_bubbles_contextually

logger = logging.getLogger(__name__)

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}


def _collect_page_images(input_path: str) -> List[str]:
    """Return sorted list of image file paths from a folder or a single image.

    Raises:
        FileNotFoundError: If the path doesn't exist.
        ValueError:        If no images are found.
    """
    p = Path(input_path).resolve()

    if not p.exists():
        raise FileNotFoundError(f"Input path does not exist: {p}")

    if p.is_file() and p.suffix.lower() in _IMAGE_EXTS:
        return [str(p)]

    if p.is_dir():
        images = sorted(
            f for f in p.iterdir()
            if f.is_file() and f.suffix.lower() in _IMAGE_EXTS
        )
        if not images:
            raise ValueError(f"No image files found in {p}")
        return [str(f) for f in images]

    raise ValueError(f"Unsupported input: {p} (expected a folder of images or a single image file)")


def _tag_dark_bubbles(image: np.ndarray, bubbles: list) -> None:
    """Set ``dark_bubble=True`` on each bubble whose interior is predominantly dark.

    Samples the inner 75% of the bubble bounding box to avoid border pixels
    and checks mean brightness.  Dark narrator boxes typically have mean < 80.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    for b in bubbles:
        bx, by, bw, bh = b["bbox"]
        x1, y1 = max(bx, 0), max(by, 0)
        x2, y2 = min(bx + bw, w), min(by + bh, h)
        roi = gray[y1:y2, x1:x2]

        # Shrink inward by ~12.5% on each side to avoid bubble border pixels
        mx = max(1, roi.shape[1] // 8)
        my = max(1, roi.shape[0] // 8)
        inner = roi[my:roi.shape[0] - my, mx:roi.shape[1] - mx]
        if inner.size == 0:
            inner = roi

        mean_brightness = float(np.mean(inner))
        b["dark_bubble"] = mean_brightness < 80
        if b["dark_bubble"]:
            logger.info("Dark bubble detected: bbox=%s  brightness=%.1f", b["bbox"], mean_brightness)


def _translate_page(
    image_path: str,
    output_path: str,
    on_step=None,
    target_lang: str = "en",
    source_lang: str = "ja",
    detected_lang_ref: list = None,
) -> str:
    """Run the full translation pipeline on a single page image.

    Args:
        on_step: Optional callback ``(step_label: str) -> None`` called
                 before each pipeline step so callers can track progress.
        target_lang: Target language code (e.g. 'en', 'fr', 'es').
        source_lang: User-selected source language; may be overridden by detection.
        detected_lang_ref: Optional list to store detected language (used on first page).

    Returns the absolute path to the saved translated image.
    """
    t0 = time.time()
    page_name = Path(image_path).name

    if on_step:
        on_step("Segmenting page")
    logger.info("── [%s] Segmenting…", page_name)
    img_for_seg = cv2.imread(image_path)
    text_mask = segment_page_safe(img_for_seg) if img_for_seg is not None else None

    if on_step:
        on_step("Detecting bubbles")
    logger.info("── [%s] Detecting bubbles…", page_name)
    bubbles = detect_bubbles(image_path, text_mask=text_mask)
    logger.info("── [%s] %d bubble(s) detected", page_name, len(bubbles))

    # Tag each bubble as dark (black narrator box) or light (white speech bubble)
    if img_for_seg is not None:
        _tag_dark_bubbles(img_for_seg, bubbles)

    if not bubbles:
        logger.warning("── [%s] No bubbles found — copying original", page_name)
        img = cv2.imread(image_path)
        cv2.imwrite(output_path, img)
        return str(Path(output_path).resolve())

    if on_step:
        on_step("Running OCR")
    logger.info("── [%s] Running OCR…", page_name)
    bubbles = extract_all_bubbles(image_path, bubbles)
    logger.info("── [%s] %d bubble(s) with text", page_name, len(bubbles))

    if not bubbles:
        logger.warning("── [%s] OCR found no text — copying original", page_name)
        img = cv2.imread(image_path)
        cv2.imwrite(output_path, img)
        return str(Path(output_path).resolve())

    # Detect source language on first page; reuse for subsequent pages
    if detected_lang_ref is not None and len(detected_lang_ref) > 0:
        effective_source = detected_lang_ref[0]
    else:
        effective_source = detect_source_language(bubbles, user_selection=source_lang)
        if detected_lang_ref is not None:
            detected_lang_ref.append(effective_source)

    if on_step:
        on_step("Translating text")
    logger.info("── [%s] Translating…", page_name)
    bubbles = translate_page_bubbles_contextually(
        bubbles, target_lang=target_lang, source_lang=effective_source
    )

    if on_step:
        on_step("Erasing Japanese text")
    logger.info("── [%s] Inpainting…", page_name)
    cleaned = inpaint_all_bubbles(image_path, bubbles, text_mask=text_mask)

    if on_step:
        on_step("Rendering translated text")
    logger.info("── [%s] Rendering translated text…", page_name)
    final = render_all_bubbles(cleaned, bubbles)

    cv2.imwrite(output_path, final)
    elapsed = time.time() - t0
    logger.info("── [%s] Done (%.1fs) → %s", page_name, elapsed, output_path)

    return str(Path(output_path).resolve())


def _images_to_pdf(image_paths: List[str], pdf_path: str) -> str:
    """Combine a list of images into a single PDF file.

    Returns the absolute path to the saved PDF.
    """
    pil_pages: List[Image.Image] = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        pil_pages.append(img)

    if not pil_pages:
        raise ValueError("No pages to combine into PDF")

    first, rest = pil_pages[0], pil_pages[1:]
    first.save(pdf_path, "PDF", save_all=True, append_images=rest, resolution=300)

    logger.info("PDF saved (%d pages) → %s", len(pil_pages), pdf_path)
    return str(Path(pdf_path).resolve())


_STEPS_PER_PAGE = 5


def run_pipeline(
    input_path: str,
    output_dir: str,
    pdf_filename: str = "translated.pdf",
    on_page_done=None,
    on_progress=None,
    target_lang: str = "en",
    source_lang: str = "ja",
) -> dict:
    """Run the full manga translation pipeline.

    Accepts a folder of page images (JPEG/PNG/WEBP/etc.) or a single
    image file.  Each page is processed through: bubble detection →
    OCR → translation → inpainting → text rendering.  The translated
    pages are combined into a single PDF.

    Args:
        input_path:   Path to a folder of page images, or a single
                      image file.
        output_dir:   Directory where translated page images and the
                      final PDF will be saved.
        pdf_filename: Name for the output PDF file.
        on_page_done: ``(idx, total, page_name) -> None`` called after
                      each page finishes.
        on_progress:  ``(pct: int, label: str) -> None`` called after
                      each pipeline step for fine-grained progress.

    Returns:
        Dict with keys:
            - ``"pages"``: list of absolute paths to translated page PNGs
            - ``"pdf"``:   absolute path to the combined PDF
            - ``"failed"``: list of page filenames that failed
            - ``"elapsed"``: total processing time in seconds
    """
    t_start = time.time()
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    logger.info("="*60)
    logger.info("PIPELINE START  input=%s  output=%s", input_path, out)
    logger.info("="*60)

    page_images = _collect_page_images(input_path)
    total = len(page_images)
    logger.info("Found %d page image(s)", total)

    translated_pages: List[str] = []
    original_pages: List[str] = []
    failed_pages: List[str] = []
    detected_lang_ref: List[str] = []

    for idx, img_path in enumerate(page_images, start=1):
        page_name = Path(img_path).name
        out_name = f"translated_page_{idx:03d}.png"
        out_path = str(out / out_name)

        # Save a copy of the original page so the frontend can show it
        orig_out = str(out / f"original_page_{idx:03d}.png")
        img_orig = cv2.imread(img_path)
        if img_orig is not None:
            cv2.imwrite(orig_out, img_orig)
            original_pages.append(orig_out)

        logger.info("━━ Page %d/%d: %s ━━", idx, total, page_name)

        step_counter = [0]

        def _make_step_cb(page_idx):
            def _step_cb(label):
                step_counter[0] += 1
                if on_progress:
                    pct = int(
                        ((page_idx - 1) * _STEPS_PER_PAGE + step_counter[0])
                        / (total * _STEPS_PER_PAGE) * 100
                    )
                    on_progress(
                        min(pct, 99),
                        f"Page {page_idx}/{total}: {label}",
                    )
            return _step_cb

        try:
            saved = _translate_page(
                img_path,
                out_path,
                on_step=_make_step_cb(idx),
                target_lang=target_lang,
                source_lang=source_lang,
                detected_lang_ref=detected_lang_ref,
            )
            translated_pages.append(saved)
        except Exception:
            logger.exception("FAILED on page %d (%s) — skipping", idx, page_name)
            failed_pages.append(page_name)

        if on_page_done:
            on_page_done(idx, total, page_name)

    if not translated_pages:
        raise RuntimeError("All pages failed — no output produced")

    pdf_path = str(out / pdf_filename)
    _images_to_pdf(translated_pages, pdf_path)

    elapsed = time.time() - t_start
    logger.info("="*60)
    logger.info(
        "PIPELINE COMPLETE  pages=%d  failed=%d  time=%.1fs",
        len(translated_pages), len(failed_pages), elapsed,
    )
    logger.info("PDF → %s", pdf_path)
    logger.info("="*60)

    return {
        "pages": translated_pages,
        "original_pages": original_pages,
        "pdf": str(Path(pdf_path).resolve()),
        "failed": failed_pages,
        "elapsed": elapsed,
        "detected_lang": detected_lang_ref[0] if detected_lang_ref else source_lang,
    }


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.orchestrator <image_folder_or_file> [output_dir]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "../outputs/translated"

    try:
        result = run_pipeline(input_path, output_dir)
        print(f"\nTranslated {len(result['pages'])} page(s)")
        if result["failed"]:
            print(f"Failed: {result['failed']}")
        print(f"PDF: {result['pdf']}")
        print(f"Time: {result['elapsed']:.1f}s")
    except Exception as err:
        print(f"\nError: {err}", file=sys.stderr)
        sys.exit(1)
