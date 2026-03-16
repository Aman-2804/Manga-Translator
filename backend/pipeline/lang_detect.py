"""Detect source language from OCR-extracted text in manga bubbles."""

import logging
from typing import List

logger = logging.getLogger(__name__)

# Map langdetect codes to our internal language codes
_LANGDETECT_TO_INTERNAL = {
    "ja": "ja",
    "ko": "ko",
    "zh-cn": "zh-CN",
    "zh-tw": "zh-TW",
    "zh": "zh-CN",  # fallback when variant not specified
}

_SUPPORTED_SOURCE_LANGS = {"ja", "ko", "zh-CN", "zh-TW"}


def detect_source_language(
    bubbles: List[dict],
    user_selection: str = "ja",
    min_text_length: int = 5,
) -> str:
    """Detect source language from OCR-extracted text in bubbles.

    Concatenates all ``japanese_text`` (or equivalent) from bubbles and runs
    langdetect. If detection is confident and differs from user selection,
    returns the detected language; otherwise keeps user's choice.

    Args:
        bubbles: List of bubble dicts with ``japanese_text`` key.
        user_selection: User-selected source language code.
        min_text_length: Minimum total character length to attempt detection.

    Returns:
        Language code (ja, ko, zh-CN, zh-TW).
    """
    try:
        from langdetect import detect, DetectorFactory

        DetectorFactory.seed = 0  # reproducible results
    except ImportError:
        logger.warning("langdetect not installed — using user selection")
        return user_selection if user_selection in _SUPPORTED_SOURCE_LANGS else "ja"

    combined = " ".join(
        b.get("japanese_text", "") or ""
        for b in bubbles
        if b.get("japanese_text", "").strip()
    ).strip()

    if len(combined) < min_text_length:
        logger.debug("Insufficient text for detection (%d chars) — using %s", len(combined), user_selection)
        return user_selection if user_selection in _SUPPORTED_SOURCE_LANGS else "ja"

    try:
        detected = detect(combined)
        internal = _LANGDETECT_TO_INTERNAL.get(detected.lower(), detected)
    except Exception as exc:
        logger.warning("Language detection failed: %s — using %s", exc, user_selection)
        return user_selection if user_selection in _SUPPORTED_SOURCE_LANGS else "ja"

    if internal not in _SUPPORTED_SOURCE_LANGS:
        logger.debug("Detected %s not in supported list — using %s", internal, user_selection)
        return user_selection if user_selection in _SUPPORTED_SOURCE_LANGS else "ja"

    if internal != user_selection:
        logger.info("Auto-detected source language: %s (user had %s)", internal, user_selection)

    return internal
