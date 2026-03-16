"""Translate Japanese text to English.

Supports two modes:
- **Contextual** (preferred): sends all bubbles on a page to Claude Haiku
  as a single prompt so the translation reads as a cohesive conversation.
- **Per-bubble fallback**: translates each bubble independently via Google
  Translate when no Anthropic API key is available.
"""

import json
import logging
import os
import re
import time
from typing import List, Optional

from deep_translator import GoogleTranslator

logger = logging.getLogger(__name__)

_SOURCE_LANG = "ja"
_TARGET_LANG = "en"
_MAX_RETRIES = 3
_RETRY_DELAY = 1.0
_RATE_LIMIT_DELAY = 0.1

_LANG_NAMES = {
    "en": "English",
    "fr": "French",
    "es": "Spanish",
    "zh-CN": "Chinese (Simplified)",
    "zh-TW": "Chinese (Traditional)",
    "ko": "Korean",
    "pt": "Portuguese",
    "de": "German",
    "hi": "Hindi",
    "ar": "Arabic",
    "ja": "Japanese",
}

_PUNCT_ONLY_RE = re.compile(
    r'^[\s。、．，.…・！？!?\-ー~～「」『』（）()\[\]【】\u3000○◯●◎]+$'
)

_JP_TO_EN_PUNCT = str.maketrans({
    '。': '.',
    '、': ',',
    '．': '.',
    '，': ',',
    '！': '!',
    '？': '?',
    '…': '...',
    '・': '...',
    '～': '~',
    '「': '"',
    '」': '"',
    '『': '"',
    '』': '"',
    '（': '(',
    '）': ')',
    '【': '[',
    '】': ']',
})


def _normalize_punct(text: str) -> str:
    """Convert Japanese punctuation to English equivalents."""
    return text.translate(_JP_TO_EN_PUNCT)


def _is_punct_only(text: str) -> bool:
    """Return True if text contains only punctuation/symbols, no real words."""
    return bool(_PUNCT_ONLY_RE.match(text.strip()))


# ── Class-based API (used by main.py pipeline) ──────────────────────────────


class Translator:
    """Translates Japanese text strings to English using Google Translate."""

    SOURCE_LANG = _SOURCE_LANG
    TARGET_LANG = _TARGET_LANG

    @staticmethod
    def translate(texts: List[str]) -> List[str]:
        """Translate a list of Japanese strings to English.

        Args:
            texts: List of Japanese text strings.

        Returns:
            List of English translations in the same order.
        """
        translated: list[str] = []

        for text in texts:
            result = translate_text(text)
            translated.append(result)
            if text.strip():
                time.sleep(_RATE_LIMIT_DELAY)

        return translated


# ── Standalone functions ─────────────────────────────────────────────────────


def translate_text(
    text: str,
    source: str = _SOURCE_LANG,
    target: str = _TARGET_LANG,
) -> str:
    """Translate a single string with retry logic.

    Retries up to 3 times on failure with a 1-second back-off between
    attempts.  Returns the original text unchanged if all retries are
    exhausted.

    Args:
        text:   Input string (typically Japanese).
        source: Source language code.
        target: Target language code.

    Returns:
        Translated string, or the original text if translation fails.
    """
    if not text or not text.strip():
        return ""

    if _is_punct_only(text):
        return _normalize_punct(text).strip()

    translator = GoogleTranslator(source=source, target=target)

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            result = translator.translate(text)
            return result.strip() if result else text
        except Exception as exc:
            logger.warning(
                "Translation attempt %d/%d failed: %s",
                attempt, _MAX_RETRIES, exc,
            )
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_DELAY)

    logger.error("All %d translation attempts failed — returning original text", _MAX_RETRIES)
    return text


def _postprocess_translation(text: str) -> str:
    """Clean up common Google Translate artifacts."""
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    words = text.split()
    if len(words) <= 2 and text.endswith('!'):
        text = text.upper()
    return text


def translate_all_bubbles(
    bubbles: list[dict],
    target_lang: str = "en",
    source_lang: str = _SOURCE_LANG,
) -> list[dict]:
    """Translate the source-language text in every bubble dict.

    For each bubble that contains a non-empty ``"japanese_text"`` key,
    ``translate_text`` is called and the result is stored under
    ``"english_text"``.

    Args:
        bubbles: List of bubble dicts, each expected to have a
                 ``"japanese_text"`` key.
        target_lang: Target language code (e.g. 'en', 'fr', 'es').
        source_lang: Source language code (e.g. 'ja', 'ko', 'zh-CN').

    Returns:
        The same list with an ``"english_text"`` key added to each
        bubble that had translatable text.
    """
    lang_name = _LANG_NAMES.get(target_lang, target_lang)
    logger.info("Translating %d bubble(s) to %s…", len(bubbles), lang_name)

    for idx, bubble in enumerate(bubbles):
        jp_text = bubble.get("japanese_text", "")

        if not jp_text or not jp_text.strip():
            bubble["english_text"] = ""
            logger.debug("Bubble #%d: empty japanese_text — skipped", idx)
            continue

        en_text = _postprocess_translation(
            translate_text(jp_text, source=source_lang, target=target_lang)
        )
        bubble["english_text"] = en_text

        logger.info('Bubble #%d: "%s" → "%s"', idx, jp_text, en_text)

        if idx < len(bubbles) - 1:
            time.sleep(_RATE_LIMIT_DELAY)

    translated_count = sum(1 for b in bubbles if b.get("english_text", "").strip())
    logger.info(
        "Translation complete: %d / %d bubble(s) translated",
        translated_count, len(bubbles),
    )
    return bubbles


# ── Contextual translation (Claude Haiku) ────────────────────────────────────

_ANTHROPIC_MODEL = "claude-3-haiku-20240307"


def _sort_manga_reading_order(bubbles: list[dict]) -> list[dict]:
    """Sort bubbles in manga reading order: right-to-left, top-to-bottom."""
    return sorted(
        bubbles,
        key=lambda b: (-b["bbox"][0], b["bbox"][1]),
    )


def _call_claude(
    japanese_texts: list[str],
    target_lang: str = "en",
    source_lang: str = _SOURCE_LANG,
) -> Optional[list[str]]:
    """Send all bubble texts to Claude Haiku and return translated list."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None

    try:
        import anthropic
    except ImportError:
        logger.warning("anthropic package not installed — falling back")
        return None

    source_name = _LANG_NAMES.get(source_lang, source_lang)
    target_name = _LANG_NAMES.get(target_lang, target_lang)
    prompt = (
        f"You are translating manga speech bubbles from {source_name} to {target_name}.\n\n"
        "Rules:\n"
        "- These bubbles are from ONE manga page, listed in reading order "
        "(right-to-left, top-to-bottom). Translate them as a cohesive "
        "conversation, not individually.\n"
        "- Keep translations short and punchy — speech bubbles have limited space.\n"
        "- Shouts/exclamations should be energetic and uppercase "
        '(e.g. "YOU BASTARD!" not "You bastard").\n'
        "- Sound effects and onomatopoeia (ドドド, バキ, ゴゴゴ, ザワザワ) should "
        "be transliterated to uppercase romaji, NOT translated "
        "(e.g. DODODO, BAKI, GOGOGO, ZAWAZAWA).\n"
        "- Short reactions (えっ, はっ, うん, ああ) should become natural "
        f"{target_name} equivalents (Huh?, Hah!, Yeah, Ahh).\n"
        "- Preserve the original energy and tone — don't make it formal or stiff.\n"
        "- If a bubble is clearly a continuation of the previous one, make them "
        "flow naturally together.\n\n"
        "Return ONLY a JSON array of translated strings in the same order as "
        "input. No explanation, no markdown, just the JSON array.\n\n"
        f"Bubbles: {json.dumps(japanese_texts, ensure_ascii=False)}"
    )

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=_ANTHROPIC_MODEL,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()

        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

        translations = json.loads(raw)

        if not isinstance(translations, list):
            logger.error("Claude returned non-list: %s", type(translations))
            return None
        if len(translations) != len(japanese_texts):
            logger.warning(
                "Claude returned %d items, expected %d — falling back",
                len(translations), len(japanese_texts),
            )
            return None

        return [str(t) for t in translations]

    except Exception as exc:
        logger.error("Claude contextual translation failed: %s", exc)
        return None


def translate_page_bubbles_contextually(
    bubbles: list[dict],
    target_lang: str = "en",
    source_lang: str = _SOURCE_LANG,
) -> list[dict]:
    """Translate all bubbles on a page as a cohesive conversation.

    Sorts bubbles into manga reading order, sends all source-language text to
    Claude Haiku in a single prompt, and maps translations back. Falls
    back to per-bubble Google Translate if no API key is set or the
    call fails.

    Args:
        bubbles: List of bubble dicts with ``"japanese_text"`` keys.
        target_lang: Target language code (e.g. 'en', 'fr', 'es').
        source_lang: Source language code (e.g. 'ja', 'ko', 'zh-CN').

    Returns:
        Same list with ``"english_text"`` added to each bubble.
    """
    if not bubbles:
        return bubbles

    target_name = _LANG_NAMES.get(target_lang, target_lang)

    translatable = [
        b for b in bubbles
        if b.get("japanese_text", "").strip()
        and not _is_punct_only(b["japanese_text"])
    ]
    punct_only = [
        b for b in bubbles
        if b.get("japanese_text", "").strip()
        and _is_punct_only(b["japanese_text"])
    ]

    for b in punct_only:
        b["english_text"] = _normalize_punct(b["japanese_text"]).strip()

    if not translatable:
        for b in bubbles:
            b.setdefault("english_text", "")
        return bubbles

    sorted_bubbles = _sort_manga_reading_order(translatable)
    jp_texts = [b["japanese_text"] for b in sorted_bubbles]

    logger.info(
        "Attempting contextual translation of %d bubble(s) to %s via Claude…",
        len(jp_texts), target_name,
    )

    translations = _call_claude(jp_texts, target_lang=target_lang, source_lang=source_lang)

    if translations is not None:
        for bubble, en_text in zip(sorted_bubbles, translations):
            bubble["english_text"] = en_text
            logger.info('Contextual: "%s" → "%s"', bubble["japanese_text"], en_text)
    else:
        logger.info("Falling back to per-bubble Google Translate")
        for idx, bubble in enumerate(translatable):
            en_text = _postprocess_translation(
                translate_text(
                    bubble["japanese_text"],
                    source=source_lang,
                    target=target_lang,
                )
            )
            bubble["english_text"] = en_text
            logger.info('Bubble #%d: "%s" → "%s"', idx, bubble["japanese_text"], en_text)
            if idx < len(translatable) - 1:
                time.sleep(_RATE_LIMIT_DELAY)

    for b in bubbles:
        b.setdefault("english_text", "")

    translated_count = sum(1 for b in bubbles if b.get("english_text", "").strip())
    logger.info("Contextual translation done: %d / %d bubble(s)", translated_count, len(bubbles))
    return bubbles


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        print(f"Input:  {text}")
        print(f"Output: {translate_text(text)}")
    else:
        sample_bubbles = [
            {"bbox": [100, 100, 200, 80], "japanese_text": "うむ まぁ当時は"},
            {"bbox": [400, 300, 180, 90], "japanese_text": "ワシもよく乱暴されたものじゃ"},
            {"bbox": [200, 500, 150, 60], "japanese_text": ""},
        ]
        results = translate_all_bubbles(sample_bubbles)
        print(f"\nTranslated {len(results)} bubble(s):")
        for i, b in enumerate(results):
            print(f'  #{i}  JP="{b.get("japanese_text", "")}"  EN="{b.get("english_text", "")}"')
