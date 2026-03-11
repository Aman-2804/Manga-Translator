#!/usr/bin/env python3
"""Download bold fonts for manga text rendering.

Fetches variable fonts (Black weight) and static Bold from Google Fonts.
Run from project root:

    python scripts/download_fonts.py
"""

import urllib.parse
import urllib.request
from pathlib import Path

FONTS_DIR = Path(__file__).resolve().parent.parent / "fonts"
BASE_URL = "https://github.com/google/fonts/raw/main/ofl"

FONTS_TO_DOWNLOAD = [
    # Variable fonts (used at Black/900 weight for bold manga text)
    ("notosanskr", "NotoSansKR[wght].ttf"),
    ("notosanssc", "NotoSansSC[wght].ttf"),
    ("notosansarabic", "NotoSansArabic[wght].ttf"),
    ("notosansdevanagari", "NotoSansDevanagari[wght].ttf"),
    # Static Bold fallbacks
    ("notosanskr", "NotoSansKR-Bold.ttf"),
    ("notosanssc", "NotoSansSC-Bold.ttf"),
    ("notosansarabic", "NotoSansArabic-Bold.ttf"),
    ("notosansdevanagari", "NotoSansDevanagari-Bold.ttf"),
    # Latin (comic-style)
    ("bangers", "Bangers-Regular.ttf"),
    ("comicneue", "ComicNeue-Bold.ttf"),
    ("outfit", "Outfit-ExtraBold.ttf"),
    ("outfit", "Outfit-Bold.ttf"),
]


def main():
    FONTS_DIR.mkdir(parents=True, exist_ok=True)

    for folder, filename in FONTS_TO_DOWNLOAD:
        dest = FONTS_DIR / filename
        if dest.is_file():
            print(f"Skip (exists): {filename}")
            continue

        url = f"{BASE_URL}/{folder}/{urllib.parse.quote(filename)}"
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"Downloaded: {filename}")
        except Exception as e:
            print(f"Failed {filename}: {e}")


if __name__ == "__main__":
    main()
