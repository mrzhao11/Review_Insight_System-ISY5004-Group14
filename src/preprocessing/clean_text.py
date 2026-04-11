"""Text cleaning utilities for review preprocessing."""

from __future__ import annotations

import html
import re

HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str | None) -> str:
    """Return a lightly normalized review string."""
    if not text:
        return ""

    normalized = html.unescape(text)
    normalized = HTML_TAG_PATTERN.sub(" ", normalized)
    normalized = URL_PATTERN.sub(" ", normalized)
    normalized = normalized.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    normalized = WHITESPACE_PATTERN.sub(" ", normalized)
    return normalized.strip()
