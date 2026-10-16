"""Simple language-code validation helper."""

from __future__ import annotations

import re


def validate_language_code(language_code: str) -> bool:
    return bool(re.fullmatch(r"[a-z]{2,3}(?:-[A-Z]{2})?", language_code))
