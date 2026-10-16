"""JSON structure validation helper."""

from __future__ import annotations

import json
from typing import Any


def validate_json(data: str | dict[str, Any], required_keys: set[str] | None = None) -> list[str]:
    try:
        parsed = json.loads(data) if isinstance(data, str) else data
    except json.JSONDecodeError as error:
        return [f"Invalid JSON: {error.msg}"]
    if not isinstance(parsed, dict):
        return ["JSON root must be an object."]
    return [f"Missing required key: {key}" for key in sorted(required_keys or set()) if key not in parsed]
