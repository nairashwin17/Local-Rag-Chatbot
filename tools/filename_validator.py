"""Filename comparison helper, ready for Localization QA."""

from __future__ import annotations


def validate_filenames(expected_assets: set[str], actual_files: set[str]) -> dict[str, list[str]]:
    return {"missing": sorted(expected_assets - actual_files), "unexpected": sorted(actual_files - expected_assets)}
