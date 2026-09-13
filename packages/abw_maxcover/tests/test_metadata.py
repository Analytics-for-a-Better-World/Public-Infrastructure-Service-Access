"""Tests that keep package metadata consistent with the code."""

from __future__ import annotations

import re
from importlib import metadata
from pathlib import Path

import pytest

import abw_maxcover

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def test_version_string_is_pep440_like() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+([ab]|rc)?\d*", abw_maxcover.__version__)


def test_installed_distribution_version_matches_package_attribute() -> None:
    try:
        installed = metadata.version("abw-maxcover")
    except metadata.PackageNotFoundError:
        pytest.skip("abw-maxcover is not installed as a distribution")
    assert installed == abw_maxcover.__version__


def test_citation_file_version_matches_package_attribute() -> None:
    citation = PACKAGE_ROOT / "CITATION.cff"
    if not citation.exists():
        pytest.skip("CITATION.cff is not shipped in this layout")
    match = re.search(r"^version:\s*(\S+)\s*$", citation.read_text(encoding="utf-8"), re.M)
    assert match is not None, "CITATION.cff has no version field"
    assert match.group(1) == abw_maxcover.__version__


def test_changelog_mentions_the_current_version() -> None:
    changelog = PACKAGE_ROOT / "CHANGELOG.md"
    if not changelog.exists():
        pytest.skip("CHANGELOG.md is not shipped in this layout")
    text = changelog.read_text(encoding="utf-8")
    assert f"## {abw_maxcover.__version__} - " in text
