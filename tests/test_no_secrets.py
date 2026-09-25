"""The public repository must not contain credentials.

This is a guard-rail, not a scanner: it fails the build when something that looks
like an API key, a token or a private key appears in a file that git tracks.  The
only exemption is a developer's own local ``.env`` (ignored by git, and asserted
to stay that way below).

Run: ``pytest tests/test_no_secrets.py``
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: Content that must never be committed.  Each pattern is deliberately loose.
PATTERNS = {
    "openai-style key": re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
    "github token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    "aws access key id": re.compile(r"\bAKIA[0-9A-Z]{12,}\b"),
    "slack token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    "private key block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "key assigned in a tracked file": re.compile(
        r"(?i)\b(api[_-]?key|access[_-]?token|client[_-]?secret)\s*[=:]\s*[\"']?[A-Za-z0-9_\-]{24,}"
    ),
}

#: Directories and binary artefacts that are not source.
SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", "node_modules", "out"}
SKIP_SUFFIXES = {".mp4", ".webm", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".woff", ".woff2"}

#: A local .env is the developer's own file: it is ignored by git (checked below)
#: and never part of a commit, so its contents are not the repository's problem.
SKIP_NAMES = {".env", ".env.local"}

#: Values that look like a key but are obviously synthetic (test fixtures use them
#: to prove that a key never reaches the trace).  Keeping them out of the scan
#: avoids renaming half the test-suite for the sake of a regex.
PLACEHOLDER_MARKERS = (
    "fake", "dummy", "test", "example", "placeholder", "redacted",
    "not-a-key", "xxxxxxxx", "0000000000", "your-key", "changeme",
)


def _is_placeholder(value: str) -> bool:
    lowered = value.lower()
    return any(marker in lowered for marker in PLACEHOLDER_MARKERS)


def _tracked_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        if set(rel.parts) & SKIP_DIRS:
            continue
        if path.suffix.lower() in SKIP_SUFFIXES or path.name in SKIP_NAMES:
            continue
        files.append(path)
    return files


@pytest.mark.parametrize("label", sorted(PATTERNS), ids=lambda s: s.replace(" ", "-"))
def test_no_credentials_in_the_repository(label: str):
    pattern = PATTERNS[label]
    hits: list[str] = []
    for path in _tracked_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):  # not text, or unreadable: not our business
            continue
        for number, line in enumerate(text.splitlines(), 1):
            for match in pattern.finditer(line):
                if _is_placeholder(match.group(0)):
                    continue
                hits.append(f"{path.relative_to(ROOT)}:{number}")
    assert not hits, f"{label} found in tracked files: {hits}"


def test_local_env_files_are_gitignored():
    """A developer's own secrets live in .env — which must never be committable."""
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    lines = {line.strip() for line in ignore.splitlines()}
    assert ".env" in lines
    assert not (ROOT / ".env.example").read_text(encoding="utf-8").count("sk-")


def test_no_absolute_developer_paths_in_docs():
    """Docs must not bake in somebody's home directory or private scratch space."""
    forbidden = ("/home/zonom", "cache/scratch", ".hermes/")
    hits: list[str] = []
    for path in sorted((ROOT / "docs").glob("*.md")) + [ROOT / "README.md"]:
        text = path.read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), 1):
            if any(token in line for token in forbidden):
                hits.append(f"{path.relative_to(ROOT)}:{number}")
    assert not hits, f"private paths leaked into the docs: {hits}"
