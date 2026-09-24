"""Token accounting.

We need a token estimate that is (a) fast, (b) dependency-free and (c) close
enough for budget decisions.  ``tiktoken`` is used when it happens to be
installed, otherwise a calibrated character heuristic: the model context is
protected by the *provider-reported* usage anyway (see ``Usage``), the estimate
is only used to decide how much page text we may attach.
"""

from __future__ import annotations

import functools
import re

#: Rough chars-per-token ratios measured on mixed English/Russian web page text.
_CHARS_PER_TOKEN_LATIN = 3.9
_CHARS_PER_TOKEN_CYRILLIC = 2.4

try:  # optional fast path
    import tiktoken  # type: ignore

    @functools.lru_cache(maxsize=1)
    def _encoding():  # pragma: no cover - depends on optional dependency
        try:
            return tiktoken.get_encoding("o200k_base")
        except Exception:
            return None

except Exception:  # pragma: no cover
    tiktoken = None  # type: ignore
    _encoding = lambda: None  # type: ignore

_CYRILLIC = re.compile(r"[\u0400-\u04ff]")
_WORDISH = re.compile(r"\w+")


def count_tokens(text: str) -> int:
    """Estimate the number of tokens in ``text``."""
    if not text:
        return 0
    enc = _encoding()
    if enc is not None:  # pragma: no cover - optional dependency
        try:
            return len(enc.encode(text, disallowed_special=()))
        except Exception:
            pass
    cyrillic = len(_CYRILLIC.findall(text))
    latin = len(text) - cyrillic
    estimate = latin / _CHARS_PER_TOKEN_LATIN + cyrillic / _CHARS_PER_TOKEN_CYRILLIC
    # punctuation-dense markup costs extra tokens; words are a decent floor
    words = len(_WORDISH.findall(text))
    return max(int(estimate), int(words * 0.9), 1)


def truncate_to_tokens(text: str, max_tokens: int, *, marker: str = "…[truncated]") -> tuple[str, bool]:
    """Hard-truncate ``text`` so that ``count_tokens(result) <= max_tokens``.

    Returns ``(text, was_truncated)``.  Cuts on line boundaries when possible.
    """
    if max_tokens <= 0:
        return "", bool(text)
    if count_tokens(text) <= max_tokens:
        return text, False
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count_tokens(text[:mid]) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    cut = text[:lo]
    newline = cut.rfind("\n")
    if newline > len(cut) * 0.6:
        cut = cut[:newline]
    return cut.rstrip() + ("\n" + marker if marker else ""), True
