"""Deterministic language identification for article bodies."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from langdetect import DetectorFactory, detect_langs
from langdetect.lang_detect_exception import LangDetectException

DetectorFactory.seed = 0  # langdetect is randomised by default

_LETTER = re.compile(r"[^\W\d_]")
_CYR = re.compile(r"[А-Яа-яЁёІіЇїЄєҐґ]")
_LAT = re.compile(r"[A-Za-z]")
_UK_ONLY = re.compile(r"[ІіЇїЄєҐґ]")
_RU_ONLY = re.compile(r"[ЫыЭэЁёЪъ]")
MIN_LETTERS = 20


def _detect_block(text: str) -> str | None:
    if len(_LETTER.findall(text)) < MIN_LETTERS:
        return None
    try:
        lang = detect_langs(text)[0].lang
    except LangDetectException:
        return None
    # langdetect occasionally confuses uk/ru/bg/mk; letter evidence wins.
    if lang in ("ru", "bg", "mk") and len(_UK_ONLY.findall(text)) > len(_RU_ONLY.findall(text)):
        lang = "uk"
    return lang


def detect_language(body_text: str, prose_blocks: list[str] | None = None) -> dict[str, Any]:
    """Return {"language": ISO 639-1 code | "unknown", "scores": {...}}.

    Each prose block (IOC/code blocks excluded) is classified separately and the
    article language is the one covering most letters. ``scores.shares`` keeps
    the per-language proportion so bilingual articles (e.g. an English note in
    a Ukrainian report) can be identified.
    """
    blocks = prose_blocks if prose_blocks is not None else body_text.split("\n\n")
    weights: Counter[str] = Counter()
    for block in blocks:
        lang = _detect_block(block)
        if lang:
            weights[lang] += len(_LETTER.findall(block))
    prose = "\n".join(blocks)
    cyr, lat = len(_CYR.findall(prose)), len(_LAT.findall(prose))
    scores: dict[str, Any] = {"cyrillic_ratio": round(cyr / (cyr + lat), 3) if cyr + lat else None}
    if not weights:
        lang = _detect_block(prose)
        return {"language": lang or "unknown", "scores": scores}
    total = sum(weights.values())
    scores["shares"] = {k: round(v / total, 3) for k, v in weights.most_common()}
    return {"language": weights.most_common(1)[0][0], "scores": scores}
