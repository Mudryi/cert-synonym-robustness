"""Section-aware text chunks for embedding / retrieval experiments.

Chunks never cross section boundaries. Within a section, text is packed by
paragraph; paragraphs longer than ``chunk_size`` are split at sentence
boundaries, and only as a last resort at whitespace. Technical identifiers
(IPs, domains, hashes, versions) are never split: sentence boundaries require
whitespace after the punctuation and a following capital letter / digit /
quote, and the hard fallback cuts only at whitespace.
"""

from __future__ import annotations

import re
from typing import Any, Iterator

from . import config, storage

# ". " / "! " / "? " / "… " followed by an upper-case letter, digit, quote or bracket.
_SENT_END = re.compile(r"(?<=[.!?…])\s+(?=[\"«“(\[A-ZА-ЯІЇЄҐ0-9])")
# Common Ukrainian/English abbreviations that end with a dot but do not end a sentence.
_ABBREV = re.compile(r"(?:\b(?:т\.\s?ч|т\.\s?д|т\.\s?п|ст|п|рис|Рис|див|напр|тис|млн|млрд|грн|р|рр|с|e\.g|i\.e|etc|vs|Mr|Dr|No))\.$")


def _sentences(text: str) -> list[tuple[int, int]]:
    spans, start = [], 0
    for m in _SENT_END.finditer(text):
        if _ABBREV.search(text[start:m.start()]):
            continue
        spans.append((start, m.start()))
        start = m.end()
    spans.append((start, len(text)))
    return [(s, e) for s, e in spans if text[s:e].strip()]


def _hard_split(text: str, base: int, size: int) -> list[tuple[int, int]]:
    """Split an over-long piece at whitespace (never inside a token)."""
    out, start = [], 0
    while len(text) - start > size:
        cut = text.rfind(" ", start, start + size)
        nl = text.rfind("\n", start, start + size)
        cut = max(cut, nl)
        if cut <= start:  # a single token longer than size: keep it whole
            m = re.compile(r"\s").search(text, start + size)
            cut = m.start() if m else len(text)
        out.append((base + start, base + cut))
        start = cut + 1
    if text[start:].strip():
        out.append((base + start, base + len(text)))
    return out


def _units(body: str, start: int, end: int, size: int) -> list[tuple[int, int]]:
    """Paragraph-level units (absolute offsets), split further when too long."""
    units: list[tuple[int, int]] = []
    pos = start
    for para in body[start:end].split("\n\n"):
        p_start, p_end = pos, pos + len(para)
        pos = p_end + 2
        if not para.strip():
            continue
        if len(para) <= size:
            units.append((p_start, p_end))
            continue
        # IOC listings are line-based; sentences elsewhere.
        pieces = _sentences(para) if "\n" not in para else [
            (m.start(), m.end()) for m in re.finditer(r"[^\n]+", para)
        ]
        for s, e in pieces:
            if e - s <= size:
                units.append((p_start + s, p_start + e))
            else:
                units.extend(_hard_split(para[s:e], p_start + s, size))
    return units


def chunk_article(record: dict[str, Any], chunk_size: int, overlap: int = 0) -> Iterator[dict[str, Any]]:
    body = record["body_text"]
    sections = record["sections"] or [{"heading": None, "path": [], "start_char": 0,
                                       "content_start_char": 0, "end_char": len(body)}]
    index = 0
    for sec in sections:
        units = _units(body, sec["content_start_char"], sec["end_char"], chunk_size)
        i = 0
        while i < len(units):
            start, end = units[i]
            j = i + 1
            while j < len(units) and units[j][1] - start <= chunk_size:
                end = units[j][1]
                j += 1
            text = body[start:end]
            yield {
                "chunk_id": f"{record['article_id']}-{index:04d}",
                "article_id": record["article_id"],
                "chunk_index": index,
                "section": sec["heading"],
                "section_path": sec.get("path", []) + ([sec["heading"]] if sec["heading"] else []),
                "text": text,
                "start_char": start,
                "end_char": end,
                "char_count": len(text),
                "language": record["language"],
                "title": record["title"],
                "publication_date": record["publication_date"],
                "url": record["url"],
            }
            index += 1
            if j >= len(units):
                break
            # Overlap: step back over whole units covering at most `overlap` characters.
            k = j
            if overlap > 0:
                while k - 1 > i and units[j - 1][1] - units[k - 1][0] <= overlap:
                    k -= 1
            i = k


def write_chunks(chunk_size: int = config.DEFAULT_CHUNK_SIZE, overlap: int = config.DEFAULT_CHUNK_OVERLAP) -> int:
    if chunk_size < 100:
        raise ValueError("chunk_size must be at least 100 characters")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    def gen() -> Iterator[dict[str, Any]]:
        for record in storage.read_jsonl(config.ARTICLES_PATH):
            for chunk in chunk_article(record, chunk_size, overlap):
                chunk["chunk_size_setting"] = chunk_size
                yield chunk

    return storage.write_jsonl(config.CHUNKS_PATH, gen())
