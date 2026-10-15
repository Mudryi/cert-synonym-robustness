"""Convert a cached CERT-UA API article into a structured dataset record.

The article body comes from the API as Quill-editor HTML (``p, strong, em, br,
ul/ol, h3, pre.ql-syntax, img, a``). We walk it in document order and emit
blocks (heading / paragraph / list / table / code / image), from which we build:

* ``body_text``      plain text, blocks separated by blank lines
* ``body_markdown``  the same content with headings, lists, tables, code fences
* ``sections``       heading-delimited spans with offsets into ``body_text``

Content is never translated or rewritten; only Unicode (NFC) and whitespace are
normalised.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup, NavigableString, Tag

from . import PARSER_VERSION, config
from .entities import extract_entities
from .langid import detect_language

# ---------------------------------------------------------------- text normalisation

_SPACE_CHARS = "             　\t\f\v"
_SPACE_TABLE = {ord(c): " " for c in _SPACE_CHARS}
_DROP_TABLE = {ord(c): None for c in "​﻿­⁠"}
_MULTI_SPACE = re.compile(r" {2,}")


def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_DROP_TABLE).translate(_SPACE_TABLE)
    return text.replace("\r\n", "\n").replace("\r", "\n")


def normalize_inline(text: str) -> str:
    """Prose: collapse whitespace runs inside lines, trim lines, drop empty lines."""
    text = normalize_unicode(text)
    lines = [_MULTI_SPACE.sub(" ", ln).strip() for ln in text.split("\n")]
    return "\n".join(ln for ln in lines if ln)


def normalize_pre(text: str) -> str:
    """Preformatted (IOC) blocks: keep line structure and column spacing, trim line ends."""
    text = normalize_unicode(text)
    lines = [ln.rstrip() for ln in text.split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def html_to_plain(html: str | None) -> str | None:
    if not html:
        return None
    soup = BeautifulSoup(html, "lxml")
    for br in soup.find_all("br"):
        br.replace_with("\n")
    parts = [normalize_inline(p.get_text()) for p in soup.find_all(["p", "li"])] or [normalize_inline(soup.get_text())]
    text = "\n\n".join(p for p in parts if p)
    return text or None


# ---------------------------------------------------------------- blocks

@dataclass
class Block:
    kind: str  # heading | paragraph | list | table | code | image
    text: str = ""  # plain-text rendering
    markdown: str = ""
    level: int = 0  # heading level


@dataclass
class _Ctx:
    blocks: list[Block] = field(default_factory=list)
    links: list[dict[str, str]] = field(default_factory=list)
    images: list[str] = field(default_factory=list)
    hidden_blocks: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


HEADING_TAGS = {f"h{i}" for i in range(1, 7)}
BLOCK_TAGS = HEADING_TAGS | {"p", "div", "ul", "ol", "pre", "table", "blockquote", "figure", "section", "article", "hr"}
MAX_PSEUDO_HEADING_LEN = 120
MAX_PSEUDO_HEADING_WORDS = 8


def _is_hidden(tag: Tag) -> bool:
    classes = tag.get("class") or []
    style = (tag.get("style") or "").replace(" ", "").lower()
    return "hidden" in classes or "display:none" in style or tag.has_attr("hidden")


def _inline(node: Tag | NavigableString, ctx: _Ctx) -> tuple[str, str]:
    """Render inline content -> (plain, markdown)."""
    if isinstance(node, NavigableString):
        if node.__class__.__name__ in ("Comment", "ProcessingInstruction", "Doctype"):
            return "", ""
        s = str(node)
        return s, s
    name = node.name
    if name == "br":
        return "\n", "  \n"
    if name == "img":
        src = node.get("src") or ""
        ctx.images.append(src)
        return "", "[image]"
    if name in ("script", "style", "noscript", "iframe"):
        return "", ""
    if _is_hidden(node):
        return "", ""
    plain_parts, md_parts = [], []
    for child in node.children:
        p, m = _inline(child, ctx)
        plain_parts.append(p)
        md_parts.append(m)
    plain, md = "".join(plain_parts), "".join(md_parts)
    if name in ("strong", "b") and md.strip():
        md = _wrap(md, "**")
    elif name in ("em", "i") and md.strip():
        md = _wrap(md, "*")
    elif name in ("code", "kbd", "samp") and md.strip():
        md = _wrap(md, "`")
    elif name == "a":
        href = (node.get("href") or "").strip()
        if href:
            ctx.links.append({"href": href, "text": normalize_inline(plain)})
            if md.strip() and not href.lower().startswith("javascript:"):
                md = f"[{md.strip()}]({href})"
    return plain, md


def _wrap(md: str, marker: str) -> str:
    """Wrap inline markdown keeping surrounding whitespace outside the markers."""
    lead = md[: len(md) - len(md.lstrip())]
    trail = md[len(md.rstrip()):]
    return f"{lead}{marker}{md.strip()}{marker}{trail}"


def _only_child_formatting(tag: Tag, names: tuple[str, ...]) -> bool:
    """True if all non-whitespace text of `tag` sits inside elements named `names`."""
    for desc in tag.descendants:
        if isinstance(desc, NavigableString) and desc.strip():
            parent = desc.parent
            ok = False
            while parent is not None and parent is not tag:
                if parent.name in names:
                    ok = True
                    break
                parent = parent.parent
            if not ok:
                return False
    return True


def _pseudo_heading_level(tag: Tag, text: str) -> int | None:
    """Quill articles mark headings as <p><strong>Heading</strong></p> (or short <em>Label:</em>)."""
    if not text or "\n" in text or len(text) > MAX_PSEUDO_HEADING_LEN:
        return None
    if tag.find("img") or tag.find("a"):
        return None
    if _only_child_formatting(tag, ("strong", "b")):
        # Short bold lines are headings; long bold sentences are emphasised prose.
        if text.endswith((":", "?")) or len(text.split()) <= MAX_PSEUDO_HEADING_WORDS:
            if text.endswith((".", ";", ",")) and not text.endswith("..."):
                return None
            return 3
        return None
    if _only_child_formatting(tag, ("em", "i", "u")) and text.endswith(":") and len(text) <= 60:
        return 4
    return None


def _list_lines(tag: Tag, ctx: _Ctx, depth: int = 0) -> tuple[list[str], list[str]]:
    ordered = tag.name == "ol"
    plain_lines, md_lines = [], []
    n = int(tag.get("start") or 1) if ordered and str(tag.get("start") or "1").isdigit() else 1
    for li in tag.find_all("li", recursive=False):
        # Quill encodes nesting as <li class="ql-indent-N"> rather than nested lists.
        indent_cls = next((c for c in li.get("class") or [] if c.startswith("ql-indent-")), None)
        level = depth + (int(indent_cls.rsplit("-", 1)[1]) if indent_cls and indent_cls.rsplit("-", 1)[1].isdigit() else 0)
        nested = [c for c in li.find_all(["ul", "ol"], recursive=False)]
        for c in nested:
            c.extract()
        plain_parts, md_parts = [], []
        for child in li.children:
            p, m = _inline(child, ctx)
            plain_parts.append(p)
            md_parts.append(m)
        plain = normalize_inline("".join(plain_parts)).replace("\n", " ")
        md = normalize_inline("".join(md_parts)).replace("\n", " ")
        marker = f"{n}." if ordered else "-"
        pad = "  " * level
        if plain:
            plain_lines.append(f"{pad}{marker} {plain}")
            md_lines.append(f"{pad}{marker} {md}")
            n += 1
        for c in nested:
            p_sub, m_sub = _list_lines(c, ctx, level + 1)
            plain_lines.extend(p_sub)
            md_lines.extend(m_sub)
    return plain_lines, md_lines


def _table_block(tag: Tag, ctx: _Ctx) -> Block | None:
    rows: list[list[str]] = []
    for tr in tag.find_all("tr"):
        cells = []
        for cell in tr.find_all(["td", "th"], recursive=False):
            p, _ = _inline(cell, ctx)
            cells.append(normalize_inline(p).replace("\n", " "))
        if any(cells):
            rows.append(cells)
    if not rows:
        return None
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    plain = "\n".join(" | ".join(c for c in r) for r in rows)
    esc = [[c.replace("|", "\\|") for c in r] for r in rows]
    md_lines = ["| " + " | ".join(esc[0]) + " |", "|" + "---|" * width]
    md_lines += ["| " + " | ".join(r) + " |" for r in esc[1:]]
    return Block("table", plain, "\n".join(md_lines))


def _walk(container: Tag, ctx: _Ctx) -> None:
    """Emit blocks for block-level children; group stray inline content into paragraphs."""
    pending: list[Tag | NavigableString] = []

    def flush() -> None:
        if not pending:
            return
        rendered = [_inline(n, ctx) for n in pending]
        plain = "".join(p for p, _ in rendered)
        md = "".join(m for _, m in rendered)
        _add_paragraph(normalize_inline(plain), normalize_inline(md), ctx)
        pending.clear()

    for child in list(container.children):
        if isinstance(child, NavigableString):
            if str(child).strip():
                pending.append(child)
            continue
        if not isinstance(child, Tag):
            continue
        name = child.name
        if name not in BLOCK_TAGS:
            pending.append(child)
            continue
        flush()
        if _is_hidden(child):
            hidden = normalize_pre(child.get_text())
            if hidden:
                ctx.hidden_blocks.append(hidden)
            continue
        if name in HEADING_TAGS:
            p, m = _inline(child, ctx)
            text = normalize_inline(p).replace("\n", " ")
            if text:
                ctx.blocks.append(Block("heading", text, "#" * int(name[1]) + " " + text, int(name[1])))
            if child.find("img"):
                ctx.blocks.append(Block("image", "", "[image]"))
        elif name == "p":
            p, m = _inline(child, ctx)
            plain, md = normalize_inline(p), normalize_inline(m)
            level = _pseudo_heading_level(child, plain)
            if level:
                ctx.blocks.append(Block("heading", plain, "#" * level + " " + plain, level))
            elif plain:
                _add_paragraph(plain, md, ctx)
            elif "[image]" in md:
                ctx.blocks.append(Block("image", "", "[image]"))
        elif name in ("ul", "ol"):
            pl, ml = _list_lines(child, ctx)
            if pl:
                ctx.blocks.append(Block("list", "\n".join(pl), "\n".join(ml)))
        elif name == "pre":
            text = normalize_pre(child.get_text())
            if text:
                ctx.blocks.append(Block("code", text, f"```\n{text}\n```"))
        elif name == "table":
            b = _table_block(child, ctx)
            if b:
                ctx.blocks.append(b)
        elif name == "hr":
            continue
        elif name == "blockquote":
            p, m = _inline(child, ctx)
            plain, md = normalize_inline(p), normalize_inline(m)
            if plain:
                ctx.blocks.append(Block("paragraph", plain, "\n".join("> " + ln for ln in md.split("\n"))))
        else:  # div / section / figure ... recurse
            _walk(child, ctx)
    flush()


def _add_paragraph(plain: str, md: str, ctx: _Ctx) -> None:
    if plain:
        ctx.blocks.append(Block("paragraph", plain, md or plain))
    elif "[image]" in md:
        ctx.blocks.append(Block("image", "", "[image]"))


def html_to_blocks(html: str) -> _Ctx:
    ctx = _Ctx()
    soup = BeautifulSoup(html or "", "lxml")
    root = soup.body or soup
    _walk(root, ctx)
    return ctx


# ---------------------------------------------------------------- assembly

def assemble(blocks: list[Block]) -> tuple[str, str, list[dict[str, Any]]]:
    """Join blocks into body_text / body_markdown and compute sections with char offsets."""
    text_parts: list[str] = []
    md_parts: list[str] = []
    sections: list[dict[str, Any]] = []
    pos = 0
    stack: list[tuple[int, str]] = []  # (level, heading) for heading paths
    current: dict[str, Any] | None = None

    def open_section(heading: str | None, level: int, start: int) -> dict[str, Any]:
        path = [h for _, h in stack]
        return {"heading": heading, "level": level, "path": path, "start_char": start,
                "content_start_char": start, "end_char": start}

    for b in blocks:
        if b.markdown:
            md_parts.append(b.markdown)
        if not b.text:
            continue
        start = pos + (2 if text_parts else 0)
        if b.kind == "heading":
            if current is not None:
                sections.append(current)
            while stack and stack[-1][0] >= b.level:
                stack.pop()
            current = open_section(b.text, b.level, start)  # path = ancestor headings only
            stack.append((b.level, b.text))
            current["content_start_char"] = start + len(b.text) + 2
        elif current is None:
            current = open_section(None, 0, start)
        text_parts.append(b.text)
        pos = start + len(b.text)
        current["end_char"] = pos
    if current is not None:
        sections.append(current)

    body_text = "\n\n".join(text_parts)
    for s in sections:
        s["content_start_char"] = min(s["content_start_char"], s["end_char"])
        s["text"] = body_text[s["content_start_char"]: s["end_char"]]
    return body_text, "\n\n".join(md_parts), sections


# ---------------------------------------------------------------- metadata helpers

_DATE_FORMATS = ("%d.%m.%Y", "%Y-%m-%d", "%d/%m/%Y", "%d.%m.%Y %H:%M", "%Y-%m-%dT%H:%M:%S")
ATTACHMENT_EXTENSIONS = (".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".odt", ".rtf",
                         ".zip", ".rar", ".7z", ".txt", ".csv", ".json", ".xml", ".stix", ".yar", ".yara", ".rules")
_UPDATED_RE = re.compile(r"^\s*(?:Оновлено|Updated|Оновлення)\s*:?\s*(\d{1,2}\.\d{1,2}\.\d{4})", re.IGNORECASE)
_ARTICLE_LINK_RE = re.compile(r"cert\.gov\.ua/article/(\d+)")


def parse_date(value: Any) -> str | None:
    if not value:
        return None
    s = str(value).strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _as_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def attachment_urls(links: list[dict[str, str]]) -> list[str]:
    out: list[str] = []
    for link in links:
        href = urljoin(config.SITE_URL + "/", link["href"])
        u = urlparse(href)
        if u.scheme not in ("http", "https") or not u.netloc.endswith("cert.gov.ua"):
            continue
        path = u.path.lower()
        if path.endswith(ATTACHMENT_EXTENSIONS) or path.startswith(("/files/", "/api/upload/", "/upload/")):
            if href not in out:
                out.append(href)
    return out


def word_count(text: str) -> int:
    return len(re.findall(r"\w+(?:[-'’ʼ]\w+)*", text))


def content_hash(text: str) -> str:
    return hashlib.sha256(unicodedata.normalize("NFC", text).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- main entry

def parse_article(envelope: dict[str, Any], meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a dataset record from a raw-cache envelope (see crawler.fetch_article)."""
    meta = meta or {}
    payload = envelope["payload"]
    article_id = str(payload.get("id") or envelope["article_id"])
    warnings: list[str] = []

    title = normalize_inline(payload.get("title") or "").replace("\n", " ") or None
    if not title and meta.get("listing_title"):
        title = normalize_inline(meta["listing_title"])
        warnings.append("title_from_listing")
    if not title:
        warnings.append("missing_title")

    raw_date = payload.get("date") or meta.get("listing_date")
    publication_date = parse_date(raw_date)
    if raw_date and not publication_date:
        warnings.append(f"unparsed_date:{raw_date}")

    ctx = html_to_blocks(payload.get("text") or "")
    body_text, body_markdown, sections = assemble(ctx.blocks)
    if not body_text:
        warnings.append("empty_body")

    summary_html = payload.get("description") or meta.get("listing_description")
    summary = html_to_plain(summary_html)

    tags = _as_list(payload.get("tags")) or [{"name": t} for t in _as_list(meta.get("listing_tags"))]
    categories = [normalize_inline(t["name"]) for t in tags if isinstance(t, dict) and t.get("name")]

    updated = _UPDATED_RE.match(body_text)
    source_lang = envelope.get("request_lang") or meta.get("source_lang") or "uk"
    # Language is judged on running prose; lists/tables (often quoted English titles) only as fallback.
    prose = [b.text for b in ctx.blocks if b.kind in ("paragraph", "heading")]
    lang = detect_language(body_text, prose if sum(map(len, prose)) >= 200 else
                           [b.text for b in ctx.blocks if b.kind != "code"])
    if lang["language"] not in ("unknown", source_lang):
        warnings.append(f"language_mismatch:{source_lang}->{lang['language']}")

    variant = payload.get("langId")
    variant_id = str(variant) if variant not in (None, "None", "", 0) and str(variant).isdigit() else None
    linked_articles = sorted({m for link in ctx.links for m in _ARTICLE_LINK_RE.findall(link["href"])} - {article_id})

    record: dict[str, Any] = {
        "article_id": article_id,
        "url": config.ARTICLE_URL_TEMPLATE.format(id=article_id),
        "title": title,
        "publication_date": publication_date,
        "categories": categories,
        "summary": summary,
        "language": lang["language"],
        "body_text": body_text,
        "body_markdown": body_markdown,
        "sections": sections,
        "entities": extract_entities(body_text, sections, title=title or ""),
        "attachment_urls": attachment_urls(ctx.links),
        "word_count": word_count(body_text),
        "content_hash": content_hash(body_text),
        "scraped_at": envelope.get("fetched_at"),
        "parser_version": PARSER_VERSION,
        # ---- extra provenance / research fields
        "source_lang": source_lang,
        "language_scores": lang["scores"],
        "variant_id": variant_id,
        "updated_date": parse_date(updated.group(1)) if updated else None,
        "char_count": len(body_text),
        "image_count": len(ctx.images),
        "outbound_links": [link["href"] for link in ctx.links],
        "linked_article_ids": linked_articles,
        "hidden_blocks": ctx.hidden_blocks,
        "api_url": envelope.get("request_url"),
        "listing_page": meta.get("listing_page"),
        "discovered_via": meta.get("discovered_via", []),
        "parse_warnings": warnings,
    }
    return record
