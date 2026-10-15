"""Deterministic (regex) extraction of technical identifiers.

These are *mentions* found in article text, not verified threat intelligence.
Each entity is grouped by its normalised value and keeps the original spellings
plus every mention's character offsets into ``body_text`` (or ``title``) and
the section heading it occurs in. Defanged indicators (``hxxp``, ``[.]``) are
kept verbatim in ``variants`` and refanged in ``normalized``. Nothing extracted
here is ever fetched or resolved.
"""

from __future__ import annotations

import re
from bisect import bisect_right
from functools import lru_cache
from importlib import resources
from typing import Any, Callable, Iterable

ENTITY_TYPES = ("cert_ua_ids", "uac_ids", "cve_ids", "file_hashes", "ipv4_addresses", "domains", "urls", "email_addresses")

# ---------------------------------------------------------------- patterns

_DOT = r"(?:\.|\[\.\]|\(\.\)|\{\.\}|\[dot\]|\(dot\))"
_DEFANG_DOT = re.compile(r"\[\.\]|\(\.\)|\{\.\}|\[dot\]|\(dot\)", re.IGNORECASE)

CERT_UA_RE = re.compile(r"(?<![\w-])CERT-UA\s?#\s?(\d{1,7})(?!\d)", re.IGNORECASE)
UAC_RE = re.compile(r"(?<![\w-])UAC-(\d{4})(?![\w-])", re.IGNORECASE)
CVE_RE = re.compile(r"(?<![\w-])CVE-(\d{4})-(\d{4,7})(?!\d)", re.IGNORECASE)
HASH_RE = re.compile(r"(?<![0-9A-Za-z])([0-9A-Fa-f]{128}|[0-9A-Fa-f]{64}|[0-9A-Fa-f]{40}|[0-9A-Fa-f]{32})(?![0-9A-Za-z])")
HASH_TYPES = {32: "md5", 40: "sha1", 64: "sha256", 128: "sha512"}
_OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
IPV4_RE = re.compile(rf"(?<![\d.\w])({_OCTET}){_DOT}({_OCTET}){_DOT}({_OCTET}){_DOT}({_OCTET})(?!\d|\.\d|\[\.\]\d)", re.IGNORECASE)
_SCHEMES = r"(?:hxxps?|hXXps?|https?|fxp|ftp|sftp|tcp|udp|smb|wss?)"
URL_RE = re.compile(
    # "(tcp)://1.2.3.4" is a CERT-UA defanging style; a bare "(" before http is prose punctuation.
    rf"(?<![\w])(?:\({_SCHEMES}\)(?:\[:\]|:)//|{_SCHEMES}(?:\[:\]|:)//|www{_DOT})[^\s<>\"'`]+",
    re.IGNORECASE,
)
_LABEL = r"[A-Za-z0-9Ѐ-ӿ](?:[A-Za-z0-9Ѐ-ӿ-]{0,61}[A-Za-z0-9Ѐ-ӿ])?"
DOMAIN_RE = re.compile(rf"(?<![\w@.\-\\/%$\])}}])((?:{_LABEL}{_DOT})+{_LABEL})(?![\w-]|{_DOT}[A-Za-z0-9Ѐ-ӿ])", re.IGNORECASE)
EMAIL_RE = re.compile(rf"(?<![\w.+-])([\w.+-]+(?:@|\[@\]|\[at\]))((?:{_LABEL}{_DOT})+{_LABEL})(?![\w-])", re.IGNORECASE)

# Extensions that are also valid TLDs: "status.zip", "run.sh" are almost always file names.
FILE_EXTENSION_TLDS = {"zip", "mov", "sh", "py", "pl", "rs", "ps", "md", "so", "app", "bat", "cab", "ini", "rar", "msi"}
# Labels that commonly appear in dotted code / product names rather than hostnames.
CODE_LIKE = re.compile(r"^(?:system|microsoft|windows|net|java|android|com\.android|org\.apache)\.", re.IGNORECASE)


@lru_cache(maxsize=1)
def valid_tlds() -> frozenset[str]:
    text = resources.files("cert_ua.resources").joinpath("tlds-alpha-by-domain.txt").read_text(encoding="utf-8")
    tlds: set[str] = set()
    for line in text.splitlines():
        line = line.strip().lower()
        if not line or line.startswith("#"):
            continue
        tlds.add(line)
        if line.startswith("xn--"):
            try:
                tlds.add(line.encode("ascii").decode("idna"))
            except UnicodeError:
                pass
    return frozenset(tlds)


# ---------------------------------------------------------------- normalisers

def refang(value: str) -> str:
    v = _DEFANG_DOT.sub(".", value)
    v = re.sub(r"\[:\]", ":", v)
    v = re.sub(r"\[@\]|\[at\]", "@", v, flags=re.IGNORECASE)
    v = re.sub(r"^\(?(h[xX]{2}p)(s?)\)?", lambda m: "http" + m.group(2), v, flags=re.IGNORECASE)
    v = re.sub(r"^fxp", "ftp", v, flags=re.IGNORECASE)
    v = re.sub(r"^\((\w+)\)(?=://)", r"\1", v)
    return v


def _is_defanged(value: str) -> bool:
    return bool(_DEFANG_DOT.search(value) or "[:]" in value or re.match(r"\(?h[xX]{2}p", value))


_TRAILING = ".,;:!?'\"»”’)]}>"
_PAIRS = {")": "(", "]": "[", "}": "{"}


def _trim_url(url: str) -> str:
    """Strip trailing punctuation that belongs to the sentence, keeping balanced brackets."""
    while url and url[-1] in _TRAILING:
        ch = url[-1]
        if ch in _PAIRS and url.count(_PAIRS[ch]) >= url.count(ch) and not url.endswith(("[.]", "(.)")):
            break
        if url.endswith(("[.]", "(.)", "{.}")):
            url = url[:-3]
            continue
        url = url[:-1]
    return url


_URL_HOST_RE = re.compile(r"^(?:\(?\w+\)?(?:\[:\]|:)//)?(?:[^@/\s]+@)?((?:[\w-]+" + _DOT + r")+[\w-]+)", re.IGNORECASE)


def _valid_domain(token: str, in_url: bool = False) -> bool:
    plain = refang(token)
    labels = plain.lower().split(".")
    if len(labels) < 2 or labels[-1] not in valid_tlds():
        return False
    defanged = _is_defanged(token)
    if all(l.isdigit() for l in labels[:-1]) and not (defanged or in_url):
        return False
    if not defanged and not in_url:
        if labels[-1] in FILE_EXTENSION_TLDS:
            return False
        # CamelCase dotted identifiers (System.Net.WebClient) are code, not hostnames.
        if token != token.lower() and token != token.upper() and not (token[0].isupper() and token[1:] == token[1:].lower()):
            return False
        if CODE_LIKE.match(plain):
            return False
    return True


# ---------------------------------------------------------------- extraction

class _Collector:
    def __init__(self, section_of: Callable[[int], str | None]):
        self.section_of = section_of
        self.groups: dict[str, dict[str, dict[str, Any]]] = {t: {} for t in ENTITY_TYPES}

    def add(self, etype: str, value: str, normalized: str, start: int, end: int, field: str, **extra: Any) -> None:
        group = self.groups[etype].setdefault(normalized, {"normalized": normalized, "value": value, "variants": [], **extra, "mentions": []})
        if value not in group["variants"]:
            group["variants"].append(value)
        group["mentions"].append({
            "start": start,
            "end": end,
            "field": field,
            "section": self.section_of(start) if field == "body_text" else None,
        })

    def result(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for etype, groups in self.groups.items():
            out[etype] = sorted(groups.values(), key=lambda g: g["mentions"][0]["start"] if g["mentions"] else 0)
        out["counts"] = {t: len(out[t]) for t in ENTITY_TYPES}
        return out


def _scan(text: str, field: str, c: _Collector) -> None:
    for m in CERT_UA_RE.finditer(text):
        c.add("cert_ua_ids", m.group(0), f"CERT-UA#{m.group(1)}", m.start(), m.end(), field)
    for m in UAC_RE.finditer(text):
        c.add("uac_ids", m.group(0), f"UAC-{m.group(1)}", m.start(), m.end(), field)
    for m in CVE_RE.finditer(text):
        c.add("cve_ids", m.group(0), f"CVE-{m.group(1)}-{m.group(2)}", m.start(), m.end(), field)
    for m in HASH_RE.finditer(text):
        h = m.group(1)
        if not (re.search(r"[a-fA-F]", h) and re.search(r"\d", h)):
            continue
        c.add("file_hashes", h, h.lower(), m.start(), m.end(), field, hash_type=HASH_TYPES[len(h)])
    for m in URL_RE.finditer(text):
        url = _trim_url(m.group(0))
        if len(url) < 8 or url.lower().rstrip("/").endswith(("://", "[:]//")):
            continue
        norm = refang(url)
        if norm.lower().startswith("www."):
            norm = "http://" + norm
        c.add("urls", url, norm, m.start(), m.start() + len(url), field, defanged=_is_defanged(url))
        # The DOMAIN_RE lookbehind skips hosts inside URLs; register them explicitly.
        host = _URL_HOST_RE.match(url)
        if host and not IPV4_RE.fullmatch(host.group(1)) and _valid_domain(host.group(1), in_url=True):
            h0 = m.start() + host.start(1)
            c.add("domains", host.group(1), refang(host.group(1)).lower(), h0, h0 + len(host.group(1)), field,
                  defanged=_is_defanged(host.group(1)))
    for m in EMAIL_RE.finditer(text):
        dom = m.group(2)
        if not _valid_domain(dom) and not _is_defanged(m.group(0)):
            continue
        val = m.group(0)
        c.add("email_addresses", val, refang(val).lower(), m.start(), m.end(), field, defanged=_is_defanged(val))
    for m in IPV4_RE.finditer(text):
        ip = m.group(0)
        # "Chrome/111.0.0.0" is a version in a User-Agent; "http://1.2.3.4" is fine.
        if m.start() >= 1 and text[m.start() - 1] == "/" and text[max(0, m.start() - 2): m.start()] != "//":
            continue
        c.add("ipv4_addresses", ip, ".".join(str(int(o)) for o in m.groups()), m.start(), m.end(), field,
              defanged=_is_defanged(ip))
    for m in DOMAIN_RE.finditer(text):
        dom = m.group(1)
        if IPV4_RE.fullmatch(dom) or not _valid_domain(dom):
            continue
        c.add("domains", dom, refang(dom).lower().rstrip("."), m.start(1), m.end(1), field, defanged=_is_defanged(dom))


def extract_entities(body_text: str, sections: Iterable[dict[str, Any]] = (), title: str = "") -> dict[str, Any]:
    secs = sorted(sections, key=lambda s: s["start_char"])
    starts = [s["start_char"] for s in secs]

    def section_of(offset: int) -> str | None:
        i = bisect_right(starts, offset) - 1
        return secs[i]["heading"] if i >= 0 else None

    c = _Collector(section_of)
    _scan(body_text, "body_text", c)
    if title:
        _scan(title, "title", c)
    return c.result()


def entity_types_present(entities: dict[str, Any]) -> set[str]:
    return {t for t in ENTITY_TYPES if entities.get(t)}
