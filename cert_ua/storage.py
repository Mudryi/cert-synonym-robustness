"""JSONL persistence, raw-response cache and resumable crawl state."""

from __future__ import annotations

import gzip
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from . import config


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # A truncated last line can appear after a hard kill; skip it.
                continue


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    lines = [json.dumps(r, ensure_ascii=False) for r in records]
    _atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


def write_json(path: Path, obj: Any) -> None:
    _atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- raw cache

def raw_path(article_id: str, raw_dir: Path | None = None) -> Path:
    return (raw_dir or config.RAW_DIR) / f"{article_id}.json.gz"


def save_raw(article_id: str, payload: dict[str, Any], raw_dir: Path | None = None) -> Path:
    path = raw_path(article_id, raw_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)
    return path


def load_raw(article_id: str, raw_dir: Path | None = None) -> dict[str, Any] | None:
    path = raw_path(article_id, raw_dir)
    if not path.exists():
        return None
    with gzip.open(path, "rb") as f:
        return json.loads(f.read().decode("utf-8"))


def cached_ids(raw_dir: Path | None = None) -> list[str]:
    d = raw_dir or config.RAW_DIR
    if not d.exists():
        return []
    return sorted(p.name.split(".")[0] for p in d.glob("*.json.gz"))


# ---------------------------------------------------------------- crawl state

class Progress:
    """Tracks which article IDs were fetched/parsed or failed, persisted atomically."""

    def __init__(self, path: Path | None = None):
        self.path = path or config.PROGRESS_PATH
        data = read_json(self.path, default={}) or {}
        self.done: set[str] = set(data.get("done", []))
        self.failed: dict[str, dict[str, Any]] = data.get("failed", {})
        self.listing_exhausted: bool = data.get("listing_exhausted", False)
        self.last_listing_page: int | None = data.get("last_listing_page")
        self.api_reported_count: int | None = data.get("api_reported_count")
        self.updated_at: str | None = data.get("updated_at")

    def mark_done(self, article_id: str) -> None:
        self.done.add(article_id)
        self.failed.pop(article_id, None)

    def mark_failed(self, article_id: str, info: dict[str, Any]) -> None:
        prev = self.failed.get(article_id, {})
        info = {**info, "attempts": prev.get("attempts", 0) + 1}
        self.failed[article_id] = info

    def save(self) -> None:
        self.updated_at = utc_now()
        write_json(
            self.path,
            {
                "done": sorted(self.done),
                "failed": self.failed,
                "listing_exhausted": self.listing_exhausted,
                "last_listing_page": self.last_listing_page,
                "api_reported_count": self.api_reported_count,
                "updated_at": self.updated_at,
            },
        )


# ---------------------------------------------------------------- discovered articles

def load_discovered(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Discovered article metadata keyed by article_id; later lines update earlier ones."""
    out: dict[str, dict[str, Any]] = {}
    for rec in read_jsonl(path or config.DISCOVERED_PATH):
        aid = str(rec["article_id"])
        if aid in out:
            merged = {**out[aid], **{k: v for k, v in rec.items() if v not in (None, [], "")}}
            merged["discovered_via"] = sorted(
                set(out[aid].get("discovered_via", [])) | set(rec.get("discovered_via", []))
            )
            out[aid] = merged
        else:
            out[aid] = rec
    return out


def save_discovered(records: dict[str, dict[str, Any]], path: Path | None = None) -> None:
    write_jsonl(path or config.DISCOVERED_PATH, records.values())


# ---------------------------------------------------------------- processed articles

def _sort_key(rec: dict[str, Any]) -> tuple[str, int]:
    aid = rec.get("article_id", "0")
    return (rec.get("publication_date") or "", int(aid) if str(aid).isdigit() else 0)


def upsert_articles(records: Iterable[dict[str, Any]], path: Path | None = None) -> int:
    """Merge records into the processed JSONL keyed by article_id (no duplicates)."""
    path = path or config.ARTICLES_PATH
    by_id = {r["article_id"]: r for r in read_jsonl(path)}
    for r in records:
        by_id[r["article_id"]] = r
    return write_jsonl(path, sorted(by_id.values(), key=_sort_key, reverse=True))
