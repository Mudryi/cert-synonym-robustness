"""Shared helpers for the synonym-robust retrieval experiments.

Everything here runs in the project's uv environment (Python 3.12). The two
perturbation-pool scripts that run in the sibling repos' venvs do not import
this module; they communicate through JSONL files only.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
ARTICLES = DATA / "cert_ua_articles.jsonl"
CHUNKS = DATA / "cert_ua_text_chunks.jsonl"
STATS = ROOT / "data" / "reports" / "dataset_statistics.json"
EXP = ROOT / "experiments"
OUT = EXP / "out"
RESOURCES = EXP / "resources"
SEED = 13

SBERT_MODEL = "lang-uk/ukr-paraphrase-multilingual-mpnet-base"
DEFAULT_LLM = os.environ.get("EXP_LLM_MODEL", "gpt-4.1-mini")
# hard OpenAI budget across all stages/processes (user has ~$15 on the account)
BUDGET_USD = float(os.environ.get("EXP_LLM_BUDGET_USD", "8"))
PRICE_PER_M = {"gpt-4.1-mini": (0.40, 1.60), "gpt-4.1-nano": (0.10, 0.40), "gpt-4o-mini": (0.15, 0.60)}
SPEND_LOG = OUT / "llm_spend.jsonl"


# ---------------------------------------------------------------- jsonl io

def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def iter_jsonl(path: Path | str) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path | str, rows: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    tmp.replace(path)
    return n


def write_json(path: Path | str, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def text_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def done(path: Path, force: bool) -> bool:
    """Stage-level resume: skip when the output already exists."""
    if path.exists() and not force:
        print(f"[skip] {path} exists (use --force to rebuild)", flush=True)
        return True
    return False


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- entities

PROTECTED_TYPES = ("cert_ua_ids", "uac_ids", "cve_ids", "file_hashes", "ipv4_addresses", "domains", "urls", "email_addresses")
LATIN_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_.\-]*")


def protected_entities(text: str) -> list[dict[str, Any]]:
    """Identifiers that must survive every perturbation verbatim.

    Regex entities from the crawler (CVE/UAC/CERT-UA IDs, hashes, IPs, domains,
    URLs, e-mails) plus Latin-script tokens (malware/tool/product names).
    """
    from cert_ua.entities import extract_entities

    ents = extract_entities(text)
    out: list[dict[str, Any]] = []
    for t in PROTECTED_TYPES:
        for e in ents.get(t, []):
            for m in e["mentions"]:
                out.append({"type": t, "value": text[m["start"]:m["end"]], "start": m["start"], "end": m["end"]})
    taken = [(e["start"], e["end"]) for e in out]
    for m in LATIN_TOKEN.finditer(text):
        if not any(s < m.end() and m.start() < e for s, e in taken):
            out.append({"type": "latin_term", "value": m.group(0), "start": m.start(), "end": m.end()})
    return sorted(out, key=lambda e: e["start"])


def has_identifier(ents: list[dict[str, Any]]) -> bool:
    return any(e["type"] != "latin_term" for e in ents)


def entities_preserved(ents: list[dict[str, Any]], new_text: str) -> bool:
    return all(e["value"] in new_text for e in ents)


# ---------------------------------------------------------------- LLM

def openai_key() -> str:
    """Read the key from .env; tolerates `OPEN_AI_KEY = sk-...` and quotes."""
    for name in ("OPENAI_API_KEY", "OPEN_AI_KEY"):
        if os.environ.get(name):
            return os.environ[name]
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if "=" not in line or line.lstrip().startswith("#"):
                continue
            k, v = line.split("=", 1)
            if k.strip().removeprefix("export ").strip() in ("OPENAI_API_KEY", "OPEN_AI_KEY"):
                return v.strip().strip('"').strip("'")
    raise RuntimeError("OpenAI key not found (.env OPEN_AI_KEY / OPENAI_API_KEY)")


class LLM:
    """OpenAI chat client with a persistent prompt-hash cache and retries.

    Responses are JSON objects. The cache makes reruns free and the benchmark
    reproducible from `llm_cache.jsonl`.
    """

    def __init__(self, model: str = DEFAULT_LLM, cache_path: Path = OUT / "llm_cache.jsonl", workers: int = 8):
        from openai import OpenAI

        self.model = model
        self.client = OpenAI(api_key=openai_key(), timeout=60, max_retries=0)
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache: dict[str, Any] = {}
        if self.cache_path.exists():
            for row in iter_jsonl(self.cache_path):
                self.cache[row["key"]] = row["response"]
        self.lock = threading.Lock()
        self.workers = workers
        self.calls = 0
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}
        self.spent = sum(r["usd"] for r in iter_jsonl(SPEND_LOG)) if SPEND_LOG.exists() else 0.0
        self.over_budget_logged = False
        log(f"LLM {model}: cache {len(self.cache)} entries, spent so far ${self.spent:.2f} of ${BUDGET_USD:.2f}")

    def _charge(self, usage) -> None:
        pin, pout = PRICE_PER_M.get(self.model, (2.0, 8.0))  # unknown model: assume expensive
        usd = (usage.prompt_tokens * pin + usage.completion_tokens * pout) / 1e6
        self.spent += usd
        with open(SPEND_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.time(), "model": self.model, "usd": usd}) + "\n")

    def _key(self, system: str, user: str, temperature: float) -> str:
        return hashlib.sha256(json.dumps([self.model, system, user, temperature], ensure_ascii=False).encode()).hexdigest()

    def ask(self, system: str, user: str, temperature: float = 0.0) -> dict[str, Any] | None:
        key = self._key(system, user, temperature)
        if key in self.cache:
            return self.cache[key]
        if self.spent >= BUDGET_USD:
            with self.lock:
                if not self.over_budget_logged:
                    log(f"LLM budget ${BUDGET_USD:.2f} reached (spent ${self.spent:.2f}); returning None for new prompts")
                    self.over_budget_logged = True
            return None
        delay = 2.0
        for attempt in range(6):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    temperature=temperature,
                    response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                )
                content = resp.choices[0].message.content or "{}"
                parsed = json.loads(content)
                with self.lock:
                    self.calls += 1
                    if resp.usage:
                        self.usage["prompt_tokens"] += resp.usage.prompt_tokens
                        self.usage["completion_tokens"] += resp.usage.completion_tokens
                        self._charge(resp.usage)
                    self.cache[key] = parsed
                    with open(self.cache_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"key": key, "model": self.model, "response": parsed}, ensure_ascii=False) + "\n")
                return parsed
            except json.JSONDecodeError:
                temperature = 0.2  # nudge a malformed answer
            except Exception as e:  # network, rate limit, 5xx
                msg = str(e)
                if "invalid_api_key" in msg or "model_not_found" in msg or "does not exist" in msg:
                    raise
                log(f"LLM error (attempt {attempt + 1}/6): {msg[:200]}")
                time.sleep(delay + random.random())
                delay = min(delay * 2, 120)
        return None

    def map(self, fn: Callable[[Any], tuple[str, str]], items: list[Any], temperature: float = 0.0, desc: str = "llm") -> list[dict[str, Any] | None]:
        """Run `fn(item) -> (system, user)` prompts concurrently, preserving order."""
        out: list[dict[str, Any] | None] = [None] * len(items)
        done_n = [0]

        def work(i: int) -> None:
            s, u = fn(items[i])
            out[i] = self.ask(s, u, temperature)
            with self.lock:
                done_n[0] += 1
                if done_n[0] % 200 == 0:
                    log(f"{desc}: {done_n[0]}/{len(items)} (api calls {self.calls})")

        with ThreadPoolExecutor(self.workers) as ex:
            list(ex.map(work, range(len(items))))
        log(f"{desc}: done {len(items)} items, {self.calls} new API calls, usage {self.usage}, total spent ${self.spent:.3f}")
        return out


# ---------------------------------------------------------------- SBERT

_sbert = None


def sbert_cos(pairs: list[tuple[str, str]], batch_size: int = 128) -> list[float]:
    """Cosine similarity of (a, b) pairs with the Ukrainian paraphrase SBERT."""
    global _sbert
    import numpy as np
    from sentence_transformers import SentenceTransformer

    if not pairs:
        return []
    if _sbert is None:
        _sbert = SentenceTransformer(SBERT_MODEL, device="cuda")
    texts = sorted({t for p in pairs for t in p})
    emb = _sbert.encode(texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    idx = {t: i for i, t in enumerate(texts)}
    return [float(np.dot(emb[idx[a]], emb[idx[b]])) for a, b in pairs]


# ---------------------------------------------------------------- tokens

TOKEN_RE = re.compile(r"(\s+|[^\w\s']+|[\w']+)")


def apply_subs(tokens: list[str], subs: dict[int, str]) -> str:
    return "".join(subs.get(i, t) for i, t in enumerate(tokens))
