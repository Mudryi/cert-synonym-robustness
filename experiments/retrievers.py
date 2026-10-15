"""Frozen retrievers with a common interface.

`score(texts) -> {level: float32 array [len(texts), N_level]}` for the levels
  chunk     rank chunks (section-level retrieval)
  doc_max   rank documents by their best chunk
  doc_full  rank documents encoded as a whole (if the model supports it)
Documents are the S0 collection (title-stripped). Corpus embeddings are cached
on disk under experiments/out/cache/ so later stages do not re-encode.
"""
from __future__ import annotations

import gc
import math
import re
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

from experiments.common import OUT, log, read_jsonl

CACHE = OUT / "cache"
LEVELS = ("chunk", "doc_max", "doc_full")


class Corpus:
    def __init__(self, corpus_dir: Path = OUT / "corpus"):
        docs = read_jsonl(Path(corpus_dir) / "corpus_docs.jsonl")
        chunks = read_jsonl(Path(corpus_dir) / "corpus_chunks.jsonl")
        self.docs = docs
        self.doc_ids = [d["doc_id"] for d in docs]
        self.doc_pos = {d: i for i, d in enumerate(self.doc_ids)}
        chunks.sort(key=lambda c: (self.doc_pos[c["doc_id"]], c["chunk_id"]))
        self.chunks = chunks
        self.chunk_ids = [c["chunk_id"] for c in chunks]
        self.chunk_pos = {c: i for i, c in enumerate(self.chunk_ids)}
        cdoc = np.array([self.doc_pos[c["doc_id"]] for c in chunks])
        self.reduce_starts = np.flatnonzero(np.r_[True, cdoc[1:] != cdoc[:-1]])
        self.reduce_docs = cdoc[self.reduce_starts]
        self.tag = f"{len(docs)}d{len(chunks)}c"

    def doc_max(self, chunk_scores: np.ndarray) -> np.ndarray:
        out = np.full((chunk_scores.shape[0], len(self.doc_ids)), -np.inf, dtype=np.float32)
        out[:, self.reduce_docs] = np.maximum.reduceat(chunk_scores, self.reduce_starts, axis=1)
        return out


# ---------------------------------------------------------------- BM25

@lru_cache(maxsize=1)
def _morph():
    import pymorphy3
    return pymorphy3.MorphAnalyzer(lang="uk")


@lru_cache(maxsize=1_000_000)
def _lemma(tok: str) -> str:
    if re.search(r"[а-яґєії]", tok):
        return _morph().parse(tok)[0].normal_form
    return tok


def analyze(text: str, lemmatize: bool) -> list[str]:
    toks = re.findall(r"[\w'’]+", text.lower().replace("’", "'"))
    return [_lemma(t) for t in toks] if lemmatize else toks


class BM25:
    """Okapi BM25 (k1=1.5, b=0.75, Lucene idf) as sparse matrix products."""

    def __init__(self, corpus: Corpus, lemmatize: bool = True, with_titles: bool = False, k1: float = 1.5, b: float = 0.75):
        from scipy import sparse

        self.corpus, self.lemmatize = corpus, lemmatize
        self.vocab: dict[str, int] = {}
        chunk_tokens = [analyze(c["text"], lemmatize) for c in corpus.chunks]
        doc_tokens = [analyze((d["title"] + "\n" if with_titles else "") + d["text"], lemmatize) for d in corpus.docs]
        for toks in chunk_tokens + doc_tokens:
            for t in toks:
                self.vocab.setdefault(t, len(self.vocab))

        def weights(token_lists):
            n = len(token_lists)
            df = Counter(t for toks in token_lists for t in set(toks))
            avgdl = sum(map(len, token_lists)) / max(1, n)
            rows, cols, vals = [], [], []
            for i, toks in enumerate(token_lists):
                dl = len(toks)
                for t, tf in Counter(toks).items():
                    idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                    rows.append(i)
                    cols.append(self.vocab[t])
                    vals.append(idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl)))
            return sparse.csr_matrix((vals, (rows, cols)), shape=(n, len(self.vocab)), dtype=np.float32)

        self.W_chunk = weights(chunk_tokens).T.tocsr()
        self.W_doc = weights(doc_tokens).T.tocsr()

    def _q(self, texts):
        from scipy import sparse

        rows, cols, vals = [], [], []
        for i, t in enumerate(texts):
            for tok, c in Counter(analyze(t, self.lemmatize)).items():
                j = self.vocab.get(tok)
                if j is not None:
                    rows.append(i)
                    cols.append(j)
                    vals.append(c)
        return sparse.csr_matrix((vals, (rows, cols)), shape=(len(texts), len(self.vocab)), dtype=np.float32)

    def score(self, texts: list[str]) -> dict[str, np.ndarray]:
        q = self._q(texts)
        chunk = (q @ self.W_chunk).toarray()
        return {"chunk": chunk, "doc_max": self.corpus.doc_max(chunk), "doc_full": (q @ self.W_doc).toarray()}

    def close(self):
        pass


# ---------------------------------------------------------------- dense

class Dense:
    def __init__(self, corpus: Corpus, name: str, model: str, q_prefix: str = "", d_prefix: str = "",
                 doc_max_len: int = 512, chunk_max_len: int = 512):
        import torch
        from sentence_transformers import SentenceTransformer

        self.corpus, self.name, self.q_prefix = corpus, name, q_prefix
        self.model = SentenceTransformer(model, device="cuda", model_kwargs={"torch_dtype": torch.float16})
        self.qcache: dict[str, np.ndarray] = {}
        CACHE.mkdir(parents=True, exist_ok=True)

        def corpus_emb(kind: str, texts: list[str], max_len: int, bs: int) -> np.ndarray:
            path = CACHE / f"emb_{name}_{kind}_{corpus.tag}_{max_len}.npy"
            if path.exists():
                return np.load(path)
            self.model.max_seq_length = max_len
            t0 = time.time()
            emb = self.model.encode([d_prefix + t for t in texts], batch_size=bs, normalize_embeddings=True,
                                    convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
            np.save(path, emb)
            log(f"{name}: encoded {len(texts)} {kind} (max_len {max_len}) in {time.time() - t0:.0f}s")
            return emb

        self.C = corpus_emb("chunks", [c["text"] for c in corpus.chunks], chunk_max_len, 32)
        self.D = corpus_emb("docs", [d["text"] for d in corpus.docs], doc_max_len, 1 if doc_max_len > 1024 else 16)
        self.model.max_seq_length = 512

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        new = [t for t in dict.fromkeys(texts) if t not in self.qcache]
        if new:
            emb = self.model.encode([self.q_prefix + t for t in new], batch_size=128, normalize_embeddings=True,
                                    convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
            self.qcache.update(zip(new, emb))
            if len(self.qcache) > 300_000:
                self.qcache.clear()
                self.qcache.update(zip(new, emb))
        return np.stack([self.qcache[t] for t in texts])

    def score(self, texts: list[str]) -> dict[str, np.ndarray]:
        q = self.encode_queries(texts)
        chunk = q @ self.C.T
        return {"chunk": chunk, "doc_max": self.corpus.doc_max(chunk), "doc_full": q @ self.D.T}

    def close(self):
        del self.model
        _free()


# ---------------------------------------------------------------- BGE-M3 multi-vector

class M3Multi:
    """BGE-M3 dense + sparse (lexical weights) + ColBERT, via FlagEmbedding.

    mode: 'sparse' | 'colbert' | 'all' (0.4 dense + 0.2 sparse + 0.4 colbert, the
    weights recommended by the BGE-M3 authors). Levels chunk and doc_max only.
    """

    _backend = None

    def __init__(self, corpus: Corpus, mode: str):
        import torch
        from scipy import sparse

        self.corpus, self.mode = corpus, mode
        if M3Multi._backend is None:
            from FlagEmbedding import BGEM3FlagModel

            model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, devices=["cuda:0"])
            t0 = time.time()
            enc = model.encode([c["text"] for c in corpus.chunks], batch_size=16, max_length=512,
                               return_dense=True, return_sparse=True, return_colbert_vecs=True)
            dense = np.asarray(enc["dense_vecs"], dtype=np.float32)
            n, vocab = len(corpus.chunks), 250_002
            rows, cols, vals = [], [], []
            for i, lw in enumerate(enc["lexical_weights"]):
                for tok, w in lw.items():
                    rows.append(i)
                    cols.append(int(tok))
                    vals.append(float(w))
            S = sparse.csr_matrix((vals, (rows, cols)), shape=(n, vocab), dtype=np.float32).T.tocsr()
            vecs = enc["colbert_vecs"]
            L = max(v.shape[0] for v in vecs)
            P = torch.zeros((n, L, vecs[0].shape[1]), dtype=torch.float16)
            mask = torch.zeros((n, L), dtype=torch.bool)
            for i, v in enumerate(vecs):
                P[i, :v.shape[0]] = torch.from_numpy(np.asarray(v, dtype=np.float16))
                mask[i, :v.shape[0]] = True
            M3Multi._backend = {"model": model, "dense": dense, "sparse": S, "colbert": P.cuda(), "mask": mask.cuda(), "vocab": vocab}
            log(f"bge-m3 multi: encoded {n} chunks in {time.time() - t0:.0f}s (colbert pad {L})")
        self.be = M3Multi._backend

    def score(self, texts: list[str]) -> dict[str, np.ndarray]:
        import torch
        from scipy import sparse

        be = self.be
        enc = be["model"].encode(texts, batch_size=64, max_length=512, return_dense=True, return_sparse=True,
                                 return_colbert_vecs=self.mode in ("colbert", "all"))
        parts = {}
        if self.mode in ("all",):
            parts["dense"] = np.asarray(enc["dense_vecs"], dtype=np.float32) @ be["dense"].T
        if self.mode in ("sparse", "all"):
            rows, cols, vals = [], [], []
            for i, lw in enumerate(enc["lexical_weights"]):
                for tok, w in lw.items():
                    rows.append(i)
                    cols.append(int(tok))
                    vals.append(float(w))
            q = sparse.csr_matrix((vals, (rows, cols)), shape=(len(texts), be["vocab"]), dtype=np.float32)
            parts["sparse"] = (q @ be["sparse"]).toarray()
        if self.mode in ("colbert", "all"):
            out = np.zeros((len(texts), be["colbert"].shape[0]), dtype=np.float32)
            with torch.no_grad():
                for i, v in enumerate(enc["colbert_vecs"]):
                    qv = torch.from_numpy(np.asarray(v, dtype=np.float16)).cuda()
                    sim = torch.einsum("qd,nld->nql", qv, be["colbert"])
                    sim = sim.masked_fill(~be["mask"][:, None, :], -1e4)
                    out[i] = sim.max(dim=2).values.float().mean(dim=1).cpu().numpy()
            parts["colbert"] = out
        if self.mode == "all":
            chunk = 0.4 * parts["dense"] + 0.2 * parts["sparse"] + 0.4 * parts["colbert"]
        else:
            chunk = parts[self.mode]
        return {"chunk": chunk, "doc_max": self.corpus.doc_max(chunk)}

    def close(self):
        pass


# ---------------------------------------------------------------- fusion

def ranks_desc(scores: np.ndarray) -> np.ndarray:
    """1-based ranks per row (ties broken by order)."""
    order = np.argsort(-scores, axis=1, kind="stable")
    r = np.empty_like(order)
    rows = np.arange(scores.shape[0])[:, None]
    r[rows, order] = np.arange(1, scores.shape[1] + 1)[None, :]
    return r


def rrf(score_list: list[np.ndarray], k: int = 60) -> np.ndarray:
    return sum(1.0 / (k + ranks_desc(s)) for s in score_list).astype(np.float32)


class Hybrid:
    def __init__(self, parts: list):
        self.parts = parts

    def score(self, texts: list[str]) -> dict[str, np.ndarray]:
        outs = [p.score(texts) for p in self.parts]
        levels = set.intersection(*(set(o) for o in outs))
        return {lv: rrf([o[lv] for o in outs]) for lv in levels}

    def close(self):
        pass


# ---------------------------------------------------------------- registry

SPECS = {
    "bm25": lambda c: BM25(c, lemmatize=True),
    "bm25_surface": lambda c: BM25(c, lemmatize=False),
    "bm25_titled": lambda c: BM25(c, lemmatize=True, with_titles=True),
    "e5_base": lambda c: Dense(c, "e5_base", "intfloat/multilingual-e5-base", "query: ", "passage: ", 512),
    "e5_large": lambda c: Dense(c, "e5_large", "intfloat/multilingual-e5-large", "query: ", "passage: ", 512),
    "bge_m3": lambda c: Dense(c, "bge_m3", "BAAI/bge-m3", "", "", 8192),
    "ukr_mpnet": lambda c: Dense(c, "ukr_mpnet", "lang-uk/ukr-paraphrase-multilingual-mpnet-base", "", "", 512),
    "para_mpnet": lambda c: Dense(c, "para_mpnet", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "", "", 512),
    "m3_sparse": lambda c: M3Multi(c, "sparse"),
    "m3_colbert": lambda c: M3Multi(c, "colbert"),
    "m3_all": lambda c: M3Multi(c, "all"),
}
_live: dict[str, object] = {}


def get(name: str, corpus: Corpus):
    if name in _live:
        return _live[name]
    if name == "hybrid":
        r = Hybrid([get("bm25", corpus), get("bge_m3", corpus)])
    elif name == "hybrid_e5":
        r = Hybrid([get("bm25", corpus), get("e5_base", corpus)])
    else:
        r = SPECS[name](corpus)
    _live[name] = r
    return r


def release_all() -> None:
    for r in _live.values():
        r.close()
    _live.clear()
    M3Multi._backend = None
    _free()


def _free() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def gold_rank(row: np.ndarray, gold: int) -> int:
    """Pessimistic rank: ties with the gold item count against it."""
    return int((row >= row[gold]).sum())


def score_batched(r, texts: list[str], batch: int = 512):
    for i in range(0, len(texts), batch):
        yield i, r.score(texts[i:i + batch])
