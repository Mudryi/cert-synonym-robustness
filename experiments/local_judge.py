"""Local LLM validity judge (MamayLM-Gemma-3-12B, 4-bit), zero API cost.

Scores P(yes) for "does the modified query preserve the meaning?" from the
next-token distribution after the chat prompt (one forward pass per item).
Used as an independent second judge (agreement with the OpenAI judge) and as
the only judge for the extra-seed GPU runs.

  python -m experiments.local_judge --out DIR --name variants   # adds local_* fields in place
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from experiments.common import log, read_jsonl, write_jsonl

MODEL = "INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v1.0"
PROMPT = """You validate a search-robustness benchmark of Ukrainian cybersecurity reports.
Original query: "{orig}"
Modified query: "{new}"
Replaced: {changed}
Does the modified query express the SAME information need, so that it should retrieve the same document? Synonyms, abbreviations and spelling variants are fine; a different entity, action, object, scope, or a word used in a wrong sense is not.
Answer with one word: Yes or No."""

_model = None


def load():
    global _model
    if _model is None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig

        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(MODEL)
        tok.padding_side = "left"
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL, device_map="cuda:0", torch_dtype=torch.bfloat16,
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                   bnb_4bit_compute_dtype=torch.bfloat16))
        model.eval()
        yes = [tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", " Yes", "yes")]
        no = [tok.encode(w, add_special_tokens=False)[0] for w in ("No", " No", "no")]
        _model = (tok, model, sorted(set(yes)), sorted(set(no)))
        log(f"local judge loaded in {time.time() - t0:.0f}s")
    return _model


def p_yes(rows: list[dict], batch_size: int = 16) -> list[float]:
    import torch

    tok, model, yes, no = load()
    prompts = []
    for r in rows:
        msg = PROMPT.format(orig=r["original_query"], new=r["synonym_query"],
                            changed="; ".join(f"{a} → {b}" for a, b in r["changed_words"]))
        prompts.append(tok.apply_chat_template([{"role": "user", "content": [{"type": "text", "text": msg}]}],
                                               tokenize=False, add_generation_prompt=True))
    order = np.argsort([len(p) for p in prompts])
    out = np.zeros(len(rows))
    t0 = time.time()
    with torch.no_grad():
        for b in range(0, len(order), batch_size):
            idx = order[b:b + batch_size]
            enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda:0")
            logits = model(**enc).logits[:, -1, :].float()
            lp = torch.log_softmax(logits, dim=-1)
            ly = torch.logsumexp(lp[:, yes], dim=-1)
            ln = torch.logsumexp(lp[:, no], dim=-1)
            out[idx] = torch.sigmoid(ly - ln).cpu().numpy()
            if (b // batch_size) % 100 == 0:
                log(f"local judge {b + len(idx)}/{len(rows)} ({time.time() - t0:.0f}s)")
    return out.tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="variants")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    path = Path(args.out) / f"{args.name}_judged.jsonl"
    rows = read_jsonl(path)
    if rows and "local_p_yes" in rows[0] and not args.force:
        log(f"[skip] {path} already has local judge scores")
        return
    ps = p_yes(rows)
    for r, p in zip(rows, ps):
        r["local_p_yes"] = round(p, 4)
        r["local_preserves_meaning"] = p >= 0.5
    write_jsonl(path, rows)
    agree = [r["local_preserves_meaning"] == r["judge_preserves_meaning"] for r in rows if not r.get("judge_missing")]
    log(f"local judge: {sum(r['local_preserves_meaning'] for r in rows)}/{len(rows)} yes; raw agreement with API judge "
        f"{np.mean(agree) if agree else float('nan'):.3f}")


if __name__ == "__main__":
    main()
