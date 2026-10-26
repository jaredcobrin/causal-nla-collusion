"""Build every concept's splits once, on CPU, before any GPU run.

    python -m cv.prepare

Writes data/splits/<concept>.jsonl (+ .meta.json), neutral.jsonl (side-effect check text),
MANIFEST.json (counts, source hashes, leakage checks).
Everything downstream reads only these files, so the splits are fixed and auditable.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from cv import data
from cv.io import SPLITS, load_cfg, load_concepts, seed_for, write_json, write_jsonl
from cv.splits import assign, check, content

SPLIT_ORDER = ("train", "val", "test", "screen")


def _sha(path: str) -> str:
    p = Path(path) if Path(path).is_absolute() else data.RAW / path
    if not p.exists():
        return "n/a"
    files = sorted(p.rglob("*")) if p.is_dir() else [p]
    d = hashlib.sha256()
    for f in files:
        if f.is_file():
            d.update(f.read_bytes())
    return d.hexdigest()[:16]


def _check_rows(tok, rows, meta) -> None:
    """Appended answers ('(A', 'Yes') are single tokens; every test prompt has what its rubric needs."""
    for r in rows:
        if r["role"] == "extract" and r.get("ans"):
            assert len(tok.encode(r["ans"], add_special_tokens=False)) == 1, r["ans"]
        if r["role"] == "eval":
            assert ("{target}" in meta["rubric"]) == (r["target"] is not None), r


def _wikitext(tok, split: str, n: int, n_tok: int, seed: int) -> list[dict]:
    from datasets import load_dataset

    paras = [t.strip() for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)["text"]
             if len(t.split()) > 120 and not t.strip().startswith("=")]
    out = []
    for i in np.random.default_rng(seed).permutation(len(paras)):
        ids = tok.encode(paras[i], add_special_tokens=False)
        if len(ids) >= n_tok:
            out.append(dict(id=f"wiki-{split}-{i}", ids=ids[:n_tok]))
        if len(out) == n:
            return out
    raise ValueError("not enough wikitext paragraphs")


def main() -> None:
    cfg = load_cfg()
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    manifest, contents = {"seed": cfg["seed"], "concepts": {}}, {}
    for c in load_concepts():
        rows, meta = data.load(c)
        quota = {**cfg["quota"], **meta.pop("quota", {})}
        rows = assign(rows, quota, seed_for(cfg["seed"], c["name"]))
        counts = check(rows)
        _check_rows(tok, rows, meta)
        write_jsonl(SPLITS / f"{c['name']}.jsonl", rows)
        write_json(SPLITS / f"{c['name']}.meta.json", {**meta, **c})
        contents[c["name"]] = {s: {content(r) for r in rows if r["split"] == s} for s in SPLIT_ORDER}
        manifest["concepts"][c["name"]] = dict(
            group=c["group"], source=c["source"], quota=quota, splits=counts,
            files={f: _sha(f) for f in meta["files"]},
            **{k: meta[k] for k in ("min_label_confidence", "dropped_duplicates") if k in meta})
        print(f"{c['name']:72s} " + " ".join(f"{s}={v['groups']}" for s, v in counts.items()))

    # Cross-concept: a prompt that is test for one concept should not be train for another.
    cross = {}
    for a, ca in contents.items():
        for b, cb in contents.items():
            n = len(ca["test"] & cb["train"]) if a != b else 0
            if n:
                cross[f"{a} test & {b} train"] = n
    manifest["cross_concept_test_train_overlap"] = cross
    print("cross-concept test/train overlaps:", cross or "none")

    write_jsonl(SPLITS / "neutral.jsonl", _wikitext(tok, "test", **_wk(cfg["neutral"]), seed=seed_for("neutral")))
    write_json(SPLITS / "MANIFEST.json", manifest)


def _wk(c: dict) -> dict:
    return dict(n=c["n"], n_tok=c["tokens"])


if __name__ == "__main__":
    main()
