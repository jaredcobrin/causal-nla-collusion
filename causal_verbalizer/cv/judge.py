"""Blind judge queue, scored by Claude Haiku 5.5 subagents (.claude/agents/judge.md, lowest effort).

A stage asks for scores of items {key, question, answer, target}. Items already scored are
returned; the rest go into shuffled queue batches with anonymous ids and no condition information
(the judge cannot tell steered from unsteered replies), and the stage reports itself as pending.
Once scores/<batch>.jsonl exist, rerunning the stage picks them up.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cv.io import RUNS, h, read_json, write_json

JDIR = RUNS / "judge"


def _known() -> dict[str, tuple]:
    """key -> (score or None for REFUSAL, coherence)."""
    out = {}
    for f in sorted((JDIR / "scores").glob("*.jsonl")):
        keys = read_json(JDIR / "keys" / f"{f.stem}.json")
        for line in f.read_text().splitlines():
            if line.strip():
                s = json.loads(line)
                v = s.get("score", s.get("trait"))
                out[keys[s["id"]]] = (None if str(v).upper() == "REFUSAL" else float(v), float(s["coherence"]))
    return out


def _queued() -> set[str]:
    return {k for f in (JDIR / "keys").glob("*.json") for k in read_json(f).values()} if (JDIR / "keys").exists() else set()


def pending_batches() -> list[Path]:
    q = JDIR / "queue"
    return [p for p in sorted(q.glob("*.json")) if not (JDIR / "scores" / f"{p.stem}.jsonl").exists()] if q.exists() else []


def scores(items: list[dict], meta: dict, batch_size: int = 100, seed: int = 0) -> dict[str, tuple] | None:
    """Returns {key: (score, coherence)} once every item is scored, else queues the rest and returns None."""
    known = _known()
    missing = [it for it in items if it["key"] not in known]
    if not missing:
        return {it["key"]: known[it["key"]] for it in items}
    queued = _queued()
    new = [it for it in missing if it["key"] not in queued]
    new = [new[i] for i in np.random.default_rng(seed).permutation(len(new))]
    for i in range(0, len(new), batch_size):
        chunk = new[i:i + batch_size]
        bid = h("".join(it["key"] for it in chunk), 10)
        ids = [h(bid + it["key"], 8) for it in chunk]
        write_json(JDIR / "keys" / f"{bid}.json", {a: it["key"] for a, it in zip(ids, chunk)})
        write_json(JDIR / "queue" / f"{bid}.json", dict(
            rubric=meta["rubric"], coherence_rubric=meta["coherence_rubric"],
            items=[{"id": a, "question": it["question"], "answer": it["answer"],
                    **({"target": it["target"]} if it.get("target") else {})} for a, it in zip(ids, chunk)]))
    return None


def retest(prefix: str, n: int = 100, seed: int = 0) -> dict | None:
    """Judge control: re-score n random already-scored items of this run under new anonymous ids
    and report how much the judge disagrees with itself."""
    known, pool = _known(), {}
    for q in sorted((JDIR / "queue").glob("*.json")):
        b, keys = read_json(q), read_json(JDIR / "keys" / f"{q.stem}.json")
        for it in b["items"]:
            k = keys[it["id"]]
            if k.startswith(prefix) and not k.endswith("|retest") and known.get(k, (None,))[0] is not None:
                pool[k] = (it, {"rubric": b["rubric"], "coherence_rubric": b["coherence_rubric"]})
    if len(pool) < n:
        return None
    pick = [sorted(pool)[i] for i in np.random.default_rng(seed).choice(len(pool), n, replace=False)]
    by_rubric: dict[str, list] = {}
    for k in pick:
        it, meta = pool[k]
        by_rubric.setdefault(json.dumps(meta), []).append({**it, "key": k + "|retest"})
    got = [scores(items, json.loads(m), seed=seed) for m, items in by_rubric.items()]
    if any(g is None for g in got):
        return None
    again = {k: v for g in got for k, v in g.items()}
    a = np.array([known[k][0] for k in pick])
    b = np.array([again[k + "|retest"][0] if again[k + "|retest"][0] is not None else np.nan for k in pick])
    ok = ~np.isnan(b)
    return dict(n=int(ok.sum()), correlation=float(np.corrcoef(a[ok], b[ok])[0, 1]),
                mean_abs_diff=float(np.mean(np.abs(a[ok] - b[ok]))),
                same_side_of_50=float(np.mean((a[ok] > 50) == (b[ok] > 50))))
