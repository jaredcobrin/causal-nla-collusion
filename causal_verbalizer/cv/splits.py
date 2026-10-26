"""Group-level, seeded, class-balanced splits with explicit leakage checks.

Groups (never rows) are assigned to splits, so near-duplicate rows from one source item cannot
straddle train and test. Groups are shuffled with a per-concept seed *within each stratum* (answer
letter or class label) and the strata are interleaved, so every prefix of a split is balanced and
no split inherits an ordering quirk of the source file.
"""

from __future__ import annotations

import itertools
import re
from collections import Counter

import numpy as np

SPLITS = ("test", "val", "train", "screen")  # assignment order for groups without a source split


def _stratum(rows: list[dict]) -> str:
    s = sorted({str(r.get("stratum", r.get("label"))) for r in rows if r["role"] != "pv_question"})
    return ",".join(s)


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def content(row: dict) -> str:
    return norm(" ".join(m["content"] for m in row["msgs"]))


def assign(rows: list[dict], quota: dict, seed: int) -> list[dict]:
    """Return rows with `split` and `rank` (position of its group within the split) filled in.
    Groups beyond the quotas are dropped (counted in the manifest as unused)."""
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r["group"], []).append(r)
    pre = {}
    for g, rs in groups.items():
        s = {r.get("split") for r in rs}
        assert len(s) == 1, f"group {g} mixes pre-assigned splits {s}"
        pre[g] = s.pop()
    rng = np.random.default_rng(seed)

    proportional = any("stratum" in r for r in rows)  # answer letters: keep their natural mix

    def ordered(gs):
        """Shuffle within strata, then merge so every prefix keeps the strata balanced (class labels)
        or in their natural proportions (answer letters)."""
        strata: dict[str, list[str]] = {}
        for g in sorted(gs):
            strata.setdefault(_stratum(groups[g]), []).append(g)
        lists = [list(rng.permutation(v)) for _, v in sorted(strata.items())]
        if proportional:
            keyed = [((i + rng.random()) / len(v), g) for v in lists for i, g in enumerate(v)]
            return [g for _, g in sorted(keyed)]
        return [g for tup in itertools.zip_longest(*lists) for g in tup if g is not None]

    split_of: dict[str, tuple[str, int]] = {}
    free = ordered([g for g, s in pre.items() if s is None])
    i = 0
    for name in SPLITS:
        for rank, g in enumerate(free[i:i + quota[name]]):
            split_of[g] = (name, rank)
        i += quota[name]
    overflow: list[str] = []
    for name in SPLITS:  # source-defined splits: keep within quota; test overflow joins screen
        gs = ordered([g for g, s in pre.items() if s == name])
        if name == "screen":
            gs = overflow + gs
        if name == "test":
            overflow = gs[quota[name]:]
        for rank, g in enumerate(gs[:quota[name]]):
            split_of[g] = (name, rank)

    out = []
    for g, rs in groups.items():
        if g in split_of:
            name, rank = split_of[g]
            for j, r in enumerate(rs):
                out.append({**r, "split": name, "rank": rank, "id": f"{g}-{j}"})
    return out


def check(rows: list[dict]) -> dict:
    """Assert no group and no prompt text appears in two splits; return per-split counts."""
    by_group, by_text = {}, {}
    for r in rows:
        for key, table in ((r["group"], by_group), (content(r), by_text)):
            table.setdefault(key, set()).add(r["split"])
    bad_g = [g for g, s in by_group.items() if len(s) > 1]
    bad_t = [t for t, s in by_text.items() if len(s) > 1]
    assert not bad_g, f"groups in several splits: {bad_g[:3]}"
    assert not bad_t, f"identical prompt text in several splits: {bad_t[:3]}"
    counts = {}
    for name in SPLITS:
        rs = [r for r in rows if r["split"] == name]
        if not rs:
            continue
        ext = [r for r in rs if r["role"] == "extract"]
        ev = [r for r in rs if r["role"] == "eval"]
        counts[name] = dict(
            groups=len({r["group"] for r in rs}),
            extract=dict(Counter(str(r["label"]) for r in ext)),
            eval=dict(Counter(str(r.get("stratum", r["label"])) for r in ev)),
            pv_questions=sum(r["role"] == "pv_question" for r in rs),
        )
    return counts
