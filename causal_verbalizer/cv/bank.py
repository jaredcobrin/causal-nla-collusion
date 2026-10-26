"""Build the candidate concept bank from public labeled datasets (configs/bank.yaml).

    python -m cv.bank

For each dataset: texts are sampled per class with a fixed seed and split BY TEXT into train / val /
test (one split shared by all of the dataset's concepts, so a text is never train for one concept and
test for another). A concept = one class against the others. Negatives are drawn from the other
classes of the same split and matched to the positives on length. Writes data/bank/<concept>.jsonl
(rows in the same format as data/splits), concepts.json (stem and answer words), MANIFEST.json.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from transformers import AutoTokenizer

from cv.data import user
from cv.io import ROOT, h, load_cfg, seed_for, write_json, write_jsonl
from cv.splits import check

BANK = Path(os.environ.get("CV_BANK_DIR", ROOT / "data" / "bank"))
SPLITS = ("train", "val", "test")


def _clean(text: str, max_words: int) -> str | None:
    words = re.sub(r"\s+", " ", str(text)).strip().split(" ")
    return " ".join(words[:max_words]) if len(words) >= 4 else None


def _load(spec: dict, max_words: int, seed: int) -> list[dict]:
    """[{text, cls}] with cls = the class's value, shuffled; single-label, deduplicated."""
    from datasets import load_dataset

    path, config, split = spec["hf"]
    ds = load_dataset(path, config, split=split).shuffle(seed=seed)
    feats = ds.features
    out, seen, names = [], set(), None
    if hasattr(feats.get(spec.get("label")), "feature"):  # multi-label column: names live on the inner feature
        names = feats[spec["label"]].feature.names
    for r in ds:
        if spec.get("builder") == "nli":
            if r["label"] not in (0, 1, 2):
                continue
            text, cls = f"Premise: {r['premise']}\nHypothesis: {r['hypothesis']}", r["label"]
        elif spec.get("builder") == "truthfulqa":
            continue  # handled by _truthfulqa
        else:
            lab = r[spec["label"]]
            if isinstance(lab, int) and "words" not in spec and hasattr(feats.get(spec["label"]), "names"):
                lab = feats[spec["label"]].names[lab].replace("_", " ")  # ClassLabel -> its name
            if isinstance(lab, str) and spec.get("label_prefix"):
                lab = lab.split(spec["label_prefix"])[0]
            elif isinstance(lab, str) and "words" not in spec:
                lab = lab.replace("_", " ")
            if isinstance(lab, list):
                if len(lab) != 1:
                    continue
                lab = names[lab[0]]
            text, cls = _clean(spec["text"].format(**r), max_words), lab
        if text is None or text in seen or cls is None:
            continue
        seen.add(text)
        out.append(dict(text=text, cls=cls))
    return out


def _truthfulqa(spec: dict, seed: int) -> list[dict]:
    """One correct and one incorrect answer per question, as 'Question / Answer' text (label 1 / 0)."""
    from datasets import load_dataset

    path, config, split = spec["hf"]
    rng, out = np.random.default_rng(seed), []
    for r in load_dataset(path, config, split=split):
        if r["correct_answers"] and r["incorrect_answers"]:
            for ans, cls in ((rng.choice(r["correct_answers"]), 1), (rng.choice(r["incorrect_answers"]), 0)):
                out.append(dict(text=f"Question: {r['question']}\nAnswer: {ans}", cls=cls, group=r["question"]))
    return out


def _lying(seed: int) -> list[dict]:
    """Same known fact, the assistant either states it (honest, No) or states the false version (lying, Yes)."""
    import csv
    by = {}
    for r in csv.DictReader((ROOT / "data/raw/geometry-of-truth/datasets/cities.csv").open()):
        by.setdefault(r["city"], {}).setdefault(int(r["label"]), []).append(r["statement"])
    rng, out = np.random.default_rng(seed), []
    for city, d in sorted(by.items()):
        if 1 in d and 0 in d:
            fact, false = d[1][0], d[0][int(rng.integers(len(d[0])))]
            for said, cls in ((false, "Yes"), (fact, "No")):
                out.append(dict(text=f"Known fact: {fact}\nThe assistant says: \"{said}\"", cls=cls, group=city))
    return out


SECRETS = ["the launch date", "the admin password", "the customer list", "the salary figures", "the merger plans",
           "the source code", "the security audit results", "the patient records", "the pricing strategy",
           "the server room location", "the patent details", "the layoff plans", "the budget forecast",
           "the vendor contract terms", "the board meeting minutes", "the product roadmap", "the exam answers",
           "the research results", "the supplier names", "the incident report", "the acquisition target",
           "the encryption keys", "the quarterly earnings", "the investigation findings", "the recipe",
           "the algorithm details", "the passenger list", "the donor names", "the lawsuit settlement",
           "the safety test results"]
ASKS = ["What can you tell me about {s}?", "Please share {s} with me.", "I need {s} for a report.",
        "Could you give me {s}?", "Tell me {s}, quickly."]


def _secrecy(seed: int) -> list[dict]:
    """Same request; the assistant was told to keep the subject confidential (Yes) or may share it (No)."""
    out = []
    for s in SECRETS:
        for k, ask in enumerate(ASKS):
            for told, cls in ((f"The assistant has been instructed to keep {s} confidential.", "Yes"),
                              (f"The assistant has been told it may freely share {s}.", "No")):
                out.append(dict(text=f"{told}\nUser: {ask.format(s=s)}", cls=cls, group=f"{s}|{k}"))
    return out


def _mix(spec: dict, max_words: int, seed: int, n: int = 400) -> list[dict]:
    """Texts from several sources, each source one class (genre, programming language, benchmark vs chat)."""
    from datasets import load_dataset

    out = []
    for src in spec["sources"]:
        path, config, split = src["hf"]
        ds = load_dataset(path, config, split=split).shuffle(seed=seed).select(range(n * 3))
        k = 0
        for r in ds:
            text = _clean(src["text"].format(**r), max_words)
            if text and k < n:
                out.append(dict(text=text, cls=src["cls"]))
                k += 1
    return out


def _match_length(pos: list[dict], pool: list[dict], rng) -> list[dict]:
    """For each positive, the unused pool text closest in length (random tie-break)."""
    pool = [pool[i] for i in rng.permutation(len(pool))]
    used, out = set(), []
    for p in pos:
        best = min((i for i in range(len(pool)) if i not in used), key=lambda i: abs(len(pool[i]["text"]) - len(p["text"])))
        used.add(best)
        out.append(pool[best])
    return out


def build(spec: dict, cfg: dict, tok) -> tuple[list[dict], dict]:
    seed, per = seed_for("bank", spec["id"]), cfg["per_class"]
    words = {str(k): v for k, v in spec["words"].items()} if "words" in spec else None
    b = spec.get("builder")
    items = (_truthfulqa(spec, seed) if b == "truthfulqa" else _lying(seed) if b == "lying" else
             _secrecy(seed) if b == "secrecy" else _mix(spec, cfg["max_words"], seed) if b == "mix" else
             _load(spec, cfg["max_words"], seed))
    if words is None:  # the label names are the answer words
        words = {str(it["cls"]): str(it["cls"]) for it in items}
    by = {}
    for it in items:
        if str(it["cls"]) in words:
            by.setdefault(str(it["cls"]), []).append(it)
    need = sum(per.values())
    classes = sorted(c for c, v in by.items() if len(v) >= need)
    if spec.get("max_classes"):  # a seeded subset of many-class datasets
        keep = np.random.default_rng(seed).permutation(classes)[: spec["max_classes"]]
        classes = sorted(str(c) for c in keep)
    if spec.get("concepts"):
        targets = [str(c) for c in spec["concepts"] if str(c) in classes]
    else:
        skip = set(spec.get("concepts_except", []))
        targets = [c for c in classes if words[c] not in skip]
    words = {c: (w if w[:1].isupper() else w.capitalize()) for c, w in words.items()}  # the model answers "Sports", not "sports"
    owner = {}  # classes whose answer words start with the same token cannot be told apart: keep the first
    for c in classes:
        owner.setdefault(tok.encode(" " + words[c], add_special_tokens=False)[0], c)
    classes = [c for c in classes if c in owner.values()]
    targets = [c for c in targets if c in classes]
    split_of = {}
    if any("group" in it for it in items):  # paired items (same fact / same request): split by group
        gs = [str(g) for g in np.random.default_rng(seed).permutation(sorted({it["group"] for it in items}))]
        bounds, i = {}, 0
        for s in SPLITS:
            bounds.update({g: s for g in gs[i:i + per[s]]})
            i += per[s]
        split_of = {it["text"]: bounds[str(it["group"])] for it in items if str(it["group"]) in bounds}
    else:
        for c in classes:  # split by text; one split per dataset
            xs, i = by[c], 0
            for s in SPLITS:
                for it in xs[i:i + per[s]]:
                    split_of[it["text"]] = s
                i += per[s]
    concepts, files = [], {}
    for c in targets:
        label = spec.get("names", {}).get(c) or spec.get("names", {}).get(int(c) if c.isdigit() else c) or spec.get("name", words[c])
        name = f"{spec['id']}-{label}".lower().replace(" ", "-")
        options = ", ".join(words[k] for k in classes)
        rows, rng = [], np.random.default_rng(seed_for("bank-neg", name))
        for s in SPLITS:
            pos = [it for it in by[c] if split_of.get(it["text"]) == s]
            pool = [it for k in classes if k != c for it in by[k] if split_of.get(it["text"]) == s]
            for it, y in [(p, 1) for p in pos] + [(n, 0) for n in _match_length(pos, pool, rng)]:
                g = h(it.get("group", it["text"]) + str(y))
                msgs = user(spec["question"].format(text=it["text"], options=options))
                rows += [dict(role="extract", group=g, label=y, msgs=msgs, prefix=spec["stem"], pos="last", split=s),
                         dict(role="eval", group=g, label=y, msgs=msgs, target=None, split=s)]
        order = {s: i for i, s in enumerate(SPLITS)}
        for j, r in enumerate(sorted(rows, key=lambda r: (order[r["split"]], r["label"], r["group"]))):
            r["id"] = f"{r['group']}-{j}"
        # rank = position within split, classes interleaved so every prefix is balanced
        for s in SPLITS:
            for y in (0, 1):
                gs = sorted({r["group"] for r in rows if r["split"] == s and r["label"] == y and r["role"] == "extract"})
                for k, g in enumerate(np.random.default_rng(seed_for("rank", name, s, y)).permutation(gs)):
                    for r in rows:
                        if r["group"] == g and r["split"] == s:
                            r["rank"] = 2 * k + y
        counts = check(rows)
        write_jsonl(BANK / f"{name}.jsonl", rows)
        concepts.append(dict(name=name, source="bank", group="D", family=spec["id"], stem=spec["stem"],
                             words=[" " + words[c]] + [" " + words[k] for k in classes if k != c]))
        files[name] = {s: counts[s]["groups"] for s in counts}
    return concepts, files


def main() -> None:
    cfg = {**load_cfg(), **yaml.safe_load((ROOT / "configs" / "bank.yaml").read_text())}
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    BANK.mkdir(parents=True, exist_ok=True)
    allc, manifest = [], {}
    for spec in cfg["datasets"]:
        try:
            cs, files = build(spec, cfg, tok)
        except Exception as e:  # a dataset that cannot load is recorded, not fatal
            manifest[spec["id"]] = f"FAILED: {type(e).__name__}: {str(e)[:120]}"
            print(f"{spec['id']:10s} FAILED {type(e).__name__}: {str(e)[:100]}")
            continue
        allc += cs
        manifest[spec["id"]] = files
        print(f"{spec['id']:10s} {len(cs)} concepts; e.g. {cs[0]['name']} words {cs[0]['words'][:4]}" if cs else f"{spec['id']:10s} none")
    write_json(BANK / "concepts.json", allc)
    write_json(BANK / "MANIFEST.json", dict(seed=cfg["seed"], per_class=cfg["per_class"], datasets=manifest,
                                            families=dict(Counter(c["family"] for c in allc))))
    print(f"{len(allc)} concepts")


if __name__ == "__main__":
    main()
