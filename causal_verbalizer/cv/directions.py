"""Directions = mean(concept activations) - mean(non-concept activations), from the train split only,
plus the data-quality gate (split-half stability, held-out-group AUROC, answer-token bias) and a
shuffled-label control direction built from the same activations."""

from __future__ import annotations

import numpy as np
import torch

from cv import judge
from cv.io import RUNS, h, read_jsonl, seed_for, write_jsonl
from cv.stats import auroc


def _cos(a, b) -> float:
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


def _diff(acts, labels, strata=None):
    """Difference of class means. With strata (the matching answer of each row's item), the
    difference is taken within each stratum and averaged, so an answer-letter component cancels."""
    if strata is None:
        return acts[labels == 1].mean(0) - acts[labels == 0].mean(0)
    return torch.stack([_diff(acts[strata == s], labels[strata == s]) for s in np.unique(strata)]).mean(0)


def gate(acts, labels, groups, seed, strata=None) -> tuple[torch.Tensor, dict]:
    v = _diff(acts, labels, strata)
    ug = np.random.default_rng(seed).permutation(sorted(set(groups)))
    half = np.isin(groups, ug[: len(ug) // 2])
    sub = (lambda m: None) if strata is None else (lambda m: strata[m])
    va, vb = _diff(acts[half], labels[half], sub(half)), _diff(acts[~half], labels[~half], sub(~half))
    auc = np.mean([auroc((acts[~half] @ (va / va.norm())).numpy(), labels[~half]),
                   auroc((acts[half] @ (vb / vb.norm())).numpy(), labels[half])])
    stats = dict(norm=float(v.norm()), split_half=_cos(va, vb), auroc=float(auc),
                 n_pos=int(labels.sum()), n_neg=int((1 - labels).sum()), n_groups=len(ug))
    if strata is not None and len(np.unique(strata)) == 2:
        # Each stratum's difference is concept + letter for one matching answer and concept - letter
        # for the other, so half their difference estimates the answer-letter direction.
        a, b = np.unique(strata)
        letter = (_diff(acts[strata == a], labels[strata == a]) - _diff(acts[strata == b], labels[strata == b])) / 2
        stats["letter_cos"] = _cos(v, letter)
    return v, stats


def _finish(acts_by_layer, labels, groups, name, strata=None):
    labels = np.asarray(labels)
    strata = None if strata is None else np.asarray(strata)
    out = {"v": {}, "shuffled": {}, "gate": {}}
    shuf = np.random.default_rng(seed_for(name, "shuffled")).permutation(labels)
    for l, acts in acts_by_layer.items():
        out["v"][l], out["gate"][l] = gate(acts, labels, groups, seed_for(name, "half"), strata)
        out["shuffled"][l] = _diff(acts, shuf)
    return out


def _paired_balance(rows):
    """For A/B and Yes/No items keep only items whose (matching, non-matching) answers are the two
    most common answers, in either order, and stratify by the matching answer: the two strata then
    contribute equally and the answer-token direction cancels exactly."""
    match = {r["group"]: r["ans"] for r in rows if r["label"] == 1}
    other = {r["group"]: r["ans"] for r in rows if r["label"] == 0}
    counts = {}
    for a in [*match.values(), *other.values()]:
        counts[a] = counts.get(a, 0) + 1
    top = set(sorted(counts, key=counts.get)[-2:])
    keep = [r for r in rows if {match[r["group"]], other[r["group"]]} == top]
    return keep, [match[r["group"]] for r in keep]


def build_static(lm, name: str, rows: list[dict], cfg: dict, keep_groups: set | None = None) -> dict:
    """Extract at the saved layers from the first `train_start` train groups; use the rest of the
    train split only if split-half stability is below threshold at any saved layer.
    keep_groups restricts the train split to those groups (see build_filtered)."""
    layers = cfg["saved_layers"]
    train = sorted((r for r in rows if r["role"] == "extract" and r["split"] == "train"), key=lambda r: r["rank"])
    if keep_groups is not None:
        train = [r for r in train if r["group"] in keep_groups]

    def acts_for(rs):
        texts = [lm.render(r["msgs"], r["prefix"]) for r in rs]
        return lm.run(texts, read_layers=layers, pos=[r["pos"] for r in rs], bs=cfg["batch"]["forward"])["acts"]

    paired = bool(train[0].get("ans"))
    if paired:
        train, _ = _paired_balance(train)

    def finish(rs, acts):
        strata = _paired_balance(rs)[1] if paired else None
        return _finish(acts, [r["label"] for r in rs], [r["group"] for r in rs], name, strata)

    if keep_groups is None:
        used = [r for r in train if r["rank"] < cfg["train_start"]]
    else:  # ranks have gaps once groups are dropped: take the first train_start kept groups
        first = set(sorted({r["rank"] for r in train})[: cfg["train_start"]])
        used = [r for r in train if r["rank"] in first]
    acts = acts_for(used)
    out, grown = finish(used, acts), False
    if min(g["split_half"] for g in out["gate"].values()) < cfg["thresholds"]["split_half"] and len(used) < len(train):
        rest = train[len(used):]
        extra = acts_for(rest)
        acts = {l: torch.cat([acts[l], extra[l]]) for l in layers}
        used += rest
        out, grown = finish(used, acts), True
    out["data"] = dict(extract_rows=len(used), train_rows_available=len(train), grown=grown)
    return out


def build_filtered(lm, name: str, rows: list[dict], meta: dict, cfg: dict) -> dict | None:
    """Like build_static, but only from train groups where the model's own greedy reply already shows
    the concept (judge score > 50 on that group's test prompt). The plain direction contrasts what the
    dataset's answers say, whether or not the model would give them; this one contrasts examples
    where the model does the behaviour. Train split only. None while judge scores are pending."""
    ev = sorted((r for r in rows if r["role"] == "eval" and r["split"] == "train"), key=lambda r: r["rank"])
    path = RUNS / "gens" / f"filter-{name}-train.jsonl"
    if path.exists():
        replies = [g["reply"] for g in read_jsonl(path)]
    else:
        replies = lm.generate([lm.render(r["msgs"]) for r in ev], max_new_tokens=cfg["replies"]["max_new_tokens"],
                              temperature=0, seed=seed_for(name, "filter"), bs=cfg["batch"]["generate"])
        write_jsonl(path, [dict(id=r["id"], reply=a) for r, a in zip(ev, replies)])
    key = lambda r, a: f"filter|{name}|{r['id']}|{h(a)}"
    sc = judge.scores([dict(key=key(r, a), question=r["msgs"][0]["content"], answer=a, target=r["target"])
                       for r, a in zip(ev, replies)], meta, seed=seed_for(name, "judge-filter"))
    if sc is None:
        return None
    kept = {r["group"] for r, a in zip(ev, replies) if sc[key(r, a)][0] is not None and sc[key(r, a)][0] > 50}
    data = dict(groups_scored=len(ev), groups_kept=len(kept))
    if len(kept) < cfg["filter"]["min_groups"]:
        return dict(v={}, shuffled={}, gate={}, data=data,
                    failed=f"only {len(kept)} train groups where the model shows the behaviour")
    out = build_static(lm, name, rows, cfg, keep_groups=kept)
    out["data"] = {**out["data"], **data}
    return out


def build_pv(lm, name: str, rows: list[dict], meta: dict, cfg: dict) -> dict | None:
    """Persona Vectors recipe: answer the extraction questions under each of the 5 positive and
    negative instructions, keep pairs the judge confirms (pos trait >= 50, neg < 50, both coherent
    >= 50), average activations over response tokens. Returns None while judge scores are pending."""
    pc, layers = cfg["pv"], cfg["saved_layers"]
    qs = [r for r in rows if r["role"] == "pv_question"]
    path = RUNS / "gens" / f"{name}-pv-extract.jsonl"
    gens = read_jsonl(path) if path.exists() else []
    n_samples = pc["samples"]
    while True:
        have = {g["k"] for g in gens}
        for k in range(n_samples):
            if k in have:
                continue
            todo = [(qi, ii, side) for qi in range(len(qs)) for ii in range(len(meta["instructions"])) for side in ("pos", "neg")]
            texts = [lm.render(qs[qi]["msgs"], system=meta["instructions"][ii][side]) for qi, ii, side in todo]
            answers = lm.generate(texts, max_new_tokens=pc["max_new_tokens"], temperature=pc["temperature"],
                                  seed=seed_for(name, "pvx", k), bs=cfg["batch"]["generate"])
            gens += [dict(key=f"pvx|{name}|{qi}|{ii}|{side}|{k}|{h(a)}", qi=qi, ii=ii, side=side, k=k, answer=a)
                     for (qi, ii, side), a in zip(todo, answers)]
            write_jsonl(path, gens)
        sc = judge.scores([dict(key=g["key"], question=qs[g["qi"]]["msgs"][0]["content"], answer=g["answer"]) for g in gens],
                          meta, seed=seed_for(name, "judge-pvx"))
        if sc is None:
            return None
        by = {(g["qi"], g["ii"], g["k"], g["side"]): g for g in gens}
        keep = []
        for (qi, ii, k, side), g in by.items():
            if side != "pos":
                continue
            n = by[(qi, ii, k, "neg")]
            (tp, cp), (tn, cn) = sc[g["key"]], sc[n["key"]]
            if tp is not None and tn is not None and tp >= 50 and tn < 50 and cp >= 50 and cn >= 50:
                keep += [(g, 1), (n, 0)]
        data = dict(generated=len(gens), kept_pairs=len(keep) // 2, samples_per_prompt=n_samples,
                    grown=n_samples > pc["samples"])
        if len(keep) < 20:
            return dict(failed="fewer than 10 judge-confirmed pairs", data=data)
        texts, starts = [], []
        for g, _ in keep:
            sys = meta["instructions"][g["ii"]][g["side"]]
            prompt = lm.render(qs[g["qi"]]["msgs"], system=sys)
            texts.append(prompt + g["answer"])
            starts.append(len(lm.tok.encode(prompt, add_special_tokens=False)))
        acts = lm.run(texts, read_layers=layers, resp_start=starts, bs=cfg["batch"]["forward"])["acts"]
        out = _finish(acts, [y for _, y in keep], [g["qi"] for g, _ in keep], name)
        ok = min(g["split_half"] for g in out["gate"].values()) >= cfg["thresholds"]["split_half"]
        if ok or n_samples >= 2 * pc["samples"]:
            out["data"] = data
            return out
        n_samples *= 2
