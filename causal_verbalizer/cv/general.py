"""General concepts (configs/general.yaml): contrast sets from public datasets, cached activations, readout tests.

    python -m cv.general build               download/sample every source -> data/general/texts.jsonl
    python -m cv.general embed               layer-20 activations of every text -> runs/general/acts.pt   (GPU)
    python -m cv.general eval [--concepts]   leave-one-source-out readout test per concept -> runs/general/

A concept's positives are the union of several sources. To test a concept's direction on a source, the
direction is built WITHOUT that source (positives and negatives), then scored on that source's own
positives against that source's own negatives (so style cannot leak), and compared with a keyword
(TF-IDF + logistic regression) baseline trained on the same data.
"""

from __future__ import annotations

import argparse
import collections
import fnmatch
import json

import numpy as np
import torch
import yaml

from cv.io import ROOT, RUNS, Log, load_cfg, read_json, read_jsonl, seed_for, write_json, write_jsonl
from cv.stats import auroc

DATA = ROOT / "data" / "general"
OUT = RUNS / "general"
MAX_WORDS = 100
PER_CLASS = 250            # texts kept per (source, class)
MIN_TEST = 40              # positives and negatives a held-out source needs to be tested


def general_cfg() -> dict:
    return yaml.safe_load((ROOT / "configs" / "general.yaml").read_text())


# ---- build: sample every source ------------------------------------------------------------------------
class _Blank(dict):
    def __missing__(self, k):
        return ""


def _names(feature):
    """Class names of a ClassLabel feature (or a list of them), else None."""
    f = getattr(feature, "feature", feature)
    return getattr(f, "names", None)


def _classes(spec: dict, row: dict, feats) -> list[str]:
    c = spec["cls"]
    if "const" in c:
        return [c["const"]]
    if c.get("pku"):  # PKU-SafeRLHF: harm categories of whichever response is unsafe; "safe" when both are safe
        cats = {k for r in (0, 1) if not row.get(f"is_response_{r}_safe", True)
                for k, v in (row.get(f"response_{r}_harm_category") or {}).items() if v}
        return (["unsafe"] + sorted(cats)) if cats else ["safe"]
    if c.get("beaver"):
        return ["safe" if row.get("is_safe") else "unsafe"] + [k for k, v in (row.get("category") or {}).items() if v]
    v = row.get(c["col"])
    if v is None:
        return []
    if "thr" in c:
        return ["high"] if v > c["thr"]["high"] else ["low"] if v < c["thr"]["low"] else []
    names = c.get("names") or (_names(feats[c["col"]]) if feats and c["col"] in feats else None)
    items = list(v) if c.get("multi") else [v]
    out = [str(names[i]) if names and isinstance(i, int) and 0 <= i < len(names) else str(i) for i in items]
    return [x for x in out if x not in c.get("drop", [])]


def sample_source(key: str, spec: dict, per_class: int, max_rows: int, seed: int = 0) -> list[dict]:
    """~per_class texts of every class of one source. Streams a shuffled buffer, except for datasets stored
    sorted by class (`ordered: true`), which are loaded whole so the buffer sees every class."""
    per_class = spec.get("cap", per_class)
    if "local" in spec:  # texts written by subagents: JSONL {label, text}; label 1 = the concept
        rows = [json.loads(l) for f in sorted(ROOT.glob(spec["local"])) for l in f.read_text().splitlines() if l.strip()]
        return [dict(src=key, cls=["pos" if r["label"] == 1 else "neg"], text=" ".join(r["text"].split()[:MAX_WORDS]))
                for r in rows if len(r["text"].split()) >= 4]
    from datasets import load_dataset
    if "configs" in spec:  # one source made of several configs; the config key is the class (languages)
        out = []
        for cls, cfg_name in spec["configs"].items():
            sub = {k: v for k, v in spec.items() if k != "configs"} | {"config": cfg_name, "cls": {"const": cls}}
            out += [dict(r, src=key) for r in sample_source(key, sub, per_class, max_rows, seed)]
        return out
    kw = dict(split=spec["split"], trust_remote_code=spec.get("trust", False))
    if spec.get("ordered"):
        ds = load_dataset(spec["hf"], spec.get("config"), **kw)
        feats = ds.features
        ds = ds.shuffle(seed=seed).select(range(min(len(ds), max_rows)))
    else:
        ds = load_dataset(spec["hf"], spec.get("config"), streaming=True, **kw)
        feats = ds.features
        ds = ds.shuffle(seed=seed, buffer_size=20000)
    item = spec.get("per_item")  # one dialogue row -> one example per utterance, with its own class
    names = spec["cls"].get("names") or (_names(feats[spec["cls"]["col"]]) if feats and spec["cls"].get("col") in (feats or {}) else None)
    keep = {k.lower() for k in spec.get("keep", [])}   # only these classes (e.g. chosen subreddits)
    seen, out, n_read = collections.Counter(), [], 0
    for row in ds:
        n_read += 1
        if n_read > max_rows:
            break
        if item:
            pairs = [(t, [str(item["names"][a])]) for t, a in zip(row[item["text"]], row[item["cls"]]) if 0 <= a < len(item["names"])]
        else:
            pairs = [(spec["text"].format_map(_Blank(row)), [c for c in _classes(spec, row, feats) if not keep or c.lower() in keep])]
        for text, cls in pairs:
            if not cls or all(seen[c] >= per_class for c in cls):
                continue
            text = " ".join(text.split()[:MAX_WORDS])
            if len(text.split()) < 4:
                continue
            for c in cls:
                seen[c] += 1
            out.append(dict(src=key, cls=cls, text=text))
        if names and len(seen) >= len(names) and all(seen[c] >= per_class for c in seen):
            break
        if keep and len(seen) >= len(keep) and all(seen[c] >= per_class for c in seen):
            break
        if "const" in spec["cls"] and seen[spec["cls"]["const"]] >= per_class:
            break
    return out


def build(only: list[str] | None = None) -> None:
    cfg, log = general_cfg(), Log(OUT)
    path = DATA / "texts.jsonl"
    have = read_jsonl(path) if path.exists() else []
    done = {r["src"] for r in have} - set(only or [])
    rows = [{k: v for k, v in r.items() if k != "i"} for r in have if r["src"] not in (only or [])]  # --sources re-samples
    for key, spec in cfg["registry"].items():
        if key in done or (only and key not in only):
            continue
        try:
            got = sample_source(key, spec, PER_CLASS, spec.get("max_rows", 60000))
        except Exception as e:  # noqa: BLE001 - one broken source must not stop the build
            log(f"{key}: FAILED {type(e).__name__}: {str(e)[:150]}")
            continue
        rows += got
        counts = collections.Counter(c for r in got for c in r["cls"])
        log(f"{key}: {len(got)} texts, {len(counts)} classes: " + ", ".join(f"{k}={v}" for k, v in counts.most_common(40)))
        write_jsonl(path, [dict(i=i, **r) for i, r in enumerate(rows)])
    write_jsonl(path, [dict(i=i, **r) for i, r in enumerate(rows)])
    log(f"build finished: {len(rows)} texts from {len({r['src'] for r in rows})} sources")


# ---- embed: layer-20 activations ---------------------------------------------------------------------
def pooled(h: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Average of a text's token activations, skipping the first REAL token (with left padding it is not
    position 0) and any token whose norm is over 3x the text's median-ish norm: those sink tokens carry
    activations ~10x larger than the rest and otherwise swamp the average (cross-source AUROC of the
    technology direction was 0.41 with the first-token bug, 0.89 without)."""
    m = attention_mask.clone().float()
    m[torch.arange(len(m)), m.argmax(1)] = 0
    norm = h.norm(dim=-1)
    ref = (norm * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp(min=1)
    m = m * (norm < 3 * ref).float()
    return (h * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp(min=1)


def embed() -> None:
    """Activations of every text; resumable: only rows beyond what acts.pt already holds are computed."""
    from cv.model import LM
    cfg, log = load_cfg(), Log(OUT)
    rows = read_jsonl(DATA / "texts.jsonl")
    lm, L = LM(cfg["model"]), cfg["layer"]
    have = torch.load(OUT / "acts.pt", weights_only=True) if (OUT / "acts.pt").exists() else None
    start = len(have["mean"]) if have else 0
    new = rows[start:]
    mean, last = torch.zeros(len(new), lm.d, dtype=torch.float16), torch.zeros(len(new), lm.d, dtype=torch.float16)
    order = np.argsort([-len(r["text"]) for r in new])
    for k in range(0, len(order), 48):
        idx = [int(j) for j in order[k:k + 48]]
        enc = lm.tok([new[j]["text"] for j in idx], return_tensors="pt", padding=True, truncation=True,
                     max_length=192, add_special_tokens=False).to(lm.device)
        with torch.no_grad():
            h = lm.model.model(**enc, output_hidden_states=True).hidden_states[L + 1].float()  # output of block L
        mean[idx] = pooled(h, enc.attention_mask).half().cpu()
        last[idx] = h[:, -1].half().cpu()
        if (k // 48) % 100 == 0:
            log(f"embed {start + k}/{len(rows)}")
    OUT.mkdir(parents=True, exist_ok=True)
    if have:
        mean, last = torch.cat([have["mean"], mean]), torch.cat([have["last"], last])
    torch.save(dict(mean=mean, last=last), OUT / "acts.pt")
    log(f"embed finished: {len(rows)} texts ({len(new)} new)")


# ---- evaluate: leave-one-source-out readout test ------------------------------------------------------
def positive_mask(spec: dict, rows: list[dict]) -> tuple[np.ndarray, list[str]]:
    """Which texts are positives of a concept, and the pattern items that matched nothing."""
    mask, unmatched = np.zeros(len(rows), bool), []
    for item in spec["pos"]:
        src, _, pat = item.partition(":")
        hit = [i for i, r in enumerate(rows) if r["src"] == src and (pat == "*" or any(fnmatch.fnmatchcase(c.lower(), pat.lower()) for c in r["cls"]))]
        if hit:
            mask[hit] = True
        else:
            unmatched.append(item)
    return mask, unmatched


def _unit(v):
    return v / np.linalg.norm(v)


def _keyword_auroc(train_pos, train_neg, test_pos, test_neg) -> float:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    vec = TfidfVectorizer(min_df=2, sublinear_tf=True)
    X = vec.fit_transform(train_pos + train_neg)
    clf = LogisticRegression(max_iter=300, C=3.0).fit(X, [1] * len(train_pos) + [0] * len(train_neg))
    s = clf.decision_function(vec.transform(test_pos + test_neg))
    return auroc(s, [1] * len(test_pos) + [0] * len(test_neg))


def _best_threshold(pos_s, neg_s) -> float:
    cand = np.quantile(np.concatenate([pos_s, neg_s]), np.linspace(0.02, 0.98, 97))
    f1 = []
    for t in cand:
        tp, fp, fn = (pos_s >= t).sum(), (neg_s >= t).sum(), (pos_s < t).sum()
        f1.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(cand[int(np.argmax(f1))])


def _contrast(X, P, N, rng, cap=300):
    """One source's concept direction: mean(positives) - mean(its own negatives). Same dataset on both sides,
    so the dataset's style cancels out of the difference."""
    P, N = rng.permutation(P)[:cap], rng.permutation(N)[:cap]
    return X[P].mean(0) - X[N].mean(0)


def _probe(X, train: dict, rng, sd, cap=300):
    """Logistic-regression direction over several sources. Each source is centred on its own mean first, so
    dataset style cannot help; features are standardised. Returned in raw activation space (w / sd), so
    X @ direction reproduces the probe's score and the vector can be ablated or added like any other."""
    from sklearn.linear_model import LogisticRegression
    Xs, ys = [], []
    for P, N in train.values():
        P, N = rng.permutation(P)[:cap], rng.permutation(N)[:cap]
        idx = np.concatenate([P, N])
        Z = X[idx] - X[idx].mean(0)
        Xs.append(Z / sd)
        ys += [1] * len(P) + [0] * len(N)
    clf = LogisticRegression(C=0.02, max_iter=400).fit(np.vstack(Xs), ys)
    return _unit(clf.coef_[0] / sd)


def _shuffled(train: dict, rng) -> dict:
    """Control: inside every training source, give the same texts random positive/negative labels (same sizes)."""
    out = {}
    for s, (P, N) in train.items():
        allv = rng.permutation(np.concatenate([P, N]))
        out[s] = (allv[: len(P)], allv[len(P):])
    return out


def evaluate_concept(name: str, spec: dict, rows: list[dict], X: np.ndarray, pos_mask: np.ndarray, rng, method: str = "mean",
                     shuffle: bool = False):
    """Direction = average over sources of (positives - that source's own negatives). Tested on a held-out
    source's own positives vs negatives (leave-one-source-out). With one usable source, a random half split
    of that source (in-distribution only; flagged)."""
    srcs = np.array([r["src"] for r in rows])
    texts = [r["text"] for r in rows]
    use = {}
    for s in sorted(set(srcs[pos_mask])):
        P, N = np.flatnonzero((srcs == s) & pos_mask), np.flatnonzero((srcs == s) & ~pos_mask)
        if len(P) >= 30 and len(N) >= 30:
            use[s] = (P, N)
    res = dict(concept=name, group=spec["group"], positives=int(pos_mask.sum()), sources=sorted(set(srcs[pos_mask])),
               contrastive_sources=sorted(use), per_source={})

    sd = X.std(0) + 1e-6

    def direction_of(train: dict):
        if shuffle:
            train = _shuffled(train, rng)
        if method == "logreg":
            return _probe(X, train, rng, sd)
        return _unit(np.mean([_contrast(X, P, N, rng) for P, N in train.values()], axis=0))

    def run(train: dict, test: dict, label: str, mode: str):
        u = direction_of(train)
        tp = np.concatenate([P for P, _ in train.values()])
        tn = np.concatenate([N for _, N in train.values()])
        thr = _best_threshold(X[rng.permutation(tp)[:600]] @ u, X[rng.permutation(tn)[:600]] @ u)
        P, N = test
        n = min(len(P), len(N), 250)
        P, N = rng.permutation(P)[:n], rng.permutation(N)[:n]
        sp, sn = X[P] @ u, X[N] @ u
        y = [1] * n + [0] * n
        tr_t = [texts[i] for i in rng.permutation(tp)[:1500]]
        tr_n = [texts[i] for i in rng.permutation(tn)[:1500]]
        res["per_source"][label] = dict(
            mode=mode, n_test=int(n), auroc=auroc(np.concatenate([sp, sn]), y),
            keyword_auroc=_keyword_auroc(tr_t, tr_n, [texts[i] for i in P], [texts[i] for i in N]),
            precision=float((sp >= thr).sum() / max((sp >= thr).sum() + (sn >= thr).sum(), 1)), recall=float((sp >= thr).mean()))

    if len(use) >= 2:
        for h in use:
            run({s: v for s, v in use.items() if s != h}, use[h], h, "leave-one-source-out")
    elif len(use) == 1:
        (s, (P, N)), = use.items()
        P, N = rng.permutation(P), rng.permutation(N)
        run({s: (P[: len(P) // 2], N[: len(N) // 2])}, (P[len(P) // 2:], N[len(N) // 2:]), s, "in-source half split")
    ok = list(res["per_source"].values())
    res["summary"] = dict(
        mode=ok[0]["mode"] if ok else None, sources_tested=len(ok),
        auroc_mean=float(np.mean([e["auroc"] for e in ok])) if ok else None,
        auroc_min=float(np.min([e["auroc"] for e in ok])) if ok else None,
        keyword_mean=float(np.mean([e["keyword_auroc"] for e in ok])) if ok else None,
        precision_mean=float(np.mean([e["precision"] for e in ok])) if ok else None,
        recall_mean=float(np.mean([e["recall"] for e in ok])) if ok else None)
    # the final direction: every contrastive source; split-half = two random halves of every source
    if not use:
        return res, None
    d = direction_of(use)
    h = []
    for _ in range(10):
        a_, b_ = [], []
        for P, N in use.values():
            P, N = rng.permutation(P), rng.permutation(N)
            a_.append((P[: len(P) // 2], N[: len(N) // 2]))
            b_.append((P[len(P) // 2:], N[len(N) // 2:]))
        h.append(float(direction_of(dict(enumerate(a_))) @ direction_of(dict(enumerate(b_)))))
    res["direction"] = dict(split_half=float(np.mean(h)), size=float(np.linalg.norm(d)), sources=len(use))
    return res, d


def evaluate(names: list[str] | None, pooling: str = "mean", method: str = "mean", shuffle: bool = False) -> None:
    cfg, log = general_cfg(), Log(OUT)
    rows = read_jsonl(DATA / "texts.jsonl")
    X = torch.load(OUT / "acts.pt", weights_only=True)[pooling].float().numpy()
    rng = np.random.default_rng(seed_for("general-eval"))
    results = []
    for name, spec in cfg["concepts"].items():
        if (names and name not in names) or spec.get("gen") or not spec["pos"]:
            continue
        mask, unmatched = positive_mask(spec, rows)
        if mask.sum() < 100:
            results.append(dict(concept=name, group=spec["group"], positives=int(mask.sum()), skipped="fewer than 100 positives", unmatched=unmatched))
            log(f"{name}: skipped, {int(mask.sum())} positives; unmatched {unmatched}")
            continue
        res, d = evaluate_concept(name, spec, rows, X, mask, rng, method, shuffle)
        res["unmatched"] = unmatched
        results.append(res)
        s = res["summary"]
        if d is None:
            log(f"{name}: no source with both positives and its own negatives")
            continue
        if not shuffle:
            (OUT / ("directions" if method == "mean" else f"directions_{method}")).mkdir(parents=True, exist_ok=True)
            torch.save(dict(v=torch.tensor(d)), OUT / ("directions" if method == "mean" else f"directions_{method}") / f"{name}.pt")
        log(f"{name}: {res['positives']} pos, {len(res['contrastive_sources'])} contrastive sources | {s['mode']} AUROC {s['auroc_mean']:.2f} "
            f"(keyword {s['keyword_mean']:.2f}) prec {s['precision_mean']:.2f} rec {s['recall_mean']:.2f} | split-half {res['direction']['split_half']:.2f}")
    tag = "_shuffled" if shuffle else ""
    write_json(OUT / (f"readout_{pooling}{tag}.json" if method == "mean" else f"readout_{pooling}_{method}{tag}.json"), results)
    log(f"eval finished: {len(results)} concepts")


# ---- causal: does the model rely on the direction to predict a concept's text? -----------------------------
def text_nll(lm, texts: list[str], edits=None, bs: int = 32) -> np.ndarray:
    """Mean next-token loss of each text (first token excluded), under optional block edits. Teacher
    forced: no generation, no judge."""
    from cv.hooks import hooked
    out = np.zeros(len(texts))
    order = np.argsort([-len(t) for t in texts])
    for k in range(0, len(order), bs):
        idx = [int(j) for j in order[k:k + bs]]
        enc = lm.tok([texts[j] for j in idx], return_tensors="pt", padding=True, truncation=True,
                     max_length=160, add_special_tokens=False).to(lm.device)
        with torch.no_grad(), hooked(lm.blocks, edits):
            logits = lm.model(**enc).logits[:, :-1].float()
        tgt, m = enc.input_ids[:, 1:], enc.attention_mask[:, 1:].float()
        loss = torch.nn.functional.cross_entropy(logits.transpose(1, 2), tgt, reduction="none")
        out[idx] = ((loss * m).sum(1) / m.sum(1).clamp(min=1)).cpu().numpy()
    return out


def _did(real_p, real_n, base_p, base_n, rand: list[tuple], n_boot=1000, rng=None):
    """Difference-in-differences: how much more the edit changes concept texts than matched non-concept
    texts, minus the same for random directions. Bootstrap over texts."""
    rng = rng or np.random.default_rng(0)
    dp, dn = real_p - base_p, real_n - base_n
    rp = np.mean([a - base_p for a, _ in rand], axis=0)
    rn = np.mean([b - base_n for _, b in rand], axis=0)
    est = (dp.mean() - dn.mean()) - (rp.mean() - rn.mean())
    boots = []
    for _ in range(n_boot):
        i, j = rng.integers(0, len(dp), len(dp)), rng.integers(0, len(dn), len(dn))
        boots.append((dp[i].mean() - dn[j].mean()) - (rp[i].mean() - rn[j].mean()))
    return float(est), [float(x) for x in np.percentile(boots, [2.5, 97.5])]


def causal(names: list[str] | None, n_test: int = 80, method: str = "mean", shuffle: bool = False) -> None:
    from cv.hooks import add
    from cv.model import LM
    cfg, log = general_cfg(), Log(OUT)
    lcfg = load_cfg()
    rows = read_jsonl(DATA / "texts.jsonl")
    X = torch.load(OUT / "acts.pt", weights_only=True)["mean"].float().numpy()
    texts = [r["text"] for r in rows]
    srcs = np.array([r["src"] for r in rows])
    lm, L = LM(lcfg["model"]), lcfg["layer"]
    rng = np.random.default_rng(seed_for("general-causal"))
    rand_dirs = torch.nn.functional.normalize(torch.randn(2, lm.d, generator=torch.Generator().manual_seed(1)), dim=1).to(lm.device)
    results = []
    for name, spec in cfg["concepts"].items():
        if (names and name not in names) or not spec["pos"]:
            continue
        mask, _ = positive_mask(spec, rows)
        use = {}
        for s in sorted(set(srcs[mask])):
            P, N = np.flatnonzero((srcs == s) & mask), np.flatnonzero((srcs == s) & ~mask)
            if len(P) >= 30 and len(N) >= 30:
                use[s] = (P, N)
        if not use:
            continue
        # held-out test set: the largest contrastive source (or one half of it when there is only one)
        h = max(use, key=lambda s: min(len(use[s][0]), len(use[s][1])))
        P, N = rng.permutation(use[h][0]), rng.permutation(use[h][1])
        if len(use) == 1:
            train = {h: (P[: len(P) // 2], N[: len(N) // 2])}
            P, N = P[len(P) // 2:], N[len(N) // 2:]
        else:
            train = {s: v for s, v in use.items() if s != h}
        if shuffle:
            train = _shuffled(train, rng)   # control: same texts, random labels; the test texts keep their true labels
        if method == "logreg":
            ud = _probe(X, train, rng, X.std(0) + 1e-6)
            # natural size along this direction: how far apart the classes sit on it, averaged over sources
            size = float(np.mean([(X[a] @ ud).mean() - (X[b] @ ud).mean() for a, b in train.values()]))
            d = ud
        else:
            d = np.mean([_contrast(X, a, b, rng) for a, b in train.values()], axis=0)
            size = float(np.linalg.norm(d))
        u = torch.tensor(_unit(d), dtype=torch.float32, device=lm.device)
        # neutral point along u: the mean component of the training negatives. Zero is not neutral (a
        # class far from zero on this axis would be changed most by projecting the axis out).
        neg_train = np.concatenate([b for _, b in train.values()])
        m0 = float(np.mean(X[rng.permutation(neg_train)[:600]] @ _unit(d)))
        mean_abl = lambda w: (lambda h: (h.float() - ((h.float() @ u) - m0)[..., None] * w).to(h.dtype))
        k = min(n_test, len(P), len(N))
        tp, tn = [texts[i] for i in P[:k]], [texts[i] for i in N[:k]]
        base_p, base_n = text_nll(lm, tp), text_nll(lm, tn)
        ed = lambda e: {L: e}
        abl_p, abl_n = text_nll(lm, tp, ed(mean_abl(u))), text_nll(lm, tn, ed(mean_abl(u)))
        # random control: the same per-token shift (<h,u> - m0), applied along a random direction
        rabl = [(text_nll(lm, tp, ed(mean_abl(r))), text_nll(lm, tn, ed(mean_abl(r)))) for r in rand_dirs]
        s_ = 2.0
        st_p, st_n = text_nll(lm, tp, ed(add(s_ * size * u))), text_nll(lm, tn, ed(add(s_ * size * u)))
        rst = [(text_nll(lm, tp, ed(add(s_ * size * r))), text_nll(lm, tn, ed(add(s_ * size * r)))) for r in rand_dirs]
        a_est, a_ci = _did(abl_p, abl_n, base_p, base_n, rabl, rng=rng)
        s_est, s_ci = _did(st_p, st_n, base_p, base_n, rst, rng=rng)
        res = dict(concept=name, group=spec["group"], test_source=h, n=int(k), direction_size=size,
                   base_nll=dict(concept=float(base_p.mean()), other=float(base_n.mean())),
                   ablation=dict(did=a_est, ci=a_ci, concept_change=float((abl_p - base_p).mean()),
                                 other_change=float((abl_n - base_n).mean()),
                                 random_concept_change=float(np.mean([(a - base_p).mean() for a, _ in rabl])),
                                 random_other_change=float(np.mean([(b - base_n).mean() for _, b in rabl]))),
                   steering=dict(strength=s_, did=s_est, ci=s_ci, concept_change=float((st_p - base_p).mean()),
                                 other_change=float((st_n - base_n).mean())))
        results.append(res)
        log(f"{name}: ablation DiD {a_est:+.4f} CI [{a_ci[0]:+.4f}, {a_ci[1]:+.4f}] | steering DiD {s_est:+.4f} CI [{s_ci[0]:+.4f}, {s_ci[1]:+.4f}] "
            f"| base NLL concept {base_p.mean():.2f} other {base_n.mean():.2f} (n={k}, held-out {h})")
        tag = "_shuffled" if shuffle else ""
        write_json(OUT / (f"causal_mean_ablation{tag}.json" if method == "mean" else f"causal_mean_ablation_{method}{tag}.json"), results)
    log(f"causal finished: {len(results)} concepts")

# ---- positions: does the direction read at ONE token, not only on a text's average? -----------------------
def select_positions(real: np.ndarray, norm: np.ndarray, sent_end: np.ndarray, rng, n_interior: int = 5):
    """Indices (into the padded sequence) of the positions to keep for one text, with their kind and depth.
    real = indices of real tokens; norm = activation norm per padded position; sent_end = bool per padded
    position (token ends a sentence). Skips the first real token and norm outliers. kind: 0 last token,
    1 sentence end, 2 interior. depth in [0, 1] = fraction of the text already read."""
    first, last = int(real[0]), int(real[-1])
    ref = norm[real[1:]].mean() if len(real) > 1 else norm[real].mean()
    ok = np.array([i for i in real[1:-1] if norm[i] < 3 * ref], dtype=int)
    out = [(last, 0)]
    ends = ok[sent_end[ok]] if len(ok) else ok
    ends = rng.permutation(ends)[:2]
    out += [(int(i), 1) for i in ends]
    rest = np.setdiff1d(ok, ends)
    out += [(int(i), 2) for i in rng.permutation(rest)[:n_interior]]
    n = max(last - first, 1)
    return [(i, k, (i - first) / n) for i, k in out]


def embed_positions(seed: int = 0) -> None:
    """One pass over every text: the pooled activations (as `embed`) AND the raw layer-L activation at ~8
    sampled positions per text -> runs/general/acts.pt and pos_acts.pt (h fp16, text, kind, depth)."""
    from cv.model import LM
    cfg, log = load_cfg(), Log(OUT)
    rows = read_jsonl(DATA / "texts.jsonl")
    lm, L = LM(cfg["model"]), cfg["layer"]
    vocab = len(lm.tok)
    sent_end = np.array([lm.tok.decode([i]).strip().endswith((".", "!", "?")) for i in range(vocab)])
    rng = np.random.default_rng(seed)
    mean = torch.zeros(len(rows), lm.d, dtype=torch.float16)
    last = torch.zeros(len(rows), lm.d, dtype=torch.float16)
    H, T_, K_, D_ = [], [], [], []
    order = np.argsort([-len(r["text"]) for r in rows])
    for k in range(0, len(order), 48):
        idx = [int(j) for j in order[k:k + 48]]
        enc = lm.tok([rows[j]["text"] for j in idx], return_tensors="pt", padding=True, truncation=True,
                     max_length=192, add_special_tokens=False).to(lm.device)
        with torch.no_grad():
            h = lm.model.model(**enc, output_hidden_states=True).hidden_states[L + 1].float()
        mean[idx] = pooled(h, enc.attention_mask).half().cpu()
        last[idx] = h[:, -1].half().cpu()
        norms, ids, mask = h.norm(dim=-1).cpu().numpy(), enc.input_ids.cpu().numpy(), enc.attention_mask.cpu().numpy()
        for b, j in enumerate(idx):
            real = np.flatnonzero(mask[b])
            if len(real) < 6:
                continue
            for i, kind, depth in select_positions(real, norms[b], sent_end[ids[b]], rng):
                H.append(h[b, i].half().cpu())
                T_.append(j), K_.append(kind), D_.append(depth)
        if (k // 48) % 100 == 0:
            log(f"positions {k}/{len(rows)}")
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(dict(mean=mean, last=last), OUT / "acts.pt")
    torch.save(dict(h=torch.stack(H), text=torch.tensor(T_, dtype=torch.int32), kind=torch.tensor(K_, dtype=torch.int8),
                    depth=torch.tensor(D_, dtype=torch.float16)), OUT / "pos_acts.pt")
    log(f"positions finished: {len(rows)} texts, {len(H)} positions")


def eval_positions(names: list[str] | None = None) -> None:
    """Leave-one-dataset-out as in the readout test, but scored on single positions: the direction is built
    from pooled activations of the other datasets, then applied to stored raw activations at the last token,
    at sentence ends and at interior tokens of the held-out dataset's own concept / other texts."""
    cfg, log = general_cfg(), Log(OUT)
    rows = read_jsonl(DATA / "texts.jsonl")
    X = torch.load(OUT / "acts.pt", weights_only=True)["mean"].float().numpy()
    pos = torch.load(OUT / "pos_acts.pt", weights_only=True)
    Hs, text, kind, depth = pos["h"], pos["text"].numpy(), pos["kind"].numpy(), pos["depth"].float().numpy()
    by_text = collections.defaultdict(list)
    for m, t in enumerate(text):
        by_text[int(t)].append(m)
    srcs = np.array([r["src"] for r in rows])
    sd = X.std(0) + 1e-6
    rng = np.random.default_rng(seed_for("positions-eval"))
    approved = {a["concept"]: a["method"] for a in read_json(OUT / "approved.json") if a["tier"] == "APPROVED"}
    groups = {"last token": lambda k, d: k == 0, "sentence end": lambda k, d: k == 1,
              "interior 0-50%": lambda k, d: (k == 2) & (d < 0.5), "interior 50-100%": lambda k, d: (k == 2) & (d >= 0.5)}
    results = []
    for name, method in approved.items():
        if names and name not in names:
            continue
        mask, _ = positive_mask(cfg["concepts"][name], rows)
        use = {}
        for s_ in sorted(set(srcs[mask])):
            P, N = np.flatnonzero((srcs == s_) & mask), np.flatnonzero((srcs == s_) & ~mask)
            if len(P) >= 30 and len(N) >= 30:
                use[s_] = (P, N)
        per = collections.defaultdict(list)
        for h_, (P, N) in use.items():
            train = {s_: v for s_, v in use.items() if s_ != h_}
            if not train:
                continue
            u = _probe(X, train, rng, sd) if method == "logreg" else _unit(np.mean([_contrast(X, a, b, rng) for a, b in train.values()], axis=0))
            n = min(len(P), len(N), 250)
            P, N = rng.permutation(P)[:n], rng.permutation(N)[:n]
            ms = [m for t in list(P) + list(N) for m in by_text.get(int(t), [])]
            lab = np.array([1 if text[m] in set(P.tolist()) else 0 for m in ms])
            sc = Hs[ms].float().numpy() @ u
            k_, d_ = kind[ms], depth[ms]
            per["text average"].append(auroc(np.concatenate([X[P] @ u, X[N] @ u]), [1] * n + [0] * n))
            for g, f in groups.items():
                sel = f(k_, d_)
                if sel.sum() > 20 and 0 < lab[sel].sum() < sel.sum():
                    per[g].append(auroc(sc[sel], lab[sel]))
        res = dict(concept=name, method=method, datasets=len(use), **{g: (float(np.mean(v)), float(np.min(v))) for g, v in per.items() if v})
        results.append(res)
        log(f"{name}: " + " | ".join(f"{g} {v[0]:.2f}" for g, v in res.items() if isinstance(v, tuple)))
    write_json(OUT / "positions.json", results)
    log(f"positions eval finished: {len(results)} concepts")


# ---- token level: directions built from, and tested on, individual token positions -------------------------
TOK = OUT / "tokens"


def token_concepts() -> list[str]:
    """Approved concepts plus near misses: >= 2 datasets and a text-level cross-dataset AUROC >= 0.75."""
    keep = {a["concept"] for a in read_json(OUT / "approved.json") if a["tier"] == "APPROVED"}
    for f in ("readout_mean.json", "readout_mean_logreg.json"):
        for r in read_json(OUT / f):
            sm = r.get("summary")
            if sm and sm.get("auroc_mean") is not None and sm["mode"].startswith("leave") and sm["auroc_mean"] >= 0.75:
                keep.add(r["concept"])
    return sorted(keep)


def build_lexicon(rows, tok, concepts: list[str], spec_all: dict, top: int = 100) -> dict[str, set]:
    """Words that mark a concept: token ids that are far more common in a source's concept texts than in the
    same source's other texts (log-odds, summed over sources). Independent of any direction."""
    masks = {c: positive_mask(spec_all[c], rows)[0] for c in concepts}
    srcs = np.array([r["src"] for r in rows])
    enc = tok([r["text"] for r in rows], add_special_tokens=False, truncation=True, max_length=192)["input_ids"]
    sets = [set(x) for x in enc]
    stop = {i for i in range(len(tok)) if not tok.decode([i]).strip().isalpha() or len(tok.decode([i]).strip()) < 3}
    lex = {}
    for c in concepts:
        score = collections.Counter()
        total = collections.Counter()
        for s in sorted(set(srcs[masks[c]])):
            idx = np.flatnonzero(srcs == s)
            P = [i for i in idx if masks[c][i]]
            N = [i for i in idx if not masks[c][i]]
            if len(P) < 30 or len(N) < 30:
                continue
            dfP, dfN = collections.Counter(), collections.Counter()
            for i in P:
                dfP.update(sets[i])
            for i in N:
                dfN.update(sets[i])
            for t, v in dfP.items():
                total[t] += v
                score[t] += np.log((v + 0.5) / (len(P) + 1)) - np.log((dfN.get(t, 0) + 0.5) / (len(N) + 1))
        cand = [(sc, t) for t, sc in score.items() if t not in stop and total[t] >= 10]
        lex[c] = {t for _, t in sorted(cand, reverse=True)[:top]}
    return lex


def tokens_pass(seed: int = 0) -> None:
    """One pass. Per text, store the layer-L activation of: up to 4 concept words for every concept the text
    is positive for (type 0), the last token (2) and up to 2 sentence ends (1), and 3 random interior tokens (3)."""
    from cv.model import LM
    cfg, log, gcfg = load_cfg(), Log(OUT), general_cfg()
    rows = read_jsonl(DATA / "texts.jsonl")
    lm, L = LM(cfg["model"]), cfg["layer"]
    concepts = token_concepts()
    log(f"token pass: {len(concepts)} concepts")
    lex = build_lexicon(rows, lm.tok, concepts, gcfg["concepts"])
    pos = {c: positive_mask(gcfg["concepts"][c], rows)[0] for c in concepts}
    pos_of = collections.defaultdict(list)
    for ci, c in enumerate(concepts):
        for i in np.flatnonzero(pos[c]):
            pos_of[int(i)].append(ci)
    sent_end = np.array([lm.tok.decode([i]).strip().endswith((".", "!", "?")) for i in range(len(lm.tok))])
    rng = np.random.default_rng(seed)
    H, rec = [], []   # rec: (text, concept index or -1, type, token index in the text, depth, token id)
    for k in range(0, len(rows), 48):
        idx = list(range(k, min(k + 48, len(rows))))
        enc = lm.tok([rows[j]["text"] for j in idx], return_tensors="pt", padding=True, truncation=True,
                     max_length=192, add_special_tokens=False).to(lm.device)
        with torch.no_grad():
            h = lm.model.model(**enc, output_hidden_states=True).hidden_states[L + 1].float()
        norms, ids, mask = h.norm(dim=-1).cpu().numpy(), enc.input_ids.cpu().numpy(), enc.attention_mask.cpu().numpy()
        for b, j in enumerate(idx):
            real = np.flatnonzero(mask[b])
            if len(real) < 8:
                continue
            first, last = int(real[0]), int(real[-1])
            ref = norms[b][real[1:]].mean()
            ok = np.array([i for i in real[1:-1] if norms[b][i] < 3 * ref], dtype=int)
            if len(ok) == 0:
                continue
            take = [(last, -1, 2)]
            ends = rng.permutation(ok[sent_end[ids[b][ok]]])[:2]
            take += [(int(i), -1, 1) for i in ends]
            for ci in pos_of.get(j, []):
                hits = ok[np.isin(ids[b][ok], list(lex[concepts[ci]]))]
                take += [(int(i), ci, 0) for i in rng.permutation(hits)[:4]]
            rest = np.setdiff1d(ok, [t[0] for t in take])
            take += [(int(i), -1, 3) for i in rng.permutation(rest)[:3]]
            n = max(last - first, 1)
            for i, ci, ty in take:
                H.append(h[b, i].half().cpu())
                rec.append((j, ci, ty, int(i - first), (i - first) / n, int(ids[b][i])))
        if (k // 48) % 200 == 0:
            log(f"tokens {k}/{len(rows)}")
    TOK.mkdir(parents=True, exist_ok=True)
    arr = np.array(rec, dtype=np.float64)
    torch.save(dict(h=torch.stack(H), text=torch.tensor(arr[:, 0], dtype=torch.int32), concept=torch.tensor(arr[:, 1], dtype=torch.int16),
                    type=torch.tensor(arr[:, 2], dtype=torch.int8), index=torch.tensor(arr[:, 3], dtype=torch.int32),
                    depth=torch.tensor(arr[:, 4], dtype=torch.float16), token=torch.tensor(arr[:, 5], dtype=torch.int32),
                    concepts=concepts), TOK / "tokens.pt")
    write_json(TOK / "lexicon.json", {c: [lm.tok.decode([t]).strip() for t in sorted(v)] for c, v in lex.items()})
    log(f"tokens finished: {len(H)} token activations")


def _token_sets(rec, rows_src, mask_c, ci, texts_P, texts_N):
    """Token indices for concept ci within one dataset: positives = its concept words + sentence ends / last
    tokens of concept texts; negatives = the matching tokens of other texts (their own concept words)."""
    t, c, ty = rec["text"], rec["concept"], rec["type"]
    inP, inN = np.isin(t, texts_P), np.isin(t, texts_N)
    pos_word = np.flatnonzero(inP & (ty == 0) & (c == ci))
    neg_word = np.flatnonzero(inN & (ty == 0) & (c != ci) & (c >= 0))
    if len(neg_word) < 20:   # datasets whose other texts have no concept of their own: use random interior tokens
        neg_word = np.flatnonzero(inN & (ty == 3))
    pos_end = np.flatnonzero(inP & np.isin(ty, (1, 2)))
    neg_end = np.flatnonzero(inN & np.isin(ty, (1, 2)))
    return pos_word, neg_word, pos_end, neg_end


def token_eval(names: list[str] | None = None, method: str = "mean") -> None:
    """Leave-one-dataset-out at single tokens. Direction from token activations of the other datasets
    (per dataset: positives minus its own negatives; or a centred logistic probe), scored on the held-out
    dataset's own tokens. Reported by token type: concept word / sentence end / last token."""
    cfg, log = general_cfg(), Log(OUT)
    rows = read_jsonl(DATA / "texts.jsonl")
    d = torch.load(TOK / "tokens.pt", weights_only=True)
    concepts = d["concepts"]
    rec = {k: d[k].numpy() for k in ("text", "concept", "type", "index", "depth", "token")}
    H = d["h"]
    srcs = np.array([r["src"] for r in rows])
    rng = np.random.default_rng(seed_for("token-eval"))
    sd = H[::20].float().numpy().std(0) + 1e-6
    approved = {a["concept"] for a in read_json(OUT / "approved.json") if a["tier"] == "APPROVED"}
    from sklearn.linear_model import LogisticRegression
    out = []
    for ci, name in enumerate(concepts):
        if names and name not in names:
            continue
        mask, _ = positive_mask(cfg["concepts"][name], rows)
        use = {}
        for s_ in sorted(set(srcs[mask])):
            P, N = np.flatnonzero((srcs == s_) & mask), np.flatnonzero((srcs == s_) & ~mask)
            if len(P) >= 30 and len(N) >= 30:
                use[s_] = (P, N, *_token_sets(rec, srcs, mask, ci, P, N))
        if len(use) < 2:
            continue
        per = collections.defaultdict(list)
        for h_ in use:
            train = [v for s_, v in use.items() if s_ != h_]
            pos_idx = np.concatenate([np.concatenate([v[2], v[4]]) for v in train])
            neg_idx = np.concatenate([np.concatenate([v[3], v[5]]) for v in train])
            if len(pos_idx) < 60 or len(neg_idx) < 60:
                continue
            def contrast(v):
                a, b = np.concatenate([v[2], v[4]]), np.concatenate([v[3], v[5]])
                if len(a) < 10 or len(b) < 10:
                    return None
                a, b = rng.permutation(a)[:600], rng.permutation(b)[:600]
                return H[a].float().numpy().mean(0) - H[b].float().numpy().mean(0)
            if method == "logreg":
                Xs, ys = [], []
                for v in train:
                    a, b = np.concatenate([v[2], v[4]]), np.concatenate([v[3], v[5]])
                    a, b = rng.permutation(a)[:400], rng.permutation(b)[:400]
                    idx = np.concatenate([a, b]); Z = H[idx].float().numpy(); Z = (Z - Z.mean(0)) / sd
                    Xs.append(Z); ys += [1] * len(a) + [0] * len(b)
                u = _unit(LogisticRegression(C=0.02, max_iter=300).fit(np.vstack(Xs), ys).coef_[0] / sd)
            else:
                cs = [c_ for c_ in (contrast(v) for v in train) if c_ is not None]
                u = _unit(np.mean(cs, axis=0))
            v = use[h_]
            for label, (a, b) in {"concept word": (v[2], v[3]), "sentence end/last token": (v[4], v[5])}.items():
                if len(a) >= 20 and len(b) >= 20:
                    n = min(len(a), len(b), 400)
                    a, b = rng.permutation(a)[:n], rng.permutation(b)[:n]
                    per[label].append(auroc(np.concatenate([H[a].float().numpy() @ u, H[b].float().numpy() @ u]), [1] * n + [0] * n))
            a, b = np.concatenate([v[2], v[4]]), np.concatenate([v[3], v[5]])
            if len(a) >= 20 and len(b) >= 20:
                n = min(len(a), len(b), 600)
                a, b = rng.permutation(a)[:n], rng.permutation(b)[:n]
                per["all single tokens"].append(auroc(np.concatenate([H[a].float().numpy() @ u, H[b].float().numpy() @ u]), [1] * n + [0] * n))
        if per.get("all single tokens"):
            res = dict(concept=name, approved=name in approved, method=method, datasets=len(use),
                       **{k: (float(np.mean(v)), float(np.min(v))) for k, v in per.items()})
            out.append(res)
            log(f"{name} [{method}]: " + " | ".join(f"{k} {v[0]:.2f}/{v[1]:.2f}" for k, v in res.items() if isinstance(v, tuple)))
    write_json(TOK / (f"token_readout_{method}_subset.json" if names else f"token_readout_{method}.json"), out)
    log(f"token eval finished: {len(out)} concepts")


def token_causal(names: list[str] | None = None, per_group: int = 60, window: int = 30) -> None:
    """Single-position ablation. For held-out texts, set the direction's component to the negatives' mean at
    ONE token position (a concept word, or a sentence end) and measure how much worse the model predicts the
    next `window` tokens; also from that position onward. Compared with matched texts from the same dataset
    (their own concept words / sentence ends) and with random directions given the same per-token shift."""
    from cv.hooks import hooked
    from cv.model import LM
    cfg, log, lcfg = general_cfg(), Log(OUT), load_cfg()
    rows = read_jsonl(DATA / "texts.jsonl")
    d = torch.load(TOK / "tokens.pt", weights_only=True)
    concepts, H = d["concepts"], d["h"]
    rec = {k: d[k].numpy() for k in ("text", "concept", "type", "index", "token")}
    srcs = np.array([r["src"] for r in rows])
    lm, L = LM(lcfg["model"]), lcfg["layer"]
    rng = np.random.default_rng(seed_for("token-causal"))
    rand_dirs = torch.nn.functional.normalize(torch.randn(2, lm.d, generator=torch.Generator().manual_seed(2)), dim=1).to(lm.device)
    results = []

    def run(items, u, m0, mode):
        """items: [(text, token index, token id)] -> mean next-token loss over the window per item, for
        mode in base / single / onward, and for u replaced by each random direction."""
        out = np.full(len(items), np.nan)
        for k in range(0, len(items), 16):
            chunk = items[k:k + 16]
            enc = lm.tok([rows[t]["text"] for t, _, _ in chunk], return_tensors="pt", padding=True, truncation=True,
                         max_length=192, add_special_tokens=False).to(lm.device)
            first = enc.attention_mask.argmax(1)
            pp = torch.tensor([int(first[b]) + chunk[b][1] for b in range(len(chunk))], device=lm.device)
            ok = torch.tensor([int(enc.input_ids[b, pp[b]]) == chunk[b][2] for b in range(len(chunk))], device=lm.device)
            T = enc.input_ids.shape[1]
            ar = torch.arange(T, device=lm.device)[None]
            sel = (ar == pp[:, None]) if mode == "single" else (ar >= pp[:, None])

            def edit(h):
                if mode == "base":
                    return h
                comp = ((h.float() @ u) - m0)[..., None] * u
                return (h.float() - comp * sel[..., None]).to(h.dtype)
            with torch.no_grad(), hooked(lm.blocks, {L: edit}):
                logits = lm.model(**enc).logits[:, :-1].float()
            loss = torch.nn.functional.cross_entropy(logits.transpose(1, 2), enc.input_ids[:, 1:], reduction="none")
            win = (ar[:, :-1] >= pp[:, None]) & (ar[:, :-1] < pp[:, None] + window) & enc.attention_mask[:, 1:].bool()
            val = ((loss * win).sum(1) / win.sum(1).clamp(min=1)).cpu().numpy()
            for b in range(len(chunk)):
                if bool(ok[b]) and win[b].sum() >= window // 2:
                    out[k + b] = val[b]
        return out

    for ci, name in enumerate(concepts):
        if names and name not in names:
            continue
        mask, _ = positive_mask(cfg["concepts"][name], rows)
        use = {}
        for s_ in sorted(set(srcs[mask])):
            P, N = np.flatnonzero((srcs == s_) & mask), np.flatnonzero((srcs == s_) & ~mask)
            if len(P) >= 30 and len(N) >= 30:
                use[s_] = (P, N, *_token_sets(rec, srcs, mask, ci, P, N))
        if len(use) < 2:
            continue
        h_ = max(use, key=lambda s_: min(len(use[s_][0]), len(use[s_][1])))
        train = [v for s_, v in use.items() if s_ != h_]
        cs = []
        for v in train:
            a, b = np.concatenate([v[2], v[4]]), np.concatenate([v[3], v[5]])
            if len(a) >= 10 and len(b) >= 10:
                cs.append(H[rng.permutation(a)[:600]].float().numpy().mean(0) - H[rng.permutation(b)[:600]].float().numpy().mean(0))
        if not cs:
            continue
        du = np.mean(cs, axis=0)
        u = torch.tensor(_unit(du), dtype=torch.float32, device=lm.device)
        negs = np.concatenate([np.concatenate([v[3], v[5]]) for v in train])
        m0 = float(np.mean(H[rng.permutation(negs)[:800]].float().numpy() @ _unit(du)))
        P, N, pw, nw, pe, ne = use[h_]
        res = dict(concept=name, test_source=h_, **{})
        for label, (pi, ni) in {"concept word": (pw, nw), "sentence end": (pe[rec["type"][pe] == 1], ne[rec["type"][ne] == 1])}.items():
            def items(idx):
                idx = rng.permutation(idx)
                got, seen = [], set()
                for m in idx:
                    t = int(rec["text"][m])
                    if t in seen:
                        continue
                    seen.add(t)
                    got.append((t, int(rec["index"][m]), int(rec["token"][m])))
                    if len(got) >= per_group:
                        break
                return got
            ip, in_ = items(pi), items(ni)
            if len(ip) < 20 or len(in_) < 20:
                continue
            allit = ip + in_
            base = run(allit, u, m0, "base")
            r = {}
            for mode in ("single", "onward"):
                real = run(allit, u, m0, mode) - base
                rnd = np.mean([run(allit, rd, m0, mode) - base for rd in rand_dirs], axis=0)
                cp, cn = real[:len(ip)], real[len(ip):]
                rp, rn = rnd[:len(ip)], rnd[len(ip):]
                cp, cn, rp, rn = (x[~np.isnan(x)] for x in (cp, cn, rp, rn))
                if min(len(cp), len(cn), len(rp), len(rn)) < 10:
                    continue
                did = (cp.mean() - cn.mean()) - (rp.mean() - rn.mean())
                boots = [(rng.choice(cp, len(cp)).mean() - rng.choice(cn, len(cn)).mean()) - (rng.choice(rp, len(rp)).mean() - rng.choice(rn, len(rn)).mean())
                         for _ in range(500)]
                r[mode] = dict(did=float(did), ci=[float(x) for x in np.percentile(boots, [2.5, 97.5])],
                               concept_change=float(cp.mean()), other_change=float(cn.mean()),
                               random_concept=float(rp.mean()), random_other=float(rn.mean()), n=[len(cp), len(cn)])
            if r:
                res[label] = r
        if len(res) > 2:
            results.append(res)
            write_json(TOK / "token_causal.json", results)
            log(f"{name}: " + " | ".join(f"{lb} {md} {v[md]['did']:+.3f} [{v[md]['ci'][0]:+.3f},{v[md]['ci'][1]:+.3f}]"
                                         for lb, v in res.items() if isinstance(v, dict) for md in v))
    log(f"token causal finished: {len(results)} concepts")



# ---- approve: the fixed rule, applied mechanically to the saved results -----------------------------------
BEHAVIOUR = {"refusal", "sycophancy", "deception", "secrecy", "eval_awareness", "jailbreak", "prompt_injection",
             "harmful_request", "toxicity", "hate_speech"}   # concepts whose "use" is behaviour, not text likelihood


def approve() -> None:
    """Tiers (fixed in advance, no tuning):
    APPROVED     >= 2 independent datasets, leave-one-dataset-out AUROC mean >= 0.85 and worst >= 0.75, AND the
                 model relies on it: mean-ablation raises the loss of concept texts more than matched others and
                 random directions (CI above 0).
    READOUT-ONLY same readout bar; ablation shows no reliance (or, for behaviour concepts, the text-likelihood
                 test is the wrong test and a behavioural test is still owed).
    SINGLE-DATASET  would pass in-distribution but has only one dataset: not approvable yet.
    NOT APPROVED everything else.
    Two direction methods are tried (mean difference, logistic probe); a concept gets its best tier, and the
    use test is always run with the direction of the method that earned the readout."""
    rank = {"APPROVED": 3, "READOUT-ONLY": 2, "SINGLE-DATASET": 1, "NOT APPROVED": 0}
    best: dict[str, dict] = {}
    for method, rfile, cfile in (("mean", "readout_mean.json", "causal_mean_ablation.json"),
                                 ("logreg", "readout_mean_logreg.json", "causal_mean_ablation_logreg.json")):
        if not (OUT / rfile).exists():
            continue
        R = {r["concept"]: r for r in read_json(OUT / rfile)}
        C = {r["concept"]: r for r in read_json(OUT / cfile)} if (OUT / cfile).exists() else {}
        for n, r in R.items():
            s = r.get("summary")
            if not s or s.get("auroc_mean") is None:
                o = dict(concept=n, group=r["group"], tier="NOT APPROVED", method=method, why="no dataset with its own negatives")
            else:
                loso = s["mode"].startswith("leave")
                read = s["auroc_mean"] >= 0.85 and (s["auroc_min"] >= 0.75 if loso else True)
                c = C.get(n)
                abl = bool(c) and c["ablation"]["ci"][0] > 0
                tier = ("APPROVED" if abl else "READOUT-ONLY") if (read and loso) else ("SINGLE-DATASET" if read else "NOT APPROVED")
                o = dict(concept=n, group=r["group"], tier=tier, method=method, loso=loso,
                         datasets=len(r.get("contrastive_sources", [])), auroc=s["auroc_mean"], auroc_min=s["auroc_min"],
                         keyword=s["keyword_mean"], ablation_did=c["ablation"]["did"] if c else None,
                         ablation_ci=c["ablation"]["ci"] if c else None, steering_did=c["steering"]["did"] if c else None,
                         steering_ci=c["steering"]["ci"] if c else None,
                         behaviour_test_owed=n in BEHAVIOUR and tier in ("READOUT-ONLY", "SINGLE-DATASET"))
            if n not in best or rank[o["tier"]] > rank[best[n]["tier"]] or (rank[o["tier"]] == rank[best[n]["tier"]] and o.get("auroc", 0) > best[n].get("auroc", 0) + 0.02 and o["method"] != best[n]["method"] and best[n]["method"] == "mean" and False):
                best[n] = o
    out = list(best.values())
    write_json(OUT / "approved.json", out)
    cnt = collections.Counter(o["tier"] for o in out)
    print(dict(cnt))
    for t in ("APPROVED", "READOUT-ONLY", "SINGLE-DATASET"):
        print(f"\n== {t}")
        for o in sorted((o for o in out if o["tier"] == t), key=lambda o: (o["group"], -o.get("auroc", 0))):
            ab = f"{o['ablation_did']:+.3f}" if o.get("ablation_did") is not None else "n/a"
            print(f"  {o['concept']:22s} {o['group']:7s} {o['method']:6s} datasets {o.get('datasets')} AUROC {o['auroc']:.2f} (min {o['auroc_min']:.2f}, kw {o['keyword']:.2f}) | ablation {ab}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["build", "embed", "eval", "causal", "approve", "embed_positions", "eval_positions", "tokens", "token_eval", "token_causal"])
    ap.add_argument("--concepts", nargs="*")
    ap.add_argument("--sources", nargs="*")
    ap.add_argument("--pooling", default="mean", choices=["mean", "last"])
    ap.add_argument("--method", default="mean", choices=["mean", "logreg"])
    ap.add_argument("--shuffle", action="store_true", help="control: train on randomly relabelled texts")
    a = ap.parse_args()
    if a.stage == "build":
        build(a.sources)
    elif a.stage == "embed":
        embed()
    elif a.stage == "causal":
        causal(a.concepts, method=a.method, shuffle=a.shuffle)
    elif a.stage == "approve":
        approve()
    elif a.stage == "tokens":
        tokens_pass()
    elif a.stage == "token_eval":
        token_eval(a.concepts, a.method)
    elif a.stage == "token_causal":
        token_causal(a.concepts)
    elif a.stage == "embed_positions":
        embed_positions()
    elif a.stage == "eval_positions":
        eval_positions(a.concepts)
    else:
        evaluate(a.concepts, a.pooling, a.method, a.shuffle)


if __name__ == "__main__":
    main()
