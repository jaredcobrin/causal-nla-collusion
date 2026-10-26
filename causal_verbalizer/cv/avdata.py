"""Activation-verbalizer training data: which approved concepts are active at a sampled position.

    python -m cv.avdata calibrate     token-level directions + thresholds (>= 90% precision) per concept
    python -m cv.avdata sample N      N labelled positions -> runs/general/av_data/pilot.jsonl + av_acts.pt

A position's concepts are active / inactive / uncertain from its projection on each approved concept's
single-token direction, standardised and thresholded. Active means precision >= 0.9 at an assumed
prevalence of 10% (positives are rarer in the wild than in the labelled datasets). Positions come from
the stored token activations (last token, sentence ends, interior tokens at depth >= 0.25).
"""

from __future__ import annotations

import argparse
import collections
import sys

import numpy as np
import torch

from cv.general import DATA, OUT, TOK, _token_sets, _unit, positive_mask, general_cfg
from cv.io import read_json, read_jsonl, seed_for, write_json, write_jsonl

AV = OUT / "av_data"
PREVALENCE = 0.10      # assumed share of positions where a given concept is really present
PRECISION = 0.90
INACTIVE_MAX_POS = 0.05
POS_TYPES = (1, 2, 3)  # sentence end, last token, interior


def _load():
    cfg = general_cfg()
    rows = read_jsonl(DATA / "texts.jsonl")
    d = torch.load(TOK / "tokens.pt", weights_only=True)
    rec = {k: d[k].numpy() for k in ("text", "concept", "type", "index", "depth", "token")}
    return cfg, rows, d["h"], rec, list(d["concepts"])


def _direction(H, train, method, rng, sd):
    """train: list of tuples (_, _, pos_word, neg_word, pos_end, neg_end) as built by _token_sets."""
    if method == "logreg":
        from sklearn.linear_model import LogisticRegression
        Xs, ys = [], []
        for v in train:
            a, b = np.concatenate([v[2], v[4]]), np.concatenate([v[3], v[5]])
            if len(a) < 10 or len(b) < 10:
                continue
            a, b = rng.permutation(a)[:400], rng.permutation(b)[:400]
            Z = H[np.concatenate([a, b])].float().numpy()
            Xs.append((Z - Z.mean(0)) / sd)
            ys += [1] * len(a) + [0] * len(b)
        return _unit(LogisticRegression(C=0.02, max_iter=300).fit(np.vstack(Xs), ys).coef_[0] / sd)
    cs = []
    for v in train:
        a, b = np.concatenate([v[2], v[4]]), np.concatenate([v[3], v[5]])
        if len(a) >= 10 and len(b) >= 10:
            cs.append(H[rng.permutation(a)[:600]].float().numpy().mean(0) - H[rng.permutation(b)[:600]].float().numpy().mean(0))
    return _unit(np.mean(cs, axis=0))


def _thresholds(pos_s, neg_s):
    """Smallest score with prevalence-adjusted precision >= PRECISION; largest score below which the
    positive share stays <= INACTIVE_MAX_POS. Scores are on the calibration scale."""
    w = ((1 - PREVALENCE) / PREVALENCE) * (len(pos_s) / len(neg_s))
    cand = np.unique(np.quantile(np.concatenate([pos_s, neg_s]), np.linspace(0.02, 0.995, 200)))
    high = None
    for t in cand:
        tp, fp = (pos_s >= t).sum(), (neg_s >= t).sum()
        if tp >= 30 and tp / (tp + w * fp) >= PRECISION:
            high = float(t)
            break
    low = None
    for t in cand[::-1]:
        fn, tn = (pos_s <= t).sum(), (neg_s <= t).sum()
        if tn >= 30 and fn / (fn + w * tn + 1e-9) <= INACTIVE_MAX_POS:
            low = float(t)
            break
    return high, low, w


def calibrate() -> None:
    cfg, rows, H, rec, concepts = _load()
    srcs = np.array([r["src"] for r in rows])
    rng = np.random.default_rng(seed_for("av-calibrate"))
    sd = H[::20].float().numpy().std(0) + 1e-6
    approved = {a["concept"] for a in read_json(OUT / "approved.json") if a["tier"] == "APPROVED"}
    tm = {r["concept"]: r for r in read_json(TOK / "token_readout_mean.json")}
    tl = {r["concept"]: r for r in read_json(TOK / "token_readout_logreg.json")}
    out = {}
    for ci, name in enumerate(concepts):
        if name not in approved:
            continue
        m = tm.get(name, {}).get("all single tokens", [0])[0]
        l = tl.get(name, {}).get("all single tokens", [0])[0]
        method = "logreg" if l > m else "mean"
        mask, _ = positive_mask(cfg["concepts"][name], rows)
        use = {}
        for s in sorted(set(srcs[mask])):
            P, N = np.flatnonzero((srcs == s) & mask), np.flatnonzero((srcs == s) & ~mask)
            if len(P) >= 30 and len(N) >= 30:
                use[s] = (P, N, *_token_sets(rec, srcs, mask, ci, P, N))
        if len(use) < 2:
            continue
        pos_s, neg_s, pos_t, neg_t, ref = [], [], [], [], {}
        for h in use:
            train = [v for s, v in use.items() if s != h]
            u = _direction(H, train, method, rng, sd)
            P, N = use[h][0], use[h][1]
            ok = np.isin(rec["type"], POS_TYPES) & (rec["depth"] >= 0.25)
            ip = np.flatnonzero(ok & np.isin(rec["text"], P))
            inn = np.flatnonzero(ok & np.isin(rec["text"], N))
            ip, inn = rng.permutation(ip)[:1500], rng.permutation(inn)[:1500]
            pos_s.append(H[ip].float().numpy() @ u)
            neg_s.append(H[inn].float().numpy() @ u)
            pos_t.append(rec["type"][ip]); neg_t.append(rec["type"][inn])
        pos_s, neg_s = np.concatenate(pos_s), np.concatenate(neg_s)
        pos_t, neg_t = np.concatenate(pos_t), np.concatenate(neg_t)
        high, low, w = _thresholds(pos_s, neg_s)
        # final direction from every dataset; thresholds carried over by matching the false-positive rate
        u_all = _direction(H, list(use.values()), method, rng, sd)
        P_all = np.concatenate([v[0] for v in use.values()])
        N_all = np.concatenate([v[1] for v in use.values()])
        ok = np.isin(rec["type"], POS_TYPES) & (rec["depth"] >= 0.25)
        ip = rng.permutation(np.flatnonzero(ok & np.isin(rec["text"], P_all)))[:4000]
        inn = rng.permutation(np.flatnonzero(ok & np.isin(rec["text"], N_all)))[:4000]
        fp_s, fn_s = H[inn].float().numpy() @ u_all, H[ip].float().numpy() @ u_all
        AV.mkdir(parents=True, exist_ok=True)
        np.savez(AV / f"cal_{name}.npz", pos_s=pos_s, neg_s=neg_s, pos_t=pos_t, neg_t=neg_t,
                 fp_s=fp_s, fn_s=fn_s, fp_t=rec["type"][inn], fn_t=rec["type"][ip])
        entry = dict(method=method, datasets=len(use), n_pos=int(len(pos_s)), n_neg=int(len(neg_s)))
        if high is not None:
            fpr = float((neg_s >= high).mean())
            tpr = float((pos_s >= high).mean())
            entry.update(t_high=float(np.quantile(fp_s, 1 - fpr)) if fpr > 0 else float(fp_s.max()), recall_at_high=tpr, fpr_at_high=fpr)
        if low is not None:
            tnr = float((neg_s <= low).mean())
            entry.update(t_low=float(np.quantile(fp_s, tnr)), tnr_at_low=tnr)
        entry["usable"] = "t_high" in entry
        out[name] = entry
        np.save(AV / f"dir_{name}.npy", u_all.astype(np.float32)) if (AV.mkdir(parents=True, exist_ok=True) or True) else None
        print(f"{name:18s} {method:6s} datasets {len(use)} | active threshold {'yes' if high is not None else 'NO'} "
              f"recall {entry.get('recall_at_high', float('nan')):.2f} fpr {entry.get('fpr_at_high', float('nan')):.3f}", flush=True)
    write_json(AV / "thresholds.json", out)
    print("usable concepts:", sum(v["usable"] for v in out.values()), "of", len(out))


def thresholds(prevalence: float, types: tuple[int, ...], tag: str) -> None:
    """Recompute thresholds from the saved calibration scores for a prevalence and the position types
    that count (1 sentence end, 2 last token, 3 interior)."""
    global PREVALENCE
    PREVALENCE = prevalence
    meta = read_json(AV / "thresholds.json")
    out = {}
    for name, m in meta.items():
        z = np.load(AV / f"cal_{name}.npz")
        kp, kn = np.isin(z["pos_t"], types), np.isin(z["neg_t"], types)
        pos_s, neg_s = z["pos_s"][kp], z["neg_s"][kn]
        if len(pos_s) < 60 or len(neg_s) < 60:
            out[name] = dict(method=m["method"], usable=False)
            continue
        high, low, w = _thresholds(pos_s, neg_s)
        fneg = z["fp_s"][np.isin(z["fp_t"], types)]
        e = dict(method=m["method"], usable=high is not None, n_pos=int(len(pos_s)), n_neg=int(len(neg_s)))
        if high is not None:
            fpr, tpr = float((neg_s >= high).mean()), float((pos_s >= high).mean())
            e.update(t_high=float(np.quantile(fneg, 1 - fpr)) if fpr > 0 else float(fneg.max()), recall_at_high=tpr, fpr_at_high=fpr)
        if low is not None:
            tnr = float((neg_s <= low).mean())
            e.update(t_low=float(np.quantile(fneg, tnr)), tnr_at_low=tnr)
        out[name] = e
    write_json(AV / f"thresholds_{tag}.json", out)
    r = [v["recall_at_high"] for v in out.values() if v.get("usable")]
    print(f"[{tag}] prevalence {prevalence}, types {types}: usable {len(r)}/{len(out)} | median recall {np.median(r):.2f} | "
          f"concepts with recall >= 0.3: {sum(x >= 0.3 for x in r)}")


def sample(n: int, seed: int = 0, per_source_cap: int = 80, tag: str = "", types: tuple[int, ...] = (1, 2)) -> None:
    cfg, rows, H, rec, concepts = _load()
    th = read_json(AV / (f"thresholds_{tag}.json" if tag else "thresholds.json"))
    names = [c for c, v in th.items() if v.get("usable", True) and "t_high" in v]
    U = np.stack([np.load(AV / f"dir_{c}.npy") for c in names])          # [k, d]
    srcs = np.array([r["src"] for r in rows])
    rng = np.random.default_rng(seed)
    ok = np.flatnonzero(np.isin(rec["type"], types) & (rec["depth"] >= 0.25))
    # spread over datasets so no single one dominates
    pick, by = [], collections.defaultdict(list)
    for i in rng.permutation(ok):
        s = srcs[rec["text"][i]]
        if len(by[s]) < per_source_cap:
            by[s].append(int(i))
    pool = [i for v in by.values() for i in v]
    pick = list(rng.permutation(pool)[:n])
    S = H[pick].float().numpy() @ U.T                                   # [n, k] projections
    hi = np.array([th[c]["t_high"] for c in names])
    lo = np.array([th[c].get("t_low", -np.inf) for c in names])
    tname = {1: "sentence end", 2: "last token", 3: "mid-text token"}
    ex, acts = [], []
    for r, i in enumerate(pick):
        act = [(names[k], float((S[r, k] - hi[k]) / (abs(hi[k]) + 1e-6))) for k in np.flatnonzero(S[r] >= hi)]
        act.sort(key=lambda x: -x[1])
        inact = [names[k] for k in np.flatnonzero(S[r] <= lo)]
        ex.append(dict(id=f"av{r:05d}", token_row=int(i), text_id=int(rec["text"][i]), source=str(srcs[rec["text"][i]]),
                       position=tname[int(rec["type"][i])], depth=float(rec["depth"][i]),
                       active=[a for a, _ in act], margins=[round(m, 2) for _, m in act], inactive=inact))
        acts.append(H[i])
    write_jsonl(AV / (f"pilot_{tag}.jsonl" if tag else "pilot.jsonl"), ex)
    torch.save(torch.stack(acts), AV / (f"av_acts_{tag}.pt" if tag else "av_acts.pt"))
    cnt = collections.Counter(len(e["active"]) for e in ex)
    cov = collections.Counter(c for e in ex for c in e["active"])
    print(f"{len(ex)} positions | active-concept count distribution {dict(sorted(cnt.items()))}")
    print("share with >=1 active concept: %.2f" % (1 - cnt.get(0, 0) / len(ex)))
    print("most frequent active:", cov.most_common(12))

def sample_enriched(n: int, per_concept: int, tag: str, seed: int = 0, multi_share: float = 0.25, none_share: float = 0.10) -> None:
    """Training pool built around where the concepts fire: up to `per_concept` positions for every concept
    (so each one appears often), a share of positions with >= 2 active concepts, and a few with nothing active.
    At most 2 positions per text and at most 15% of a concept's examples from any one dataset."""
    cfg, rows, H, rec, concepts = _load()
    th = read_json(AV / f"thresholds_{tag}.json")
    names = [c for c, v in th.items() if "t_high" in v]
    U = np.stack([np.load(AV / f"dir_{c}.npy") for c in names])
    srcs = np.array([r["src"] for r in rows])
    rng = np.random.default_rng(seed)
    ok = np.flatnonzero(np.isin(rec["type"], (1, 2)) & (rec["depth"] >= 0.25))
    S = H[ok].float().numpy() @ U.T
    hi = np.array([th[c]["t_high"] for c in names])
    lo = np.array([th[c].get("t_low", -np.inf) for c in names])
    act = S >= hi
    nact = act.sum(1)
    text, src = rec["text"][ok], srcs[rec["text"][ok]]
    used_text = collections.Counter()
    chosen: list[int] = []
    chosen_set: set[int] = set()

    def take(i: int) -> bool:
        if i in chosen_set or used_text[text[i]] >= 2:
            return False
        chosen.append(int(i)); chosen_set.add(int(i)); used_text[text[i]] += 1
        return True

    # 1. a few positions per concept, prefering ones where several concepts fire
    for k, c in enumerate(names):
        idx = np.flatnonzero(act[:, k])
        idx = rng.permutation(idx)
        by, got = collections.Counter(), 0
        for i in idx:
            if by[src[i]] >= max(2, int(0.15 * per_concept)):
                continue
            if take(i):
                by[src[i]] += 1; got += 1
            if got >= per_concept:
                break
    # 2. extra multi-concept positions
    multi = np.flatnonzero(nact >= 2)
    for i in rng.permutation(multi):
        if len(chosen) >= int(n * (1 - none_share)):
            break
        if sum(1 for j in chosen if nact[j] >= 2) >= int(n * multi_share):
            break
        take(i)
    # 3. some with nothing active (spread over datasets)
    none = np.flatnonzero(nact == 0)
    for i in rng.permutation(none)[: int(n * none_share) * 3]:
        if len([j for j in chosen if nact[j] == 0]) >= int(n * none_share):
            break
        take(i)
    tname = {1: "sentence end", 2: "last token"}
    ex, acts = [], []
    for r, j in enumerate(chosen):
        i = ok[j]
        a = sorted(((names[k], float((S[j, k] - hi[k]) / (abs(hi[k]) + 1e-6))) for k in np.flatnonzero(act[j])), key=lambda x: -x[1])
        ex.append(dict(id=f"en{r:05d}", token_row=int(i), text_id=int(rec["text"][i]), source=str(src[j]), position=tname[int(rec["type"][i])],
                       depth=float(rec["depth"][i]), active=[x for x, _ in a], margins=[round(m, 2) for _, m in a],
                       inactive=[names[k] for k in np.flatnonzero(S[j] <= lo)]))
        acts.append(H[i])
    write_jsonl(AV / f"enriched_{tag}.jsonl", ex)
    torch.save(torch.stack(acts), AV / f"enriched_acts_{tag}.pt")
    cnt = collections.Counter(len(e["active"]) for e in ex)
    cov = collections.Counter(c for e in ex for c in e["active"])
    print(f"{len(ex)} positions | actives per position {dict(sorted(cnt.items()))}")
    print(f"with >=1 active: {1 - cnt.get(0, 0) / len(ex):.2f} | with >=2: {sum(v for k, v in cnt.items() if k >= 2) / len(ex):.2f}")
    print("examples per concept: min %d, median %d, max %d (concepts: %d)" % (min(cov.values()), np.median(list(cov.values())), max(cov.values()), len(cov)))
    print("datasets represented:", len({e['source'] for e in ex}), "| top:", collections.Counter(e['source'] for e in ex).most_common(6))



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["calibrate", "thresholds", "sample", "enriched"])
    ap.add_argument("n", nargs="?", type=int, default=2000)
    ap.add_argument("--prevalence", type=float, default=0.10)
    ap.add_argument("--types", default="1,2,3")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.stage == "calibrate":
        calibrate()
    elif a.stage == "thresholds":
        thresholds(a.prevalence, tuple(int(x) for x in a.types.split(",")), a.tag or "custom")
    elif a.stage == "enriched":
        sample_enriched(a.n, per_concept=60, tag=a.tag or "global10")
    else:
        sample(a.n, tag=a.tag)
    sys.exit(0)
