"""Fresh activations for RL: positions from texts never used before (not in the 2,650-position pool, not from the test datasets).

Same extraction as cv.general.tokens_pass (layer-20 output, last token and one sentence end per text, outlier tokens skipped).
Positions where any held-out concept is active are dropped; positions with at least one trained concept active are kept
(a quota per concept, like the original pool). RL needs only the activations, no labels or sentences.

    python -m cv.fresh_acts --n_texts 30000 --keep 3000 --out runs/general/av_data/fresh
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cv.general import DATA, load_cfg, read_jsonl
from cv.model import LM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_texts", type=int, default=30000)
    ap.add_argument("--keep", type=int, default=3000)
    ap.add_argument("--hidden", default="music,law,mental_health,vehicles,hedging,german")
    ap.add_argument("--test_sources", default="clinc,massive,arxiv,pku,beaver,fintopic")
    ap.add_argument("--av", default="runs/general/av_data")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--hf", default="", help="public datasets to add, 'name|config|split|column|n' separated by ';'")
    ap.add_argument("--only_hf", action="store_true", help="use only the --hf datasets (no local texts)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    av = Path(a.av)
    hidden = set(a.hidden.split(","))
    test_src = set(a.test_sources.split(","))
    used = {json.loads(l)["text_id"] for l in open(av / "enriched_sft_global10.jsonl")}
    th = json.load(open(av / "thresholds_global10.json"))
    names = [c for c, v in th.items() if "t_high" in v]
    U = np.stack([np.load(av / f"dir_{c}.npy") for c in names])
    hi = np.array([th[c]["t_high"] for c in names])
    is_hidden = np.array([c in hidden for c in names])
    rows = read_jsonl(DATA / "texts.jsonl")
    cand = [i for i, r in enumerate(rows) if r["src"] not in test_src and r["i"] not in used and len(r["text"].split()) >= 8]
    cand = list(rng.permutation(cand)[: a.n_texts]) if not a.only_hf else []
    items = [dict(text=rows[j]["text"], src=rows[j]["src"], i=int(rows[j]["i"])) for j in cand]
    if a.hf:
        from datasets import load_dataset
        for spec in a.hf.split(";"):
            name, cfgname, split, col, n = spec.split("|")
            ds = load_dataset(name, cfgname or None, split=split, streaming=True)
            got = 0
            seen = set()
            for r in ds:
                t = str(r.get(col, "")).strip()
                if len(t.split()) >= 8 and t not in seen:
                    seen.add(t)
                    items.append(dict(text=t[:2500], src=name.split("/")[-1], i=-1))
                    got += 1
                if got >= int(n):
                    break
            print(f"{name}: {got} texts", flush=True)
        rng.shuffle(items)
    cand = list(range(len(items)))
    rows = items
    print(f"{len(cand)} candidate texts", flush=True)
    cfg = load_cfg()
    lm = LM(cfg["model"])
    sent_end = np.array([lm.tok.decode([i]).strip().endswith((".", "!", "?")) for i in range(len(lm.tok))])
    H, meta = [], []
    for k in range(0, len(cand), 48):
        idx = cand[k: k + 48]
        enc = lm.tok([rows[j]["text"] for j in idx], return_tensors="pt", padding=True, truncation=True, max_length=192,
                     add_special_tokens=False).to(lm.device)
        with torch.no_grad():
            h = lm.model.model(**enc, output_hidden_states=True).hidden_states[a.layer + 1].float()
        norms, ids, mask = h.norm(dim=-1).cpu().numpy(), enc.input_ids.cpu().numpy(), enc.attention_mask.cpu().numpy()
        for b, j in enumerate(idx):
            real = np.flatnonzero(mask[b])
            if len(real) < 8:
                continue
            first, last = int(real[0]), int(real[-1])
            ref = norms[b][real[1:]].mean()
            ok = np.array([i for i in real[1:-1] if norms[b][i] < 3 * ref], dtype=int)
            if len(ok) == 0 or norms[b][last] > 3 * ref:
                continue
            take = [(last, "last token")]
            ends = rng.permutation(ok[sent_end[ids[b][ok]]])[:1]
            take += [(int(i), "sentence end") for i in ends]
            for i, ty in take:
                H.append(h[b, i].half().cpu())
                meta.append(dict(text_id=int(rows[j]["i"]), source=rows[j]["src"], position=ty, depth=float((i - first) / max(last - first, 1))))
        if (k // 48) % 100 == 0:
            print(f"{k}/{len(cand)}", flush=True)
    H = torch.stack(H)
    S = H.float().numpy() @ U.T
    act = S >= hi
    keep = np.flatnonzero((act.sum(1) >= 1) & ~(act & is_hidden).any(1))
    print(f"{len(H)} positions | with a trained concept active and no held-out one: {len(keep)}", flush=True)
    # quota per concept so common concepts do not dominate
    per = max(1, a.keep // 33 * 2)
    chosen, count = [], {}
    for i in rng.permutation(keep):
        cs = [names[k] for k in np.flatnonzero(act[i])]
        if all(count.get(c, 0) >= per for c in cs):
            continue
        chosen.append(int(i))
        for c in cs:
            count[c] = count.get(c, 0) + 1
        if len(chosen) >= a.keep:
            break
    out = Path(a.out)
    torch.save(H[chosen], str(out) + "_acts.pt")
    with open(str(out) + "_meta.jsonl", "w") as f:
        for n, i in enumerate(chosen):
            f.write(json.dumps(dict(id=f"fr{n:05d}", **meta[i], active=[names[k] for k in np.flatnonzero(act[i])])) + "\n")
    print(f"saved {len(chosen)} fresh positions; per-concept counts min/max:", min(count.values()), max(count.values()), flush=True)


if __name__ == "__main__":
    main()
