"""Activation pool from public datasets that SFT and the AR never saw.

For every text: the layer-20 output at the last token and at one random sentence end (same extraction as
cv.general.tokens_pass: first token and outlier tokens skipped, max 192 tokens), plus the token ids up to and including
that position (needed to patch a rebuilt activation back into the model for the behavioural reward). No selection here;
cv.select_pool chooses the positions.

    python -m cv.pool_acts --hf "rajpurkar/squad||train|context|4000;..." --out runs/general/av_data/pool
"""
import argparse
import json

import numpy as np
import torch

from cv.general import load_cfg
from cv.model import LM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True, help="'name|config|split|column|n' separated by ';'")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    from datasets import load_dataset
    items = []
    for spec in a.hf.split(";"):
        name, cfgname, split, col, n = spec.split("|")
        src = name.split("/")[-1] + (("_" + cfgname.split(".")[-1]) if cfgname else "")
        try:
            ds = load_dataset(name, cfgname or None, split=split, streaming=True)
            got, seen = 0, set()
            for r in ds:
                t = str(r.get(col, "")).strip()
                if len(t.split()) >= 8 and t not in seen:
                    seen.add(t)
                    items.append(dict(text=t[:2500], src=src))
                    got += 1
                if got >= int(n):
                    break
            print(f"{src}: {got} texts", flush=True)
        except Exception as e:
            print(f"SKIPPED {name} {cfgname}: {str(e)[:100]}", flush=True)
    order = rng.permutation(len(items))
    items = [items[i] for i in order]
    lm = LM(load_cfg()["model"])
    sent_end = np.array([lm.tok.decode([i]).strip().endswith((".", "!", "?")) for i in range(len(lm.tok))])
    H, meta, prefix = [], [], []
    for k in range(0, len(items), 48):
        batch = items[k: k + 48]
        enc = lm.tok([x["text"] for x in batch], return_tensors="pt", padding=True, truncation=True, max_length=192,
                     add_special_tokens=False).to(lm.device)
        with torch.no_grad():
            h = lm.model.model(**enc, output_hidden_states=True).hidden_states[a.layer + 1].float()
        norms, ids, mask = h.norm(dim=-1).cpu().numpy(), enc.input_ids.cpu().numpy(), enc.attention_mask.cpu().numpy()
        for b, x in enumerate(batch):
            real = np.flatnonzero(mask[b])
            if len(real) < 8:
                continue
            first, last = int(real[0]), int(real[-1])
            ref = norms[b][real[1:]].mean()
            ok = np.array([i for i in real[1:-1] if norms[b][i] < 3 * ref], dtype=int)
            if len(ok) == 0 or norms[b][last] > 3 * ref:
                continue
            take = [(last, "last token")]
            take += [(int(i), "sentence end") for i in rng.permutation(ok[sent_end[ids[b][ok]]])[:1]]
            for i, ty in take:
                H.append(h[b, i].half().cpu())
                prefix.append(torch.tensor(ids[b][first: i + 1], dtype=torch.int32))
                meta.append(dict(id=f"pl{len(H) - 1:06d}", source=x["src"], position=ty, depth=float((i - first) / max(last - first, 1))))
        if (k // 48) % 100 == 0:
            print(f"{k}/{len(items)}", flush=True)
    torch.save(torch.stack(H), a.out + "_acts.pt")
    torch.save(prefix, a.out + "_prefix.pt")
    with open(a.out + "_meta.jsonl", "w") as f:
        for m in meta:
            f.write(json.dumps(m) + "\n")
    print(f"saved {len(H)} positions", flush=True)


if __name__ == "__main__":
    main()
