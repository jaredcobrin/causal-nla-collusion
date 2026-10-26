"""Merge select_pool outputs (untrained / random / trained / hidden) into B's pool: 90% for RL training, 10% held out for retention tests.
   python merge_b.py out_prefix sel_u sel_r sel_t sel_h"""
import json, sys, hashlib, torch
out, parts = sys.argv[1], sys.argv[2:]
A, P, M = [], [], []
for p in parts:
    A.append(torch.load(p + "_acts.pt")); P += torch.load(p + "_prefix.pt"); M += [json.loads(l) for l in open(p + "_meta.jsonl")]
A = torch.cat(A)
tr, ev = [], []
for k, m in enumerate(M):
    m = dict(m); m["orig_id"] = m["id"]; m["id"] = f"bp{k:05d}"; M[k] = m
    (ev if int(hashlib.md5(m["id"].encode()).hexdigest(), 16) % 10 == 0 else tr).append(k)
for name, ix in (("train", tr), ("eval", ev)):
    torch.save(A[ix], f"{out}_{name}_acts.pt"); torch.save([P[i] for i in ix], f"{out}_{name}_prefix.pt")
    with open(f"{out}_{name}_meta.jsonl", "w") as f:
        for i in ix: f.write(json.dumps(M[i]) + "\n")
    print(name, len(ix), "positions; with >=1 trained concept active:", sum(1 for i in ix if M[i]["trained_active"]), "| selected_by held-out concept:", sum(1 for i in ix if any(c in ("music","law","mental_health","vehicles","hedging","german") for c in M[i].get("selected_by", []))))
