"""Collusion direction = mean(collusive) - mean(independent+open), train split only, layer 20.
Gates (as in cv/directions.py): split-half cosine over train scenarios, held-out AUROC (val, test; also per negative version),
shuffled-label control, random-direction control. Cosine with the repo's 53 existing directions.
Zero-shot transfer: projections of Core/Transfer agents (mean over public replay messages) and Verdict episodes (mean over messages)
onto the direction - never fit on those. Outputs: dirs/collusion_{last,mean}.npy, proj.json, gate.json.
   python directions_concept.py concept_texts.jsonl acts_prefix"""
import json, sys, glob, os, numpy as np, torch
from sklearn.metrics import roc_auc_score
rows = [json.loads(l) for l in open(sys.argv[1])]; pre = sys.argv[2]
y = np.array([r["label"] for r in rows]); sp = np.array([r["split"] for r in rows]); ver = np.array([r["version"] for r in rows]); grp = np.array([r["group"] for r in rows])
D = "/root/data"; V = "/root/verdict"; AD = "/root/cv_repo/runs/general/av_data"
os.makedirs("dirs", exist_ok=True)
def diff(A, lab): return A[lab == 1].mean(0) - A[lab == 0].mean(0)
exist = {os.path.basename(f)[4:-4]: np.load(f).astype(np.float32) for f in glob.glob(f"{AD}/dir_*.npy")}
gate = {}; proj = {"core": {}, "verdict": {}}
units = [json.loads(l) for l in open(f"{D}/units.jsonl")]; meta = [json.loads(l) for l in open(f"{D}/acts_meta.jsonl")]; idx = {m["mid"]: i for i, m in enumerate(meta)}
R = torch.load(f"{D}/acts_replay.pt").float().numpy()
ev = [json.loads(l) for l in open(f"{V}/events.jsonl")]; M = torch.load(f"{V}/msg_L20.pt").float().numpy()
eids = sorted({e["eid"] for e in ev}); byep = {}
for i, e in enumerate(ev): byep.setdefault(e["eid"], []).append(i)
rng = np.random.default_rng(0)
for mode in ("last", "mean"):
    A = torch.load(f"{pre}_{mode}.pt").float().numpy(); tr = sp == "train"
    v = diff(A[tr], y[tr]); n = np.linalg.norm(v); u = v / n
    g = {"norm": float(n), "n_train": int(tr.sum())}
    ug = np.unique(grp[tr]); cs = []
    for _ in range(20):
        h = rng.permutation(ug)[: len(ug) // 2]; m1 = np.isin(grp, h) & tr; m2 = ~np.isin(grp, h) & tr
        a, b = diff(A[m1], y[m1]), diff(A[m2], y[m2]); cs.append(float(a @ b / np.linalg.norm(a) / np.linalg.norm(b)))
    g["split_half_cos_mean"] = float(np.mean(cs))
    for s in ("val", "test"):
        m = sp == s; sc = A[m] @ u; g[f"auroc_{s}"] = float(roc_auc_score(y[m], sc))
        for ng in ("independent", "open"):
            mm = m & ((ver == ng) | (ver == "collusive")); g[f"auroc_{s}_vs_{ng}"] = float(roc_auc_score(y[mm], A[mm] @ u))
    sh = rng.permutation(y[tr]); vs = diff(A[tr], sh); vs /= np.linalg.norm(vs)
    g["shuffled_label_auroc_test"] = float(roc_auc_score(y[sp == "test"], A[sp == "test"] @ vs))
    ra = []
    for _ in range(20):
        r = rng.normal(size=u.shape); r /= np.linalg.norm(r); ra.append(roc_auc_score(y[sp == "test"], A[sp == "test"] @ r))
    g["random_dir_auroc_test_mean_abs_dev"] = float(np.mean(np.abs(np.array(ra) - 0.5)))
    # only-version directions (is it 'agreement'?)
    for ng in ("independent", "open"):
        mk = tr & ((ver == ng) | (ver == "collusive")); w = diff(A[mk], y[mk]); w /= np.linalg.norm(w); g[f"cos_with_dir_vs_{ng}_only"] = float(u @ w)
    cos = {k: float(u @ (e / np.linalg.norm(e))) for k, e in exist.items()}
    g["top_cos_existing"] = sorted(cos.items(), key=lambda kv: -abs(kv[1]))[:8]
    gate[mode] = g; np.save(f"dirs/collusion_{mode}.npy", v.astype(np.float32))
    ucore = {}
    for un in units:
        ii = [idx[m] for m in un["mids"] if "/pub" in m and m in idx]
        if ii: ucore[un["uid"]] = float(R[ii].mean(0) @ u)
    proj["core"][mode] = ucore
    proj["verdict"][mode] = {e: float(M[byep[e]].mean(0) @ u) for e in eids}
    print(mode, json.dumps(g)[:900], flush=True)
json.dump(gate, open("gate.json", "w"), indent=1); json.dump(proj, open("proj.json", "w"))
print("saved")
