"""Choose RL positions from the extracted pool (cv.pool_acts) in one of three ways, all from the same candidates:

  random     random positions
  untrained  positions where one of the 13 approved-but-unused directions is strongly active (top 2% of its scores)
  trained    positions where one of the 34 trained concepts fires (probe score >= its threshold), like the SFT pool

Outlier activations are always dropped; positions where a held-out concept fires are dropped unless --keep_hidden; mode 'hidden' picks
positions where one of the 6 held-out concepts fires.

    python -m cv.select_pool --pool runs/general/av_data/pool --mode untrained --n 3000 --out runs/general/av_data/rlU
"""
import argparse
import collections
import json

import numpy as np
import torch

HIDDEN = ["music", "law", "mental_health", "vehicles", "hedging", "german"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--mode", required=True, choices=["random", "untrained", "trained", "hidden"])
    ap.add_argument("--keep_hidden", action="store_true", help="do not drop positions where a held-out concept fires")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--av", default="runs/general/av_data")
    ap.add_argument("--top", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    H = torch.load(a.pool + "_acts.pt")
    meta = [json.loads(l) for l in open(a.pool + "_meta.jsonl")]
    prefix = torch.load(a.pool + "_prefix.pt")
    th = json.load(open(f"{a.av}/thresholds_global10.json"))
    names = [c for c, v in th.items() if "t_high" in v]
    hi = np.array([th[c]["t_high"] for c in names])
    unused = sorted(c for c in (p.split("dir_")[-1][:-4] for p in __import__("glob").glob(f"{a.av}/dir_*.npy")) if c not in names)
    U = np.stack([np.load(f"{a.av}/dir_{c}.npy") for c in names])
    S = H.float().numpy() @ U.T
    is_hid = np.array([c in HIDDEN for c in names])
    act = S >= hi
    ok = (H.float().norm(dim=-1).numpy() < 300) if a.keep_hidden else (~(act & is_hid).any(1) & (H.float().norm(dim=-1).numpy() < 300))
    print(f"{len(H)} positions; {int(ok.sum())} without a held-out concept firing", flush=True)
    if a.mode == "random":
        pick = list(rng.permutation(np.flatnonzero(ok))[: a.n])
        why = [[] for _ in pick]
    elif a.mode == "hidden":
        hid_ix = [k for k in range(len(names)) if is_hid[k]]
        cand = np.flatnonzero(ok & act[:, hid_ix].any(1))
        per = max(1, a.n // len(hid_ix) * 2)
        count, pick, why = collections.Counter(), [], []
        for i in rng.permutation(cand):
            cs = [names[k] for k in hid_ix if act[i, k]]
            if all(count[c] >= per for c in cs):
                continue
            pick.append(i); why.append(cs)
            count.update(cs)
            if len(pick) >= a.n:
                break
    elif a.mode == "trained":
        trained_ix = [k for k, c in enumerate(names) if not is_hid[k]]
        cand = np.flatnonzero(ok & act[:, trained_ix].any(1))
        per = max(1, a.n // len(trained_ix) * 2)
        count, pick, why = collections.Counter(), [], []
        for i in rng.permutation(cand):
            cs = [names[k] for k in trained_ix if act[i, k]]
            if all(count[c] >= per for c in cs):
                continue
            pick.append(i); why.append(cs)
            count.update(cs)
            if len(pick) >= a.n:
                break
    else:
        Uu = np.stack([np.load(f"{a.av}/dir_{c}.npy") for c in unused])
        Su = H.float().numpy() @ Uu.T
        thr = np.quantile(Su[ok], 1 - a.top, axis=0)
        actu = Su >= thr
        cand = np.flatnonzero(ok & actu.any(1))
        per = max(1, a.n // len(unused) * 2)
        count, pick, why = collections.Counter(), [], []
        for i in rng.permutation(cand):
            cs = [unused[k] for k in np.flatnonzero(actu[i])]
            if all(count[c] >= per for c in cs):
                continue
            pick.append(i); why.append(cs)
            count.update(cs)
            if len(pick) >= a.n:
                break
        print("unused directions used for selection:", unused, flush=True)
    pick = [int(i) for i in pick]
    torch.save(H[pick], a.out + "_acts.pt")
    torch.save([prefix[i] for i in pick], a.out + "_prefix.pt")
    with open(a.out + "_meta.jsonl", "w") as f:
        for n, (i, w) in enumerate(zip(pick, why)):
            f.write(json.dumps(dict(id=f"rl{a.mode[0]}{n:05d}", source=meta[i]["source"], position=meta[i]["position"], selected_by=w,
                                    trained_active=[names[k] for k in np.flatnonzero(act[i]) if not is_hid[k]])) + "\n")
    src = collections.Counter(meta[i]["source"] for i in pick)
    print(f"saved {len(pick)} positions ({a.mode}); sources {dict(src.most_common(12))}", flush=True)
    if why and why[0] != []:
        print("selected-by counts:", dict(collections.Counter(c for w in why for c in w).most_common(15)), flush=True)
    tr_any = np.mean([act[i, [k for k in range(len(names)) if not is_hid[k]]].any() for i in pick])
    print(f"share of selected positions where a TRAINED concept also fires: {tr_any:.0%}", flush=True)


if __name__ == "__main__":
    main()
