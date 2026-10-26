"""Pod. Concept retention for B3 (= B), coll, ctl on B's HELD-OUT positions (poolB_eval; never trained on) and on the original SFT positions.
Decoder = the repo's cheap TF-IDF reader fit on SFT sentences -> active trained concepts (cv.av_sft.fit_decoder), same as the built-in F1.
   cd /root/cv_repo && python /root/cc/retention.py"""
import json, re, sys, collections, numpy as np
sys.path.insert(0, "/root/cv_repo")
from cv.av_sft import fit_decoder, prf, mentions
G = "/root/cv_repo/runs/general"; AD = f"{G}/av_data"; N = "/root/data"
HIDDEN = ["music", "law", "mental_health", "vehicles", "hedging", "german"]
rows = {json.loads(l)["id"]: json.loads(l) for l in open(f"{AD}/enriched_sft_global10.jsonl")}
desc = {json.loads(l)["id"]: json.loads(l)["description"] for l in open(f"{AD}/th_descriptions.jsonl") if l.strip()}
drop = {json.loads(l)["id"] for l in open(f"{AD}/strict_drop_ids.jsonl")}
ids = [i for i in rows if i in desc and i not in drop and not (set(rows[i]["active"]) & set(HIDDEN)) and not mentions(desc[i], HIDDEN)]
trained = sorted({c for i in ids for c in rows[i]["active"]})
dec = fit_decoder([desc[i] for i in ids], [rows[i]["active"] for i in ids], trained)
meta = [json.loads(l) for l in open("/root/cc/poolB_eval_meta.jsonl")]
truth = [set(m["trained_active"]) & set(trained) for m in meta]
hid_sel = [set(m.get("selected_by", [])) & set(HIDDEN) for m in meta]
def lines(p): return [json.loads(l)["text"] for l in open(p)]
KW = {"music": r"\bmusic|musical|song|singer|guitar|band\b|concert|melod", "law": r"\blaw\b|legal|court|judge|attorney|lawyer", "mental_health": r"mental|depress|anxiety|therap|psycholog", "vehicles": r"vehicle|\bcar\b|truck|motor|driving|aircraft|bike", "hedging": r"hedg|uncertain|perhaps|might", "german": r"german|deutsch"}
print(f"{'adapter':6s} | trained-concept F1 on B's held-out positions (n={len(meta)}) | named-held-out-concept rate on positions picked for a held-out concept | sentences distinct")
for ad in ("B3", "coll", "ctl"):
    S = lines(f"{N}/nla3_bev_{ad}.jsonl")
    p = prf(dec(S), truth)
    hp = [i for i, h in enumerate(hid_sel) if h]
    named = sum(any(re.search(KW[c], S[i], re.I) for c in hid_sel[i]) for i in hp)
    print(f"{ad:6s} | F1 {p['f1']:.3f} (P {p['p']:.3f} R {p['r']:.3f}) | {named}/{len(hp)} | distinct {len(set(S))}/{len(S)}")
# original enriched positions: music (63) naming + overall
en = {json.loads(l)["id"]: i for i, l in enumerate(open(f"{AD}/enriched_global10.jsonl"))}
mus = [i for i, r in rows.items() if "music" in r["active"] or "'music'" in str(r["active"])]
print("\noriginal music-active positions (n=%d), sentence mentions music-related word / on all 2650 positions:" % len(mus))
for ad in ("B3", "coll", "ctl"):
    S = lines(f"{N}/nla3_enr_{ad}.jsonl"); k = re.compile(KW["music"], re.I)
    print(f"  {ad:5s} {sum(bool(k.search(S[en[i]])) for i in mus)}/{len(mus)}   all positions: {sum(bool(k.search(s)) for s in S)}")
