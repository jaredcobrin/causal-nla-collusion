"""Pod. Retention of B's skills for each evaluated adapter on B's held-out positions (poolB_eval, 307; never trained on).
   python retention_ev.py name1 name2 ...   reads /root/data/ev_bev_<name>.jsonl. Uses the repo's TF-IDF concept decoder (fit on SFT sentences), same as the built-in F1."""
import re, sys
src = open("/root/cc/ret8b.py").read().split("print(f\"{'adapter'")[0]
exec(src)
print(f"{'adapter':12s} | trained-concept F1 (n=307) | held-out concept named (of 68 picked for one) | distinct sentences")
for ad in sys.argv[1:]:
    S = lines(f"/root/data/ev_bev_{ad}.jsonl"); p = prf(dec(S), truth)
    hp = [i for i, h in enumerate(hid_sel) if h]; named = sum(any(re.search(KW[c], S[i], re.I) for c in hid_sel[i]) for i in hp)
    print(f"{ad:12s} | F1 {p['f1']:.3f} | {named}/{len(hp)} | distinct {len(set(S))}/{len(S)}")
