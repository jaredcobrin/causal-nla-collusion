"""Blind judge analysis. python analyze_judge.py <judge_dir> <pass tags e.g. p1 p2>.  Per model: mean score by cell, pair AUROC (collusive>twin, ties .5) with pair-bootstrap CI, honest-with-cue vs collusive-with-cue AUROC; correlation with keyword flag."""
import json, glob, os, sys, random, collections, statistics as st
sys.path.insert(0, os.path.dirname(__file__)); from flag import FLAG, FLAG_OFFICIAL
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); J = sys.argv[1]; tags = sys.argv[2:]
key = json.load(open(f"{R}/results/judge_key.json")); T = [json.loads(l) for l in open(f"{R}/data/locked_truth.jsonl")]
sc = collections.defaultdict(list)
for tg in tags:
    for f in glob.glob(f"{J}/out_{tg}_*.json"):
        for k, v in json.load(open(f)).items(): sc[k].append(float(v))
byid = {it["id"]: it for it in key}
S = collections.defaultdict(dict)   # model -> idx -> mean score
miss = 0
for k, it in byid.items():
    if k in sc: S[it["model"]][it["idx"]] = st.mean(sc[k])
    else: miss += 1
pairs = {}
for i, t in enumerate(T):
    if t["cell"] in ("collusive_nocue", "honest_nocue"): pairs.setdefault(t["pair"], {})[t["cell"]] = i
P = [(v["collusive_nocue"], v["honest_nocue"]) for v in pairs.values() if len(v) == 2]
CC = [i for i, t in enumerate(T) if t["cell"] == "collusive_cue"]; HC = [i for i, t in enumerate(T) if t["cell"] == "honest_cue"]
def auc(a, b): return sum((x > y) + .5 * (x == y) for x in a for y in b) / (len(a) * len(b))
out = ["# Blind-judge report (Haiku, sentence-only, passes: %s; missing ids: %d)\n" % (",".join(tags), miss),
       "| model | n pairs | mean score: collusive-no-cue / honest-no-cue | **pair AUROC no-cue** [95% CI] | collusive-with-cue vs honest-with-cue AUROC | mean score HC / CC |", "|---|---|---|---|---|---|"]
for m in S:
    d = S[m]; pp = [(c, h) for c, h in P if c in d and h in d]
    if not pp: continue
    f = lambda ps: sum((d[c] > d[h]) + .5 * (d[c] == d[h]) for c, h in ps) / len(ps)
    rng = random.Random(0); b = sorted(f(rng.choices(pp, k=len(pp))) for _ in range(2000))
    cc = [d[i] for i in CC if i in d]; hc = [d[i] for i in HC if i in d]
    out.append("| %s | %d | %.1f / %.1f | **%.3f** [%.3f, %.3f] | %.3f | %.1f / %.1f |" % (m, len(pp), st.mean(d[c] for c, h in pp), st.mean(d[h] for c, h in pp), f(pp), b[50], b[1950], auc(cc, hc), st.mean(hc), st.mean(cc)))
open(f"{R}/results/judge_report.md", "w").write("\n".join(out)); print("\n".join(out))
