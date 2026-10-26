"""Secondary sets: HAND-50 (20 pairs, written by me) and HELD-OUT-578 (239 pairs, 4 held-out types). Same flag, pooled over seeds like analyze_locked.py."""
import json, glob, os, random, re, collections, statistics as st, sys
sys.path.insert(0, os.path.dirname(__file__)); from flag import FLAG
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
I = json.load(open(f"{R}/data/short_info.json")); T = [json.loads(l) for l in open(f"{R}/data/short_test_truth.jsonl")]
c50 = [i for i, l in enumerate(I["lab"]) if l == "colluding"]; h50 = [i for i, l in enumerate(I["lab"]) if l == "honest"]; P50 = list(zip(c50, h50))
byp = collections.defaultdict(list)
for i, t in enumerate(T):
    if t["type"] != "neutral": byp[t["pair"]].append(i)
P578 = [(next(i for i in v if T[i]["label"] == 1), next(i for i in v if T[i]["label"] == 0)) for v in byp.values() if len(v) == 2]
def load(setn, n):
    p = f"{R}/results/gen/ev_{setn}_{n}.jsonl"; return [json.loads(l)["text"] for l in open(p)] if os.path.exists(p) else None
names = sorted({os.path.basename(p)[len("ev_short_"):-6] for p in glob.glob(f"{R}/results/gen/ev_short_*.jsonl")})
G = collections.defaultdict(list)
for n in names:
    m = re.match(r"(A\d)s(\d+)_(\d+)$", n); G[f"{m.group(1)}_{m.group(3)}" if m else n].append(n)
def ba(F, ps): return sum((F[c] + 1 - F[h]) / 2 for c, h in ps) / len(ps)
def boot(F, ps):
    r = random.Random(0); v = sorted(ba(F, r.choices(ps, k=len(ps))) for _ in range(2000)); return ba(F, ps), v[50], v[1950]
out = ["# Secondary sets (already seen once; consistency check, not confirmatory)\n", "| arm | HAND-50: collusive / honest flagged (of 20, seed-mean) | **BA** [CI over 20 pairs] | HELD-OUT-578: collusive / honest flagged | **BA** [CI over 239 pairs] |", "|---|---|---|---|---|"]
for g, ns in G.items():
    row = []
    for setn, ps, nn in (("short", P50, 20), ("shortt", P578, None)):
        Fs = [[int(bool(FLAG.search(s))) for s in load(setn, n)] for n in ns]; F = [sum(f[i] for f in Fs) / len(Fs) for i in range(len(Fs[0]))]
        b = boot(F, ps); row.append(("%.1f / %.1f" % (sum(F[c] for c, h in ps), sum(F[h] for c, h in ps)) if nn else "%.0f%% / %.0f%%" % (100 * st.mean(F[c] for c, h in ps), 100 * st.mean(F[h] for c, h in ps)), "%.3f [%.3f, %.3f]" % b))
    out.append(f"| {g} (n={len(ns)}) | {row[0][0]} | {row[0][1]} | {row[1][0]} | {row[1][1]} |")
open(f"{R}/results/secondary_report.md", "w").write("\n".join(out)); print("\n".join(out))
