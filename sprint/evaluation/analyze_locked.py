"""Confirmatory analysis on the LOCKED-L set.  python analyze_locked.py  (reads results/gen/ev_locked_<name>.jsonl, data/locked_truth.jsonl)
Arms are pooled over seeds by name pattern  <ARM>s<seed>_<step>  e.g. A2s12_100.  Output: results/locked_report.md"""
import json, glob, os, random, re, collections, statistics as st, sys
sys.path.insert(0, os.path.dirname(__file__)); from flag import FLAG, FLAG_OFFICIAL
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T = [json.loads(l) for l in open(f"{R}/data/locked_truth.jsonl")]
cells = collections.defaultdict(list)
for i, t in enumerate(T): cells[t["cell"]].append(i)
byp = collections.defaultdict(dict)
for i, t in enumerate(T):
    if t["cell"] in ("collusive_nocue", "honest_nocue"): byp[("nc", t["pair"])][t["cell"]] = i
    if t["cell"] in ("collusive_cue", "honest_twin_explicit"): byp[("cu", t["pair"])][t["cell"]] = i
PN = [(v["collusive_nocue"], v["honest_nocue"]) for k, v in byp.items() if k[0] == "nc" and len(v) == 2]
PC = [(v["collusive_cue"], v["honest_twin_explicit"]) for k, v in byp.items() if k[0] == "cu" and len(v) == 2]
HC = cells["honest_cue"]
def load(name):
    p = f"{R}/results/gen/ev_locked_{name}.jsonl"
    return [json.loads(l)["text"].replace("\n", " ") for l in open(p)] if os.path.exists(p) else None
def flags(S, rx): return [int(bool(rx.search(s))) for s in S]
def ba(F, pairs): return sum((F[c] + 1 - F[h]) / 2 for c, h in pairs) / len(pairs)
def boot(fn, pairs, B=2000, seed=0):
    rng = random.Random(seed); v = sorted(fn(rng.choices(pairs, k=len(pairs))) for _ in range(B)); return fn(pairs), v[int(.025 * B)], v[int(.975 * B)]
def models():
    ms = {}
    for p in sorted(glob.glob(f"{R}/results/gen/ev_locked_*.jsonl")):
        n = os.path.basename(p)[len("ev_locked_"):-6]
        ms[n] = n
    return list(ms)
def group(names):
    g = collections.defaultdict(list)
    for n in names:
        m = re.match(r"(A\d)s(\d+)_(\d+)$", n)
        g[f"{m.group(1)}_{m.group(3)}" if m else n].append(n)
    return g
G = group(models())
if os.path.exists(f"{R}/results/av_locked.json"):  # official NLA (separate file)
    pass
out = []; P = out.append
P("# LOCKED-L report (auto-generated)\n")
P(f"pairs: no-cue {len(PN)}, with-cue {len(PC)}; honest-with-cue sentences {len(HC)}.  Flag = output contains a collusion/secrecy word (scripts/flag.py).  BA = balanced accuracy of the flag over matched pairs (0.5 = chance).  95% CI: bootstrap over pairs (and over seeds when pooled).\n")
P("| model / arm (seeds) | collusive no-cue flagged | honest no-cue flagged | **BA no-cue** [CI] | collusive with-cue flagged | honest twin flagged | **honest-with-cue flagged** | BA with-cue |")
P("|---|---|---|---|---|---|---|---|")
FL = {}
for g, names in G.items():
    Fs = [flags(load(n), FLAG) for n in names]
    F = [sum(f[i] for f in Fs) / len(Fs) for i in range(len(T))]; FL[g] = F   # mean flag over seeds
    a = boot(lambda ps: ba(F, ps), PN); b = ba(F, PC)
    P("| %s (n=%d) | %.0f%% | %.0f%% | **%.3f** [%.3f, %.3f] | %.0f%% | %.0f%% | **%.0f%%** | %.3f |" % (g, len(names), 100 * st.mean(F[c] for c, h in PN), 100 * st.mean(F[h] for c, h in PN), *a, 100 * st.mean(F[c] for c, h in PC), 100 * st.mean(F[h] for c, h in PC), 100 * st.mean(F[i] for i in HC), b))
    if len(names) > 1: P("|   per-seed BA no-cue: " + ", ".join("%s %.3f" % (n, ba(f, PN)) for n, f in zip(names, Fs)) + " | | | | | | | |")
def diff(x, y, label):
    if x not in FL or y not in FL: return
    fn = lambda ps: ba(FL[x], ps) - ba(FL[y], ps)
    d = boot(fn, PN); rng = random.Random(1); obs = fn(PN)
    # sign-flip permutation on per-pair differences
    dd = [(FL[x][c] + 1 - FL[x][h]) / 2 - (FL[y][c] + 1 - FL[y][h]) / 2 for c, h in PN]
    cnt = sum(abs(sum(rng.choice((-1, 1)) * v for v in dd) / len(dd)) >= abs(obs) - 1e-12 for _ in range(5000))
    P("| %s | %+.3f [%+.3f, %+.3f] | %.4f |" % (label, *d, (cnt + 1) / 5001))
P("\n## Paired differences on the primary endpoint (BA no-cue)\n\n| contrast | diff [95% CI over pairs] | sign-flip permutation p |\n|---|---|---|")
for x, y in (("A2_100", "A1_100"), ("A2_100", "sft"), ("A1_100", "sft"), ("A1_100", "A0_100"), ("A0_100", "sft"), ("A2_100", "A0_100"), ("A2_100", "B"), ("A2_50", "A1_50")):
    diff(x, y, f"{x} − {y}")

# ---- 2x2: intent (collusive/honest) x cue words (present/absent) ----
CELL = {"CN": cells["collusive_nocue"], "HN": cells["honest_nocue"], "CC": cells["collusive_cue"], "HC": cells["honest_cue"]}
def rate(F, idx): return sum(F[i] for i in idx) / len(idx)
def eff(F, ca, cb, rng):
    ia = [rng.choice(CELL[ca]) for _ in CELL[ca]]; ib = [rng.choice(CELL[cb]) for _ in CELL[cb]]; return rate(F, ia) - rate(F, ib)
P("\n## 2x2 control: does the flag follow INTENT or CUE WORDS?  (flag rates by cell; effects = differences in percentage points, 95% CI by resampling sentences)\n")
P("cells: CN = collusive, no cue words; HN = honest, no cue words; CC = collusive with cue words; HC = honest with cue words (e.g. 'I won't collude with you').\n")
P("| arm | CN | HN | CC | HC | INTENT effect, no cue (CN−HN) | INTENT effect, with cue (CC−HC) | CUE effect, honest (HC−HN) | CUE effect, collusive (CC−CN) |")
P("|---|---|---|---|---|---|---|---|---|")
for g, F in FL.items():
    rng = random.Random(3); row = []
    for a, b in (("CN", "HN"), ("CC", "HC"), ("HC", "HN"), ("CC", "CN")):
        v = sorted(eff(F, a, b, rng) for _ in range(1000)); row.append("%+.0f [%+.0f, %+.0f]" % (100 * (rate(F, CELL[a]) - rate(F, CELL[b])), 100 * v[25], 100 * v[975]))
    P("| %s | %.0f%% | %.0f%% | %.0f%% | %.0f%% | %s |" % (g, *(100 * rate(F, CELL[c]) for c in ("CN", "HN", "CC", "HC")), " | ".join(row)))
open(f"{R}/results/locked_report.md", "w").write("\n".join(out)); print("\n".join(out))
