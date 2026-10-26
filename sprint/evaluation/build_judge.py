"""Build blind judge chunks. The judge sees ONLY an opaque id and the verbalizer's output sentence(s). Key (id -> model, input index, cell) stays here, never given to the judge."""
import json, random, os, sys
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); J = sys.argv[1]; os.makedirs(J, exist_ok=True)
T = [json.loads(l) for l in open(f"{R}/data/locked_truth.jsonl")]
rng = random.Random(42)
byc = {}
for i, t in enumerate(T): byc.setdefault(t["cell"], []).append(i)
pairs = {}
for i, t in enumerate(T):
    if t["cell"] in ("collusive_nocue", "honest_nocue"): pairs.setdefault(t["pair"], {})[t["cell"]] = i
pk = sorted(pairs); rng.shuffle(pk); sel = []
for k in pk[:100]: sel += [pairs[k]["collusive_nocue"], pairs[k]["honest_nocue"]]
sel += rng.sample(byc["collusive_cue"], 40) + rng.sample(byc["honest_cue"], 40)
MODELS = sys.argv[2:]
def load(m):
    if m == "official": return [t.replace("\n", " ")[:600] for t in json.load(open(f"{R}/results/av_locked.json"))["text"]]
    return [json.loads(l)["text"].replace("\n", " ") for l in open(f"{R}/results/gen/ev_locked_{m}.jsonl")]
items = []
for m in MODELS:
    S = load(m)
    for i in sel: items.append({"model": m, "idx": i, "text": S[i]})
rng.shuffle(items)
for n, it in enumerate(items): it["id"] = f"j{n:05d}"
json.dump(items, open(f"{R}/results/judge_key.json", "w"))
CH = 140
for k in range(0, len(items), CH):
    json.dump([{"id": it["id"], "text": it["text"]} for it in items[k:k + CH]], open(f"{J}/chunk_{k // CH}.json", "w"))
print(len(items), "items,", (len(items) + CH - 1) // CH, "chunks")
