"""concept_texts3 (blatant: collusive / independent / open) + rewritten collusive versions that contain the word -> concept_texts4.jsonl.
New rows: version 'collusive_word' (label 1), same group/split. Rewrites kept only if the word (collu*) appears in >= 3 messages and the speakers/length are intact."""
import json, glob, re, collections
R = [json.loads(l) for l in open("concept_texts3.jsonl")]
rw = {}
for f in sorted(glob.glob("gen4/rw_out_*.jsonl")):
    for l in open(f):
        try: d = json.loads(l)
        except Exception: continue
        rw[d["id"]] = d["msgs"]
orig = {r["group"]: r for r in R if r["version"] == "collusive"}
kept = dropped = 0; out = list(R)
for g, msgs in rw.items():
    if g not in orig: dropped += 1; continue
    ok = isinstance(msgs, list) and len(msgs) == len(orig[g]["msgs"]) and all(isinstance(m, dict) and "text" in m and "speaker" in m for m in msgs)
    n = sum(bool(re.search(r"collu", m["text"], re.I)) for m in msgs) if ok else 0
    if not ok or n < 3: dropped += 1; continue
    r = dict(orig[g]); r["tid"] = g + "/collusive_word"; r["version"] = "collusive_word"; r["msgs"] = [{"speaker": m["speaker"], "text": m["text"].strip()} for m in msgs]
    out.append(r); kept += 1
with open("concept_texts4.jsonl", "w") as f:
    for r in out: f.write(json.dumps(r) + "\n")
c = collections.Counter((r["version"], r["split"]) for r in out)
print("rewrites kept", kept, "dropped", dropped, "| rows by (version, split):", dict(sorted(c.items())))
