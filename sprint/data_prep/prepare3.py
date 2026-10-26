"""v1 + v2 scenarios -> concept_texts3.jsonl (one row per transcript version, with per-message structure kept).
Splits by scenario md5 (same rule as v1): test 25%, val 15%, train 60%. v2 formats have 2-9 messages."""
import json, glob, hashlib, re, collections
rows = []; bad = collections.Counter()
for f in sorted(glob.glob("gen3/g3_*.jsonl")):
    for ln, l in enumerate(open(f)):
        try: s = json.loads(l)
        except Exception: bad["json"] += 1; continue
        ok = all(k in s and isinstance(s[k], list) and 2 <= len(s[k]) <= 14 and all(isinstance(m, dict) and "speaker" in m and "text" in m for m in s[k]) for k in ("collusive", "independent", "open"))
        if not ok: bad["shape"] += 1; continue
        txt = " ".join(m["text"] for k in ("collusive", "independent", "open") for m in s[k]).lower()
        if re.search(r"collu|conspir|cartel", txt): bad["banned"] += 1; continue
        s["_src"] = "blatant"; rows.append(s)
ids = [s["id"] for s in rows]; assert len(ids) == len(set(ids)), "dup ids"
out = []
for s in rows:
    h = int(hashlib.md5(s["id"].encode()).hexdigest(), 16) % 100
    split = "test" if h < 25 else "val" if h < 40 else "train"
    for k, lab in (("collusive", 1), ("independent", 0), ("open", 0)):
        out.append(dict(tid=f"{s['id']}/{k}", group=s["id"], src=s["_src"], theme=s["theme"], fmt=s.get("format", -1), version=k, label=lab, split=split,
                        msgs=[{"speaker": m["speaker"], "text": m["text"].strip()} for m in s[k]]))
with open("concept_texts3.jsonl", "w") as f:
    for r in out: f.write(json.dumps(r) + "\n")
c = collections.Counter((r["src"], r["split"]) for r in out if r["version"] == "collusive")
print("kept scenarios", len(rows), "dropped", dict(bad), "| scenarios by (source, split):", dict(sorted(c.items())))
print("themes", len({s['theme'] for s in rows}), "| formats", dict(collections.Counter(r['fmt'] for r in out if r['version']=='collusive')))
print("messages per transcript: mean %.1f" % (sum(len(r['msgs']) for r in out) / len(out)))
