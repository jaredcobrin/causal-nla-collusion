"""Validate the Haiku-written scenarios and build concept_texts.jsonl (one row per transcript version).
   python prepare.py  -> concept_texts.jsonl, prepare_report.txt
label 1 = collusive; label 0 = independent / open. Splits by scenario (md5): test 25%, val 15%, train 60%."""
import json, glob, hashlib, re, collections
rows = []; bad = []
for f in sorted(glob.glob("gen/gen_*.jsonl")):
    for ln, l in enumerate(open(f)):
        try: s = json.loads(l)
        except Exception as e: bad.append((f, ln, "json")); continue
        ok = all(k in s and isinstance(s[k], list) and 6 <= len(s[k]) <= 12 for k in ("collusive", "independent", "open"))
        ok = ok and all(isinstance(m, dict) and "speaker" in m and "text" in m for k in ("collusive", "independent", "open") for m in s.get(k, []))
        if not ok: bad.append((f, ln, "shape")); continue
        txt = " ".join(m["text"] for k in ("collusive", "independent", "open") for m in s[k]).lower()
        if re.search(r"collu|conspir|cartel", txt): bad.append((f, ln, "banned word")); continue
        rows.append(s)
ids = [s["id"] for s in rows]; assert len(ids) == len(set(ids)), "duplicate ids"
out = []
for s in rows:
    h = int(hashlib.md5(s["id"].encode()).hexdigest(), 16) % 100
    split = "test" if h < 25 else "val" if h < 40 else "train"
    for k, lab in (("collusive", 1), ("independent", 0), ("open", 0)):
        text = "\n".join(f"{m['speaker']}: {m['text'].strip()}" for m in s[k])
        out.append(dict(tid=f"{s['id']}/{k}", group=s["id"], theme=s["theme"], version=k, label=lab, split=split, text=text))
with open("concept_texts.jsonl", "w") as f:
    for r in out: f.write(json.dumps(r) + "\n")
c = collections.Counter((r["split"], r["version"]) for r in out)
words = [len(r["text"].split()) for r in out]
rep = [f"scenarios kept {len(rows)}; dropped {len(bad)}: {collections.Counter(b[2] for b in bad)}", f"rows {len(out)}; by split/version {dict(sorted(c.items()))}",
       f"words per transcript: mean {sum(words)/len(words):.0f} min {min(words)} max {max(words)}", f"themes {len({s['theme'] for s in rows})}"]
open("prepare_report.txt", "w").write("\n".join(rep)); print("\n".join(rep))
