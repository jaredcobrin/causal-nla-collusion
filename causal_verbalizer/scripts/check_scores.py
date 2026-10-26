"""Validate judge score files against their queue batches (ids, ranges); list batches still unscored."""
import json
import sys
from pathlib import Path

J = Path(__file__).resolve().parents[1] / "runs" / "judge"
bad = 0
for q in sorted((J / "queue").glob("*.json")):
    s = J / "scores" / f"{q.stem}.jsonl"
    if not s.exists():
        print("unscored", q.stem)
        continue
    ids = {i["id"] for i in json.loads(q.read_text())["items"]}
    rows = [json.loads(l) for l in s.read_text().splitlines() if l.strip()]
    val = lambda r: r.get("score", r.get("trait"))  # the judge agent writes "score"; older files used "trait"
    ok = ({r["id"] for r in rows} == ids and len(rows) == len(ids) and all(
        val(r) is not None and (val(r) == "REFUSAL" or 0 <= val(r) <= 100) and 0 <= r["coherence"] <= 100 for r in rows))
    bad += not ok
    print("ok " if ok else "BAD", q.stem, len(rows))
sys.exit(bad)
