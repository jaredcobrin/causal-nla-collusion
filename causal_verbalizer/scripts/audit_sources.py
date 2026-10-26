"""Audit the general-concept taxonomy (configs/general.yaml) against the real datasets.

For every dataset in the registry: does it exist, how many rows in the chosen split, and how many rows
per class (Hugging Face datasets-server statistics; if the class column is free text with many values,
a spread-out sample of 600 rows is used instead and the counts are scaled, marked ~). Then for every
concept: how many independent sources actually matched, and how many positives it has.

    python scripts/audit_sources.py            # writes runs/overnight/source_audit.json and prints the table
"""

from __future__ import annotations

import collections
import fnmatch
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
API = "https://datasets-server.huggingface.co"
MIN_POS = {"GREEN": 1500, "AMBER": 600}   # estimated positives; GREEN also needs >= 3 sources, AMBER >= 2


def get(path: str) -> dict:
    try:
        with urllib.request.urlopen(f"{API}/{path}", timeout=90) as r:
            return json.load(r)
    except Exception as e:  # noqa: BLE001 - report, never crash the audit
        return {"error": str(e)[:120]}


def q(**kw) -> str:
    return urllib.parse.urlencode(kw)


def class_counts(ds: str, cfg: str, split: str, col: str, n_rows: int) -> tuple[dict | None, bool, str]:
    """({class name: rows}, approximate?, note)."""
    s = get("statistics?" + q(dataset=ds, config=cfg, split=split))
    for c in s.get("statistics", []):
        if c["column_name"] == col and c["column_type"] in ("class_label", "string_label"):
            return {str(k): int(v) for k, v in c["column_statistics"]["frequencies"].items()}, False, ""
        if c["column_name"] == col and c["column_type"] == "bool":
            f = c["column_statistics"]
            t = round(f.get("true_proportion", 0) * n_rows) if "true_proportion" in f else None
            if t is not None:
                return {"1": t, "0": n_rows - t, "true": t, "false": n_rows - t}, False, ""
    # free-text class column: sample rows spread over the split
    seen, step = collections.Counter(), max(n_rows // 6, 1)
    total = 0
    for off in range(0, n_rows, step):
        r = get("rows?" + q(dataset=ds, config=cfg, split=split, offset=off, length=100))
        for row in r.get("rows", []):
            v = row["row"].get(col)
            if isinstance(v, (str, int, float, bool)):
                seen[str(v)] += 1
                total += 1
    if not total:
        return None, False, "no class counts available"
    return {k: round(v * n_rows / total) for k, v in seen.items()}, True, "sampled"


def audit_registry(reg: dict) -> dict:
    out = {}
    for key, spec in reg.items():
        ds, split = spec["hf"], spec["split"]
        cfg = spec.get("config")
        sp = get("splits?" + q(dataset=ds))
        if "splits" not in sp:
            out[key] = dict(ok=False, note=sp.get("error", "not found"))
            continue
        configs = sorted({s["config"] for s in sp["splits"]})
        cfg = cfg or ("default" if "default" in configs else configs[0])
        size = get("size?" + q(dataset=ds, config=cfg))
        n = None
        for s in size.get("size", {}).get("splits", []):
            if s["split"] == split and s["config"] == cfg:
                n = s["num_rows"]
        if n is None:
            have = sorted({s["split"] for s in sp["splits"] if s["config"] == cfg})
            out[key] = dict(ok=False, note=f"split {split!r} not in {have}")
            continue
        counts, approx, note = class_counts(ds, cfg, split, spec["col"], n)
        out[key] = dict(ok=True, rows=n, config=cfg, classes=counts, approx=approx, note=note)
        print(f"  {key:10s} {ds:50s} {n:>9,} rows  {len(counts) if counts else '?':>4} classes {'~' if approx else ''}{note}", flush=True)
    return out


def match(src: dict, pattern: str) -> tuple[int, bool]:
    """Rows of source `src` whose class matches `pattern` (glob), and whether any class matched."""
    if pattern == "*":
        return src["rows"], True
    if not src.get("classes"):
        return 0, False
    hit = [v for k, v in src["classes"].items() if fnmatch.fnmatchcase(k.lower(), pattern.lower())]
    return sum(hit), bool(hit)


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "general.yaml").read_text())
    print(f"auditing {len(cfg['registry'])} datasets ...", flush=True)
    reg = audit_registry(cfg["registry"])
    rows = []
    for name, c in cfg["concepts"].items():
        per_src, missing = collections.defaultdict(int), []
        for item in c["pos"]:
            key, _, pat = item.partition(":")
            src = reg.get(key)
            if not src or not src["ok"]:
                missing.append(f"{item} (dataset unavailable)")
                continue
            n, found = match(src, pat)
            if found:
                per_src[key] += n
            else:
                missing.append(f"{item} (class not found)")
        total = sum(per_src.values())
        if c.get("gen"):
            flag = "GEN"
        elif len(per_src) >= 3 and total >= MIN_POS["GREEN"]:
            flag = "GREEN"
        elif len(per_src) >= 2 and total >= MIN_POS["AMBER"]:
            flag = "AMBER"
        else:
            flag = "RED"
        rows.append(dict(concept=name, group=c["group"], flag=flag, sources=len(per_src), positives=total,
                         per_source=dict(per_src), problems=missing))
    (ROOT / "runs" / "overnight").mkdir(parents=True, exist_ok=True)
    (ROOT / "runs" / "overnight" / "source_audit.json").write_text(json.dumps(dict(registry=reg, concepts=rows), indent=1))
    by = collections.Counter(r["flag"] for r in rows)
    print("\nconcept".ljust(26), "grp".ljust(8), "flag".ljust(6), "src", "positives", " problems")
    for r in sorted(rows, key=lambda r: (r["group"], r["flag"], -r["positives"])):
        print(f"{r['concept']:25s} {r['group']:8s} {r['flag']:6s} {r['sources']:3d} {r['positives']:9,d}  {'; '.join(r['problems'])[:90]}")
    print("\nflags:", dict(by), "| total concepts:", len(rows))


if __name__ == "__main__":
    sys.exit(main())
