"""summary.csv (every number) and REPORT.md (the scores per concept), rebuilt after every concept."""

from __future__ import annotations

import csv
from collections import Counter

from cv import judge
from cv.io import read_json


def _f(x, nd=1):
    return "" if x is None else f"{x:.{nd}f}"


def _row(r: dict) -> dict:
    d = r["direction"]
    out = {"concept": r["concept"], "group": r["group"], "verdict": r["verdict"], "failed checks": "; ".join(r.get("failed", [])),
           "split-half": _f(d.get("split_half"), 3), "auroc": _f(d.get("auroc"), 3), "letter cos": _f(d.get("letter_cos"), 2),
           "rows used": d.get("extract_rows", d.get("kept_pairs")), "replies": r["usage"]["generated"], "minutes": _f(r["usage"]["seconds"] / 60)}
    if "t1" in r:
        t1, t2, t3, t4 = r["t1"], r["t2"], r["t3"], r["t4"]
        lv = t1["mean_score"]
        out.update({
            "score at -1 / 0 / +0.5 / +1 / +2": " / ".join(_f(lv[k]) for k in ("-1.0", "0.0", "0.5", "1.0", "2.0")),
            "T1 effect at +1 [CI vs random]": f"{t1['effect_plus1']:+.1f} [{t1['ci_vs_random'][0]:+.1f}, {t1['ci_vs_random'][1]:+.1f}]",
            "T1 best random": _f(max(t1["random_effects"]), 1), "T1 spearman": _f(t1["spearman"], 2),
            "T2 coherence -1/+1 (base)": f"{t2['coherence']['-1.0']:.0f}/{t2['coherence']['1.0']:.0f} ({t2['base_coherence']:.0f})",
            "T2 max spillover": (max(t2["spillover"].items(), key=lambda x: x[1]) if t2["spillover"] else ("-", 0)),
            "T2 neutral loss change": "/".join(_f(v, 2) for v in t2["neutral_loss_change"].values()),
            "T3 replies showing it": t3["n_present"],
            "T3 score present -> removed (absent)": (f"{t3['score_present']:.0f} -> {t3['score_removed']:.0f} ({_f(t3['score_absent'], 0)})"
                                                     if t3.get("testable") else ""),
            "T3 gap closed [CI]": f"{t3['gap_closed']:.2f} [{t3['ci_gap_closed'][0]:.2f}, {t3['ci_gap_closed'][1]:.2f}]" if t3.get("testable") else "",
            "T3 best random drop": _f(max(t3["random_drops"]), 1) if t3.get("testable") else "",
            "T4 change (best random)": f"{t4['change']:.3f} ({max(t4['random_changes']):.3f})"})
        who = {"qwen": "Qwen2.5-7B-Instruct (local, unsteered)", "haiku": "Claude Haiku 5.5",
               "rule": "first-word rule, no model ('Certainly' = 100, 'No' = 0)"}
        out["scored by"] = who[r.get("judge", "haiku")]
        q = [o for o, w in t2.get("spillover_judges", {}).items() if w == "qwen"]
        if q:
            out["spillover scored by Qwen on"] = ", ".join(q)
        name, ratio = out["T2 max spillover"]
        out["T2 max spillover"] = f"{ratio:.2f} ({name})" if name != "-" else "-"
    return out


def write(run_dir, phase: str, expected: int) -> None:
    results = [read_json(p) for p in sorted((run_dir / "results").glob("*.json"))]
    rows = [_row(r) for r in results]
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with (run_dir / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(rows)
    counts = Counter(r["verdict"] for r in results if "#" not in r["concept"])
    rel = run_dir / "judge_reliability.json"
    lines = [f"# {phase} run (layer {results[0]['layer'] if results else '?'}, {results[0]['split'] if results else '?'} split)", "",
             f"Done {len(results)} of {expected}. Judge batches waiting: {len(judge.pending_batches())}.",
             "Verdicts: " + ", ".join(f"{k} {v}" for k, v in counts.most_common()), ""]
    if rel.exists():
        j = read_json(rel)
        lines += [f"Judge re-score control: correlation {j['correlation']:.2f}, mean |difference| {j['mean_abs_diff']:.1f} points, "
                  f"same side of 50 {100 * j['same_side_of_50']:.0f}% (n={j['n']}).", ""]
    lines += ["Scores are the judge's 0-100 rating of how clearly the reply shows the concept (averaged over 40 replies).",
              "Replies to prompts from the refusal and harmfulness datasets are scored by the local, unsteered "
              "Qwen2.5-7B-Instruct (100 x P(Yes) to a Yes/No question), because Claude Haiku's safety filter rejects "
              "them; harmfulness replies are one word, so a first-word rule scores them exactly. Each result below says "
              "which judge scored it ('scored by').", ""]
    for r in rows:
        lines += [f"## {r['concept']} — **{r['verdict']}**", ""]
        lines += [f"- {k}: {v}" for k, v in r.items() if k not in ("concept", "verdict") and v not in ("", None)]
        lines.append("")
    (run_dir / "REPORT.md").write_text("\n".join(lines) + "\n")
