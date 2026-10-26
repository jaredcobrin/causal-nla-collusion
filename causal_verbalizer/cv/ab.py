"""A/B choice tests: does a concept direction control the model's choice between answers?

For each concept the model's next token (the token before it answers) is compared between the
concept answer and the other answer(s). We report the chance of picking the concept answer,
p = P(concept) / (P(concept) + P(other)), as a percentage. No judge, no free generation. The direction
is built from train groups only; every test prompt comes from the val split (asserted disjoint).

  steering   add s x size x u at block L, s in -1, +0.5, +1, +2 (size = the direction's natural length)
  removal    on prompts where the model already picks the concept answer (p > 50%): remove u
  controls   5 random unit directions (same size / same displacement) and the shuffled-label direction
  position   edit only the last token (just before the answer), or every token of the prompt

Pass rules (fixed before the first run; thresholds from run.yaml):
  steering   +1x raises p by >= ab.min_effect, more than every control, CI vs the random mean above 0,
             p rises with strength (Spearman >= t1_monotone)
  removal    >= t3_present prompts start above 50%; removal lowers p by >= ab.min_effect, more than every
             control, CI above 0
"""

from __future__ import annotations

import argparse
import csv
import re

import numpy as np
import torch

from cv.bank import BANK
from cv.data import EMOTIONS, HARM_Q, user
from cv.directions import build_static
from cv.hooks import ablate, add, displace, last_only
from cv.io import RUNS, SPLITS, Log, load_cfg, load_concepts, read_json, read_jsonl, seed_for, write_json
from cv.model import LM
from cv.splits import content
from cv.stats import boot_ci, spearman

# Sources whose items can be asked as "concept answer vs other answer". Everything else
# (Persona Vectors traits, refusal, sycophantic-praise) has no such pair and is left out.
AB_SOURCES = {"caa", "persona", "airisk", "syco_split", "truth", "sentiment", "emotion", "harmfulness", "refusal", "bank"}
APPROVED = {"truth", "sentiment", "fear", "sadness", "happiness", "disgust", "surprise"}  # user-approved earlier; not re-run
MIN_COVERAGE = 0.3  # a pass only counts if the answer words hold at least this much of the next-token probability


def eligible(c: dict) -> bool:
    return c["source"] in AB_SOURCES and c.get("label") != "pr"


def _agree_pairs(ext: list[dict]) -> list[dict]:
    """Sycophancy rows -> minimal pairs: for each prompt, the forced reply 'I agree' (label 1: agreeing
    in the way this concept defines) against 'I disagree' (label 0), same prompt."""
    dis = {(r["group"], r["msgs"][0]["content"]): r for r in ext if r["prefix"].startswith("I disagree")}
    out = []
    for r in ext:
        d = dis.get((r["group"], r["msgs"][0]["content"]))
        if r["label"] == 1 and d is not None:
            out += [{**r, "prefix": "I agree", "label": 1}, {**d, "prefix": "I disagree", "label": 0}]
    return out


def _spec(lm: LM, c: dict, rows: list[dict], n_items: int, split: str = "val", recipe: str = "A"):
    """(rows to build the direction from, test texts, token ids [concept, other...], groups).
    A stem puts the answer word next; the direction is read at the stem's last token, i.e. where it is
    later edited. Recipe B (binary bank concepts) instead forces the right / wrong answer word on
    identical prompts and reads at that word, the contrast being only whether the answer is right."""
    tok = lambda s: lm.tok.encode(s, add_special_tokens=False)[0]
    ev = sorted((r for r in rows if r["role"] == "eval" and r["split"] == split), key=lambda r: r["rank"])[:n_items]
    ext, src, stem = [r for r in rows if r["role"] == "extract"], c["source"], ""
    if src in ("caa", "persona", "airisk"):  # direction data unchanged; only the test prompt gets a stem
        ans = {(r["group"], r["label"]): r["ans"] for r in ext}
        yes_no = ans[ev[0]["group"], 1] in ("Yes", "No")
        stem = "My answer is:" if yes_no else "My answer is: ("
        fix = (lambda a: tok(" " + a)) if yes_no else (lambda a: tok(a[1:]))
        ids = [[fix(ans[r["group"], 1]), fix(ans[r["group"], 0])] for r in ev]
    elif src == "syco_split":  # the model's own words: "I" then " agree" / " disagree"
        ext, stem, words = _agree_pairs(ext), "I", [" agree", " disagree"]
    elif src == "harmfulness":  # the instruction wrapped in the question the model is asked at test time
        ext = [{**r, "msgs": user(HARM_Q.format(r["msgs"][0]["content"][0].lower() + r["msgs"][0]["content"][1:])),
                "pos": "last"} for r in ext]
        words = ["Certainly", "No"]
    elif src == "refusal":  # Arditi's readout: does the reply start with "I" (I'm sorry / I cannot) or comply
        words = ["I", "Sure", "Certainly", "Here", "To", "Creating", "Yes"]
    elif src == "truth":
        stem, words = "The statement is", [" true", " false"]
    elif src == "sentiment":
        stem, words = "The sentiment of the review is", [" Positive", " Negative"]
    elif src == "bank":
        stem, words = c["stem"], c["words"]
    else:  # emotion: this emotion against the other five
        stem, words = "The emotion evoked is", [" " + c["name"], *[" " + e for e in EMOTIONS if e != c["name"]]]
    if src not in ("caa", "persona", "airisk"):
        ids = [[tok(w) for w in words]] * len(ev)
    if src in ("truth", "sentiment", "emotion"):
        ext = [{**r, "prefix": stem} for r in ext]
    if recipe == "B":
        assert src == "bank" and len(words) == 2, "recipe B needs a binary bank concept"
        forced = []
        for r in ext:
            right, wrong = (words[0], words[1]) if r["label"] == 1 else (words[1], words[0])
            forced += [{**r, "label": 1, "prefix": stem + right, "ans": right},
                       {**r, "label": 0, "prefix": stem + wrong, "ans": wrong, "id": r["id"] + "-wrong"}]
        ext = forced
    train = {r["group"] for r in ext if r["split"] == "train"}
    assert not train & {r["group"] for r in ev}, "train and test groups overlap"
    assert not {content(r) for r in ext if r["split"] == "train"} & {content(r) for r in ev}, "prompt in train and test"
    return ext, [lm.render(r["msgs"], stem) for r in ev], ids, [r["group"] for r in ev]


def _stats(d: dict, groups, th: dict, min_effect: float, n_boot: int) -> dict:
    """d: condition -> logit difference per prompt (base, s<strength>, rand<j>, shuf, rm, rand-rm<j>, shuf-rm).
    Everything is converted to p = chance of the concept answer."""
    p = {k: 1 / (1 + np.exp(-x)) for k, x in d.items()}
    base, g = p["base"], np.array(groups)
    strengths = sorted(float(k[1:]) for k in p if re.fullmatch(r"s-?[0-9.]+", k))
    rand = sorted(k for k in p if re.fullmatch(r"rand[0-9]+", k))
    rrm = sorted(k for k in p if k.startswith("rand-rm"))
    gain = lambda k: p[k] - base
    levels = {0.0: 0.0, **{s: float(gain(f"s{s}").mean()) for s in strengths}}
    r_eff = [float(gain(k).mean()) for k in rand]
    eff, shuf = levels[1.0], float(gain("shuf").mean())
    ci = boot_ci(gain("s1.0") - np.mean(r_eff), g, n_boot)
    rho = round(spearman(sorted(levels), [levels[s] for s in sorted(levels)]), 6)
    other, present = base < 0.5, base > 0.5
    flip = lambda k, m: float((p[k][m] > 0.5).mean()) if m.any() else None
    steer = dict(p_at={s: float(base.mean() + levels[s]) for s in sorted(levels)}, change=levels, effect_plus1=eff,
                 random_p_plus1=float(base.mean() + np.mean(r_eff)), random_effects=r_eff, shuffled_effect=shuf,
                 ci_vs_random=ci, spearman=rho, n_other=int(other.sum()),
                 flipped_to_concept={"+1": flip("s1.0", other), "+2": flip("s2.0", other),
                                     "random +1": float(np.mean([flip(k, other) for k in rand])) if other.any() else None})
    steer["passed"] = bool(eff >= min_effect and eff > max(r_eff) and eff > shuf and ci[0] > 0 and rho >= th["t1_monotone"])
    out = dict(steering=steer)
    if present.sum() >= th["t3_present"]:
        drop = base[present] - p["rm"][present]
        r_drop = [float((base[present] - p[k][present]).mean()) for k in rrm]
        s_drop = float((base[present] - p["shuf-rm"][present]).mean())
        ci3 = boot_ci(drop - np.mean(r_drop), g[present], n_boot)
        out["removal"] = dict(n_present=int(present.sum()), p_before=float(base[present].mean()),
                              p_after=float(p["rm"][present].mean()), drop=float(drop.mean()), random_drops=r_drop,
                              shuffled_drop=s_drop, ci_vs_random=ci3,
                              flipped_to_other={"real": float((p["rm"][present] < 0.5).mean()),
                                                "random": float(np.mean([(p[k][present] < 0.5).mean() for k in rrm]))},
                              passed=bool(drop.mean() >= min_effect and drop.mean() > max(r_drop) and drop.mean() > s_drop and ci3[0] > 0))
    else:
        out["removal"] = dict(n_present=int(present.sum()), passed=False, untestable=True)
    return out


def _build(lm: LM, cfg: dict, c: dict, split: str, recipe: str):
    """Test prompts and the direction (train split only) for one concept."""
    ab = cfg["ab"]
    path = (BANK if c["source"] == "bank" else SPLITS) / f"{c['name']}.jsonl"
    ext, texts, ids, groups = _spec(lm, c, read_jsonl(path), ab["items"], split, recipe)
    return texts, ids, groups, build_static(lm, f"ab-{c['name']}-{recipe}", ext, dict(cfg, train_start=ab["groups"]))


def run_concept(lm: LM, cfg: dict, c: dict, rand: torch.Tensor, log, split: str = "val", recipe: str = "A",
                save=None) -> dict:
    L, bs = cfg["layer"], cfg["batch"]["forward"]
    texts, ids, groups, d = _build(lm, cfg, c, split, recipe)
    gate = d["gate"][L]
    log(f"{c['name']}: direction from {d['data']['extract_rows']} rows | split-half {gate['split_half']:.3f} "
        f"auroc {gate['auroc']:.3f}" + (f" letter cos {gate['letter_cos']:.2f}" if "letter_cos" in gate else ""))
    v, shuf = d["v"][L].float().to(lm.device), d["shuffled"][L].float().to(lm.device)
    size, ssize = float(v.norm()), float(shuf.norm())
    u, su = v / size, shuf / ssize
    if save is not None:
        save.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(v=v.cpu(), shuffled=shuf.cpu()), save)

    def logits(edit):
        return lm.choice_logits(texts, ids, edits=None if edit is None else {L: edit}, bs=bs)

    base, mass = logits(None)
    p0 = 1 / (1 + np.exp(-base))
    res = dict(concept=c["name"], recipe=recipe, split=split, direction=dict(gate, **d["data"], size=size), n_prompts=len(texts),
               baseline=dict(prefers_concept_answer=float((p0 > 0.5).mean()), mean_p=float(p0.mean()),
                             coverage=float(mass.mean())))
    for pos, wrap in (("last token", last_only), ("every token", lambda e: e)):
        conds = {"base": base}
        for s in cfg["strengths"]:
            conds[f"s{s}"] = logits(wrap(add(s * size * u)))[0]
        conds["shuf"] = logits(wrap(add(ssize * su)))[0]
        for j, r in enumerate(rand):
            conds[f"rand{j}"] = logits(wrap(add(size * r)))[0]
        conds["rm"] = logits(wrap(ablate(u)))[0]
        conds["shuf-rm"] = logits(wrap(ablate(su)))[0]
        for j, r in enumerate(rand):
            conds[f"rand-rm{j}"] = logits(wrap(displace(u, r)))[0]
        res[pos] = _stats(conds, groups, cfg["thresholds"], cfg["ab"]["min_effect"], cfg["bootstrap"])
        res[pos]["raw_logit_differences"] = {k: np.round(x, 3).tolist() for k, x in conds.items()}
    # causal: steering and removal both pass at the same edit position, and the answer words are the
    # model's actual next token often enough for the readout to mean something
    res["causal"] = bool(res["baseline"]["coverage"] >= MIN_COVERAGE and
                         any(res[pos]["steering"]["passed"] and res[pos]["removal"]["passed"] for pos in ("last token", "every token")))
    return res


def stable(r: dict) -> bool:
    return r["direction"]["split_half"] >= 0.9  # the repo's direction-stability gate (run.yaml thresholds.split_half)


def _pct(x, nd=0):
    return "-" if x is None else f"{100 * x:.{nd}f}%"


def _pts(x):
    return "-" if x is None else f"{100 * x:+.0f} pts"


def write_report(run_dir, excluded: list[str]) -> None:
    rs = [read_json(p) for p in sorted((run_dir / "results").glob("*.json"))]
    ok = lambda b: "pass" if b else "fail"
    table, csv_rows = [], []
    for r in rs:
        b = r["baseline"]
        row = {"concept": r["concept"], "causal": ("YES" if stable(r) else "yes (unstable direction)") if r.get("causal") else "no", "split-half": f"{r['direction']['split_half']:.2f}", "coverage": _pct(b["coverage"]),
               "picks concept answer (no edit)": _pct(b["prefers_concept_answer"])}
        for pos in ("last token", "every token"):
            s, m = r[pos]["steering"], r[pos]["removal"]
            row[f"steer +1x, {pos}"] = f"{_pts(s['effect_plus1'])} (best random {_pts(max(s['random_effects']))}) {ok(s['passed'])}"
            row[f"removal, {pos}"] = ("untestable" if m.get("untestable") else
                                      f"{_pct(m['p_before'])} -> {_pct(m['p_after'])}, {_pct(m['flipped_to_other']['real'])} flipped {ok(m['passed'])}")
        csv_rows.append(row)
    cols = list(csv_rows[0]) if csv_rows else []
    with (run_dir / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(csv_rows)
    n_causal, n_strict = sum(bool(r.get("causal")) for r in rs), sum(bool(r.get("causal")) and stable(r) for r in rs)
    L = [f"# A/B choice tests (block 20): {n_causal} causal of {len(rs)}; {n_strict} of them with a stable direction (split-half >= 0.9)", "",
         "p = the model's chance of picking the concept answer over the other answer(s), at the token before it answers "
         "(averaged over prompts). 'Coverage' = how much of the model's own next-token probability the answer words hold; "
         "if it is low the model would usually start its reply some other way. 'pts' = percentage points. "
         "Controls: 5 random directions of the same size, and the shuffled-label direction.", "",
         f"Causal = steering and removal both pass at the same edit position, coverage >= {MIN_COVERAGE:.0%}. "
         f"Left out (no two-answer form): {', '.join(excluded) or 'none'}.", "",
         "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    L += ["| " + " | ".join(r[c] for c in cols) + " |" for r in csv_rows] + [""]
    for r in rs:
        b, g = r["baseline"], r["direction"]
        L += [f"## {r['concept']}", "",
              f"- direction: {g['extract_rows']} train rows, split-half {g['split_half']:.3f}, AUROC {g['auroc']:.3f}"
              + (f", letter cos {g['letter_cos']:.2f}" if "letter_cos" in g else ""),
              f"- {r['n_prompts']} val prompts; picks the concept answer on {_pct(b['prefers_concept_answer'])} of them; "
              f"average chance {_pct(b['mean_p'])}; answer words hold {_pct(b['coverage'])} of its next-token probability", ""]
        for pos in ("last token", "every token"):
            s, m = r[pos]["steering"], r[pos]["removal"]
            pa = s["p_at"]
            L += [f"**Edit at {pos}**", "", "| steering strength | -1x | 0 | +0.5x | +1x | +2x |", "|---|---|---|---|---|---|",
                  f"| average chance of concept answer | {_pct(pa['-1.0'])} | {_pct(pa['0.0'])} | {_pct(pa['0.5'])} | {_pct(pa['1.0'])} | {_pct(pa['2.0'])} |",
                  "", f"- +1x: {_pts(s['effect_plus1'])}; random directions {_pts(min(s['random_effects']))} to "
                  f"{_pts(max(s['random_effects']))} (mean chance {_pct(s['random_p_plus1'])}); shuffled {_pts(s['shuffled_effect'])}; "
                  f"Spearman {s['spearman']:.2f}"]
            f = s["flipped_to_concept"]
            if f["+1"] is not None:
                L.append(f"- of the {s['n_other']} prompts where it picks the other answer, now pick the concept answer: "
                         f"{_pct(f['+1'])} at +1x, {_pct(f['+2'])} at +2x (random directions at +1x: {_pct(f['random +1'])})")
            L.append(f"- **steering {ok(s['passed'])}**")
            if m.get("untestable"):
                L.append(f"- removal untestable: only {m['n_present']} prompts pick the concept answer")
            else:
                L += [f"- removal on the {m['n_present']} prompts that pick the concept answer: chance {_pct(m['p_before'])} -> "
                      f"{_pct(m['p_after'])} ({_pts(-m['drop'])}); random removals {_pts(-max(m['random_drops']))} to "
                      f"{_pts(-min(m['random_drops']))}; shuffled {_pts(-m['shuffled_drop'])}",
                      f"- prompts that now pick the other answer: {_pct(m['flipped_to_other']['real'])} "
                      f"(random removals: {_pct(m['flipped_to_other']['random'])})",
                      f"- **removal {ok(m['passed'])}**"]
            L.append("")
    (run_dir / "REPORT.md").write_text("\n".join(L) + "\n")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["existing", "bank"], default="existing")
    ap.add_argument("--out", default="ab")
    ap.add_argument("--split", default="val")
    ap.add_argument("--recipe", default="A", choices=["A", "B"])
    ap.add_argument("--names", nargs="*")
    ap.add_argument("--confirm-from", help="results dir; run only the concepts it marks causal")
    args = ap.parse_args(argv)
    cfg = load_cfg()
    pool = read_json(BANK / "concepts.json") if args.set == "bank" else load_concepts()
    todo, excluded = [c for c in pool if eligible(c)], [c["name"] for c in pool if not eligible(c)]
    todo = [c for c in todo if c["name"] not in APPROVED]
    if args.names:
        todo = [c for c in todo if c["name"] in args.names]
    if args.confirm_from:
        keep = {p.stem for p in (RUNS / args.confirm_from / "results").glob("*.json") if read_json(p).get("causal")}
        todo = [c for c in todo if c["name"] in keep]
    run_dir = RUNS / args.out
    log = Log(run_dir)
    lm = LM(cfg["model"])
    g = torch.Generator().manual_seed(seed_for("random-directions"))
    rand = torch.nn.functional.normalize(torch.randn(cfg["n_random"], lm.d, generator=g), dim=1).to(lm.device)
    log(f"start {args.out}: {len(todo)} concepts ({args.set}), recipe {args.recipe}, {args.split} split, layer {cfg['layer']}, "
        f"{cfg['ab']['groups']} train groups, {cfg['ab']['items']} prompts")
    for c in todo:
        out, dirf = run_dir / "results" / f"{c['name']}.json", run_dir / "directions" / f"{c['name']}.pt"
        if out.exists():
            if not dirf.exists():  # results from before directions were saved: rebuild the direction only
                d = _build(lm, cfg, c, args.split, args.recipe)[3]
                dirf.parent.mkdir(parents=True, exist_ok=True)
                torch.save(dict(v=d["v"][cfg["layer"]].cpu(), shuffled=d["shuffled"][cfg["layer"]].cpu()), dirf)
            continue
        try:
            try:
                res = run_concept(lm, cfg, c, rand, log, args.split, args.recipe, save=dirf)
            except torch.cuda.OutOfMemoryError:  # long texts: retry once with a small batch
                torch.cuda.empty_cache()
                res = run_concept(lm, dict(cfg, batch={**cfg["batch"], "forward": 4}), c, rand, log, args.split,
                                  args.recipe, save=dirf)
        except Exception as e:  # one broken concept must not stop the queue
            log(f"{c['name']}: ERROR {type(e).__name__}: {str(e)[:150]}")
            continue
        write_json(out, res)
        s, m = res["last token"], res["every token"]
        log(f"{c['name']}: {'CAUSAL' if res['causal'] else 'no'} | coverage {_pct(res['baseline']['coverage'])} | steering last/every "
            f"{ok_(s['steering'])}/{ok_(m['steering'])} | removal last/every {ok_(s['removal'])}/{ok_(m['removal'])}")
        try:
            write_report(run_dir, excluded)
        except Exception as e:  # the report is rebuilt at the end; never lose a run over it
            log(f"report error {type(e).__name__}: {e}")
    write_report(run_dir, excluded)
    log("done")


def ok_(x):
    return "pass" if x["passed"] else "fail"


if __name__ == "__main__":
    main()
