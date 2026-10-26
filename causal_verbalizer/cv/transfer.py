"""Transfer test: do the directions still control the model on scenarios unlike the ones they were built on?

The A/B tests used the multiple-choice prompts the directions were extracted from. Here the model answers
free-form chats (configs/transfer.yaml: system prompts, multi-turn, other languages, many tasks), greedy,
and a blind Claude Haiku judge scores each reply 0-100 for the concept and for coherence.

  steering   add s x size x u (s in run.yaml transfer.strengths) on prompts whose unsteered reply does NOT show it
  removal    remove u at block L on prompts whose unsteered reply DOES show it
  controls   2 random unit directions (same size) for steering; the same amount moved along them for removal
  where      "last" = last prompt token only; "pred" = last prompt token + every generated token (every
             position that predicts a token); "all" = every position, prompt and generated
  bleed      this direction's +2 on the other concepts' prompts, scored by *their* judge target
  logits     probability of concept / opposite first-reply words, and the top-3 first tokens

    python -m cv.transfer gen   [--concepts a b]    GPU: replies + first-token probabilities -> runs/transfer/gens
    python -m cv.transfer score [--concepts a b]    local: queue judge batches, then the report
"""

from __future__ import annotations

import argparse
from itertools import cycle

import numpy as np
import torch
import yaml

from cv import judge
from cv.ab import _build
from cv.bank import BANK
from cv.hooks import ablate, add, displace, last_only
from cv.io import ROOT, RUNS, SPLITS, Log, h, load_cfg, load_concepts, read_json, read_jsonl, seed_for, write_json, write_jsonl
from cv.model import LM
from cv.stats import boot_ci

OUT = RUNS / "transfer"
SAVED = ("existing_test", "bank_test", "existing_val", "bank_val")  # where the A/B runs saved directions


def _suffix(variant: str | None) -> str:
    return f"@{variant}" if variant else ""


def scenarios() -> dict:
    return yaml.safe_load((ROOT / "configs" / "transfer.yaml").read_text())


def msgs(p: dict) -> list[dict]:
    system = [{"role": "system", "content": p["sys"]}] if "sys" in p else []
    return system + [{"role": r, "content": t} for r, t in zip(cycle(("user", "assistant")), p["turns"])]


def question(p: dict) -> str:
    """What the judge sees as the request: the whole conversation."""
    return "\n".join(f"[{m['role']}] {m['content']}" for m in msgs(p))


def at(edit, mode: str):
    """Restrict an edit to where `mode` says. HF generation runs one prefill pass over the prompt
    (hidden states [B, T, d]) and then one single-token pass per generated token ([B, 1, d])."""
    if mode == "all":
        return edit
    last = last_only(edit)  # prefill: final prompt position; decode: the one new token
    if mode == "pred":
        return last
    return lambda x: x if x.shape[1] == 1 else last(x)  # "last": the final prompt position, once


def conditions(u: torch.Tensor, size: float, rand: torch.Tensor, tc: dict, layer: int) -> dict:
    """condition name -> {block: edit}, for one direction (u unit, size = its natural length); every edit is
    at block `layer`, the block the direction was built at."""
    c = {}
    for mode in tc["modes"]:
        for s in tc["strengths"]:
            c[f"s{s:+g}-{mode}"] = {layer: at(add(s * size * u), mode)}
        c[f"remove-{mode}"] = {layer: at(ablate(u), mode)}
        for j, r in enumerate(rand):
            for s in tc["random_at"]:
                c[f"rand{j}-s{s:+g}-{mode}"] = {layer: at(add(s * size * r), mode)}
            c[f"rand{j}-remove-{mode}"] = {layer: at(displace(u, r), mode)}
    return c


def direction(lm: LM, cfg: dict, name: str, source: str) -> tuple[torch.Tensor, str]:
    """The direction the A/B run validated: the saved vector if there is one, else rebuilt the same way
    (recipe A, train groups only)."""
    for d in SAVED:
        p = RUNS / d / "directions" / f"{name}.pt"
        if p.exists():
            return torch.load(p, weights_only=True)["v"].float(), f"saved ({d})"
    p = OUT / "directions" / f"{name}.pt"
    if p.exists():
        return torch.load(p, weights_only=True)["v"].float(), "rebuilt"
    pool = read_json(BANK / "concepts.json") if source == "bank" else load_concepts()
    c = next(c for c in pool if c["name"] == name)
    v = _build(lm, cfg, c, "val", "A")[3]["v"][cfg["layer"]].float()
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(v=v.cpu()), p)
    return v, "rebuilt"


def stability(pos: np.ndarray, neg: np.ndarray, grid: list[int], splits: int = 20, seed: int = 0) -> list[dict]:
    """How the mean-difference direction settles as texts are added. For each n: its cosine with the
    full-sample direction, and the cosine between directions built from two disjoint random halves
    (mean over `splits` random splits) - the repo's split-half stability, at several sample sizes."""
    unit = lambda x: x / np.linalg.norm(x)
    full = unit(pos.mean(0) - neg.mean(0))
    rng = np.random.default_rng(seed)
    out = []
    for n in grid:
        d = unit(pos[:n].mean(0) - neg[:n].mean(0))
        halves = []
        for _ in range(splits):
            ip, ineg = rng.permutation(n), rng.permutation(n)
            a = pos[ip[: n // 2]].mean(0) - neg[ineg[: n // 2]].mean(0)
            b = pos[ip[n // 2:]].mean(0) - neg[ineg[n // 2:]].mean(0)
            halves.append(float(unit(a) @ unit(b)))
        out.append(dict(n_per_class=n, cos_with_full=float(d @ full), split_half=float(np.mean(halves)),
                        split_half_sd=float(np.std(halves))))
    return out


def plain_direction(lm: LM, cfg: dict, spec: dict, how: str) -> tuple[torch.Tensor, dict]:
    """The concept as an idea in plain text: the same kind of texts as the labelled bank, fed raw (no
    question, no answer slot), mean(concept) - mean(other) at the direction's block. how = "last"
    (final token) or "mean" (average over the text's tokens, first token excluded: its activations are
    huge). Texts: `spec['n']` per class from the dataset's train split, which the bank (test split) never uses."""
    from datasets import load_dataset
    path = OUT / "directions" / f"plain-{how}.pt"
    if path.exists():
        d = torch.load(path, weights_only=True)
        return d["v"].float(), d["curve"]
    ds = load_dataset(spec["hf"], split=spec["split"])
    lab, rng, n = np.array(ds["label"]), np.random.default_rng(0), spec["n"]
    others = [c for c in sorted(set(lab.tolist())) if c != spec["label"]]
    pos = rng.choice(np.flatnonzero(lab == spec["label"]), n, replace=False)
    neg = np.concatenate([rng.choice(np.flatnonzero(lab == c), n // len(others) + (i < n % len(others)), replace=False)
                          for i, c in enumerate(others)])
    clip = lambda t: " ".join(t.split()[: spec["max_words"]])
    texts = [clip(ds[int(i)]["text"]) for i in [*pos, *neg]]
    acts = []
    for i in range(0, len(texts), 32):
        enc = lm.tok(texts[i:i + 32], return_tensors="pt", padding=True, add_special_tokens=False).to(lm.device)
        with torch.no_grad():
            h = lm.model.model(**enc, output_hidden_states=True).hidden_states[cfg["layer"] + 1].float()  # output of block L
        if how == "last":
            acts.append(h[:, -1].cpu())
        else:
            from cv.general import pooled
            acts.append(pooled(h, enc.attention_mask).cpu())
    a = torch.cat(acts).numpy()
    ap, an = a[:n], a[n:]
    grid = [g for g in (25, 50, 100, 250, 500, 1000, 2000) if g <= n] or [n]
    curve = stability(ap, an, grid)
    v = torch.tensor(ap.mean(0) - an.mean(0))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(v=v, curve=curve), path)
    return v.float(), curve


# ---- GPU stage ---------------------------------------------------------------------------------------
def gen(names: list[str], only: list[str] | None = None, modes: list[str] | None = None,
        variant: str | None = None, strengths: list[float] | None = None, random_at: list[float] | None = None) -> None:
    """only: run just the unsteered base and conditions whose name contains one of these words.
    A runs/transfer/select/<name>.json (see `select`) restricts those conditions to its prompts.
    variant "plain-last" / "plain-mean": use the plain-text idea direction instead of the bank's."""
    cfg, sc = load_cfg(), scenarios()
    over = {k: v for k, v in dict(modes=modes, strengths=strengths, random_at=random_at).items() if v}
    tc, L = dict(cfg["transfer"], **over), cfg["layer"]
    log = Log(OUT)
    lm = LM(cfg["model"])
    g = torch.Generator().manual_seed(seed_for("transfer-random"))
    rand = torch.nn.functional.normalize(torch.randn(tc["n_random"], lm.d, generator=g), dim=1).to(lm.device)
    texts = {n: [lm.render(msgs(p)) for p in sc[n]["prompts"]] for n in sc}
    for name in names:
        path = OUT / "gens" / f"{name}{_suffix(variant)}.jsonl"
        rows = read_jsonl(path) if path.exists() else []
        done = {(r["concept"], r["cond"]) for r in rows}
        if variant:
            v, curve = plain_direction(lm, cfg, sc[name]["plain"], variant.removeprefix("plain-"))
            how = "plain-text idea direction; split-half by sample size " + ", ".join(
                f"{c['n_per_class']}: {c['split_half']:.2f}" for c in curve)
        else:
            v, how = direction(lm, cfg, sc[name].get("direction", name), sc[name]["source"])
        v = v.to(lm.device)
        size = float(v.norm())
        u = v / size
        log(f"{name}: direction {how}, size {size:.2f}, cos with random {float(u @ rand[0]):+.3f}")
        toks = {w: lm.tok.encode(w, add_special_tokens=False)[0] for w in sc[name]["up"] + sc[name]["down"]}
        ids = sorted(set(toks.values()))
        up = [ids.index(toks[w]) for w in sc[name]["up"]]
        down = [ids.index(toks[w]) for w in sc[name]["down"]]
        jobs = [(name, "base", None)] + [(name, c, e) for c, e in conditions(u, size, rand, tc, L).items()]
        for o in sc:  # bleed: this direction on the other concepts' prompts, next to their unsteered replies
            jobs += [] if variant or "direction" in sc[o] or "direction" in sc[name] else [(o, "base", None), (o, f"bleed-{name}", {L: at(add(tc["bleed"] * size * u), "all")})] if o != name else []
        picked = read_json(OUT / "select" / f"{name}.json") if (OUT / "select" / f"{name}.json").exists() else None
        for concept, cond, edit in jobs:
            if (concept, cond) in done or (only and cond != "base" and not any(k in cond for k in only)):
                continue
            pids = list(range(len(texts[concept]))) if cond == "base" or picked is None or concept != name else picked
            ed = edit
            tx = [texts[concept][i] for i in pids]
            out = lm.generate(tx, edits=ed, max_new_tokens=tc["max_new_tokens"], temperature=0,
                              seed=seed_for("transfer", concept, cond), bs=cfg["batch"]["generate"])
            p, tops = lm.next_token(tx, ids, edits=ed, bs=cfg["batch"]["forward"])
            rows += [dict(concept=concept, pid=i, cond=cond, reply=a, p_up=float(p[k, up].sum()),
                          p_down=float(p[k, down].sum()), top=tops[k]) for k, (i, a) in enumerate(zip(pids, out))]
            write_jsonl(path, rows)  # resumable after every condition
            log(f"{name}: {concept} / {cond} done")
    log("gen finished")


# ---- local stage: judge queue, then statistics --------------------------------------------------------
def _meta(sc: dict) -> dict:
    m = read_json(SPLITS / "sentiment.meta.json")  # the shared judge templates ({question}, {answer}, {target})
    return dict(rubric=m["rubric"], coherence_rubric=m["coherence_rubric"])


def _key(concept: str, pid: int, reply: str) -> str:
    return f"transfer|{concept}|{pid}|{h(reply)}"  # identical replies to one prompt are scored once


def score(names: list[str], variant: str | None = None) -> None:
    sc, suf = scenarios(), _suffix(variant)
    recs = {n: read_jsonl(OUT / "gens" / f"{n}{suf}.jsonl") for n in names}
    items = {}
    for rows in recs.values():
        for r in rows:
            p = sc[r["concept"]]["prompts"][r["pid"]]
            items.setdefault(_key(r["concept"], r["pid"], r["reply"]),
                             dict(key=_key(r["concept"], r["pid"], r["reply"]), question=question(p),
                                  answer=r["reply"], target=sc[r["concept"]]["target"]))
    got = judge.scores(list(items.values()), _meta(sc), seed=0)
    if got is None:
        waiting = judge.pending_batches()
        print(f"{len(items)} replies to judge; {len(waiting)} batches waiting in runs/judge/queue -> score them, rerun")
        return
    for n in names:
        res = analyze(n, recs[n], got, sc)
        write_json(OUT / "results" / f"{n}{suf}.json", res)
        (OUT / "results" / f"{n}{suf}.md").write_text(report(n, res, sc, variant))
        write_replies(n, recs[n], got, sc, suf)
    print("scored:", ", ".join(names))


def _arr(rows, got, concept: str, cond: str, j: int, n: int) -> np.ndarray:
    by = {r["pid"]: r for r in rows if r["concept"] == concept and r["cond"] == cond}
    out = np.full(n, np.nan)
    for i, r in by.items():
        v = got[_key(concept, i, r["reply"])][j]
        out[i] = np.nan if v is None else v
    return out


def _eff(x, base, mask) -> np.ndarray:
    d = (x - base)[mask]
    return d[~np.isnan(d)]


def _has(rows, concept: str, cond: str) -> bool:
    return any(r["concept"] == concept and r["cond"] == cond for r in rows)


def analyze(name: str, rows: list[dict], got: dict, sc: dict) -> dict:
    tc = load_cfg()["transfer"]
    n = len(sc[name]["prompts"])
    S = lambda cond, concept=name: _arr(rows, got, concept, cond, 0, n)
    C = lambda cond, concept=name: _arr(rows, got, concept, cond, 1, n)
    base = S("base")
    absent, present = base < 50, base > 50
    pids = np.arange(n)
    res = dict(concept=name, n_prompts=n, n_absent=int(absent.sum()), n_present=int(present.sum()),
               base_mean_score=float(np.nanmean(base)), base_coherence=float(np.nanmean(C("base"))),
               unscored=int(np.isnan(base).sum()), modes={})
    for mode in tc["modes"]:
        m = {"steer": {}}
        for s in tc["strengths"]:
            if not _has(rows, name, f"s{s:+g}-{mode}"):
                continue
            x = S(f"s{s:+g}-{mode}")
            d = x - base
            ok = absent & ~np.isnan(d)
            e = dict(mean_on_absent=float(np.nanmean(x[absent])) if absent.any() else None, n=int(ok.sum()),
                     effect=float(d[ok].mean()) if ok.any() else None,
                     crossed_50=float((x[ok] > 50).mean()) if ok.any() else None,
                     coherence=float(np.nanmean(C(f"s{s:+g}-{mode}"))))
            rc = [f"rand{j}-s{s:+g}-{mode}" for j in range(tc["n_random"])]
            if all(_has(rows, name, c) for c in rc):
                rd = np.nanmean([S(c) - base for c in rc], axis=0)
                ok2 = ok & ~np.isnan(rd)
                e.update(random_effect=float(rd[ok2].mean()) if ok2.any() else None,
                         random_coherence=float(np.nanmean([C(c) for c in rc])),
                         ci_vs_random=boot_ci(d[ok2] - rd[ok2], pids[ok2], 2000) if ok2.sum() > 1 else None)
            m["steer"][f"{s:+g}"] = e
        for tag, cond in (("removal", f"remove-{mode}"),):
            if not _has(rows, name, cond):
                continue
            x = S(cond)
            drop = base - x
            ok = present & ~np.isnan(drop)
            e = dict(n=int(ok.sum()), score_before=float(base[ok].mean()) if ok.any() else None,
                     score_after=float(x[ok].mean()) if ok.any() else None,
                     drop=float(drop[ok].mean()) if ok.any() else None,
                     crossed_below_50=float((x[ok] < 50).mean()) if ok.any() else None,
                     coherence=float(np.nanmean(C(cond))))
            rc = [cond.replace("remove", f"rand{j}-remove", 1) for j in range(tc["n_random"])]
            if all(_has(rows, name, c) for c in rc):
                rdrop = np.nanmean([base - S(c) for c in rc], axis=0)
                ok2 = ok & ~np.isnan(rdrop)
                e.update(random_drop=float(rdrop[ok2].mean()) if ok2.any() else None,
                         ci_vs_random=boot_ci(drop[ok2] - rdrop[ok2], pids[ok2], 2000) if ok2.sum() > 1 else None)
            m[tag] = e
        res["modes"][mode] = m
    bleed = {}
    for o in sc:
        if o == name or not _has(rows, o, f"bleed-{name}"):
            continue
        ob = S("base", o)
        d = S(f"bleed-{name}", o) - ob
        ok = (ob < 50) & ~np.isnan(d)
        bleed[o] = dict(n=int(ok.sum()), effect_on_other_concept=float(d[ok].mean()) if ok.any() else None,
                        coherence=float(np.nanmean(C(f"bleed-{name}", o))))
    res["bleed"] = bleed
    mean_of = lambda c, k: float(np.mean([r[k] for r in rows if r["concept"] == name and r["cond"] == c]))
    res["first_token"] = {c: dict(p_up=mean_of(c, "p_up"), p_down=mean_of(c, "p_down"))
                          for c in sorted({r["cond"] for r in rows if r["concept"] == name})}
    return res


def _f(x, pct=False, signed=False) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "n/a"
    return f"{x:+.0f}" if signed else (f"{100 * x:.0f}%" if pct else f"{x:.0f}")


def report(name: str, r: dict, sc: dict, variant: str | None = None) -> str:
    tc = load_cfg()["transfer"]
    L = [f"# {name}: transfer test" + (f" ({variant} direction)" if variant else ""), "",
         f"Judge target: the reply {sc[name]['target']}.",
         f"Prompts: {r['n_prompts']} ({r['n_absent']} where the unsteered reply does not show it, {r['n_present']} where it does, "
         f"{r['unscored']} unscored). Unsteered mean score {_f(r['base_mean_score'])}, coherence {_f(r['base_coherence'])}.", "",
         "Scores are 0-100. Steering effect = change in the concept score on the prompts that did not show it; "
         "removal = drop on the prompts that did. Random = mean of 2 random directions of the same size/amount. "
         "CI = 95% bootstrap interval of (real - random) over prompts; wide at this sample size.", ""]
    for mode, m in r["modes"].items():
        L += [f"## edit position: {mode}", "",
              "| steering | score | effect | random effect | real - random [CI] | crossed 50 | coherence | random coherence |",
              "|---|---|---|---|---|---|---|---|"]
        for s, e in m["steer"].items():
            ci = e.get("ci_vs_random")
            L.append(f"| {s}x | {_f(e['mean_on_absent'])} | {_f(e['effect'], signed=True)} | {_f(e.get('random_effect'), signed=True)} | "
                     f"{('[%d, %d]' % (round(ci[0]), round(ci[1]))) if ci else 'n/a'} | {_f(e['crossed_50'], pct=True)} | "
                     f"{_f(e['coherence'])} | {_f(e.get('random_coherence'))} |")
        L += ["", "| removal (prompts that showed it) | before | after | drop | random drop | real - random [CI] | now below 50 | coherence |",
              "|---|---|---|---|---|---|---|---|"]
        for tag, label in (("removal", "remove at block %d" % load_cfg()["layer"]),):
            if tag in m:
                e = m[tag]
                ci = e.get("ci_vs_random")
                L.append(f"| {label} (n={e['n']}) | {_f(e['score_before'])} | {_f(e['score_after'])} | {_f(e['drop'], signed=True)} | "
                         f"{_f(e.get('random_drop'), signed=True)} | {('[%d, %d]' % (round(ci[0]), round(ci[1]))) if ci else 'n/a'} | "
                         f"{_f(e['crossed_below_50'], pct=True)} | {_f(e['coherence'])} |")
        L.append("")
    L += ["## bleed: this direction at +%g on other concepts' prompts (their own target)" % tc["bleed"], "",
          "| other concept | change in its score | coherence |", "|---|---|---|"]
    L += [f"| {o} | {_f(b['effect_on_other_concept'], signed=True)} (n={b['n']}) | {_f(b['coherence'])} |" for o, b in r["bleed"].items()]
    L += ["", "## first reply token: probability mass of concept words (up) vs opposite words (down)", "",
          "| condition | up | down |", "|---|---|---|"]
    for c, e in r["first_token"].items():
        if (c == "base" or c.endswith("-all")) and "rand" not in c:
            L.append(f"| {c} | {_f(e['p_up'], pct=True)} | {_f(e['p_down'], pct=True)} |")
    return "\n".join(L) + "\n"


def write_replies(name: str, rows: list[dict], got: dict, sc: dict, suf: str = "") -> None:
    """Every prompt with its unsteered reply and the strongest edits' replies, with judge scores."""
    tc = load_cfg()["transfer"]
    show = ["base"] + [f"s{s:+g}-all" for s in tc["strengths"]] + ["remove-all"]
    by = {(r["concept"], r["pid"], r["cond"]): r for r in rows}
    L = [f"# {name}: replies", ""]
    for i, p in enumerate(sc[name]["prompts"]):
        L += [f"## {i:02d} [{p['expect']}] {question(p)}", ""]
        for cond in show:
            r = by.get((name, i, cond))
            if r:
                s, k = got[_key(name, i, r["reply"])]
                L += [f"**{cond}** (concept {s if s is not None else 'REFUSAL'}, coherence {k:.0f}; first tokens {r['top']}):", "",
                      "> " + r["reply"].replace("\n", "\n> "), ""]
    (OUT / "results" / f"{name}{suf}-replies.md").write_text("\n".join(L))


def select(names: list[str]) -> None:
    """Prompts whose judged unsteered reply shows the concept (> 50) -> runs/transfer/select/<name>.json."""
    sc = scenarios()
    for n in names:
        rows = [r for r in read_jsonl(OUT / "gens" / f"{n}.jsonl") if r["cond"] == "base"]
        items = [dict(key=_key(n, r["pid"], r["reply"]), question=question(sc[n]["prompts"][r["pid"]]),
                      answer=r["reply"], target=sc[n]["target"]) for r in rows]
        got = judge.scores(items, _meta(sc), seed=0)
        if got is None:
            print(f"{n}: {len(judge.pending_batches())} judge batches waiting -> score them, rerun")
            continue
        keep = [r["pid"] for r in rows if (got[_key(n, r["pid"], r["reply"])][0] or 0) > 50]
        write_json(OUT / "select" / f"{n}.json", keep)
        print(f"{n}: {len(keep)} of {len(rows)} prompts get a reply that shows the concept: {keep}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["gen", "score", "select"])
    ap.add_argument("--concepts", nargs="*")
    ap.add_argument("--only", nargs="*", help="gen: just the base and conditions containing these words")
    ap.add_argument("--modes", nargs="*", help="gen: override the edit positions")
    ap.add_argument("--variant", choices=["plain-last", "plain-mean"], help="use the plain-text idea direction")
    ap.add_argument("--strengths", nargs="*", type=float, help="gen: override the steering strengths")
    ap.add_argument("--random-at", nargs="*", type=float, help="gen: override the random-control strengths")
    a = ap.parse_args()
    names = a.concepts or list(scenarios())
    if a.stage == "gen":
        gen(names, a.only, a.modes, a.variant, a.strengths, a.random_at)
    elif a.stage == "score":
        score(names, a.variant)
    else:
        select(names)


if __name__ == "__main__":
    main()
