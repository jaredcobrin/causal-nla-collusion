"""The four causal tests at one layer, scored by the judge on the model's free replies.

Every reply gets two judge scores: the concept score (0-100; higher = the reply shows the concept)
and coherence (0-100). Controls are random unit directions: steering controls add a random
direction of the same size; removal controls move the activation by the same amount (<h,u>) along a
random direction. Effects are paired per reply (same prompt, same sampling seed).

T1 steering      add s x size x u for s in -1, +0.5, +1, +2 (0 = no steering).
                 Pass: the +1x effect beats every random direction, its 95% CI vs the random mean
                 is above 0, and the mean score rises with s (Spearman >= t1_monotone).
T2 side effects  mean coherence at +1x and -1x >= t2_coherence; and on prompts of the 3 most similar
                 other concepts, +1x steering moves their score <= t2_spillover x this concept's own
                 +1x effect. Neutral-text loss change is reported only.
T3 removal       on replies that show the concept by themselves (score > 50), remove u at the layer.
                 Pass: >= t3_present such replies; it closes >= t3_gap_frac of the gap to replies that
                 do not show it (CI > 0); and it drops the score more than every random removal.
T4 downstream    relative change in what later blocks write at the prompt's last token when u is
                 removed. Pass: beats every random removal and the CI vs the random mean is above 0.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from cv import judge, qwen_judge, rule_judge
from cv.hooks import ablate, add, displace
from cv.io import RUNS, h, read_jsonl, seed_for, write_jsonl
from cv.stats import boot_ci, spearman


class Concept:
    """A concept's test prompts in one split and its direction at the tested layer."""

    def __init__(self, name, meta, rows, split, v, cfg):
        self.name, self.meta, self.v = name, meta, v
        prompts = sorted((r for r in rows if r["role"] == "eval" and r["split"] == split), key=lambda r: (r["rank"], r["id"]))
        n = cfg["replies"]["per_condition"]
        self.sampled = len(prompts) < n  # too few prompts: several temperature-1 replies per prompt
        reps = math.ceil(n / len(prompts)) if self.sampled else 1
        self.items = [(r, k) for r in prompts[:n] for k in range(reps)][:n]


class Battery:
    def __init__(self, lm, cfg, phase: str, concepts: dict, neutral, base_nll):
        self.lm, self.cfg, self.phase, self.L = lm, cfg, phase, cfg["layer"]
        self.th, self.concepts, self.neutral, self.base_nll = cfg["thresholds"], concepts, neutral, base_nll
        g = torch.Generator().manual_seed(seed_for("random-directions"))
        self.rand = torch.nn.functional.normalize(torch.randn(cfg["n_random"], lm.d, generator=g), dim=1).to(lm.device)

    # ---- replies and scores ----------------------------------------------------------------------
    def _replies(self, c: Concept, tag: str, edit, items) -> list[str]:
        """Replies to c's prompts under an edit at the layer; cached per (concept, condition)."""
        path = RUNS / "gens" / f"{self.phase}-{c.name}-{tag}.jsonl"
        if path.exists():
            return [g["reply"] for g in read_jsonl(path)][: len(items)]
        texts = [self.lm.render(r["msgs"]) for r, _ in items]
        out = self.lm.generate(texts, edits={self.L: edit} if edit is not None else None,
                               max_new_tokens=self.cfg["replies"]["max_new_tokens"], temperature=1.0 if c.sampled else 0,
                               seed=seed_for(self.phase, c.name, "replies"), bs=self.cfg["batch"]["generate"])
        write_jsonl(path, [dict(id=r["id"], k=k, reply=a) for (r, k), a in zip(items, out)])
        return out

    def _scored(self, c: Concept, conds: dict, items) -> dict | None:
        """conds: tag -> replies. Returns tag -> (scores, coherence) arrays (nan = REFUSAL), or None."""
        # keys include a hash of the reply, so a regenerated reply can never reuse an old score
        key = lambda tag, i: f"{self.phase}|{c.name}|{tag}|{i}|{h(conds[tag][i])}"
        req = [dict(key=key(tag, i), question=r["msgs"][0]["content"], answer=a, target=r.get("target"))
               for tag, rep in conds.items() for i, ((r, _), a) in enumerate(zip(items, rep))]
        # The judge is fixed per prompt dataset, so a baseline and its steered replies always share one.
        if c.meta.get("judge") == "qwen":
            sc = qwen_judge.scores(self.lm, req, self.cfg["batch"]["forward"])
        elif c.meta.get("judge") == "rule":
            sc = rule_judge.scores(req)
        else:
            sc = judge.scores(req, c.meta, seed=seed_for(self.phase, c.name, "judge"))
        if sc is None:
            return None
        arr = lambda tag, j: np.array([np.nan if sc[key(tag, i)][j] is None else sc[key(tag, i)][j] for i in range(len(items))])
        return {tag: (arr(tag, 0), arr(tag, 1)) for tag in conds}

    @staticmethod
    def _paired(x, base):
        ok = ~np.isnan(x) & ~np.isnan(base)
        return x[ok] - base[ok], ok

    def _nearest(self, u, exclude: str) -> list[str]:
        """Other concepts whose directions are most similar to u (near-duplicates excluded)."""
        cos = {n: float(u @ (c.v / c.v.norm()).to(u.device)) for n, c in self.concepts.items()
               if n != exclude and c.v is not None}
        cos = {n: x for n, x in cos.items() if abs(x) < self.cfg["spillover"]["duplicate_cos"]}
        return sorted(cos, key=lambda n: -abs(cos[n]))[: self.cfg["spillover"]["concepts"]]

    # ---- the tests -----------------------------------------------------------------------------
    def evaluate(self, name: str, src: str, v: torch.Tensor) -> dict | None:
        """Run T1-T4 for direction v on concept src's prompts; None while judge scores are pending.
        `name` differs from `src` for a control (e.g. "refusal#shuffled" on refusal's prompts)."""
        c, L = self.concepts[src], self.L
        v = v.float().to(self.lm.device)
        size = float(v.norm())
        u = v / size
        tag = lambda t: t if name == src else f"{name}-{t}"  # the unsteered baseline is shared

        # Round 1: steering replies, random-direction replies, spillover replies -> judge.
        conds = {"base": None, **{tag(f"s{s:+}"): add(s * size * u) for s in self.cfg["strengths"]},
                 **{tag(f"rand{j}"): add(size * r) for j, r in enumerate(self.rand)}}
        reps = {t: self._replies(c, t, e, c.items) for t, e in conds.items()}
        spill_reps = {}
        for o in self._nearest(u, src):
            oc = self.concepts[o]
            it = oc.items[: self.cfg["spillover"]["prompts"]]
            spill_reps[o] = (it, {"base": self._replies(oc, "base", None, oc.items)[: len(it)],
                                  f"{name}-s+1.0": self._replies(oc, f"{name}-s+1.0", add(size * u), it)})
        sc = self._scored(c, reps, c.items)
        spill = {o: self._scored(self.concepts[o], r, it) for o, (it, r) in spill_reps.items()}
        if sc is None or any(x is None for x in spill.values()):
            return None

        base, groups = sc["base"][0], [r["group"] for r, _ in c.items]
        # T1
        levels = {0.0: float(np.nanmean(base)), **{s: float(np.nanmean(sc[tag(f"s{s:+}")][0])) for s in self.cfg["strengths"]}}
        d1, ok1 = self._paired(sc[tag("s+1.0")][0], base)
        eff1 = float(np.mean(d1))
        rand_eff = [float(np.mean(self._paired(sc[tag(f"rand{j}")][0], base)[0])) for j in range(len(self.rand))]
        ci1 = boot_ci(d1 - np.mean(rand_eff), [g for g, o in zip(groups, ok1) if o], self.cfg["bootstrap"])
        rho = round(spearman(sorted(levels), [levels[s] for s in sorted(levels)]), 6)  # 0.9 must not become 0.8999...
        t1_checks = dict(beats_random=eff1 > max(rand_eff), ci_above_random=ci1[0] > 0, monotone=rho >= self.th["t1_monotone"])
        t1 = dict(mean_score=levels, effect_plus1=eff1, ci_vs_random=ci1, random_effects=rand_eff, spearman=rho,
                  checks=t1_checks, passed=all(t1_checks.values()))
        # T2
        coh = {s: float(np.nanmean(sc[tag(f"s{s:+}")][1])) for s in (1.0, -1.0)}
        ratios = {o: abs(float(np.mean(self._paired(r[f"{name}-s+1.0"][0], r["base"][0])[0]))) / abs(eff1) if eff1 else math.inf
                  for o, r in spill.items()}
        dce = {s: self.lm.nll(self.neutral, edits={L: add(s * size * u)}) - self.base_nll for s in (1.0, -1.0)}
        t2_checks = dict(coherent=min(coh.values()) >= self.th["t2_coherence"],
                         low_spillover=max(ratios.values(), default=0.0) <= self.th["t2_spillover"])
        t2 = dict(coherence=coh, base_coherence=float(np.nanmean(sc["base"][1])), spillover=ratios,
                  spillover_judges={o: self.concepts[o].meta.get("judge", "haiku") for o in spill},
                  neutral_loss_change=dce, checks=t2_checks, passed=all(t2_checks.values()))
        # T3 (needs the judged baseline, so its replies are made in round 2)
        present = [i for i in range(len(base)) if base[i] > 50]
        absent = [i for i in range(len(base)) if base[i] < 50]
        t3 = dict(n_present=len(present), n_absent=len(absent))
        if len(present) < self.th["t3_present"]:
            t3.update(testable=False, passed=False)
        else:
            P = [c.items[i] for i in present]
            rconds = {tag("remove"): ablate(u), **{tag(f"rand-remove{j}"): displace(u, r) for j, r in enumerate(self.rand)}}
            r3 = self._scored(c, {t: self._replies(c, t, e, P) for t, e in rconds.items()}, P)
            if r3 is None:
                return None
            bp = base[present]
            drop, ok3 = self._paired(bp, r3[tag("remove")][0])  # baseline - after removal (> 0 = removal lowered it)
            rand_drop = [float(np.nanmean(bp - r3[tag(f"rand-remove{j}")][0])) for j in range(len(self.rand))]
            gap = float(np.mean(bp) - (np.mean(base[absent]) if absent else 0.0))
            frac = float(np.mean(drop)) / gap
            ci3 = boot_ci(drop / gap, [groups[i] for i, o in zip(present, ok3) if o], self.cfg["bootstrap"])
            t3_checks = dict(closes_gap=frac >= self.th["t3_gap_frac"] and ci3[0] > 0, beats_random=float(np.mean(drop)) > max(rand_drop))
            t3.update(testable=True, score_present=float(np.mean(bp)), score_removed=float(np.nanmean(r3[tag("remove")][0])),
                      score_absent=float(np.mean(base[absent])) if absent else None, gap_closed=frac, ci_gap_closed=ci3,
                      random_drops=rand_drop, checks=t3_checks, passed=all(t3_checks.values()))
        t4 = self._t4(u, c, groups, present)
        return dict(t1=t1, t2=t2, t3=t3, t4=t4, direction_size=size, judge=c.meta.get("judge", "haiku"))

    def _t4(self, u, c: Concept, groups, present) -> dict:
        """Relative change in what blocks L+1.. write at the prompt's last token when u is removed."""
        idx = present if len(present) >= self.th["t3_present"] else list(range(len(c.items)))
        texts = [self.lm.render(c.items[i][0]["msgs"]) for i in idx]
        W = list(range(self.L + 1, self.lm.n_layers))
        bs = self.cfg["batch"]["forward"]
        clean = self.lm.run(texts, write_layers=W, bs=bs)["writes"]

        def change(edit):
            w = self.lm.run(texts, edits={self.L: edit}, write_layers=W, bs=bs)["writes"]
            return (sum((w[l] - clean[l]).norm(dim=1) for l in W) / sum(clean[l].norm(dim=1) for l in W)).numpy()

        real = change(ablate(u))
        rand = [float(change(displace(u, r)).mean()) for r in self.rand]
        ci = boot_ci(real - np.mean(rand), [groups[i] for i in idx], self.cfg["bootstrap"])
        checks = dict(beats_random=float(real.mean()) > max(rand), ci_above_random=ci[0] > 0)
        return dict(prompts="showing the concept" if idx is present else "all", n=len(idx), change=float(real.mean()),
                    random_changes=rand, ci_vs_random=ci, checks=checks, passed=all(checks.values()))


def verdict(r: dict) -> tuple[str, list[str]]:
    """causal: T1+T2+T3. steerable only: T1+T2. not specific: T1 without T2. internal only: T4
    without T1. failed: none of these. Also returns every check that failed."""
    failed = [f"{t}:{k}" for t in ("t1", "t2", "t3", "t4") for k, ok in r[t].get("checks", {}).items() if not ok]
    if r["t3"].get("testable") is False:
        failed.append(f"t3:untestable ({r['t3']['n_present']} replies show the concept)")
    if not r["t1"]["passed"]:
        return ("internal only" if r["t4"]["passed"] else "failed"), failed
    if not r["t2"]["passed"]:
        return "not specific", failed
    return ("causal" if r["t3"]["passed"] else "steerable only"), failed
