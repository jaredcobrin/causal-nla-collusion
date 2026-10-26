"""Pipeline. Resumable: every step writes its own file and is skipped when that file exists, so
rerun the same command after judge batches are scored.

    python -m cv.run check    # 10 pilots + 3 shuffled-label controls, val split: does the pipeline work?
    python -m cv.run main     # all 42 concepts, test split
    python -m cv.run ab       # A/B logit-shift pilot (cv/ab.py), val split
"""

from __future__ import annotations

import argparse
import platform
import time

import torch

from cv import judge, report
from cv.battery import Battery, Concept, verdict
from cv.directions import build_filtered, build_pv, build_static
from cv.io import RUNS, SPLITS, Log, load_cfg, load_concepts, read_json, read_jsonl, write_json
from cv.model import LM

CONTROLS = ["refusal", "truth", "survival-instinct"]  # their shuffled-label directions must fail
# Concepts whose dataset labels are not what the model itself does; also tried with a direction built only
# from train groups where the model's own reply shows the behaviour (named "<concept>+filtered").
FILTERED = ["survival-instinct", "sycophantic-agreement", "corrigible-neutral-HHH", "desire-for-no-human-monitoring"]


def _freeze(run_dir, cfg) -> None:
    """Pre-registration: the first run of a phase stores the config; later runs must match it."""
    path = run_dir / "config.json"
    if path.exists() and read_json(path)["thresholds"] != cfg["thresholds"]:
        raise SystemExit(f"thresholds changed since {path} was written; refusing to rescore")
    if not path.exists():
        write_json(path, cfg)


def directions(lm, cfg, names, rows, meta, log) -> dict:
    """Build (or load) each concept's direction from its train split; shared by both phases."""
    out = {}
    for n in names:
        pt, js = RUNS / "directions" / f"{n}.pt", RUNS / "directions" / f"{n}.json"
        if pt.exists():
            out[n] = {**torch.load(pt, weights_only=True), **read_json(js)}
            continue
        t0, f0, g0 = time.time(), lm.forwarded, lm.generated
        src = n.split("+")[0]
        if n.endswith("+filtered"):
            d = build_filtered(lm, n, rows[src], meta[src], cfg)
        elif meta[n]["source"] == "pv":
            d = build_pv(lm, n, rows[n], meta[n], cfg)
        else:
            d = build_static(lm, n, rows[n], cfg)
        if d is None:
            log(f"direction {n}: waiting for judge scores")
            continue
        info = dict(gate={str(l): s for l, s in d.get("gate", {}).items()}, data=d["data"], failed=d.get("failed"),
                    usage=dict(forwarded=lm.forwarded - f0, generated=lm.generated - g0, seconds=round(time.time() - t0, 1)))
        write_json(js, info)
        torch.save({"v": d.get("v", {}), "shuffled": d.get("shuffled", {})}, pt)
        out[n] = {**d, **info}
        g = info["gate"].get(str(cfg["layer"]), {})
        log(f"direction {n}: {d['data']} | split-half {g.get('split_half', float('nan')):.3f} auroc {g.get('auroc', float('nan')):.3f}"
            + (f" | {d['failed']}" if d.get("failed") else ""))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["check", "main", "ab"])
    args, rest = ap.parse_known_args()
    if args.phase == "ab":
        from cv.ab import main as ab_main
        return ab_main(rest)
    cfg, concepts = load_cfg(), {c["name"]: c for c in load_concepts()}
    split, L = cfg["phases"][args.phase]["split"], cfg["layer"]
    names = [n for n, c in concepts.items() if c.get("pilot")] if args.phase == "check" else list(concepts)
    run_dir = RUNS / args.phase
    log = Log(run_dir)
    _freeze(run_dir, cfg)
    rows = {n: read_jsonl(SPLITS / f"{n}.jsonl") for n in names}
    meta = {n: read_json(SPLITS / f"{n}.meta.json") for n in names}
    lm = LM(cfg["model"])
    import transformers
    write_json(run_dir / "environment.json", dict(python=platform.python_version(), torch=torch.__version__,
               transformers=transformers.__version__, gpu=torch.cuda.get_device_name()))
    log(f"start {args.phase}: {len(names)} concepts, layer {L}, {split} split")

    filtered = [f"{n}+filtered" for n in FILTERED if n in names] if args.phase == "check" else []
    dirs = directions(lm, cfg, names + filtered, rows, meta, log)
    stable = {n for n in dirs if not dirs[n].get("failed") and dirs[n]["gate"][str(L)]["split_half"] >= cfg["thresholds"]["split_half"]}
    pool = {n: Concept(n, meta[n], rows[n], split, dirs[n]["v"][L] if n in stable else None, cfg) for n in names}
    neutral = [r["ids"] for r in read_jsonl(SPLITS / "neutral.jsonl")]
    bat = Battery(lm, cfg, args.phase, pool, neutral, lm.nll(neutral))
    subjects = ([(n, n, "v") for n in names] + [(f, f.split("+")[0], "v") for f in filtered]
                + ([(f"{n}#shuffled", n, "shuffled") for n in CONTROLS] if args.phase == "check" else []))

    for name, src, which in subjects:
        out = run_dir / "results" / f"{name}.json"
        key = name if name in dirs else src  # "<concept>+filtered" has its own direction
        if out.exists() or key not in dirs:
            continue
        info = dirs[key]
        res = dict(concept=name, group=concepts[src]["group"], layer=L, split=split,
                   direction=dict(info["gate"].get(str(L), {}), **info["data"]))
        t0, f0, g0 = time.time(), lm.forwarded, lm.generated
        if which == "v" and key not in stable:
            res.update(verdict="failed (unstable direction)", failed=[info.get("failed") or "split_half"])
        else:  # controls skip the stability rule: the point is whether the causal tests reject them
            tests = bat.evaluate(name, src, info[which][L])
            if tests is None:
                log(f"{name}: waiting for judge scores")
                continue
            v, failed = verdict(tests)
            res.update(tests, verdict=v, failed=failed)
        res["usage"] = dict(forwarded=lm.forwarded - f0, generated=lm.generated - g0, seconds=round(time.time() - t0, 1))
        write_json(out, res)
        log(f"{name}: {res['verdict']}" + (f" | failed {res['failed']}" if res["failed"] else "") + f" | {res['usage']}")
        report.write(run_dir, args.phase, len(subjects))

    if not judge.pending_batches() and len(list((run_dir / "results").glob("*.json"))) == len(subjects):
        rel = judge.retest(f"{args.phase}|", seed=0)
        if rel is not None:
            write_json(run_dir / "judge_reliability.json", rel)
    report.write(run_dir, args.phase, len(subjects))
    waiting = judge.pending_batches()
    log(f"pass finished; judge batches waiting: {len(waiting)}" + (" -> score them, then rerun" if waiting else ""))


if __name__ == "__main__":
    main()
