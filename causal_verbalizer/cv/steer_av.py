"""Causal steering test for the AV: does its explanation follow a direction, independent of any text?

Take held-out activations, add alpha * (a concept direction) at the injected position, let the AV write its explanation, and
count how often it mentions the concept. A random unit direction pushed by the same amount is the control. Directions the AV
was never trained or rewarded on (the hidden concepts) show whether it describes a direction it was never taught; trained
concepts are the positive control. The AV sees no text; the only thing that changes is the activation.

    python -m cv.steer_av --av runs/general/av_prose_gist --out runs/general/steer_sft
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from cv.ar import load_dirs
from cv.av_sft import KEYWORDS, NAME, PROMPTS

CONTROL = {"sports": r"sport|player|team|league|athlet", "cryptocurrency": r"crypto|bitcoin|blockchain|token",
           "weather": r"weather|rain|forecast|storm|climate|temperature", "religion": r"religio|faith|church|scripture|\bgod\b",
           "food": r"\bfood|cook|recipe|meal|dish|ingredient", "pets": r"\bpets?\b|\bdogs?\b|\bcats?\b|animal"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--av", required=True, help="AV run dir with lora/ and heldout_generations.jsonl (its ids are the held-out rows)")
    ap.add_argument("--acts", default="runs/general/av_data/enriched_acts_global10.pt")
    ap.add_argument("--dirs", default="runs/general/av_data")
    ap.add_argument("--rows", type=int, default=80)
    ap.add_argument("--alpha", type=float, default=4.0, help="push size in standard deviations of the activations' projection on the direction")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    acts_all = torch.load(a.acts).float()
    held = [json.loads(l)["id"] for l in open(f"{a.av}/heldout_generations.jsonl")][: a.rows]
    X = torch.stack([acts_all[int(i[2:])] for i in held])
    dirs = load_dirs(a.dirs)
    concepts = {**{c: KEYWORDS[c] for c in KEYWORDS}, **CONTROL}

    tok = AutoTokenizer.from_pretrained(NAME)
    base = AutoModelForCausalLM.from_pretrained(NAME, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    model = PeftModel.from_pretrained(base, f"{a.av}/lora").eval()
    pre = tok.apply_chat_template([{"role": "user", "content": "@@"}], tokenize=False, add_generation_prompt=True)
    head, tail = pre.split("@@")
    pre_ids = tok(head + " ?", add_special_tokens=False).input_ids
    pos = len(pre_ids) - 1
    prompt = pre_ids + tok(PROMPTS["prose"] + tail, add_special_tokens=False).input_ids
    inj = {"act": None}

    def hook(_, __, output):
        h = output[0] if isinstance(output, tuple) else output
        if inj["act"] is None or h.shape[1] <= 1:
            return output
        h = h.clone()
        h[:, pos] = inj["act"].to(h.dtype)
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

    model.base_model.model.model.layers[20].register_forward_hook(hook)

    @torch.no_grad()
    def generate(acts):
        texts = []
        for k in range(0, len(acts), 40):
            inj["act"] = acts[k: k + 40].cuda()
            x = torch.tensor([prompt] * len(acts[k: k + 40]), device="cuda")
            g = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=140, do_sample=False, pad_token_id=tok.pad_token_id)
            texts += tok.batch_decode(g[:, x.shape[1]:], skip_special_tokens=True)
        inj["act"] = None
        return texts

    rate = lambda texts, pat: float(np.mean([bool(re.search(pat, t, re.I)) for t in texts]))
    base_texts = generate(X)
    res = {"baseline": {c: rate(base_texts, p) for c, p in concepts.items()}}
    samples = {}
    for c, pat in concepts.items():
        v = torch.tensor(dirs[c])
        sd = float((acts_all @ v).std())
        r = torch.tensor(rng.standard_normal(v.shape[0]), dtype=torch.float32)
        r = r / r.norm()  # random unit direction, same push
        plus = generate(X + a.alpha * sd * v)
        rand = generate(X + a.alpha * sd * r)
        res[c] = dict(taught=c in CONTROL, push_sd=sd, baseline=res["baseline"][c], plus=rate(plus, pat), random=rate(rand, pat),
                      plus_n=int(sum(bool(re.search(pat, t, re.I)) for t in plus)), rows=len(X))
        samples[c] = dict(plus=plus[:3], random=rand[:3], baseline=base_texts[:3])
        print(c, json.dumps(res[c]), flush=True)
    json.dump(res, open(out / "steer.json", "w"), indent=1)
    json.dump(samples, open(out / "samples.json", "w"), indent=1)


if __name__ == "__main__":
    main()
