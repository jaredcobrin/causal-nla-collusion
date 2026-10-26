"""Scale test of never-named directions: pick public-pool positions where each of the 19 directions the AV was never
taught to name (6 held-out + 13 approved-but-unused) is strongly active (top 1% of its scores), plus random positions,
and generate a sentence for each from several models (LoRA adapters, or the untrained base model with a short question).
Positions used for RL (rl_combo) are excluded. The text read so far is saved for label checks.

    python -m cv.gen_any --models base,runs/general/th2_av,... --out runs/general/scale
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from cv.av_sft import NAME, PROMPTS

HIDDEN = ["music", "law", "mental_health", "vehicles", "hedging", "german"]
BASE_PROMPT = "In a few words, what topic or kind of content is this? Answer with a short phrase."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="runs/general/av_data/pool")
    ap.add_argument("--exclude", default="runs/general/av_data/rl_combo")
    ap.add_argument("--av", default="runs/general/av_data")
    ap.add_argument("--per", type=int, default=50)
    ap.add_argument("--n_random", type=int, default=300)
    ap.add_argument("--top", type=float, default=0.01)
    ap.add_argument("--models", required=True, help="comma list: 'base' or run dirs containing lora/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    tok = AutoTokenizer.from_pretrained(NAME)
    sel_path = a.out + "_sel.jsonl"
    H = torch.load(a.pool + "_acts.pt")
    if not os.path.exists(sel_path):
        prefix = torch.load(a.pool + "_prefix.pt")
        ex = {bytes(x.numpy().tobytes()) for x in torch.load(a.exclude + "_acts.pt")}
        ok = np.array([bytes(H[i].numpy().tobytes()) not in ex for i in range(len(H))]) & (H.float().norm(dim=-1).numpy() < 300)
        th = json.load(open(f"{a.av}/thresholds_global10.json"))
        names = HIDDEN + sorted(os.path.basename(f)[4:-4] for f in glob.glob(f"{a.av}/dir_*.npy") if os.path.basename(f)[4:-4] not in th)
        S = H.float().numpy() @ np.stack([np.load(f"{a.av}/dir_{c}.npy") for c in names]).T
        thr = np.quantile(S[ok], 1 - a.top, axis=0)
        used, sel = set(), []
        for k, c in enumerate(names):
            cand = [i for i in rng.permutation(np.flatnonzero(ok & (S[:, k] >= thr[k]))) if i not in used][: a.per]
            for i in cand:
                used.add(int(i))
                sel.append(dict(idx=int(i), concept=c, also=[names[j] for j in np.flatnonzero(S[i] >= thr) if names[j] != c]))
        rest = [i for i in rng.permutation(np.flatnonzero(ok)) if i not in used][: a.n_random]
        sel += [dict(idx=int(i), concept="random", also=[names[j] for j in np.flatnonzero(S[i] >= thr)]) for i in rest]
        with open(sel_path, "w") as f:
            for s in sel:
                s["text"] = tok.decode(prefix[s["idx"]].tolist())[-400:]
                f.write(json.dumps(s) + "\n")
        print(f"selected {len(sel)} positions ({len(names)} directions x up to {a.per}, + {a.n_random} random)", flush=True)
    sel = [json.loads(l) for l in open(sel_path)]
    acts = torch.stack([H[s["idx"]] for s in sel]).float()

    base = AutoModelForCausalLM.from_pretrained(NAME, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    pre = tok.apply_chat_template([{"role": "user", "content": "@@"}], tokenize=False, add_generation_prompt=True)
    head, tail = pre.split("@@")
    pre_ids = tok(head + " ?", add_special_tokens=False).input_ids
    pos = len(pre_ids) - 1
    inj = {"act": None}

    def hook(_, __, output):
        h = output[0] if isinstance(output, tuple) else output
        if inj["act"] is None or h.shape[1] <= 1:
            return output
        h = h.clone()
        h[:, pos] = inj["act"].to(h.dtype)
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

    base.model.layers[20].register_forward_hook(hook)
    model = None
    for spec in a.models.split(","):
        name = "base" if spec == "base" else os.path.basename(spec.rstrip("/"))
        outp = f"{a.out}_gen_{name}.jsonl"
        if os.path.exists(outp):
            continue
        if spec == "base":
            gen_model, prompt = base, pre_ids + tok(BASE_PROMPT + tail, add_special_tokens=False).input_ids
        else:
            if model is None:
                model = PeftModel.from_pretrained(base, f"{spec}/lora", adapter_name=name)
            else:
                model.load_adapter(f"{spec}/lora", adapter_name=name)
            model.set_adapter(name)
            gen_model, prompt = model, pre_ids + tok(PROMPTS["prose"] + tail, add_special_tokens=False).input_ids
        gen_model.eval()
        texts = []
        with torch.no_grad():
            for k in range(0, len(sel), 48):
                inj["act"] = acts[k: k + 48].cuda()
                x = torch.tensor([prompt] * len(acts[k: k + 48]), device="cuda")
                if spec == "base" and model is not None:
                    with model.disable_adapter():
                        g = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=40, do_sample=False, pad_token_id=tok.pad_token_id)
                else:
                    g = gen_model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=40, do_sample=False, pad_token_id=tok.pad_token_id)
                texts += tok.batch_decode(g[:, x.shape[1]:], skip_special_tokens=True)
        inj["act"] = None
        with open(outp, "w") as f:
            for s, t in zip(sel, texts):
                f.write(json.dumps(dict(idx=s["idx"], concept=s["concept"], text=t.strip())) + "\n")
        print(f"{name}: {len(texts)} sentences", flush=True)


if __name__ == "__main__":
    main()
