"""Activation reconstructor (AR): text -> layer-20 activation.

Qwen2.5-7B-Instruct + LoRA reads a description (or, for the baseline, only the list of active concept names) and a
linear head on the last token's final hidden state predicts the centered activation. Trained on the AV SFT split.

Scores on held-out positions: cosine of the centered activation, the same cosine after removing the trained concept
directions ("leftover", what the RL reward pays for), and per-direction correlation of the projections (trained, hidden
and unused directions).

    python -m cv.ar --data ... --acts ... --descriptions ... --split runs/general/av_prose_gist --hidden music,law,... --out runs/general/ar_text
    python -m cv.ar ... --input labels --out runs/general/ar_labels
"""
import argparse
import glob
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from cv.av_sft import NAME, mentions


def load_dirs(d):
    return {os.path.basename(f)[4:-4]: np.load(f) for f in sorted(glob.glob(f"{d}/dir_*.npy"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--acts", required=True)
    ap.add_argument("--descriptions", required=True)
    ap.add_argument("--split", required=True, help="AV run dir; its heldout_generations.jsonl lists the held-out ids")
    ap.add_argument("--dirs", default="runs/general/av_data")
    ap.add_argument("--drop_ids", help="jsonl of extra ids dropped from training")
    ap.add_argument("--val_ids", help="jsonl of ids kept out of training (in-distribution validation rows)")
    ap.add_argument("--hidden", default="")
    ap.add_argument("--input", default="text", choices=["text", "labels"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    hidden = [c for c in a.hidden.split(",") if c]

    rows = {json.loads(l)["id"]: json.loads(l) for l in open(a.data)}
    acts_all = torch.load(a.acts).float()
    desc = {json.loads(l)["id"]: json.loads(l)["description"] for l in open(a.descriptions) if l.strip()}
    gens = {json.loads(l)["id"]: json.loads(l)["text"] for l in open(f"{a.split}/heldout_generations.jsonl")}
    ids = [i for i in rows if i in desc]
    act = {i: acts_all[int(i[2:])] for i in ids}
    va = [i for i in ids if i in gens]
    val = {json.loads(l)["id"] for l in open(a.val_ids)} if a.val_ids else set()
    if a.drop_ids:
        val |= {json.loads(l)["id"] for l in open(a.drop_ids)}
    tr = [i for i in ids if i not in gens and i not in val and float(act[i].norm()) < 300
          and not (set(rows[i]["active"]) & set(hidden)) and not mentions(desc[i], hidden)]
    mu = torch.stack([act[i] for i in tr]).mean(0)
    sd = torch.stack([act[i] - mu for i in tr]).std()
    print(f"train {len(tr)} | held-out {len(va)} | input {a.input}", flush=True)

    def text_of(i, src=None):
        if a.input == "labels":
            return "Active concepts: " + (", ".join(rows[i]["active"]) or "none")
        return src if src is not None else desc[i]

    tok = AutoTokenizer.from_pretrained(NAME)
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(NAME, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM"))
    d = model.config.hidden_size
    head = torch.nn.Linear(d, d).cuda().float()
    torch.nn.init.zeros_(head.bias)
    torch.nn.init.normal_(head.weight, std=0.01)

    def encode(texts):
        msgs = [tok.apply_chat_template([{"role": "user", "content": "Reconstruct the model's internal state from this description:\n" + t}], tokenize=False, add_generation_prompt=True) for t in texts]
        b = tok(msgs, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        h = model(**b, output_hidden_states=True).hidden_states[-1]
        last = b["attention_mask"].sum(1) - 1
        return head(h[torch.arange(len(texts)), last].float())

    def target(ids_):
        return torch.stack([(act[i] - mu) / sd for i in ids_]).cuda()

    def loss_fn(pred, y):
        return F.mse_loss(pred, y) + (1 - F.cosine_similarity(pred, y, dim=-1)).mean()

    params = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": a.lr}, {"params": head.parameters(), "lr": 1e-3}]
    opt = torch.optim.AdamW(params, weight_decay=0.0)
    steps = a.epochs * ((len(tr) + a.bs - 1) // a.bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * max(0.0, 1 - s / steps))
    step = 0
    for ep in range(a.epochs):
        model.train()
        order = tr[:]
        rng.shuffle(order)
        for k in range(0, len(order), a.bs):
            b = order[k: k + a.bs]
            loss = loss_fn(encode([text_of(i) for i in b]), target(b))
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in params for p in g["params"]], 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % 25 == 0:
                print(f"ep {ep} step {step}/{steps} loss {loss.item():.3f}", flush=True)
    model.save_pretrained(out / "lora")
    torch.save(head.state_dict(), out / "head.pt")
    torch.save({"mu": mu, "sd": sd}, out / "norm.pt")

    # evaluation
    dirs = load_dirs(a.dirs)
    trained = sorted({c for i in tr for c in rows[i]["active"]})
    Q, _ = np.linalg.qr(np.stack([dirs[c] for c in trained if c in dirs]).T)  # basis of the trained directions

    def score(src_texts, tag):
        model.eval()
        with torch.no_grad():
            pred = torch.cat([encode([text_of(i, s) for i, s in zip(va[k: k + 16], src_texts[k: k + 16])]).cpu() for k in range(0, len(va), 16)])
        y = torch.stack([(act[i] - mu) / sd for i in va])
        cos = F.cosine_similarity(pred, y, dim=-1).mean().item()
        P, Y = pred.numpy(), y.numpy()
        left = lambda M: M - (M @ Q) @ Q.T
        lcos = float(np.mean([np.dot(p, q) / (np.linalg.norm(p) * np.linalg.norm(q) + 1e-9) for p, q in zip(left(P), left(Y))]))
        corr = {}
        for c, v in dirs.items():
            pp, yy = P @ v, Y @ v
            corr[c] = float(np.corrcoef(pp, yy)[0, 1]) if pp.std() > 0 and yy.std() > 0 else None
        grp = lambda cs: float(np.nanmean([corr[c] for c in cs if c in corr and corr[c] is not None])) if cs else None
        unused = [c for c in dirs if c not in trained and c not in hidden]
        r = dict(cosine=cos, leftover_cosine=lcos, corr_trained=grp(trained), corr_hidden=grp(hidden), corr_unused=grp(unused), per_direction=corr)
        print(tag, json.dumps({k: v for k, v in r.items() if k != "per_direction"}), flush=True)
        return r

    res = {"gold_text": score([desc[i] for i in va], "gold description")}
    if a.input == "text":
        res["av_text"] = score([gens[i] for i in va], "AV generation")
    json.dump(res, open(out / "heldout.json", "w"), indent=1)


if __name__ == "__main__":
    main()
