"""SFT of the activation verbalizer (AV): layer-20 activation -> text.

The raw activation is written into the residual stream at the output of block 20 (where it was recorded) at one
placeholder token; Qwen2.5-7B-Instruct + LoRA writes the target. Held-out positions (split by source text) are scored
against the probe states.

  --target active_first  'ACTIVE: a, b' then the description (v2); scored by parsing the ACTIVE line
  --target prose         the description only; scored by a TF-IDF decoder fit on the training descriptions

--hidden removes concepts from training: rows where one is active, or whose description mentions one, are dropped.
Held-out rows keep them, so the generations can be checked for naming a concept the AV was never taught.

    python -m cv.av_sft --data runs/general/av_data/enriched_sft_global10.jsonl --acts runs/general/av_data/enriched_acts_global10.pt
"""
import argparse
import json
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

NAME = "Qwen/Qwen2.5-7B-Instruct"
PROMPTS = {"active_first": "Describe what the model is representing at this position, then list which concepts are active and inactive.",
           "prose": "Describe what the model is thinking about at this position."}
# words that give a hidden concept away in free text
KEYWORDS = {"music": r"music|song|singer|band|concert|instrument|melod", "law": r"\blaw|legal|lawyer|court|lawsuit|attorney|judge|statute",
            "mental_health": r"mental health|anxiety|depress|therap|psycholog|emotional wellbeing|coping", "vehicles": r"vehicle|\bcars?\b|truck|automobile|motorcycle|driving",
            "hedging": r"hedg", "german": r"german|deutsch"}


def parse(text: str):
    m = re.search(r"ACTIVE:\s*([^\n]*)", text)
    if not m:
        return None
    return {x.strip() for x in m.group(1).split(",") if x.strip() and x.strip() != "none"}


def mentions(text: str, concepts) -> bool:
    return any(re.search(KEYWORDS[c], text, re.I) for c in concepts if c in KEYWORDS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--acts", required=True)
    ap.add_argument("--descriptions", help="jsonl of {id, description} replacing the descriptions in --data")
    ap.add_argument("--target", default="active_first", choices=["active_first", "prose"])
    ap.add_argument("--test_sources", default="", help="comma-separated dataset names held out entirely as the test set (generation is scored on these; early stopping uses a random validation slice of the rest)")
    ap.add_argument("--drop_ids", help="jsonl of extra ids dropped from training (texts from the held-out concepts' own datasets)")
    ap.add_argument("--hidden", default="", help="comma-separated concepts removed from training")
    ap.add_argument("--out", default="runs/general/av_sft_v2")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--holdout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    hidden = [c for c in a.hidden.split(",") if c]

    rows = [json.loads(l) for l in open(a.data)]
    acts_all = torch.load(a.acts).float()
    # enriched jsonl rows keep their original order: id en00000.. indexes the acts
    acts = acts_all[[int(r["id"][2:]) for r in rows]]
    if a.descriptions:
        desc = {}
        for l in open(a.descriptions):
            if l.strip():
                d = json.loads(l)
                desc[d["id"]] = d["description"]
        keep = [i for i, r in enumerate(rows) if r["id"] in desc]
        print(f"descriptions for {len(keep)}/{len(rows)} rows", flush=True)
        rows, acts = [rows[i] for i in keep], acts[keep]
        for r in rows:
            r["description"] = desc[r["id"]]
    for r in rows:
        r["target"] = r["description"] if a.target == "prose" else "ACTIVE: " + (", ".join(r["active"]) or "none") + "\n" + r["description"]
    trained = sorted({c for r in rows for c in r["active"]} - set(hidden))

    texts = sorted({r["text_id"] for r in rows})
    rng.shuffle(texts)
    held = set(texts[: int(len(texts) * a.holdout)])
    big = set(torch.nonzero(acts.norm(dim=-1) > 300).flatten().tolist())  # outlier rows, dropped from training
    leaky = {i for i, r in enumerate(rows) if set(r["active"]) & set(hidden) or mentions(r["description"], hidden)}
    if a.drop_ids:
        extra = {json.loads(l)["id"] for l in open(a.drop_ids)}
        leaky |= {i for i, r in enumerate(rows) if r["id"] in extra}
    tr = [i for i, r in enumerate(rows) if r["text_id"] not in held and i not in big and i not in leaky]
    va = [i for i, r in enumerate(rows) if r["text_id"] in held]  # validation: early stopping
    test_sources = {x for x in a.test_sources.split(",") if x}
    te = va
    if test_sources:
        te = [i for i, r in enumerate(rows) if r["source"] in test_sources]
        tes = set(te)
        tr = [i for i in tr if i not in tes]
        va = [i for i in va if i not in tes]
    print(f"train {len(tr)} (dropped {len(leaky)} hidden-concept rows) | validation {len(va)} | test {len(te)} | trained concepts {len(trained)}", flush=True)

    tok = AutoTokenizer.from_pretrained(NAME)
    model = AutoModelForCausalLM.from_pretrained(NAME, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM"))

    pre = tok.apply_chat_template([{"role": "user", "content": "@@"}], tokenize=False, add_generation_prompt=True)
    head, tail = pre.split("@@")
    pre_ids = tok(head + " ?", add_special_tokens=False).input_ids  # last token is the placeholder
    pos = len(pre_ids) - 1
    suf_ids = tok(PROMPTS[a.target] + tail, add_special_tokens=False).input_ids

    # write the raw activation into the residual stream at block 20's output, at the placeholder position
    inj = {"act": None}

    def hook(_, __, output):
        h = output[0] if isinstance(output, tuple) else output
        if inj["act"] is None or h.shape[1] <= 1:
            return output
        h = h.clone()
        h[:, pos] = inj["act"].to(h.dtype)
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

    model.base_model.model.model.layers[20].register_forward_hook(hook)

    def batch(ids, shuffle=False):
        seqs = [(i, tok(rows[i]["target"] + "<|im_end|>", add_special_tokens=False).input_ids) for i in ids]
        L = max(len(pre_ids) + len(suf_ids) + len(t) for _, t in seqs)
        X = torch.full((len(seqs), L), tok.pad_token_id, dtype=torch.long, device="cuda")
        M = torch.zeros(len(seqs), L, dtype=torch.long, device="cuda")
        Y = torch.full((len(seqs), L), -100, dtype=torch.long, device="cuda")
        src = list(ids)
        if shuffle:
            src = src[1:] + src[:1]
        for b, (i, t) in enumerate(seqs):
            e = pre_ids + suf_ids + t
            X[b, : len(e)] = torch.tensor(e)
            M[b, : len(e)] = 1
            Y[b, len(e) - len(t): len(e)] = torch.tensor(t)
        inj["act"] = torch.stack([acts[j] for j in src]).cuda()
        return X, M, Y

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    steps = a.epochs * ((len(tr) + a.bs - 1) // a.bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * max(0.0, 1 - s / steps))

    def loss_of(ids, shuffle=False):
        X, M, Y = batch(ids, shuffle)
        lg = model(input_ids=X, attention_mask=M).logits.float()
        return F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), Y[:, 1:].reshape(-1), ignore_index=-100)

    def heldout_loss(shuffle=False):
        with torch.no_grad():
            return sum(loss_of(va[k: k + a.bs], shuffle).item() * len(va[k: k + a.bs]) for k in range(0, len(va), a.bs)) / len(va)

    step, best, best_state = 0, 1e9, None
    for ep in range(a.epochs):
        model.train()
        order = tr[:]
        rng.shuffle(order)
        for k in range(0, len(order), a.bs):
            loss = loss_of(order[k: k + a.bs])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % 25 == 0:
                print(f"ep {ep} step {step}/{steps} loss {loss.item():.3f}", flush=True)
        model.eval()
        vl = heldout_loss()
        print(f"== epoch {ep}: held-out loss {vl:.3f}", flush=True)
        if vl < best:
            best, best_ep = vl, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.named_parameters() if v.requires_grad}
    model.load_state_dict({k: v.cuda() for k, v in best_state.items()}, strict=False)
    model.save_pretrained(out / "lora")
    print(f"== best epoch {best_ep} (held-out loss {best:.3f}); shuffled-activation loss {heldout_loss(True):.3f}", flush=True)

    # held-out generation
    model.eval()
    gens = []
    for i in te:
        inj["act"] = acts[i][None].cuda()
        x = torch.tensor([pre_ids + suf_ids], device="cuda")
        with torch.no_grad():
            g = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=110 if a.target == "active_first" else 220, do_sample=False)
        gens.append(tok.decode(g[0][x.shape[1]:], skip_special_tokens=True))

    res = dict(n=len(te), best_epoch=best_ep, heldout_loss=best)
    gold = [set(rows[i]["active"]) & set(trained) for i in te]
    if a.target == "active_first":
        pred = [parse(t) or set() for t in gens]
    else:
        # decoder: TF-IDF + logistic regression on the training descriptions
        decode = fit_decoder([rows[i]["description"] for i in tr], [rows[i]["active"] for i in tr], trained)
        pred = decode(gens)
        ceil = decode([rows[i]["description"] for i in te])  # decoder on the Haiku held-out descriptions
        res["decoder_on_haiku_text"] = prf(ceil, gold)
    res.update(prf(pred, gold))
    # naming test for hidden concepts: does the generated text mention a concept the AV was never taught?
    for c in hidden:
        pos_rows = [k for k, i in enumerate(te) if c in rows[i]["active"]]
        neg_rows = [k for k, i in enumerate(te) if not set(rows[i]["active"]) & set(hidden)]
        res[f"hidden_{c}"] = dict(n=len(pos_rows), named_when_active=float(np.mean([mentions(gens[k], [c]) for k in pos_rows])) if pos_rows else None,
                                 named_otherwise=float(np.mean([mentions(gens[k], [c]) for k in neg_rows])) if neg_rows else None)
    print(json.dumps(res, indent=1), flush=True)
    json.dump(res, open(out / "heldout.json", "w"), indent=1)
    with open(out / "heldout_generations.jsonl", "w") as f:
        for k, i in enumerate(te):
            f.write(json.dumps(dict(id=rows[i]["id"], gold=sorted(set(rows[i]["active"])), pred=sorted(pred[k]), text=gens[k])) + "\n")


def fit_decoder(descs, labels, trained):
    """TF-IDF + logistic regression from a description to the set of trained concepts it implies."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    X = vec.fit_transform(descs)
    clf = {}
    for c in trained:
        y = np.array([c in l for l in labels])
        if y.sum() >= 10:
            clf[c] = LogisticRegression(C=10, max_iter=1000).fit(X, y)

    def decode(texts_):
        Z = vec.transform(texts_)
        return [{c for c, m in clf.items() if m.predict_proba(Z[k])[0, 1] > 0.5} for k in range(len(texts_))]
    return decode


def prf(pred, gold):
    tp = sum(len(p & g) for p, g in zip(pred, gold))
    fp = sum(len(p - g) for p, g in zip(pred, gold))
    fn = sum(len(g - p) for p, g in zip(pred, gold))
    p, r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return dict(precision=p, recall=r, f1=2 * p * r / max(p + r, 1e-9), exact_match=float(np.mean([a == b for a, b in zip(pred, gold)])))


if __name__ == "__main__":
    main()
