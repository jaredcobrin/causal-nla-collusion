"""RL for the activation verbalizer: AV samples descriptions, a frozen AR reads a paraphrase of each, reward = how well the
rebuilt activation matches the real one on the part the trained concept directions do not explain, plus consistency along
those directions. GRPO-style: several samples per activation, advantage = reward minus the group mean, KL to the SFT AV.

One Qwen base carries three LoRA adapters: "av" (trained), "ref" (frozen copy of the SFT AV, for the KL) and "ar"
(frozen reconstructor). The paraphraser is the base model with adapters disabled.

    python -m cv.rl --data ... --acts ... --descriptions ... --av runs/general/av_prose_mono --ar runs/general/ar_text --out runs/general/rl_mono
"""
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from cv.ar import load_dirs
from cv.av_sft import NAME, PROMPTS, fit_decoder, mentions, prf

PARAPHRASE = "Rewrite the following in different words, keeping the same meaning and the same level of detail. Output only the rewrite.\n\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--acts", required=True)
    ap.add_argument("--descriptions", required=True)
    ap.add_argument("--av", required=True, help="AV SFT run dir (lora/ + heldout_generations.jsonl)")
    ap.add_argument("--ref", help="AV run dir for the KL reference (default: --av); use the SFT AV when resuming from an RL adapter")
    ap.add_argument("--ar", required=True, help="AR run dir (lora/, head.pt, norm.pt)")
    ap.add_argument("--dirs", default="runs/general/av_data")
    ap.add_argument("--fresh", help="prefix of fresh activations (cv.fresh_acts output): RL trains on these instead of the SFT positions")
    ap.add_argument("--drop_ids", help="jsonl of extra ids dropped from the SFT-position pool")
    ap.add_argument("--val_ids", help="jsonl of ids kept out of RL training and evaluated as a second held-out set (seen datasets, unseen texts)")
    ap.add_argument("--hidden", default="music,law,mental_health,vehicles,hedging,german")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--prompts", type=int, default=8, help="activations per step")
    ap.add_argument("--group", type=int, default=4, help="samples per activation")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--max_new", type=int, default=140)
    ap.add_argument("--w_left", type=float, default=0.6)
    ap.add_argument("--w_cons", type=float, default=0.4)
    ap.add_argument("--w_concept", type=float, default=0.0, help="weight of the concept-agreement term (F1 between the concepts the decoder reads in the sampled sentence and the trained concepts the probes say are active; needs --fresh meta with trained_active)")
    ap.add_argument("--w_behav", type=float, default=0.0, help="weight of the behavioural term (needs --fresh with a *_prefix.pt)")
    ap.add_argument("--eval_every", type=int, default=30)
    ap.add_argument("--eval_n", type=int, default=64)
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
    held = [json.loads(l)["id"] for l in open(f"{a.av}/heldout_generations.jsonl")]
    ids = [i for i in rows if i in desc]
    act = {i: acts_all[int(i[2:])] for i in ids}
    va = [i for i in held if i in act]
    val = [json.loads(l)["id"] for l in open(a.val_ids)] if a.val_ids else []
    dropped = {json.loads(l)["id"] for l in open(a.drop_ids)} if a.drop_ids else set()
    tr = [i for i in ids if i not in set(held) and i not in set(val) and i not in dropped and float(act[i].norm()) < 300
          and not (set(rows[i]["active"]) & set(hidden)) and not mentions(desc[i], hidden)]
    pool = tr
    prefix, fresh_active = {}, {}
    if a.fresh:
        fa = torch.load(a.fresh + "_acts.pt").float()
        fids = [json.loads(l)["id"] for l in open(a.fresh + "_meta.jsonl")]
        for k, i in enumerate(fids):
            act[i] = fa[k]
        pool = fids
        fresh_active = {}
        for l in open(a.fresh + "_meta.jsonl"):
            m = json.loads(l)
            fresh_active[m["id"]] = set(m.get("trained_active", []))
        pf = a.fresh + "_prefix.pt"
        if os.path.exists(pf):
            pl = torch.load(pf)
            prefix = {i: pl[k] for k, i in enumerate(fids)}
        print(f"RL pool: {len(pool)} fresh positions (SFT positions are used only for the decoder and concept list)", flush=True)
    trained = sorted({c for i in tr for c in rows[i]["active"]})
    norm = torch.load(f"{a.ar}/norm.pt")
    mu, sd = norm["mu"].cuda(), norm["sd"].cuda()
    dirs = load_dirs(a.dirs)
    V = torch.tensor(np.stack([dirs[c] for c in trained]), dtype=torch.float32).cuda()  # trained concept directions
    Q = torch.linalg.qr(V.T)[0]  # basis of their span
    print(f"RL positions {len(tr)} | held-out {len(va)} | trained concepts {len(trained)}", flush=True)

    tok = AutoTokenizer.from_pretrained(NAME)
    base = AutoModelForCausalLM.from_pretrained(NAME, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    model = PeftModel.from_pretrained(base, f"{a.av}/lora", adapter_name="av", is_trainable=True)
    model.load_adapter(f"{a.ref or a.av}/lora", adapter_name="ref", is_trainable=False)
    model.load_adapter(f"{a.ar}/lora", adapter_name="ar", is_trainable=False)
    for n, p in model.named_parameters():
        p.requires_grad = ".av." in n and "lora_" in n
    params = [p for p in model.parameters() if p.requires_grad]
    head = torch.nn.Linear(base.config.hidden_size, base.config.hidden_size).cuda().float()
    head.load_state_dict(torch.load(f"{a.ar}/head.pt"))

    pre = tok.apply_chat_template([{"role": "user", "content": "@@"}], tokenize=False, add_generation_prompt=True)
    head_s, tail = pre.split("@@")
    pre_ids = tok(head_s + " ?", add_special_tokens=False).input_ids
    pos = len(pre_ids) - 1
    prompt_ids = pre_ids + tok(PROMPTS["prose"] + tail, add_special_tokens=False).input_ids
    inj = {"act": None}

    def hook(_, __, output):
        h = output[0] if isinstance(output, tuple) else output
        if inj["act"] is None or h.shape[1] <= 1:
            return output
        h = h.clone()
        h[:, pos] = inj["act"].to(h.dtype)
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

    model.base_model.model.model.layers[20].register_forward_hook(hook)

    patch = {"pos": None, "act": None}

    def patch_hook(_, __, output):
        if patch["act"] is None:
            return output
        h = output[0] if isinstance(output, tuple) else output
        h = h.clone()
        h[torch.arange(len(patch["pos"])), patch["pos"]] = patch["act"].to(h.dtype)
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

    model.base_model.model.model.layers[20].register_forward_hook(patch_hook)

    @torch.no_grad()
    def next_logp(ids_, acts_):
        """Log-probs of the next token at each position's last token, optionally with the layer-20 activation replaced."""
        outs = []
        for k in range(0, len(ids_), 16):
            seqs = [prefix[i].long() for i in ids_[k: k + 16]]
            L = max(len(x) for x in seqs)
            X = torch.full((len(seqs), L), tok.pad_token_id, dtype=torch.long, device="cuda")
            M = torch.zeros_like(X)
            for b, x in enumerate(seqs):
                X[b, : len(x)] = x.cuda()
                M[b, : len(x)] = 1
            last = torch.tensor([len(x) - 1 for x in seqs], device="cuda")
            if acts_ is not None:
                patch["pos"], patch["act"] = last, acts_[k: k + 16].cuda()
            with model.disable_adapter():
                lg = model(input_ids=X, attention_mask=M).logits[torch.arange(len(seqs)), last].float()
            patch["act"] = None
            outs.append(torch.log_softmax(lg, -1))
        return torch.cat(outs)

    beh_cache = {}

    @torch.no_grad()
    def behav(pred, ids_):
        """1 - KL(true next-token distribution || distribution with the rebuilt activation) / the same KL with the mean activation."""
        need = [i for i in dict.fromkeys(ids_) if i not in beh_cache]
        if need:
            lt = next_logp(need, None)
            lm_ = next_logp(need, mu.expand(len(need), -1))
            for n_, i in enumerate(need):
                beh_cache[i] = (lt[n_].half().cpu(), float((lt[n_].exp() * (lt[n_] - lm_[n_])).sum()))
        lt = torch.stack([beh_cache[i][0].cuda().float() for i in ids_])
        klm = torch.tensor([beh_cache[i][1] for i in ids_], device="cuda")
        lh = next_logp(ids_, pred * sd + mu)
        kl = (lt.exp() * (lt - lh)).sum(-1)
        return (1 - kl / klm.clamp(min=1e-3)).clamp(-1, 1)

    if prefix and a.w_behav > 0:
        t_ids = pool[:8]
        lt = next_logp(t_ids, None)
        lh = next_logp(t_ids, torch.stack([act[i] for i in t_ids]))
        print("behavioural self-test, KL with the TRUE activation patched in (should be ~0):", [round(float(x), 4) for x in (lt.exp() * (lt - lh)).sum(-1)], flush=True)

    @torch.no_grad()
    def sample(act_batch, n, greedy=False):
        """n samples per activation; returns token id lists (generated part only)."""
        model.set_adapter("av")
        model.eval()
        inj["act"] = act_batch.repeat_interleave(n, 0).cuda()
        x = torch.tensor([prompt_ids] * (len(act_batch) * n), device="cuda")
        kw = dict(do_sample=False) if greedy else dict(do_sample=True, temperature=1.0, top_p=0.95)
        g = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=a.max_new, pad_token_id=tok.pad_token_id, **kw)[:, x.shape[1]:]
        res = []
        for row in g:
            t = row.tolist()
            e = [k for k, v in enumerate(t) if v in (tok.eos_token_id, tok.pad_token_id, 151645)]
            res.append(t[: e[0] + 1] if e else t)  # keep the end token so its probability is trained
        inj["act"] = None
        return res

    @torch.no_grad()
    def paraphrase(texts):
        out_t = []
        for k in range(0, len(texts), 16):
            msgs = [tok.apply_chat_template([{"role": "user", "content": PARAPHRASE + t}], tokenize=False, add_generation_prompt=True) for t in texts[k: k + 16]]
            tok.padding_side = "left"
            b = tok(msgs, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            with model.disable_adapter():
                g = model.generate(**b, max_new_tokens=a.max_new + 40, do_sample=False, pad_token_id=tok.pad_token_id)[:, b["input_ids"].shape[1]:]
            out_t += tok.batch_decode(g, skip_special_tokens=True)
        return out_t

    @torch.no_grad()
    def reconstruct(texts):
        model.set_adapter("ar")
        model.eval()
        tok.padding_side = "right"
        preds = []
        for k in range(0, len(texts), 16):
            msgs = [tok.apply_chat_template([{"role": "user", "content": "Reconstruct the model's internal state from this description:\n" + t}], tokenize=False, add_generation_prompt=True) for t in texts[k: k + 16]]
            b = tok(msgs, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            h = model(**b, output_hidden_states=True).hidden_states[-1]
            last = b["attention_mask"].sum(1) - 1
            preds.append(head(h[torch.arange(len(msgs)), last].float()))
        return torch.cat(preds)

    def metrics(pred, ids_):
        y = torch.stack([(act[i].cuda() - mu) / sd for i in ids_])
        left = lambda M: M - (M @ Q) @ Q.T
        c_left = F.cosine_similarity(left(pred), left(y), dim=-1)
        c_cons = F.cosine_similarity(pred @ V.T, y @ V.T, dim=-1)
        return c_left, c_cons

    def logprobs(seqs, adapter, grad):
        model.set_adapter(adapter)
        model.eval()
        L = max(len(s) for s in seqs)
        X = torch.full((len(seqs), len(prompt_ids) + L), tok.pad_token_id, dtype=torch.long, device="cuda")
        M = torch.zeros_like(X)
        G = torch.zeros(X.shape, dtype=torch.bool, device="cuda")
        for b, s in enumerate(seqs):
            X[b, : len(prompt_ids) + len(s)] = torch.tensor(prompt_ids + s)
            M[b, : len(prompt_ids) + len(s)] = 1
            G[b, len(prompt_ids): len(prompt_ids) + len(s)] = True
        with torch.set_grad_enabled(grad):
            lg = model(input_ids=X, attention_mask=M).logits[:, :-1].float()
            lp = torch.log_softmax(lg, -1).gather(-1, X[:, 1:, None])[..., 0]
        return lp, G[:, 1:]

    dec = fit_decoder([desc[i] for i in tr], [rows[i]["active"] for i in tr], trained)

    def evaluate(step):
        evaluate_set(step, "test", va[: a.eval_n])
        if val:
            evaluate_set(step, "val", val[: a.eval_n])

    def evaluate_set(step, name, sub):
        gens_ids = sample(torch.stack([act[i] for i in sub]), 1, greedy=True)
        texts = [tok.decode(t, skip_special_tokens=True) for t in gens_ids]
        pred_raw = reconstruct(texts)
        pred_par = reconstruct(paraphrase(texts))
        y = torch.stack([(act[i].cuda() - mu) / sd for i in sub])
        res = dict(step=step, n=len(sub), split=name)
        for tag, p in (("raw", pred_raw), ("paraphrased", pred_par)):
            cl, cc = metrics(p, sub)
            P, Y = p.cpu().numpy(), y.cpu().numpy()
            corr = {}
            for c, v in dirs.items():
                pp, yy = P @ v, Y @ v
                corr[c] = float(np.corrcoef(pp, yy)[0, 1]) if pp.std() > 0 and yy.std() > 0 else np.nan
            g = lambda cs: float(np.nanmean([corr[c] for c in cs if c in corr]))
            if tag == "raw":  # per-row projections on the hidden directions, for bootstrap intervals
                res["proj_hidden"] = {c: [(P @ dirs[c]).tolist(), (Y @ dirs[c]).tolist()] for c in hidden}
            res[tag] = dict(cosine=float(F.cosine_similarity(p, y, dim=-1).mean()), leftover=float(cl.mean()), consistency=float(cc.mean()),
                            corr_trained=g(trained), corr_hidden=g(hidden), corr_unused=g([c for c in dirs if c not in trained and c not in hidden]))
        with open(out / f"texts_{name}_{step}.jsonl", "w") as f:
            for i, t in zip(sub, texts):
                f.write(json.dumps(dict(id=i, text=t)) + "\n")
        res["decoder_f1"] = prf(dec(texts), [set(rows[i]["active"]) & set(trained) for i in sub])["f1"]
        res["mean_tokens"] = float(np.mean([len(t) for t in gens_ids]))
        res["sample"] = texts[:3]
        print("EVAL", json.dumps(res), flush=True)
        with open(out / f"eval_{name}.jsonl", "a") as f:
            f.write(json.dumps(res) + "\n")

    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    evaluate(0)
    for step in range(1, a.steps + 1):
        batch = rng.sample(pool, a.prompts)
        a_b = torch.stack([act[i] for i in batch])
        seqs = sample(a_b, a.group)
        texts = [tok.decode(s, skip_special_tokens=True) for s in seqs]
        pred = reconstruct(paraphrase(texts))
        rep_ids = [i for i in batch for _ in range(a.group)]
        c_left, c_cons = metrics(pred, rep_ids)
        reward = a.w_left * c_left + a.w_cons * c_cons
        c_con = None
        if a.w_concept > 0 and fresh_active:
            read = dec(texts)
            tset = set(trained)
            def f1(p, g):
                g = g & tset
                return 1.0 if not p and not g else 2 * len(p & g) / (len(p) + len(g))
            c_con = torch.tensor([f1(read[n_], fresh_active[rep_ids[n_]]) for n_ in range(len(texts))], device="cuda")
            reward = reward + a.w_concept * c_con
        c_beh = None
        if a.w_behav > 0 and prefix:
            c_beh = behav(pred, rep_ids)
            reward = reward + a.w_behav * c_beh
        # reward the group's relative quality
        r = reward.view(a.prompts, a.group)
        adv = ((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-4)).view(-1)
        # policy update, one sequence micro-batch at a time
        inj["act"] = None
        opt.zero_grad()
        kl_tot = 0.0
        for k in range(0, len(seqs), 8):
            sl = slice(k, k + 8)
            inj["act"] = a_b.repeat_interleave(a.group, 0)[sl].cuda()
            # reference first: switching adapters toggles requires_grad, so the policy pass must be last before backward
            lp_ref, _ = logprobs(seqs[sl], "ref", False)
            lp, mask = logprobs(seqs[sl], "av", True)
            kl = (torch.exp(lp_ref - lp) - (lp_ref - lp) - 1)
            tok_loss = -(adv[sl, None].cuda() * lp) + a.beta * kl
            loss = (tok_loss * mask).sum(1) / mask.sum(1).clamp(min=1)
            (loss.sum() / len(seqs)).backward()
            kl_tot += float(((kl * mask).sum(1) / mask.sum(1).clamp(min=1)).sum())
        inj["act"] = None
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        print(f"step {step} reward {float(reward.mean()):.3f} (left {float(c_left.mean()):.3f}, cons {float(c_cons.mean()):.3f}{'' if c_beh is None else f', behav {float(c_beh.mean()):.3f}'}{'' if c_con is None else f', concept {float(c_con.mean()):.3f}'}) kl {kl_tot / len(seqs):.4f} tokens {np.mean([len(s) for s in seqs]):.0f}", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            evaluate(step)
    model.set_adapter("av")
    model.save_pretrained(out / "lora", selected_adapters=["av"])


if __name__ == "__main__":
    main()
