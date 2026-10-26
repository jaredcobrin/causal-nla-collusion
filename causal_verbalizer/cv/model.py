"""Qwen wrapper: chat rendering, batched forward passes that return activations and per-block
writes, generation, and neutral-text loss, all under optional hook edits."""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cv.hooks import hooked


class LM:
    def __init__(self, name: str, device: str = "cuda", dtype=torch.bfloat16):
        self.tok = AutoTokenizer.from_pretrained(name)
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, attn_implementation="sdpa")
        self.model.to(device).eval()
        self.blocks = self.model.model.layers
        self.n_layers, self.d, self.device = len(self.blocks), self.model.config.hidden_size, device
        suffix = self.render([{"role": "user", "content": "X"}]).split("X")[-1]
        self.inst_end = -(len(self.tok.encode(suffix, add_special_tokens=False)) + 1)  # last instruction token
        self.forwarded = 0  # items run through the model (reported as data/compute usage)
        self.generated = 0

    def render(self, msgs: list[dict], prefix: str = "", system: str | None = None) -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + msgs
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + prefix

    def _batches(self, texts: list[str], bs: int):
        """Length-sorted batches (less padding); yields (original indices, encoded batch)."""
        lens = [len(self.tok.encode(t, add_special_tokens=False)) for t in texts]
        order = np.argsort(lens)[::-1]
        for i in range(0, len(order), bs):
            idx = [int(j) for j in order[i:i + bs]]
            enc = self.tok([texts[j] for j in idx], return_tensors="pt", padding=True, add_special_tokens=False)
            yield idx, enc.to(self.device)

    @torch.no_grad()
    def run(self, texts, *, edits=None, read_layers=(), pos="last", resp_start=None, write_layers=(), bs=32) -> dict:
        """One forward pass per text.
        read_layers: block outputs at `pos` ("last" / "inst_end" / per-text list) or, with
                 resp_start (token index where the response starts), averaged over the response.
        write_layers: what each block added to the residual (output - input) at the last token.
        """
        n = len(texts)
        acts = {l: torch.zeros(n, self.d) for l in read_layers}
        writes = {l: torch.zeros(n, self.d) for l in write_layers}
        for idx, enc in self._batches(texts, bs):
            T = enc.input_ids.shape[1]
            pad = T - enc.attention_mask.sum(1)

            def sel(h):
                if resp_start is not None:
                    start = pad + torch.tensor([resp_start[j] for j in idx], device=h.device)
                    mask = (torch.arange(T, device=h.device)[None] >= start[:, None]).float()
                    return (h.float() * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
                p = [pos if isinstance(pos, str) else pos[j] for j in idx]
                p = [-1 if q == "last" else self.inst_end if q == "inst_end" else q for q in p]
                return h.float()[torch.arange(len(idx)), torch.tensor(p, device=h.device) + T]

            def reader(l):
                def f(inp, h):
                    if l in acts:
                        acts[l][idx] = sel(h).cpu()
                    if l in writes:
                        writes[l][idx] = (h.float() - inp.float())[:, -1].cpu()
                return f

            readers = {l: reader(l) for l in {*read_layers, *write_layers}}
            with hooked(self.blocks, edits, readers):
                self.model.model(**enc)  # residual stream only; no need for logits
        self.forwarded += n
        return {"acts": acts, "writes": writes}

    @torch.no_grad()
    def generate(self, texts, *, edits=None, max_new_tokens=256, temperature=1.0, seed=0, bs=48) -> list[str]:
        """temperature 0 = greedy (the model's most likely reply); otherwise plain sampling. Qwen's
        default generation config (top_k=20, top_p=0.8, repetition_penalty=1.05) is overridden."""
        sampling = dict(do_sample=True, temperature=temperature, top_p=1.0, top_k=0) if temperature > 0 else dict(do_sample=False)
        res = [None] * len(texts)
        for k, (idx, enc) in enumerate(self._batches(texts, bs)):
            torch.manual_seed(seed + k)
            with hooked(self.blocks, edits):
                ids = self.model.generate(**enc, **sampling, repetition_penalty=1.0, max_new_tokens=max_new_tokens,
                                          pad_token_id=self.tok.pad_token_id)
            for b, j in enumerate(idx):
                res[j] = self.tok.decode(ids[b, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
        self.generated += len(texts)
        return res

    @torch.no_grad()
    def yes_no(self, texts, bs=16):
        """Next-token P(Yes) among {Yes, No}, and the total probability of those two tokens."""
        ids = [self.tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", "No")]
        p_yes, mass = np.zeros(len(texts)), np.zeros(len(texts))
        for idx, enc in self._batches(texts, bs):
            lp = torch.log_softmax(self.model(**enc, logits_to_keep=1).logits[:, -1].float(), -1)
            y, n = lp[:, ids[0]].exp().cpu().numpy(), lp[:, ids[1]].exp().cpu().numpy()
            p_yes[idx], mass[idx] = y / (y + n + 1e-12), y + n
        self.forwarded += len(texts)
        return p_yes, mass

    @torch.no_grad()
    def choice_logits(self, texts, ids, *, edits=None, bs=32):
        """Next-token log P(concept answer) - log P(other answers), per text under optional edits.
        ids[i] = [concept token, other token(s)...]; several other tokens are pooled (log-sum-exp).
        Also returns the probability mass all of these tokens hold (how much of the model's own
        next-token choice they cover)."""
        ids = np.asarray(ids)
        diff, mass = np.zeros(len(texts)), np.zeros(len(texts))
        for idx, enc in self._batches(texts, bs):
            with hooked(self.blocks, edits):
                lp = torch.log_softmax(self.model(**enc, logits_to_keep=1).logits[:, -1].float(), -1)
            sel = lp.gather(1, torch.tensor(ids[idx], device=lp.device))
            c, o = sel[:, 0], torch.logsumexp(sel[:, 1:], 1)
            diff[idx], mass[idx] = (c - o).cpu().numpy(), sel.exp().sum(1).cpu().numpy()
        self.forwarded += len(texts)
        return diff, mass

    @torch.no_grad()
    def next_token(self, texts, ids, *, edits=None, top=3, bs=32):
        """Next-token probability of each token id in `ids`, [len(texts), len(ids)], and the `top` most
        likely tokens as (text, probability) per text, under optional edits."""
        p, tops = np.zeros((len(texts), len(ids))), [None] * len(texts)
        for idx, enc in self._batches(texts, bs):
            with hooked(self.blocks, edits):
                pr = torch.softmax(self.model(**enc, logits_to_keep=1).logits[:, -1].float(), -1)
            p[idx] = pr[:, ids].cpu().numpy()
            tp, ti = pr.topk(top, -1)
            for b, j in enumerate(idx):
                tops[j] = [(self.tok.decode(int(t)), round(float(q), 4)) for t, q in zip(ti[b], tp[b])]
        self.forwarded += len(texts)
        return p, tops

    @torch.no_grad()
    def nll(self, ids_list: list[list[int]], *, edits=None, bs=8) -> float:
        """Mean next-token loss (nats/token) on raw token sequences of equal length (neutral text)."""
        total, count = 0.0, 0
        for i in range(0, len(ids_list), bs):
            x = torch.tensor(ids_list[i:i + bs], device=self.device)
            with hooked(self.blocks, edits):
                logits = self.model(input_ids=x).logits[:, :-1].float()
            loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), x[:, 1:].flatten(), reduction="sum")
            total, count = total + loss.item(), count + x[:, 1:].numel()
        self.forwarded += len(ids_list)
        return total / count
