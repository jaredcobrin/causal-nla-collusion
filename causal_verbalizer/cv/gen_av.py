"""Generate the AV's sentence for every position that has a description (greedy), from a saved LoRA.

    python -m cv.gen_av --av runs/general/th_av --out runs/general/gen_sft.jsonl
"""
import argparse
import json

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from cv.av_sft import NAME, PROMPTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--av", required=True, help="run dir containing lora/")
    ap.add_argument("--data", default="runs/general/av_data/enriched_sft_global10.jsonl")
    ap.add_argument("--acts", default="runs/general/av_data/enriched_acts_global10.pt")
    ap.add_argument("--descriptions", default="runs/general/av_data/th_descriptions.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_new", type=int, default=60)
    a = ap.parse_args()
    ids = [json.loads(l)["id"] for l in open(a.descriptions) if l.strip()]
    acts = torch.load(a.acts).float()
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
    with open(a.out, "w") as f, torch.no_grad():
        for k in range(0, len(ids), 48):
            b = ids[k: k + 48]
            inj["act"] = torch.stack([acts[int(i[2:])] for i in b]).cuda()
            x = torch.tensor([prompt] * len(b), device="cuda")
            g = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=a.max_new, do_sample=False, pad_token_id=tok.pad_token_id)
            for i, t in zip(b, tok.batch_decode(g[:, x.shape[1]:], skip_special_tokens=True)):
                f.write(json.dumps(dict(id=i, text=t.strip())) + "\n")
            print(k, flush=True)


if __name__ == "__main__":
    main()
