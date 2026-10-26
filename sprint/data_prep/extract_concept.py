"""Layer-20 activations (hidden_states[21]) for concept_texts.jsonl, raw text (no chat template), last <=192 tokens window:
   acts_last.pt : activation at the final token;  acts_mean.pt : mean over the window.   (pod: python extract_concept.py in.jsonl out_prefix)"""
import json, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
NAME = "Qwen/Qwen2.5-7B-Instruct"; L = 20
rows = [json.loads(l) for l in open(sys.argv[1])]; out = sys.argv[2]
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForCausalLM.from_pretrained(NAME, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
last, mean = [], []
with torch.no_grad():
    for k, r in enumerate(rows):
        ids = tok(r["text"], add_special_tokens=False).input_ids[-192:]
        h = model.model(input_ids=torch.tensor([ids], device="cuda"), output_hidden_states=True).hidden_states[L + 1][0].float()
        last.append(h[-1].half().cpu()); mean.append(h.mean(0).half().cpu())
        if k % 200 == 0: print(k, len(rows), flush=True)
torch.save(torch.stack(last), f"{out}_last.pt"); torch.save(torch.stack(mean), f"{out}_mean.pt"); print("saved", len(rows))
