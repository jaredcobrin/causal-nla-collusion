"""Pod. Build an RL/eval position pool from single sentences (raw text, no speaker prefix, no chat template; activation at the LAST token, layer 20 = hidden_states[21]) - same convention as the 50-sentence test.
   python build_short.py <in.jsonl with {"text":...,"id":...}> <out_prefix>
Writes {out}_acts.pt, {out}_prefix.pt, {out}_meta.jsonl (id, trained_active from the repo's concept probes). Labels/types stay ONLY in the input file's other fields (not written to meta)."""
import json, sys, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
NAME = "Qwen/Qwen2.5-7B-Instruct"; L = 20
inp, out = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open(inp)]
AD = "/root/cv_repo/runs/general/av_data"
th = json.load(open(f"{AD}/thresholds_global10.json")); hidden = {"music", "law", "mental_health", "vehicles", "hedging", "german"}
names = [c for c, v in th.items() if "t_high" in v and c not in hidden]
U = np.stack([np.load(f"{AD}/dir_{c}.npy") for c in names]).astype(np.float32); hi = np.array([th[c]["t_high"] for c in names])
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForCausalLM.from_pretrained(NAME, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
acts, prefixes, meta = [], [], []
with torch.no_grad():
    for k, r in enumerate(rows):
        ids = tok(r["text"], add_special_tokens=False).input_ids[-192:]
        h = model.model(input_ids=torch.tensor([ids], device="cuda"), output_hidden_states=True).hidden_states[L + 1][0, -1].float().cpu()
        if float(h.norm()) > 300: continue
        s = h.numpy() @ U.T; act = [names[i] for i in np.flatnonzero(s >= hi)]
        acts.append(h.half()); prefixes.append(torch.tensor(ids, dtype=torch.int32)); meta.append({"id": r["id"], "trained_active": act})
torch.save(torch.stack(acts), f"{out}_acts.pt"); torch.save(prefixes, f"{out}_prefix.pt")
with open(f"{out}_meta.jsonl", "w") as f:
    for m in meta: f.write(json.dumps(m) + "\n")
print(out, "positions", len(acts), "| with >=1 trained concept active:", sum(1 for m in meta if m["trained_active"]))
