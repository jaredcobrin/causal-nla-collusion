"""Positions for RL / evaluation (pod). Raw-text format like the verbalizer's training data: text = cumulative transcript up to the end of message k
(no chat template), last <=192 tokens, activation at the LAST token, layer 20 (hidden_states[21]).  NO labels are read when writing pool files.
   python build_pool2.py concept_texts2.jsonl <split: train|test|val> <mode: all|noncollusive> <max_positions_per_transcript> <out_prefix>
Writes {out}_acts.pt, {out}_prefix.pt (token ids, for the behavioural reward), {out}_meta.jsonl (id, trained_active from the repo's concept probes; label/version kept ONLY in {out}_truth.jsonl for evaluation)."""
import json, sys, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
NAME = "Qwen/Qwen2.5-7B-Instruct"; L = 20
inp, split, mode, maxp, out = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
rows = [json.loads(l) for l in open(inp)]
rows = [r for r in rows if r["split"] == split and (mode == "all" or r["label"] == 0)]
AD = "/root/cv_repo/runs/general/av_data"
th = json.load(open(f"{AD}/thresholds_global10.json")); hidden = {"music", "law", "mental_health", "vehicles", "hedging", "german"}
names = [c for c, v in th.items() if "t_high" in v and c not in hidden]
U = np.stack([np.load(f"{AD}/dir_{c}.npy") for c in names]).astype(np.float32); hi = np.array([th[c]["t_high"] for c in names])
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForCausalLM.from_pretrained(NAME, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
acts, prefixes, meta, truth = [], [], [], []
with torch.no_grad():
    for k, r in enumerate(rows):
        n = len(r["msgs"]); pos = sorted(set(np.linspace(0, n - 1, min(maxp, n)).round().astype(int).tolist()))
        for j in pos:
            text = "\n".join(f"{m['speaker']}: {m['text']}" for m in r["msgs"][: j + 1])
            ids = tok(text, add_special_tokens=False).input_ids[-192:]
            h = model.model(input_ids=torch.tensor([ids], device="cuda"), output_hidden_states=True).hidden_states[L + 1][0, -1].float().cpu()
            if float(h.norm()) > 300: continue
            s = h.numpy() @ U.T; act = [names[i] for i in np.flatnonzero(s >= hi)]
            pid = f"{split[:2]}{len(acts):06d}"
            acts.append(h.half()); prefixes.append(torch.tensor(ids, dtype=torch.int32))
            meta.append({"id": pid, "trained_active": act}); truth.append({"id": pid, "tid": r["tid"], "group": r["group"], "version": r["version"], "label": r["label"], "msg_index": j, "n_msgs": n, "src": r["src"], "fmt": r["fmt"]})
        if k % 100 == 0: print(k, len(rows), len(acts), flush=True)
torch.save(torch.stack(acts), f"{out}_acts.pt"); torch.save(prefixes, f"{out}_prefix.pt")
with open(f"{out}_meta.jsonl", "w") as f:
    for m in meta: f.write(json.dumps(m) + "\n")
with open(f"{out}_truth.jsonl", "w") as f:
    for t in truth: f.write(json.dumps(t) + "\n")
print("positions", len(acts), "| with >=1 trained concept active:", sum(1 for m in meta if m["trained_active"]))
