"""Does the (frozen) AR map a bare word to the right direction? For single words/phrases, the AR's rebuilt activation is projected
on each concept direction; reported as a z-score against all the phrases tried. Shows what words the RL reward can pay for."""
import glob, json, os, sys
import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from cv.av_sft import NAME

PH = {"music": ["music", "songs and music", "Music, in passing."], "law": ["law and legal matters", "legal", "Law."],
      "mental_health": ["mental health", "anxiety and depression", "Mental health."], "vehicles": ["cars and vehicles", "vehicles", "Cars."],
      "hedging": ["hedged and uncertain wording", "hedging", "Maybe, perhaps."], "german": ["German", "the German language", "German, a language other than English."],
      "weather": ["weather", "weather and rain", "Weather."], "food": ["food and cooking", "food", "Cooking."], "sports": ["sports", "football and sports", "Sports."],
      "religion": ["religion", "faith and belief", "Religion."], "cryptocurrency": ["cryptocurrency", "bitcoin and blockchain", "Crypto."],
      "japanese": ["Japanese", "the Japanese language", "Japanese."], "arabic": ["Arabic", "the Arabic language", "Arabic."]}
ar = sys.argv[1] if len(sys.argv) > 1 else "runs/general/th_ar"
tok = AutoTokenizer.from_pretrained(NAME); tok.padding_side = "right"
base = AutoModelForCausalLM.from_pretrained(NAME, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
model = PeftModel.from_pretrained(base, ar + "/lora").eval()
head = torch.nn.Linear(base.config.hidden_size, base.config.hidden_size).cuda().float(); head.load_state_dict(torch.load(ar + "/head.pt"))
nm = torch.load(ar + "/norm.pt"); mu, sd = nm["mu"].cuda(), nm["sd"].cuda()
dirs = {os.path.basename(f)[4:-4]: np.load(f) for f in glob.glob("runs/general/av_data/dir_*.npy")}
texts, owner = [], []
for c, ps in PH.items():
    for p in ps: texts.append(p); owner.append(c)
msgs = [tok.apply_chat_template([{"role": "user", "content": "Reconstruct the model's internal state from this description:\n" + t}], tokenize=False, add_generation_prompt=True) for t in texts]
with torch.no_grad():
    b = tok(msgs, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    h = model(**b, output_hidden_states=True).hidden_states[-1]
    last = b["attention_mask"].sum(1) - 1
    pred = head(h[torch.arange(len(texts)), last].float())
raw = (pred * sd + mu).cpu().numpy()
P = {c: raw @ dirs[c] for c in PH}
print("z-score of the AR's output along each concept's own direction, for phrases that name it (0 = like the average phrase; held-out concepts first):")
for c in PH:
    z = (P[c] - P[c].mean()) / (P[c].std() + 1e-9)
    mine = [z[i] for i in range(len(texts)) if owner[i] == c]
    rank = np.mean([(z > zz).sum() for zz in mine])
    print(f"  {c:14} {'HELD-OUT' if c in ('music','law','mental_health','vehicles','hedging','german') else 'trained ':8} mean z {np.mean(mine):+.2f}   (its phrases rank {rank:.1f} of {len(texts)} on this direction; 0 = top)")
