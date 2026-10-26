"""Run the OFFICIAL kitft Qwen2.5-7B L20 NLA (AV greedy + AR cosine) on activations from our experiments.
Activations are HF hidden_states[21] at the last token (= residual output of block 20), same as the official extraction."""
import sys, json, glob, random, torch, re, yaml
sys.path.insert(0, "/root/official_nla/nla-inference")
from nla_inference import inject_at_marked_positions, normalize_activation, NLACritic
from transformers import AutoTokenizer, AutoModelForCausalLM
R="/root/official_nla"; I=R+"/inputs/root"
AV=glob.glob(R+"/hf/hub/models--kitft--nla-qwen2.5-7b-L20-av/snapshots/*")[0]; AR=glob.glob(R+"/hf/hub/models--kitft--nla-qwen2.5-7b-L20-ar/snapshots/*")[0]
meta=yaml.safe_load(open(AV+"/nla_meta.yaml")); tk=meta["tokens"]; scale=meta["extraction"]["injection_scale"]
tok=AutoTokenizer.from_pretrained(AV); model=AutoModelForCausalLM.from_pretrained(AV,dtype=torch.bfloat16).cuda().eval()
content=meta["prompt_templates"]["av"].format(injection_char=tk["injection_char"])
_t=tok.apply_chat_template([{"role":"user","content":content}],tokenize=False,add_generation_prompt=True); ids=tok(_t,return_tensors="pt",add_special_tokens=False)["input_ids"]
if not torch.is_tensor(ids) or ids.dim()!=2: ids=torch.as_tensor(ids)
emb=model.get_input_embeddings()
@torch.no_grad()
def gen(V, bs=24, max_new=200):
    outs=[]
    for s in range(0,len(V),bs):
        E=[]
        for v in V[s:s+bs]:
            v=v.float().view(1,-1).cuda(); vs=normalize_activation(v,scale)
            e=emb(ids.cuda()).float(); e=inject_at_marked_positions(ids.cuda(),e,vs,tk["injection_token_id"],tk["injection_left_neighbor_id"],tk["injection_right_neighbor_id"])
            E.append(e.to(torch.bfloat16))
        E=torch.cat(E); m=torch.ones(E.shape[:2],dtype=torch.long,device="cuda")
        o=model.generate(inputs_embeds=E,attention_mask=m,max_new_tokens=max_new,do_sample=False,pad_token_id=tok.pad_token_id or tok.eos_token_id)
        outs+=tok.batch_decode(o,skip_special_tokens=True)
        print("gen",s+len(E),"/",len(V),flush=True)
    return outs
def expl(t):
    m=re.search(r"<explanation>(.*?)(</explanation>|$)",t,re.S); return (m.group(1) if m else t).strip()
sets={}
sets["short"]=torch.load(I+"/cc/short_acts.pt").float()
sets["blt"]=torch.load(I+"/cc/pool_blt_acts.pt").float()
sets["bev"]=torch.load(I+"/cc/poolB_eval_acts.pt").float()
A=torch.load(I+"/data/acts_replay.pt").float(); M=[json.loads(l) for l in open(I+"/data/acts_meta.jsonl")]
rng=random.Random(0); ci=[i for i,m in enumerate(M) if m["family"]=="core" and m["role"]=="colluder"]; hi=[i for i,m in enumerate(M) if m["family"]=="core" and m["role"]=="honest"]
pick=sorted(rng.sample(ci,300)+rng.sample(hi,300)); json.dump(pick,open(R+"/outputs/core_pick.json","w")); sets["core"]=A[pick]
res={}
for name,V in sets.items():
    raw=gen(V); res[name]={"raw":raw,"text":[expl(t) for t in raw]}
json.dump({k:{"text":v["text"],"raw":v["raw"]} for k,v in res.items()},open(R+"/outputs/av_outputs.json","w"))
del model; torch.cuda.empty_cache()
crit=NLACritic(AR,device="cuda")
cos={}
for name,V in sets.items():
    cos[name]=[crit.score(t,v)[1] for t,v in zip(res[name]["text"],V)]
json.dump(cos,open(R+"/outputs/ar_cos.json","w")); print("ALL DONE")
