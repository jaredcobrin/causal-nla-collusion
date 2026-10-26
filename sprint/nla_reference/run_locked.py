import sys,re,json,torch
src=open("/root/official_nla/run_official.py").read().split("sets={}")[0]
exec(src)
V=torch.load("/root/official_nla/inputs/locked_acts.pt").float()
raw=gen(V); txt=[expl(t) for t in raw]
json.dump({"text":txt,"raw":raw},open("/root/official_nla/outputs/av_locked.json","w"))
del model; torch.cuda.empty_cache()
crit=NLACritic(AR,device="cuda"); cos=[crit.score(t,v)[1] for t,v in zip(txt,V)]
json.dump(cos,open("/root/official_nla/outputs/ar_cos_locked.json","w")); print("DONE",len(txt))
