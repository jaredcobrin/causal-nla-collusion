import sys,re,json,torch
src=open("/root/official_nla/run_official.py").read().split("sets={}")[0]
exec(src)
V=torch.load("/root/official_nla/inputs/shortt_acts.pt").float()
raw=gen(V); txt=[expl(t) for t in raw]
json.dump({"text":txt,"raw":raw},open("/root/official_nla/outputs/av_shortt.json","w")); print("DONE",len(txt))
