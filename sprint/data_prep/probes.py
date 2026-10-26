import json, numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
d=np.load("/root/cc/collusion_last.npy").astype(np.float32).ravel(); d/=np.linalg.norm(d)
A=torch.load("/root/data/acts_replay.pt").float().numpy(); M=[json.loads(l) for l in open("/root/data/acts_meta.jsonl")]
idx=[i for i,m in enumerate(M) if m["family"]=="core"]; y=np.array([1 if M[i]["role"]=="colluder" else 0 for i in idx])
sc=StandardScaler().fit(A[idx]); lr=LogisticRegression(C=0.01,max_iter=2000).fit(sc.transform(A[idx]),y)
print("core probe train acc",lr.score(sc.transform(A[idx]),y))
out={}
for nm,f in (("short","/root/cc/short_acts.pt"),("shortt","/root/cc/pool_shortt_acts.pt")):
    X=torch.load(f).float().numpy()
    out[nm]={"dir":(X@d).tolist(),"core":lr.decision_function(sc.transform(X)).tolist()}
json.dump(out,open("/root/cc/probe_scores.json","w")); print("ok")
