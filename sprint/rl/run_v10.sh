#!/bin/bash
# v10: mixed pool (blatant long transcripts + short sentences, unlabeled), 150 steps from base SFT, stronger KL (beta 0.15), lower lr (1e-5), checkpoints every 25.
cd /root/cc; export HF_HOME=/root/hf_cache PYTHONPATH=/root/cv_repo PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/vllm_env/bin/python; G=/root/cv_repo/runs/general; D=/root/cc
$PY - <<'PYEOF'
import torch, json
D="/root/cc"
A=torch.cat([torch.load(f"{D}/pool_bl_acts.pt"),torch.load(f"{D}/pool_short_acts.pt")]); P=torch.load(f"{D}/pool_bl_prefix.pt")+torch.load(f"{D}/pool_short_prefix.pt")
M=[json.loads(l) for l in open(f"{D}/pool_bl_meta.jsonl")]+[json.loads(l) for l in open(f"{D}/pool_short_meta.jsonl")]
torch.save(A,f"{D}/pool_mix_acts.pt"); torch.save(P,f"{D}/pool_mix_prefix.pt")
open(f"{D}/pool_mix_meta.jsonl","w").write("\n".join(json.dumps(m) for m in M)+"\n"); print("mix",len(A),len(P),len(M))
PYEOF
sed 's/step % 50 == 0 and step < a.steps/step % 25 == 0 and step < a.steps/' /root/overnight/rl2_v2.py > /root/overnight/rl2_v3.py
S=$G/startSFT; rm -rf $G/rl3_mix
COMMON="--data $G/av_data/enriched_sft_global10.jsonl --acts $G/av_data/enriched_acts_global10.pt --descriptions $G/av_data/th_descriptions.jsonl \
 --ar $G/th2_ar --dirs $G/av_data --drop_ids $G/av_data/strict_drop_ids.jsonl --val_ids $G/av_data/th_val_ids.jsonl \
 --steps 150 --eval_every 25 --eval_n 64 --max_new 60 --seed 11 --av $S --ref $S --mix_orig 0.4 --mix_fresh $D/poolB_train --lr 1e-5 \
 --w_left 0.4 --w_cons 0.25 --w_behav 0.15 --w_concept 0.2 --beta 0.15 --w_div 0.5 --div_thr 0.5 --w_empty 0.3 --no_concept_fresh --max_kl 0.8"
cd /root/cv_repo
$PY /root/overnight/rl2_v3.py $COMMON --fresh $D/pool_mix --div_acts /root/data/acts_replay.pt --out $G/rl3_mix > /root/overnight/logs_rl3_mix.log 2>&1
echo "RL DONE" >> /root/overnight/logs_rl3_mix.log
