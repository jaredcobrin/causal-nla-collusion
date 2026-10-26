#!/bin/bash
# v9: 100 steps from the base SFT (KL reference = SFT). Normal steps 40% (B pool, full B reward); short-collusion steps 60% from the SHORT pool (collusive + honest twins + neutral sentences, unlabeled), no concept reward on those steps. Label-free except SFT directions.
cd /root/cc; export HF_HOME=/root/hf_cache PYTHONPATH=/root/cv_repo PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/vllm_env/bin/python; G=/root/cv_repo/runs/general; D=/root/cc
S=$G/startSFT; rm -rf $G/rl3_short
COMMON="--data $G/av_data/enriched_sft_global10.jsonl --acts $G/av_data/enriched_acts_global10.pt --descriptions $G/av_data/th_descriptions.jsonl \
 --ar $G/th2_ar --dirs $G/av_data --drop_ids $G/av_data/strict_drop_ids.jsonl --val_ids $G/av_data/th_val_ids.jsonl \
 --steps 100 --eval_every 25 --eval_n 64 --max_new 60 --seed 11 --av $S --ref $S --mix_orig 0.4 --mix_fresh $D/poolB_train \
 --w_left 0.4 --w_cons 0.25 --w_behav 0.15 --w_concept 0.2 --beta 0.05 --w_div 0.5 --div_thr 0.5 --w_empty 0.3 --no_concept_fresh --max_kl 0.8"
cd /root/cv_repo
$PY /root/overnight/rl2_v2.py $COMMON --fresh $D/pool_short --div_acts /root/data/acts_replay.pt --out $G/rl3_short > /root/overnight/logs_rl3_short.log 2>&1
echo "RL DONE" >> /root/overnight/logs_rl3_short.log
