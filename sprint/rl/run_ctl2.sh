#!/bin/bash
# control: identical to v8 (long run) but the unlabeled fresh pool contains ONLY honest transcripts (independent + open). 400-step schedule, stopped after step 125; ckpt every 25.
cd /root/cc; export HF_HOME=/root/hf_cache PYTHONPATH=/root/cv_repo PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/vllm_env/bin/python; G=/root/cv_repo/runs/general; D=/root/cc
$PY build_pool2.py concept_texts4.jsonl train noncollusive 4 $D/pool_blc > $D/log_pool_blc.txt 2>&1
S=$G/startSFT; rm -rf $G/rl3_ctl2
COMMON="--data $G/av_data/enriched_sft_global10.jsonl --acts $G/av_data/enriched_acts_global10.pt --descriptions $G/av_data/th_descriptions.jsonl \
 --ar $G/th2_ar --dirs $G/av_data --drop_ids $G/av_data/strict_drop_ids.jsonl --val_ids $G/av_data/th_val_ids.jsonl \
 --steps 400 --eval_every 100 --eval_n 64 --max_new 60 --seed 11 --av $S --ref $S --mix_orig 0.4 --mix_fresh $D/poolB_train \
 --w_left 0.4 --w_cons 0.25 --w_behav 0.15 --w_concept 0.2 --beta 0.05 --w_div 0.5 --div_thr 0.5 --w_empty 0.3 --no_concept_fresh --max_kl 0.8"
cd /root/cv_repo
$PY /root/overnight/rl2_v3.py $COMMON --fresh $D/pool_blc --div_acts /root/data/acts_replay.pt --out $G/rl3_ctl2 > /root/overnight/logs_rl3_ctl2.log 2>&1 &
PID=$!
until grep -qE "^step 126 " /root/overnight/logs_rl3_ctl2.log || ! kill -0 $PID 2>/dev/null; do sleep 10; done
kill $PID 2>/dev/null
echo "CTL DONE" >> /root/overnight/logs_rl3_ctl2.log
