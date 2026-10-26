#!/bin/bash
# usage: run_arm.sh <A0|A1|A2> <seed> <python>   (identical on both pods; layout /root/cv_repo, /root/cc, /root/data, /root/overnight)
# All arms: start = base SFT (startSFT), KL ref = SFT, 400-step schedule, stopped after step ~101, ckpt at 50 and 100.
ARM=$1; SEED=$2; PY=${3:-/root/vllm_env/bin/python}
export HF_HOME=/root/hf_cache PYTHONPATH=/root/cv_repo PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
G=/root/cv_repo/runs/general; D=/root/cc; S=$G/startSFT; OUT=$G/arm_${ARM}_s${SEED}; LOG=/root/overnight/logs_arm_${ARM}_s${SEED}.log
case $ARM in A2) FRESH=$D/pool_bl; MIX=0.4;; A1) FRESH=$D/pool_blc; MIX=0.4;; A0) FRESH=$D/pool_blc; MIX=1.0;; esac
sed 's/step % 25 == 0 and step < a.steps/step % 50 == 0 and step < a.steps/' /root/overnight/rl2_v3.py > /root/overnight/rl2_v4.py 2>/dev/null || cp /root/overnight/rl2_v2.py /root/overnight/rl2_v4.py
rm -rf $OUT
COMMON="--data $G/av_data/enriched_sft_global10.jsonl --acts $G/av_data/enriched_acts_global10.pt --descriptions $G/av_data/th_descriptions.jsonl \
 --ar $G/th2_ar --dirs $G/av_data --drop_ids $G/av_data/strict_drop_ids.jsonl --val_ids $G/av_data/th_val_ids.jsonl \
 --steps 400 --eval_every 100 --eval_n 64 --max_new 60 --seed $SEED --av $S --ref $S --mix_orig $MIX --mix_fresh $D/poolB_train \
 --w_left 0.4 --w_cons 0.25 --w_behav 0.15 --w_concept 0.2 --beta 0.05 --w_div 0.5 --div_thr 0.5 --w_empty 0.3 --no_concept_fresh --max_kl 0.8"
cd /root/cv_repo
$PY /root/overnight/rl2_v4.py $COMMON --fresh $FRESH --div_acts /root/data/acts_replay.pt --out $OUT > $LOG 2>&1 &
PID=$!
until grep -qE "^step 102 " $LOG || ! kill -0 $PID 2>/dev/null; do sleep 10; done
kill $PID 2>/dev/null; sleep 3
for c in 50 100; do P=$OUT/ckpt_$c; [ -f $P/lora/adapter_config.json ] || cp $P/lora/av/* $P/lora/ 2>/dev/null; done
echo "ARM DONE $ARM $SEED" >> $LOG
