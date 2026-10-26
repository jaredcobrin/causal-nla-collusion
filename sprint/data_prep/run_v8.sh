#!/bin/bash
# v8: base SFT start (KL reference = SFT). Normal steps 40% (B's pool at B's category mix, B's full reward); collusion steps 60% from the BLATANT pool (collusive no-word + collusive with-word + independent + open, unlabeled), no concept reward on collusion steps. 400 steps.
cd /root/cc; export HF_HOME=/root/hf_cache PYTHONPATH=/root/cv_repo PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/root/vllm_env/bin/python; G=/root/cv_repo/runs/general; D=/root/cc
rm -f $D/pool_bl_* $D/pool_blt_* /root/overnight/rl3_done.txt
$PY build_pool2.py concept_texts4.jsonl train all 4 $D/pool_bl > $D/log_pool_bl.txt 2>&1
$PY build_pool2.py concept_texts4.jsonl test all 3 $D/pool_blt > $D/log_pool_blt.txt 2>&1
S=$G/startSFT; rm -rf $S; mkdir -p $S; cp -r $G/th2_av/lora $S/lora; cp $G/th2_av/heldout_generations.jsonl $S/
rm -rf $G/rl3_sft
COMMON="--data $G/av_data/enriched_sft_global10.jsonl --acts $G/av_data/enriched_acts_global10.pt --descriptions $G/av_data/th_descriptions.jsonl \
 --ar $G/th2_ar --dirs $G/av_data --drop_ids $G/av_data/strict_drop_ids.jsonl --val_ids $G/av_data/th_val_ids.jsonl \
 --steps 400 --eval_every 100 --eval_n 64 --max_new 60 --seed 11 --av $S --ref $S --mix_orig 0.4 --mix_fresh $D/poolB_train \
 --w_left 0.4 --w_cons 0.25 --w_behav 0.15 --w_concept 0.2 --beta 0.05 --w_div 0.5 --div_thr 0.5 --w_empty 0.3 --no_concept_fresh --max_kl 0.8"
cd /root/cv_repo
$PY /root/overnight/rl2_v2.py $COMMON --fresh $D/pool_bl --div_acts /root/data/acts_replay.pt --out $G/rl3_sft > /root/overnight/logs_rl3_sft.log 2>&1
echo "RL DONE" >> /root/overnight/logs_rl3_sft.log
echo ALL_RL_DONE > /root/overnight/rl3_done.txt
