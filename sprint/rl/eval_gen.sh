#!/bin/bash
# usage: eval_gen.sh <python> <outprefix> name=adapterdir [name=adapterdir ...]   (adapter dirs relative to /root/cv_repo)
PY=$1; OUT=$2; shift 2
cd /root/cv_repo; export HF_HOME=/root/hf_cache PYTHONPATH=.
$PY /root/overnight/gen_nla.py --bs 32 --acts locked=/root/cc/pool_locked_acts.pt shortt=/root/cc/pool_shortt_acts.pt short=/root/cc/short_acts.pt bev=/root/cc/poolB_eval_acts.pt --adapters "$@" --out $OUT
