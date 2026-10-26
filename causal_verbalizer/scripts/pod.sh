#!/usr/bin/env bash
# Pod helper. Connection details live in scripts/pod.env (not tracked).
#   scripts/pod.sh setup          install requirements on the pod
#   scripts/pod.sh push           code, configs, tests, splits and judge scores -> pod
#   scripts/pod.sh pull           pod runs/ -> local runs/ (everything except activation caches)
#   scripts/pod.sh run <args>     start `python -m cv.run <args>` in the background on the pod
#   scripts/pod.sh log <phase>    tail the pod's progress log
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/pod.env
DIR=/root/causal_verbalizer  # pod-local disk (the /workspace FUSE volume breaks rsync); results are pulled back
SSH="ssh -p $POD_PORT -i $POD_KEY $POD_HOST"
RSYNC=(rsync -az -e "ssh -p $POD_PORT -i $POD_KEY")
VENV=/root/venv  # reuses the image's CUDA torch via --system-site-packages
ENV="export HF_HOME=/root/hf_cache PYTHONUNBUFFERED=1 PATH=$VENV/bin:\$PATH; cd $DIR"

case "$1" in
  setup) $SSH "python3 -m venv --system-site-packages $VENV && $VENV/bin/pip install -q -r $DIR/requirements.txt" ;;
  push)
    $SSH "mkdir -p $DIR/data/splits $DIR/data/bank $DIR/runs/judge/scores"
    "${RSYNC[@]}" --delete data/bank/ "$POD_HOST:$DIR/data/bank/"
    "${RSYNC[@]}" --delete --exclude __pycache__ cv configs tests scripts/overnight.sh requirements.txt "$POD_HOST:$DIR/"
    "${RSYNC[@]}" --delete data/splits/ "$POD_HOST:$DIR/data/splits/"
    # keys go with scores: a scores file without its keys file crashes the pod's run
    for d in keys scores; do [ -d runs/judge/$d ] && "${RSYNC[@]}" runs/judge/$d/ "$POD_HOST:$DIR/runs/judge/$d/" || true; done ;;
  pull) mkdir -p runs && "${RSYNC[@]}" --exclude cache/ "$POD_HOST:$DIR/runs/" runs/ ;;
  run) shift; $SSH -f "$ENV && mkdir -p runs && setsid nohup python -m cv.run $* > runs/run-\$(date +%H%M%S).out 2>&1 < /dev/null &" ;;
  log) $SSH "tail -n ${3:-30} $DIR/runs/$2/progress.log" ;;
  *) echo "usage: $0 setup|push|pull|run <args>|log <phase>"; exit 1 ;;
esac
