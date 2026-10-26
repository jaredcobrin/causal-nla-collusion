#!/usr/bin/env bash
# Pod-side: wait for the running queue, install the rebuilt bank (data/bank_new), redo tqa (its split changed), rerun the queue.
cd /root/causal_verbalizer
while pgrep -f overnight.sh > /dev/null; do sleep 30; done
cp -r data/bank_new/. data/bank/
rm -f runs/bank_val/results/tqa-correct-answer.json runs/bank_test/results/tqa-correct-answer.json runs/overnight_done
bash scripts/overnight.sh
