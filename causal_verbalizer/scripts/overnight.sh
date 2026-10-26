#!/usr/bin/env bash
# Pod-side queue (no tokens needed): screen every candidate on val, then confirm the causal ones once on test.
cd /root/causal_verbalizer
export HF_HOME=/root/hf_cache PYTHONUNBUFFERED=1 PATH=/root/venv/bin:$PATH
python -m cv.run ab --set bank --out bank_val
python -m cv.run ab --set existing --out existing_val
python -m cv.run ab --set bank --split test --out bank_test --confirm-from bank_val
python -m cv.run ab --set existing --split test --out existing_test --confirm-from existing_val
echo ALL-DONE > runs/overnight_done
