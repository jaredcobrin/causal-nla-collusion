"""How many of the confirmed directions are genuinely different? Greedy count at cosine < 0.8, plus every pair above it.

    python scripts/distinct.py runs/bank_test runs/existing_test
"""
import json
import sys
from pathlib import Path

import torch

names, vecs = [], []
for d in map(Path, sys.argv[1:]):
    for f in sorted((d / "results").glob("*.json")):
        if json.loads(f.read_text()).get("causal") and (d / "directions" / f"{f.stem}.pt").exists():
            names.append(f.stem)
            vecs.append(torch.nn.functional.normalize(torch.load(d / "directions" / f"{f.stem}.pt")["v"].float(), dim=0))
V = torch.stack(vecs)
cos = V @ V.T
kept = []
for i in range(len(names)):
    if all(cos[i, j] < 0.8 for j in kept):
        kept.append(i)
print(f"{len(names)} confirmed causal directions, {len(kept)} distinct at cosine < 0.8 (greedy)")
pairs = sorted(((float(cos[i, j]), names[i], names[j]) for i in range(len(names)) for j in range(i) if cos[i, j] >= 0.8), reverse=True)
for c, a, b in pairs:
    print(f"  cos {c:.2f}  {a}  ~  {b}")
