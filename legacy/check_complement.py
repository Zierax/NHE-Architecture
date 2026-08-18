import json
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

base = {r["question"]: r for r in json.load(open("results/eval_africa_baseline.json", encoding="utf-8"))["results"]}
static = {r["question"]: r for r in json.load(open("results/eval_africa_k32_midwrong.json", encoding="utf-8"))["results"]}
rt = {r["question"]: r for r in json.load(open("results/eval_runtime_africa_jump_gt_L19_t90_mask.json", encoding="utf-8"))["results"]}

print("question | static_fix | runtime_early_fix")
for q, b in base.items():
    if b["correct"]:
        continue
    s = static[q]["correct"]
    r = rt[q]["correct"]
    sf = "FIX" if (s and not r) else ("-")
    rf = "FIX" if (r and not s) else ("-")
    print(f"  {q:<45} static={sf:<4} runtime={rf:<4} (static hall={not s}, rt hall={not r})")