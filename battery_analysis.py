import json
import sys

import numpy as np
from scipy.stats import binomtest

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SEEDS = [1000, 1001, 1002, 1003, 1004]

def load(prefix, seed):
    ev = json.load(open(f"results/eval_runtime_africa_{prefix}_s{seed}.json", encoding="utf-8"))
    return {r["question"]: r["correct"] for r in ev["results"]}, ev

none = {}
mask = {}
for s in SEEDS:
    none[s], _ = load("jump_gt_L19_t90_none", s)
    mask[s], _ = load("jump_gt_L19_t90_mask_sft0.3", s)

qs = list(none[SEEDS[0]].keys())
assert len(qs) == 54

def majority(d):
    return {q: int(sum(d[s][q] for s in SEEDS) >= 3) for q in qs}

mn = majority(none)
mm = majority(mask)

n_hall_n = sum(1 for q in qs if not mn[q])
n_hall_m = sum(1 for q in qs if not mm[q])
print(f"africa majority-of-5 (54 items): baseline {n_hall_n}/54={n_hall_n/54:.4f} -> soft-mask {n_hall_m}/54={n_hall_m/54:.4f}")

fixed = [q for q in qs if (not mn[q]) and mm[q]]
broken = [q for q in qs if mn[q] and (not mm[q])]
print(f"  fixes (W2C): {len(fixed)} {fixed}")
print(f"  breaks(C2W): {len(broken)} {broken}")

# sample-level (270) significance
yn = np.array([none[s][q] for s in SEEDS for q in qs], dtype=int)
ym = np.array([mask[s][q] for s in SEEDS for q in qs], dtype=int)
hn, hm = 1 - yn, 1 - ym
print(f"\nsample-level (n=270): hall none={hn.mean():.4f} ({hn.sum()}) mask={hm.mean():.4f} ({hm.sum()})")

def mcnemar(a, b):
    b01 = int(((a == 1) & (b == 0)).sum())  # correct -> wrong
    b10 = int(((a == 0) & (b == 1)).sum())  # wrong -> correct
    if b01 + b10 == 0:
        return float("nan")
    return binomtest(min(b01, b10), b01 + b10, 0.5, alternative="two-sided").pvalue, b10 - b01

p, net = mcnemar(yn, ym)
print(f"  McNemar: W2C={int(((yn==0)&(ym==1)).sum())} C2W={int(((yn==1)&(ym==0)).sum())} net={net} p={p:.4f}")

# bootstrap on sample-level diff
rng = np.random.default_rng(0)
diffs = []
for _ in range(5000):
    idx = rng.integers(0, len(hn), len(hn))
    diffs.append((hm[idx].mean() - hn[idx].mean()))
diffs = np.array(diffs)
print(f"  bootstrap 95% CI (mask - none hall): [{np.percentile(diffs,2.5):.4f}, {np.percentile(diffs,97.5):.4f}]")

# item-level majority McNemar
w2c = sum(1 for q in qs if (not mn[q]) and mm[q])
c2w = sum(1 for q in qs if mn[q] and (not mm[q]))
if w2c + c2w == 0:
    ip = 1.0
else:
    ip = binomtest(min(w2c, c2w), w2c + c2w, 0.5, alternative="two-sided").pvalue
print(f"item-level majority: W2C={w2c} C2W={c2w} p={ip:.4f}")

# static comparison
st = json.load(open("results/eval_africa_baseline_s5.json", encoding="utf-8"))
print(f"\nstatic baseline_s5 majority rate={st.get('hallucination_rate')} (reference)")
for m in ["k32_midwrong", "k128_wrong"]:
    try:
        x = json.load(open(f"results/eval_africa_{m}_s5.json", encoding="utf-8"))
        print(f"  static {m} s5 majority={x.get('hallucination_rate')}")
    except FileNotFoundError:
        pass