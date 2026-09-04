import json
import os
import sys
from math import comb
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, "results")

def load(topic, tag):
    return json.load(open(os.path.join(RES, f"eval_{topic}_{tag}.json"), encoding="utf-8"))

def wilson(k, n):
    z = 1.96
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (p, c - h, c + h)

def mcnemar(b_fix, m_fix):
    b2w = sum(1 for q, f in m_fix.items() if q in b_fix and not b_fix[q] and f)
    w2b = sum(1 for q, f in m_fix.items() if q in b_fix and b_fix[q] and not f)
    n = b2w + w2b
    if n == 0:
        return None
    if b2w == 0 or w2b == 0:
        p = 2 ** (1 - n)
    else:
        p = 2 * sum(comb(n, k) for k in range(min(b2w, w2b) + 1)) / (2 ** n)
    return {"fixed": b2w, "broke": w2b, "exact_p": min(p, 1.0)}

def sample_pairs(b, m):
    bp = {r["question"]: r["correct"] for r in b["results"]}
    sp = {r["question"]: r["correct"] for r in m["results"]}
    b_fix, m_fix = {}, {}
    for q, flags in bp.items():
        if q in sp:
            for s, f in enumerate(flags):
                key = f"{q}::#{s}"
                b_fix[key] = f
                m_fix[key] = sp[q][s]
    return b_fix, m_fix

def bootstrap_net(b_fix, m_fix, n_boot=5000, seed=0):
    keys = [k for k in b_fix if k in m_fix]
    b0 = np.array([b_fix[k] for k in keys])
    m0 = np.array([m_fix[k] for k in keys])
    rng = np.random.default_rng(seed)
    nets = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(keys), len(keys))
        b, m = b0[idx], m0[idx]
        fix = ((b == 0) & (m == 1)).sum()
        brk = ((b == 1) & (m == 0)).sum()
        nets.append(fix - brk)
    nets = np.array(nets)
    return (float(np.percentile(nets, 2.5)), float(np.percentile(nets, 97.5)))

TOPICS = ["africa", "europe", "elements", "asia", "us_states"]
MASKS = ["k32_midwrong", "k128_wrong"]

print(f"{'topic':<10}{'mask':<14}{'n':>4}{'hall_b':>8}{'hall_m':>8}{'fix':>5}{'brk':>5}{'p':>8}{'net95CI':>18}")
for topic in TOPICS:
    b = load(topic, "baseline_s5")
    for mask in MASKS:
        m = load(topic, mask + "_s5")
        b_fix, m_fix = sample_pairs(b, m)
        k_h_b = sum(1 for v in b_fix.values() if not v)
        k_h_m = sum(1 for v in m_fix.values() if not v)
        n = len(b_fix)
        mc = mcnemar(b_fix, m_fix)
        ci = bootstrap_net(b_fix, m_fix)
        mc_s = f"{mc['fixed']:>5}{mc['broke']:>5}{mc['exact_p']:>8.3f}" if mc else f"{'-':>5}{'-':>5}{'1.000':>8}"
        print(f"{topic:<10}{mask:<14}{n:>4}{k_h_b/n:>8.3f}{k_h_m/n:>8.3f}{mc_s}{f'[{ci[0]:.0f},{ci[1]:.0f}]':>18}")

print("\nbaseline sampled hall rates:")
for topic in TOPICS:
    b = load(topic, "baseline_s5")
    k = sum(1 for r in b["results"] for f in r["correct"] if not f)
    n = sum(len(r["correct"]) for r in b["results"])
    p, lo, hi = wilson(k, n)
    print(f"  {topic:<10} {p:.3f} [{lo:.3f},{hi:.3f}] n={n}")