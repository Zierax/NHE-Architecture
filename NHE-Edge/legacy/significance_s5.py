import json
import os
import sys
from math import comb

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
    sp = {}
    bp = {}
    for r in b["results"]:
        bp[r["question"]] = r["correct"]
    for r in m["results"]:
        sp[r["question"]] = r["correct"]
    b_fix = {}
    m_fix = {}
    for q, flags_b in bp.items():
        if q in sp:
            for s, f in enumerate(flags_b):
                key = f"{q}::#{s}"
                b_fix[key] = f
                m_fix[key] = sp[q][s]
    return b_fix, m_fix

print("=== MAJORITY (item-level, 5 samples) ===")
print(f"{'topic':<10}{'mask':<16}{'hall':>8}{'wilson95':>18}{'mcnemar':>28}")
rows = [("africa", "baseline"), ("africa", "k32_midwrong"), ("africa", "k128_wrong"),
        ("europe", "baseline"), ("europe", "k32_midwrong"), ("europe", "k128_wrong"),
        ("elements", "baseline"), ("elements", "k32_midwrong"), ("elements", "k128_wrong")]
maj = {}
for topic, tag in rows:
    d = load(topic, tag + "_s5")
    n = d["n"]
    k_hall = n - d["n_correct"]
    p, lo, hi = wilson(k_hall, n)
    maj[(topic, tag)] = (k_hall, n)
    print(f"{topic:<10}{tag:<16}{p:>8.3f}{f'[{lo:.3f},{hi:.3f}]':>18}")

print("\n=== SAMPLE-LEVEL (270 instances per cell) ===")
print(f"{'topic':<10}{'mask':<16}{'hall':>8}{'wilson95':>18}{'mcnemar':>28}")
for topic, tag in rows:
    if tag == "baseline":
        continue
    b = load(topic, "baseline_s5")
    m = load(topic, tag + "_s5")
    b_fix, m_fix = sample_pairs(b, m)
    k_hall = sum(1 for v in m_fix.values() if not v)
    n = len(m_fix)
    p, lo, hi = wilson(k_hall, n)
    mc = mcnemar(b_fix, m_fix)
    mc_s = f"fix={mc['fixed']} brk={mc['broke']} p={mc['exact_p']:.3f}" if mc else "no discordant"
    print(f"{topic:<10}{tag:<16}{p:>8.3f}{f'[{lo:.3f},{hi:.3f}]':>18}{mc_s:>28}")