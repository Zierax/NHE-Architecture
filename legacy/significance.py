import json
import sys
from math import comb

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def load(topic, tag):
    return {r["question"]: r for r in json.load(open(f"results/eval_{topic}_{tag}.json", encoding="utf-8"))["results"]}

def wilson(k, n):
    z = 1.96
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (p, c - h, c + h)

def mcnemar(b, m):
    b2w = sum(1 for q, r in m.items() if q in b and not b[q]["correct"] and r["correct"])
    w2b = sum(1 for q, r in m.items() if q in b and b[q]["correct"] and not r["correct"])
    n = b2w + w2b
    if n == 0:
        return None
    p = 2 ** (n - 1) if b2w == 0 or w2b == 0 else None
    if p is None:
        p = sum(comb(n, k) for k in range(min(b2w, w2b) + 1)) / (2 ** n) * 2
    return {"fixed": b2w, "broke": w2b, "exact_p": min(p, 1.0)}

pairs = [
    ("africa", "k32_midwrong"),
    ("africa", "k128_midwrong"),
    ("africa", "k32_wrong"),
    ("africa", "k128_wrong"),
    ("europe", "k32_midwrong"),
    ("europe", "k128_wrong"),
    ("elements", "k32_midwrong"),
    ("elements", "k128_wrong"),
]

print(f"{'topic':<10}{'mask':<14}{'n':>4}{'hall':>8}{'wilson95':>18}{'mcnemar':>30}")
for topic, tag in pairs:
    b = load(topic, "baseline")
    m = load(topic, tag)
    n = len(b)
    k_hall = sum(1 for r in m.values() if not r["correct"])
    p, lo, hi = wilson(k_hall, n)
    mc = mcnemar(b, m)
    mc_s = f"fix={mc['fixed']} brk={mc['broke']} p={mc['exact_p']:.3f}" if mc else "no discordant"
    print(f"{topic:<10}{tag:<14}{n:>4}{p:>8.3f}{f'[{lo:.3f},{hi:.3f}]':>18}{mc_s:>30}")

print("\nbaseline hall rates (wilson 95%):")
for topic in ["africa", "europe", "elements"]:
    b = load(topic, "baseline")
    n = len(b)
    k = sum(1 for r in b.values() if not r["correct"])
    p, lo, hi = wilson(k, n)
    print(f"  {topic:<10} {p:.3f} [{lo:.3f},{hi:.3f}] n={n}")