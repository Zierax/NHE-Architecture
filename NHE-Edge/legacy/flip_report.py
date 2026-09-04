import json
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def load(topic, tag):
    return {r["question"]: r for r in json.load(open(f"results/eval_{topic}_{tag}.json", encoding="utf-8"))["results"]}

base_a = load("africa", "baseline")
for tag in ["k32_midwrong", "k128_midwrong", "k128_wrong", "k32_wrong"]:
    ev = load("africa", tag)
    print(f"=== africa {tag} ===")
    for q, r in ev.items():
        b = base_a[q]
        if r["correct"] != b["correct"]:
            kind = "W2C" if r["correct"] else "C2W"
            print(f"  [{kind}] {q}")
            print(f"      base : {b['generated'][:70]!r}")
            print(f"      mask : {r['generated'][:70]!r}")

base_e = load("europe", "baseline")
for tag in ["k128_wrong", "k32_wrong", "k128_midwrong"]:
    ev = load("europe", tag)
    flips = [(q, r) for q, r in ev.items() if r["correct"] != base_e[q]["correct"]]
    print(f"=== europe {tag}: {len(flips)} collateral flips ===")
    for q, r in flips:
        print(f"  [C2W] {q}: base={base_e[q]['generated'][:60]!r} -> mask={r['generated'][:60]!r}")