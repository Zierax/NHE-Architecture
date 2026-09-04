import json
import os
import re
import sys
import unicodedata

import numpy as np
from scipy.stats import binomtest

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import topics

RES = os.path.join(BASE, "results")
SEEDS = [1000, 1001, 1002, 1003, 1004, 1006]
ARMS = ["none", "mask", "abstain"]
WINDOW = 5
SCALE = 0.3

ALTS = {}
for name in ["AFRICA_LARGEST", "WORLD_CAP_TRAPS", "WORLD_LARGEST"]:
    for tup in getattr(topics, name):
        ALTS[tup[0]] = list(tup[1:])

BENCH = json.load(open(os.path.join(RES, "bench_hard.json"), encoding="utf-8"))["items"]


def norm(s):
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def fs(gen):
    m = re.split(r"[.\n]", gen.strip())
    return norm(m[0]) if m and m[0].strip() else None


def tag(arm, seed):
    t = f"jump_gt_L19_t90_{arm}"
    if arm == "mask":
        t += f"_sft{SCALE}"
    return f"{t}_s{seed}"


def load_arm(topic, arm, seed):
    p = os.path.join(RES, f"eval_runtime_{topic}_{tag(arm, seed)}.json")
    if not os.path.exists(p):
        return None
    ev = json.load(open(p, encoding="utf-8"))
    out = {}
    for r in ev["results"]:
        a = [r["answer"]] + list(ALTS.get(r["question"], []))
        f = fs(r["generated"])
        sc = 1 if (f and any(norm(x) in f for x in a)) else 0
        out[r["id"]] = {"correct": sc, "abstained": bool(r.get("abstained", False)),
                        "fired_at": r["fired_at"]}
    return out


def wrong_of(v):
    return 1 - v["correct"] - (1 if v["abstained"] else 0)


# rows: (topic, item_id, seed)
rows = []
for t, i in BENCH:
    for s in SEEDS:
        rows.append((t, i, s))

# collect per arm
data = {a: {} for a in ARMS}
missing = []
for t, i, s in rows:
    for a in ARMS:
        if (t, s) not in data[a]:
            data[a][(t, s)] = load_arm(t, a, s)
            if data[a][(t, s)] is None:
                missing.append((t, a, s))


def mcnemar(a, b):
    b01 = int(((a == 1) & (b == 0)).sum())
    b10 = int(((a == 0) & (b == 1)).sum())
    if b01 + b10 == 0:
        return float("nan"), 0
    return binomtest(min(b01, b10), b01 + b10, 0.5, alternative="two-sided").pvalue, b10 - b01


def report(completed, label=""):
    print(f"\n===== BENCH ANALYSIS (sampled battery) {label} =====")
    print(f"bench items: {len(BENCH)} | seeds: {SEEDS}")
    print(f"missing files: {missing if missing else 'none'}")
    if not completed:
        print("NO complete runs yet.")
        return

    # per-item-stable strict vectors across seeds
    for a in ARMS:
        pass

    # pooled per-arm rates (sample-level)
    print("\n-- pooled sample-level (strict metric) --")
    rates = {}
    for a in ARMS:
        acc = []
        for (t, i, s) in rows:
            d = data[a].get((t, s))
            if d is None or i not in d:
                continue
            acc.append(d[i])
        n = len(acc)
        if n == 0:
            print(f"  {a:<8} no data yet")
            continue
        wrong = sum(wrong_of(v) for v in acc)
        abst = sum(1 for v in acc if v["abstained"])
        corr = sum(1 for v in acc if v["correct"])
        fired = sum(1 for v in acc if v["fired_at"] is not None)
        rates[a] = (n, wrong, abst, corr, fired)
        print(f"  {a:<8} n={n:>4} wrong={wrong:>4} ({wrong/n:.3f})  abstain={abst:>4}  correct={corr:>4} ({corr/n:.3f})  fired={fired:>4}")

    # paired comparisons (only rows present in both arms)
    def paired_vec(a1, a2, what):
        va, vb = [], []
        for (t, i, s) in rows:
            d1 = data[a1].get((t, s)); d2 = data[a2].get((t, s))
            if d1 is None or d2 is None or i not in d1 or i not in d2:
                continue
            va.append(1 - wrong_of(d1[i]))
            vb.append(1 - wrong_of(d2[i]))
        return np.array(va, dtype=int), np.array(vb, dtype=int)

    print("\n-- paired significance (strict, abstained counts as not-wrong) --")
    for a1, a2, lab in [("none", "mask", "none vs mask (wrong)"),
                        ("none", "abstain", "none vs abstain (wrong)"),
                        ("mask", "abstain", "mask vs abstain (wrong)")]:
        va, vb = paired_vec(a1, a2, "wrong")
        p, net = mcnemar(va, vb)
        w2c = int(((va == 0) & (vb == 1)).sum()); c2w = int(((va == 1) & (vb == 0)).sum())
        print(f"  {lab:<26}: W2C={w2c:>4} C2W={c2w:>4} net={net:+d}  p={p:.4f}  (n={len(va)})")

    # correct-rate comparison (utility): none vs mask, none vs abstain
    print("\n-- paired significance (correct = strict-correct, abstained NOT correct) --")
    for a1, a2, lab in [("none", "mask", "none vs mask (correct)"),
                        ("none", "abstain", "none vs abstain (correct)")]:
        c1 = np.array([data[a1][(t, s)][i]["correct"] for (t, i, s) in rows
                       if data[a1].get((t, s)) and i in data[a1][(t, s)]
                       and data[a2].get((t, s)) and i in data[a2][(t, s)]], dtype=int)
        c2 = np.array([data[a2][(t, s)][i]["correct"] for (t, i, s) in rows
                       if data[a1].get((t, s)) and i in data[a1][(t, s)]
                       and data[a2].get((t, s)) and i in data[a2][(t, s)]], dtype=int)
        p, net = mcnemar(c1, c2)
        w2c = int(((c1 == 0) & (c2 == 1)).sum()); c2w = int(((c1 == 1) & (c2 == 0)).sum())
        print(f"  {lab:<26}: W2C={w2c:>4} C2W={c2w:>4} net={net:+d}  p={p:.4f}  (n={len(c1)})")

    # bootstrap CI on wrong-rate diff (none vs mask)
    # simpler bootstrap on the paired wrong vectors
    va, vb = paired_vec("none", "mask", "wrong")
    h_a, h_b = 1 - va, 1 - vb
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(5000):
        idx = rng.integers(0, len(h_a), len(h_a))
        diffs.append(h_b[idx].mean() - h_a[idx].mean())
    diffs = np.array(diffs)
    print(f"  bootstrap 95% CI (mask-none wrong-rate): [{np.percentile(diffs,2.5):.4f}, {np.percentile(diffs,97.5):.4f}]")

    # majority-of-seeds (strict, abstained=not-wrong), per item
    print("\n-- majority-of-seeds per item (strict) --")
    for a in ARMS:
        maj = []
        for (t, i) in BENCH:
            vals = [(data[a].get((t, s)) or {}).get(i) for s in SEEDS]
            vals = [v for v in vals if v is not None]
            if len(vals) < max(3, len(SEEDS) // 2):
                continue
            wrongs = sum(wrong_of(v) for v in vals)
            maj.append(1 if wrongs > len(vals) / 2 else 0)
        if maj:
            print(f"  {a:<8} majority-wrong = {sum(maj)}/{len(maj)} = {sum(maj)/len(maj):.3f}")

    # fired-item breakdown
    print("\n-- fired-item behaviour (strict) --")
    fi = []
    for (t, i) in BENCH:
        for s in SEEDS:
            d0 = (data["none"].get((t, s)) or {}).get(i)
            dm = (data["mask"].get((t, s)) or {}).get(i)
            da = (data["abstain"].get((t, s)) or {}).get(i)
            if d0 is not None and d0["fired_at"] is not None and dm is not None and da is not None:
                fi.append((d0, dm, da))
    if fi:
        n = len(fi)
        none_wrong = sum(1 for d0, _, _ in fi if wrong_of(d0))
        mask_wrong = sum(1 for _, dm, _ in fi if wrong_of(dm))
        mask_corr = sum(1 for _, dm, _ in fi if dm["correct"])
        abst_n = sum(1 for _, _, da in fi if da["abstained"])
        fix = sum(1 for d0, dm, _ in fi if wrong_of(d0) and dm["correct"])
        print(f"  fired paired samples n={n}: none_wrong={none_wrong} mask_wrong={mask_wrong} "
              f"mask_correct={mask_corr} abstain_refused={abst_n} | wrong->correct fixes={fix}")


def main():
    any_file = any(
        os.path.exists(os.path.join(RES, f"eval_runtime_{t}_{tag(a, s)}.json"))
        for t in {x[0] for x in BENCH} for a in ARMS for s in SEEDS
    )
    report(any_file)


if __name__ == "__main__":
    main()