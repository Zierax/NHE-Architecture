"""Unified bench tool. Replaces bench_build.py, bench_random_build.py,
bench_driver.py, bench_greedy.py, bench_random_greedy.py, bench_full_sampled.py,
bench_analysis.py (moved to legacy/ — kept for history, not deleted).

Filename conventions are preserved exactly so all committed result files
remain valid:
  hard greedy:   eval_runtime_{topic}_{tag}.json
  hard sampled:  ..._{tag}_s{seed}.json
  random:        ...{_rand}.json suffix
  static merged: _static0 in tag
  mask arm:      _sft{scale}

Usage:
  python bench.py build [--bench hard|random|all]
  python bench.py run --bench hard|random [--mode greedy|sampled]
      [--arms none,mask,abstain] [--seeds 1000,1001,...] [--static] [--transfer]
      --transfer: greedy full-topic transfer runs (cap_traps full 134 etc.)
  python bench.py analyze --bench hard|random [--static]
"""
import argparse
import json
import os
import random
import re
import sys
import time
import unicodedata
from collections import defaultdict

import numpy as np
from scipy.stats import binomtest

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import topics  # noqa: E402

import runtime_rollback as rr  # noqa: E402

RES = os.path.join(BASE, "results")
WINDOW = 5
SCALE = 0.3
SEEDS = [1000, 1001, 1002, 1003, 1004, 1006]
ARMS = ["none", "mask", "abstain"]
POOL = (
    [["africa_largest", i] for i in range(54)]
    + [["world_cap_traps", i] for i in range(134)]
    + [["world_largest", i] for i in range(173)]
)
# bench_greedy.py transfer subsets (kept for reproducibility)
TRANSFER_SUBSETS = {
    "africa_largest": list(range(54)),
    "world_cap_traps": list(range(134)),
}
ALTS = {}
for _name in ["AFRICA", "EUROPE", "AFRICA_LARGEST", "WORLD_TRICKY",
              "WORLD_CAP_TRAPS", "WORLD_LARGEST"]:
    for _tup in getattr(topics, _name):
        ALTS[_tup[0]] = list(_tup[1:])


def norm(s):
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def fs(gen):
    m = re.split(r"[.\n]", gen.strip())
    return norm(m[0]) if m and m[0].strip() else None


def _rp(name):
    return os.path.join(RES, name)


# ---------------------------------------------------------------- build ---
def strict_wrong(path, topic_name):
    ev = json.load(open(path, encoding="utf-8"))
    idx = []
    for r in ev["results"]:
        a = [r["answer"]] + list(ALTS.get(r["question"], []))
        f = fs(r["generated"])
        if not (f and any(norm(x) in f for x in a)):
            idx.append(r["id"])
    print(f"{topic_name}: {len(idx)}/{len(ev['results'])} strict-wrong")
    return idx


def build_hard():
    hard = {
        "africa_largest": list(range(54)),
        "world_cap_traps": strict_wrong(_rp("eval_world_cap_traps_baseline.json"), "world_cap_traps"),
        "world_largest": strict_wrong(_rp("eval_world_largest_baseline.json"), "world_largest"),
    }
    bench = {"items": [[t, i] for t, ids in hard.items() for i in ids]}
    with open(_rp("bench_hard.json"), "w", encoding="utf-8") as fh:
        json.dump(bench, fh, indent=1)
    print(f"bench_hard.json: {len(bench['items'])} items across {list(hard.keys())}")


def build_random(seed=42, n=99):
    pool = [list(p) for p in POOL]
    random.seed(seed)
    random.shuffle(pool)
    items = [[t, i] for (t, i) in sorted([tuple(p) for p in pool[:n]])]
    with open(_rp("bench_random.json"), "w", encoding="utf-8") as fh:
        json.dump({"items": items}, fh, indent=1)
    hard = set(tuple(x) for x in json.load(open(_rp("bench_hard.json"), encoding="utf-8"))["items"])
    print(f"bench_random.json: {len(items)} items, overlap hard: {len(set(map(tuple, items)) & hard)}")


def cmd_build(args):
    if args.bench in ("hard", "all"):
        build_hard()
    if args.bench in ("random", "all"):
        build_random()


# ------------------------------------------------------------------ run ---
def load_bench(name):
    return json.load(open(os.path.join(RES, f"bench_{name}.json"), encoding="utf-8"))["items"]


def subsets_of(bench_items):
    subs = defaultdict(list)
    for t, i in bench_items:
        subs[t].append(i)
    return dict(subs)


def expected_file(topic, arm, seed=None, static=False, suffix="", window=WINDOW, scale=SCALE):
    t = f"jump_gt_L19_t90_{arm}"
    if static:
        t += "_static0"
    if window != 5:
        t += f"_w{window}"
    if arm == "mask":
        t += f"_sft{scale}"
    if seed is not None:
        t += f"_s{seed}"
    t += suffix
    return os.path.join(RES, f"eval_runtime_{topic}_{t}.json")


def make_det(arm, seed=None, static=False, suffix="", window=WINDOW, scale=SCALE):
    dg = json.load(open(os.path.join(RES, "detector_greedy.json"), encoding="utf-8"))
    det = dict(dg["detector_early"])
    det["threshold"] = det["threshold_t90"]
    det["threshold_key"] = "t90"
    det["mode"] = arm
    det["window"] = window
    det["scale"] = scale if arm == "mask" else 0.0
    det["sample"] = seed is not None
    det["seed"] = seed if seed is not None else 0
    if suffix:
        det["bench_suffix"] = suffix
    if static:
        det["static_mask"] = os.path.join(RES, "mask_k32_midwrong.json")
        det["static_scale"] = 0.0
    return det


def cmd_run(args):
    bench_items = load_bench(args.bench)
    suffix = "_rand" if args.bench == "random" else ""
    if args.transfer:
        if args.bench != "hard":
            sys.exit("--transfer only applies to the hard bench topics")
        hard_lg = [i for t, i in bench_items if t == "world_largest"]
        subsets = dict(TRANSFER_SUBSETS)
        subsets["world_largest"] = hard_lg
    else:
        subsets = subsets_of(bench_items)
    seeds = [None] if args.mode == "greedy" else args.seeds
    arms = args.arms.split(",")
    t0 = time.time()
    total, done, skipped = 0, 0, 0
    for topic, idxs in subsets.items():
        for arm in arms:
            for seed in seeds:
                total += 1
                if os.path.exists(expected_file(topic, arm, seed, args.static, suffix)):
                    skipped += 1
                    continue
                det = make_det(arm, seed, args.static, suffix)
                det["subset"] = idxs
                print(f"\n[{time.time()-t0:.0f}s] RUN bench={args.bench} topic={topic} arm={arm} "
                      f"seed={seed} static={args.static} items={len(idxs)}", flush=True)
                rr.run_topic(topic, det)
                done += 1
    print(f"\nDONE total={total} done={done} skipped={skipped} elapsed={time.time()-t0:.0f}s", flush=True)


# --------------------------------------------------------------- analyze ---
def load_arm(topic, arm, seed, suffix="", static=False):
    t = f"jump_gt_L19_t90_{arm}"
    if static:
        t += "_static0"
    if arm == "mask":
        t += f"_sft{SCALE}"
    p = os.path.join(RES, f"eval_runtime_{topic}_{t}_s{seed}{suffix}.json")
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


def mcnemar(a, b):
    b01 = int(((a == 1) & (b == 0)).sum())
    b10 = int(((a == 0) & (b == 1)).sum())
    if b01 + b10 == 0:
        return float("nan"), 0
    return binomtest(min(b01, b10), b01 + b10, 0.5, alternative="two-sided").pvalue, b10 - b01


def cmd_analyze(args):
    bench = load_bench(args.bench)
    suffix = "_rand" if args.bench == "random" else ""
    static = args.static
    arms = ["none", "mask", "abstain"]
    rows = [(t, i, s) for (t, i) in bench for s in SEEDS]
    data = {a: {} for a in arms}
    missing = []
    for t, i, s in rows:
        for a in arms:
            if (t, s) not in data[a]:
                data[a][(t, s)] = load_arm(t, a, s, suffix, static and a != "none")
                if data[a][(t, s)] is None:
                    missing.append((t, a, s))
    print(f"\n===== BENCH ANALYSIS ({args.bench}, static={static}) =====")
    print(f"bench items: {len(bench)} | seeds: {SEEDS}")
    print(f"missing files: {missing if missing else 'none'}")
    if missing and all(data[a].get(k) is None for a in arms for k in list(data[a])):
        print("NO complete runs yet.")
        return

    print("\n-- pooled sample-level (strict metric) --")
    for a in arms:
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
        print(f"  {a:<8} n={n:>4} wrong={wrong:>4} ({wrong/n:.3f})  abstain={abst:>4}  correct={corr:>4} ({corr/n:.3f})  fired={fired:>4}")

    def paired_vec(a1, a2):
        va, vb = [], []
        for (t, i, s) in rows:
            d1 = data[a1].get((t, s))
            d2 = data[a2].get((t, s))
            if d1 is None or d2 is None or i not in d1 or i not in d2:
                continue
            va.append(1 - wrong_of(d1[i]))
            vb.append(1 - wrong_of(d2[i]))
        return np.array(va, dtype=int), np.array(vb, dtype=int)

    print("\n-- paired significance (strict, abstained counts as not-wrong) --")
    for a1, a2, lab in [("none", "mask", "none vs mask (wrong)"),
                        ("none", "abstain", "none vs abstain (wrong)"),
                        ("mask", "abstain", "mask vs abstain (wrong)")]:
        va, vb = paired_vec(a1, a2)
        p, net = mcnemar(va, vb)
        w2c = int(((va == 0) & (vb == 1)).sum())
        c2w = int(((va == 1) & (vb == 0)).sum())
        print(f"  {lab:<26}: W2C={w2c:>4} C2W={c2w:>4} net={net:+d}  p={p:.4f}  (n={len(va)})")

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
        w2c = int(((c1 == 0) & (c2 == 1)).sum())
        c2w = int(((c1 == 1) & (c2 == 0)).sum())
        print(f"  {lab:<26}: W2C={w2c:>4} C2W={c2w:>4} net={net:+d}  p={p:.4f}  (n={len(c1)})")

    va, vb = paired_vec("none", "mask")
    h_a, h_b = 1 - va, 1 - vb
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(5000):
        idx = rng.integers(0, len(h_a), len(h_a))
        diffs.append(h_b[idx].mean() - h_a[idx].mean())
    diffs = np.array(diffs)
    print(f"  bootstrap 95% CI (mask-none wrong-rate): [{np.percentile(diffs,2.5):.4f}, {np.percentile(diffs,97.5):.4f}]")
    print("  NOTE: per-draw tests treat correlated draws as independent (overstates power). Item-level below is primary.")

    print("\n-- PRIMARY: item-majority McNemar (strict) --")
    for a1, a2, lab in [("none", "mask", "none vs mask"),
                        ("none", "abstain", "none vs abstain")]:
        m1, m2 = [], []
        for (t, i) in bench:
            v1 = [(data[a1].get((t, s)) or {}).get(i) for s in SEEDS]
            v2 = [(data[a2].get((t, s)) or {}).get(i) for s in SEEDS]
            v1 = [v for v in v1 if v is not None]
            v2 = [v for v in v2 if v is not None]
            if len(v1) < max(3, len(SEEDS) // 2) or len(v2) < max(3, len(SEEDS) // 2):
                continue
            m1.append(1 if sum(wrong_of(v) for v in v1) > len(v1) / 2 else 0)
            m2.append(1 if sum(wrong_of(v) for v in v2) > len(v2) / 2 else 0)
        m1 = np.array(m1, dtype=int)
        m2 = np.array(m2, dtype=int)
        p, net = mcnemar(1 - m1, 1 - m2)
        w2c = int(((m1 == 1) & (m2 == 0)).sum())
        c2w = int(((m1 == 0) & (m2 == 1)).sum())
        print(f"  {lab:<18}: W2C={w2c:>3} C2W={c2w:>3} net={net:+d}  p={p:.4f}  (n_items={len(m1)})")

    print("\n-- cluster bootstrap by item (mask-none wrong-rate diff) --")
    items = sorted({(t, i) for (t, i, s) in rows})
    item_diff = []
    for (t, i) in items:
        a_vals, b_vals = [], []
        for s in SEEDS:
            d1 = (data["none"].get((t, s)) or {}).get(i)
            d2 = (data["mask"].get((t, s)) or {}).get(i)
            if d1 is not None and d2 is not None:
                a_vals.append(wrong_of(d1))
                b_vals.append(wrong_of(d2))
        if a_vals:
            item_diff.append(float(np.mean(b_vals)) - float(np.mean(a_vals)))
    item_diff = np.array(item_diff)
    c_diffs = []
    for _ in range(5000):
        idx = rng.integers(0, len(item_diff), len(item_diff))
        c_diffs.append(item_diff[idx].mean())
    c_diffs = np.array(c_diffs)
    print(f"  cluster bootstrap 95% CI (n_items={len(item_diff)}): [{np.percentile(c_diffs,2.5):.4f}, {np.percentile(c_diffs,97.5):.4f}]")

    print("\n-- majority-of-seeds per item (strict) --")
    for a in arms:
        maj = []
        for (t, i) in bench:
            vals = [(data[a].get((t, s)) or {}).get(i) for s in SEEDS]
            vals = [v for v in vals if v is not None]
            if len(vals) < max(3, len(SEEDS) // 2):
                continue
            wrongs = sum(wrong_of(v) for v in vals)
            maj.append(1 if wrongs > len(vals) / 2 else 0)
        if maj:
            print(f"  {a:<8} majority-wrong = {sum(maj)}/{len(maj)} = {sum(maj)/len(maj):.3f}")

    print("\n-- fired-item behaviour (strict) --")
    fi = []
    for (t, i) in bench:
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


def main(argv=None):
    ap = argparse.ArgumentParser(prog="bench.py", description="Bench build/run/analyze (hard + random).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_b = sub.add_parser("build", help="freeze bench_hard.json and/or bench_random.json")
    p_b.add_argument("--bench", choices=["hard", "random", "all"], default="all")
    p_r = sub.add_parser("run", help="run arms over a bench (skips existing files)")
    p_r.add_argument("--bench", choices=["hard", "random"], required=True)
    p_r.add_argument("--mode", choices=["greedy", "sampled"], required=True)
    p_r.add_argument("--arms", default="none,mask,abstain")
    p_r.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    p_r.add_argument("--static", action="store_true", help="merged static k32 + temporal")
    p_r.add_argument("--transfer", action="store_true",
                     help="greedy full-topic transfer runs (hard bench only)")
    p_a = sub.add_parser("analyze", help="strict analysis + McNemar + bootstrap")
    p_a.add_argument("--bench", choices=["hard", "random"], required=True)
    p_a.add_argument("--static", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "build":
        cmd_build(args)
    elif args.cmd == "run":
        args.seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
        cmd_run(args)
    elif args.cmd == "analyze":
        cmd_analyze(args)


if __name__ == "__main__":
    main()
