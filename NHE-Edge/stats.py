"""Unified stats tool. Replaces strict_final.py, battery_analysis.py,
significance_final.py, significance_s5.py (moved to legacy/ — kept for
history, not deleted). Output is identical to the originals.

Usage:
  python stats.py strict          # strict-first-sentence scoring tables
  python stats.py battery         # Africa 5-seed battery (270 draws)
  python stats.py significance    # static-arm McNemar + Wilson tables
"""
import argparse
import json
import os
import re
import sys
import unicodedata
from math import comb

import numpy as np
from scipy.stats import binomtest

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import topics  # noqa: E402

RES = os.path.join(BASE, "results")

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


def answers(q, r):
    return [r["answer"]] + list(ALTS.get(q, []))


def strict_score(path, label):
    path = path if os.path.isabs(path) else os.path.join(BASE, path)
    ev = json.load(open(path, encoding="utf-8"))
    n = len(ev["results"])
    subj = 0
    strict = 0
    for r in ev["results"]:
        a = answers(r["question"], r)
        g = r["generated"] if "generated" in r else r["samples"][0]
        if any(norm(x) in norm(g) for x in a):
            subj += 1
        if fs(g) and any(norm(x) in fs(g) for x in a):
            strict += 1
    print(f"{label:<18} n={n:>3}  substring_hall={1-subj/n:>7.3f}  strict_firstsent_hall={1-strict/n:>7.3f}")
    return n, subj, strict


STRICT_TABLES = [
    ("== AFRICA greedy, strict (first-sentence) metric with alternatives ==", [
        ("results/eval_africa_baseline.json", "baseline"),
        ("results/eval_africa_k32_midwrong.json", "static k32"),
        ("results/eval_africa_k128_wrong.json", "static k128"),
        ("results/eval_runtime_africa_jump_gt_L19_t90_mask.json", "rt w5 hard"),
        ("results/eval_runtime_africa_jump_gt_L19_t90_mask_sft0.3.json", "rt w5 soft"),
        ("results/eval_runtime_africa_jump_gt_L19_t90_mask_w4.json", "rt w4 hard"),
        ("results/eval_runtime_africa_jump_gt_L19_t90_mask_w3.json", "rt w3 hard"),
    ]),
    ("\n== AFRICA_LARGEST / WORLD_TRICKY greedy, strict ==", [
        ("results/eval_africa_largest_baseline.json", "al baseline"),
        ("results/eval_runtime_africa_largest_jump_gt_L19_t90_mask_sft0.3.json", "al rt soft"),
        ("results/eval_world_tricky_baseline.json", "wt baseline"),
        ("results/eval_world_tricky_k32_midwrong.json", "wt k32"),
        ("results/eval_world_tricky_k128_wrong.json", "wt k128"),
        ("results/eval_runtime_world_tricky_jump_gt_L19_t90_mask_sft0.3.json", "wt rt soft"),
    ]),
    ("\n== EUROPE greedy, strict ==", [
        ("results/eval_europe_baseline.json", "europe baseline"),
        ("results/eval_europe_k32_midwrong.json", "europe k32"),
        ("results/eval_europe_k128_wrong.json", "europe k128"),
        ("results/eval_runtime_europe_jump_gt_L19_t90_mask_sft0.3.json", "europe rt soft"),
    ]),
    ("\n== NEW error-rich bench (greedy, strict) ==", [
        ("results/eval_runtime_world_cap_traps_jump_gt_L19_t90_none.json", "cap_traps baseline"),
        ("results/eval_runtime_world_cap_traps_jump_gt_L19_t90_mask_sft0.3.json", "cap_traps rt mask"),
        ("results/eval_runtime_world_cap_traps_jump_gt_L19_t90_abstain.json", "cap_traps rt abstain"),
        ("results/eval_runtime_world_largest_jump_gt_L19_t90_none.json", "world_largest baseline"),
        ("results/eval_runtime_world_largest_jump_gt_L19_t90_mask_sft0.3.json", "world_largest rt mask"),
        ("results/eval_runtime_world_largest_jump_gt_L19_t90_abstain.json", "world_largest rt abstain"),
    ]),
]


def cmd_strict(_args):
    for title, rows in STRICT_TABLES:
        print(title)
        for f, label in rows:
            strict_score(f, label)


BATTERY_SEEDS = [1000, 1001, 1002, 1003, 1004]


def _battery_load(prefix, seed):
    ev = json.load(open(os.path.join(RES, f"eval_runtime_africa_{prefix}_s{seed}.json"), encoding="utf-8"))
    return {r["question"]: r["correct"] for r in ev["results"]}, ev


def cmd_battery(_args):
    none, mask = {}, {}
    for s in BATTERY_SEEDS:
        none[s], _ = _battery_load("jump_gt_L19_t90_none", s)
        mask[s], _ = _battery_load("jump_gt_L19_t90_mask_sft0.3", s)
    qs = list(none[BATTERY_SEEDS[0]].keys())
    assert len(qs) == 54

    def majority(d):
        return {q: int(sum(d[s][q] for s in BATTERY_SEEDS) >= 3) for q in qs}

    mn, mm = majority(none), majority(mask)
    n_hall_n = sum(1 for q in qs if not mn[q])
    n_hall_m = sum(1 for q in qs if not mm[q])
    print(f"africa majority-of-5 (54 items): baseline {n_hall_n}/54={n_hall_n/54:.4f} -> soft-mask {n_hall_m}/54={n_hall_m/54:.4f}")
    fixed = [q for q in qs if (not mn[q]) and mm[q]]
    broken = [q for q in qs if mn[q] and (not mm[q])]
    print(f"  fixes (W2C): {len(fixed)} {fixed}")
    print(f"  breaks(C2W): {len(broken)} {broken}")

    yn = np.array([none[s][q] for s in BATTERY_SEEDS for q in qs], dtype=int)
    ym = np.array([mask[s][q] for s in BATTERY_SEEDS for q in qs], dtype=int)
    hn, hm = 1 - yn, 1 - ym
    print(f"\nsample-level (n=270): hall none={hn.mean():.4f} ({hn.sum()}) mask={hm.mean():.4f} ({hm.sum()})")

    def mcnemar(a, b):
        b01 = int(((a == 1) & (b == 0)).sum())
        b10 = int(((a == 0) & (b == 1)).sum())
        if b01 + b10 == 0:
            return float("nan")
        return binomtest(min(b01, b10), b01 + b10, 0.5, alternative="two-sided").pvalue, b10 - b01

    p, net = mcnemar(yn, ym)
    print(f"  McNemar: W2C={int(((yn==0)&(ym==1)).sum())} C2W={int(((yn==1)&(ym==0)).sum())} net={net} p={p:.4f}")

    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(5000):
        idx = rng.integers(0, len(hn), len(hn))
        diffs.append((hm[idx].mean() - hn[idx].mean()))
    diffs = np.array(diffs)
    print(f"  bootstrap 95% CI (mask - none hall): [{np.percentile(diffs,2.5):.4f}, {np.percentile(diffs,97.5):.4f}]")

    w2c = sum(1 for q in qs if (not mn[q]) and mm[q])
    c2w = sum(1 for q in qs if mn[q] and (not mm[q]))
    ip = 1.0 if w2c + c2w == 0 else binomtest(min(w2c, c2w), w2c + c2w, 0.5, alternative="two-sided").pvalue
    print(f"item-level majority: W2C={w2c} C2W={c2w} p={ip:.4f}")

    st = json.load(open(os.path.join(RES, "eval_africa_baseline_s5.json"), encoding="utf-8"))
    print(f"\nstatic baseline_s5 majority rate={st.get('hallucination_rate')} (reference)")
    for m in ["k32_midwrong", "k128_wrong"]:
        try:
            x = json.load(open(os.path.join(RES, f"eval_africa_{m}_s5.json"), encoding="utf-8"))
            print(f"  static {m} s5 majority={x.get('hallucination_rate')}")
        except FileNotFoundError:
            pass


def _sig_load(topic, tag):
    return json.load(open(os.path.join(RES, f"eval_{topic}_{tag}.json"), encoding="utf-8"))


def _wilson(k, n):
    z = 1.96
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (p, c - h, c + h)


def _sig_mcnemar(b_fix, m_fix):
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


def _sample_pairs(b, m):
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


def _bootstrap_net(b_fix, m_fix, n_boot=5000, seed=0):
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


def cmd_significance(_args):
    topics5 = ["africa", "europe", "elements", "asia", "us_states"]
    masks = ["k32_midwrong", "k128_wrong"]
    print(f"{'topic':<10}{'mask':<14}{'n':>4}{'hall_b':>8}{'hall_m':>8}{'fix':>5}{'brk':>5}{'p':>8}{'net95CI':>18}")
    for topic in topics5:
        b = _sig_load(topic, "baseline_s5")
        for mask in masks:
            m = _sig_load(topic, mask + "_s5")
            b_fix, m_fix = _sample_pairs(b, m)
            k_h_b = sum(1 for v in b_fix.values() if not v)
            k_h_m = sum(1 for v in m_fix.values() if not v)
            n = len(b_fix)
            mc = _sig_mcnemar(b_fix, m_fix)
            ci = _bootstrap_net(b_fix, m_fix)
            mc_s = f"{mc['fixed']:>5}{mc['broke']:>5}{mc['exact_p']:>8.3f}" if mc else f"{'-':>5}{'-':>5}{'1.000':>8}"
            print(f"{topic:<10}{mask:<14}{n:>4}{k_h_b/n:>8.3f}{k_h_m/n:>8.3f}{mc_s}{f'[{ci[0]:.0f},{ci[1]:.0f}]':>18}")

    print("\nbaseline sampled hall rates:")
    for topic in topics5:
        b = _sig_load(topic, "baseline_s5")
        k = sum(1 for r in b["results"] for f in r["correct"] if not f)
        n = sum(len(r["correct"]) for r in b["results"])
        p, lo, hi = _wilson(k, n)
        print(f"  {topic:<10} {p:.3f} [{lo:.3f},{hi:.3f}] n={n}")

    print("\n=== MAJORITY (item-level, 5 samples) ===")
    print(f"{'topic':<10}{'mask':<16}{'hall':>8}{'wilson95':>18}{'mcnemar':>28}")
    rows = [("africa", "baseline"), ("africa", "k32_midwrong"), ("africa", "k128_wrong"),
            ("europe", "baseline"), ("europe", "k32_midwrong"), ("europe", "k128_wrong"),
            ("elements", "baseline"), ("elements", "k32_midwrong"), ("elements", "k128_wrong")]
    for topic, tag in rows:
        d = _sig_load(topic, tag + "_s5")
        n = d["n"]
        k_hall = n - d["n_correct"]
        p, lo, hi = _wilson(k_hall, n)
        print(f"{topic:<10}{tag:<16}{p:>8.3f}{f'[{lo:.3f},{hi:.3f}]':>18}")

    print("\n=== SAMPLE-LEVEL (270 instances per cell) ===")
    print(f"{'topic':<10}{'mask':<16}{'hall':>8}{'wilson95':>18}{'mcnemar':>28}")
    for topic, tag in rows:
        if tag == "baseline":
            continue
        b = _sig_load(topic, "baseline_s5")
        m = _sig_load(topic, tag + "_s5")
        b_fix, m_fix = _sample_pairs(b, m)
        k_hall = sum(1 for v in m_fix.values() if not v)
        n = len(m_fix)
        p, lo, hi = _wilson(k_hall, n)
        mc = _sig_mcnemar(b_fix, m_fix)
        mc_s = f"fix={mc['fixed']} brk={mc['broke']} p={mc['exact_p']:.3f}" if mc else "no discordant"
        print(f"{topic:<10}{tag:<16}{p:>8.3f}{f'[{lo:.3f},{hi:.3f}]':>18}{mc_s:>28}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="stats.py", description="Strict scoring, battery, significance.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("strict", help="strict-first-sentence scoring tables")
    sub.add_parser("battery", help="Africa 5-seed battery (270 draws)")
    sub.add_parser("significance", help="static-arm McNemar + Wilson tables")
    args = ap.parse_args(argv)
    if args.cmd == "strict":
        cmd_strict(args)
    elif args.cmd == "battery":
        cmd_battery(args)
    elif args.cmd == "significance":
        cmd_significance(args)


if __name__ == "__main__":
    main()
