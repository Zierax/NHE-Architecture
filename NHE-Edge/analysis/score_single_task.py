"""Score the single-task benchmark arms: paired significance, landing stats,
and the taxonomy split between transient drifts and static voids.

WHY THIS EXISTS
    The Africa headline (7/54 -> 5/54, 2 fixes, 0 breaks) is clean but not
    statistically significant: exact McNemar on 2 discordant pairs is p = 0.5.
    `power_single_task.py` showed that 6 fixes with 0 breaks is the minimum for
    p < 0.05. This script answers that requirement against real data, and reports
    the per-family breakdown because the whole taxonomy claim rests on transient
    drifts being the repairable subset.

METRICS, in order of authority
  primary   strict first sentence, whole-token match, union answer set
            (analysis/scoring.py, tested in analysis/test_scoring.py)
  secondary legacy substring, reported only to show the gap

ANALYSIS
  - W2C / C2W with an EXACT McNemar test (the honest paired test)
  - Wilson confidence interval on the net change, because a point estimate on a
    paired binary outcome is misleading when the discordant count is small
  - per-family breakdown (capital / largest / element)
  - firing behaviour: how many items fired, and the repair rate PER FIRED item,
    which is the quantity that generalises
  - the landing split: of the items still wrong after the arm, how many fired at
    all (candidate transient drifts) vs never fired (candidate static voids)

Usage:
  python analysis/score_single_task.py --arms none mask
  python analysis/score_single_task.py --arms none mask --family capital
"""
import argparse
import json
import math
import os
import sys

from scipy.stats import binomtest

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE = os.path.dirname(os.path.abspath(__file__))
EDGE_ROOT = os.path.dirname(BASE)
RES = os.path.join(EDGE_ROOT, "results")


def exact_mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return float(binomtest(min(b, c), n, 0.5).pvalue)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def load_arm(name):
    p = name if os.path.isabs(name) else os.path.join(RES, name)
    if not p.endswith(".json"):
        p += ".json"
    if not os.path.exists(p):
        sys.exit(f"missing arm file: {p}")
    d = json.load(open(p, encoding="utf-8"))
    return p, d


def index(d, key="correct_strict"):
    return {r["question"]: r for r in d["results"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["eval_single_task_none_full",
                                                   "eval_single_task_mask_L19_w5_sft0.3"])
    ap.add_argument("--baseline", default=None,
                    help="arm used as the paired baseline (default: first --arms)")
    ap.add_argument("--out", default=os.path.join(RES, "single_task_score.json"))
    ap.add_argument("--metric", default="strict", choices=["strict", "substring"])
    a = ap.parse_args()

    paths, datas = [], []
    for name in a.arms:
        p, d = load_arm(name)
        paths.append(os.path.basename(p))
        datas.append(d)
    base_name = a.baseline or paths[0]
    bi = paths.index(base_name)
    base = datas[bi]
    base_i = index(base, f"correct_{a.metric}")

    print(f"metric: {a.metric}")
    print(f"baseline: {os.path.basename(base_name)}  "
          f"n={base['n']} wrong={base[f'strict_wrong'] if a.metric=='strict' else base['substring_wrong']}")

    out_rows = []
    for p, d in zip(paths, datas):
        if p == base_name:
            continue
        arm_i = index(d, f"correct_{a.metric}")
        common = sorted(set(base_i) & set(arm_i))
        b = sum(1 for q in common if not base_i[q][f"correct_{a.metric}"]
                and arm_i[q][f"correct_{a.metric}"])
        c = sum(1 for q in common if base_i[q][f"correct_{a.metric}"]
                and not arm_i[q][f"correct_{a.metric}"])
        p_val = exact_mcnemar(b, c)
        lo, hi = wilson(b - c, len(common))
        fired = [q for q in common if arm_i[q].get("fired_at") is not None]
        fired_wrong_before = [q for q in fired
                              if not base_i[q][f"correct_{a.metric}"]]
        repairs_per_fire = (b / len(fired_wrong_before)) if fired_wrong_before else 0.0

        # landing split among items still wrong after this arm
        still_wrong = [q for q in common if not arm_i[q][f"correct_{a.metric}"]]
        landed = [q for q in still_wrong if arm_i[q].get("fired_at") is not None]
        quiet = [q for q in still_wrong if arm_i[q].get("fired_at") is None]

        # per-family
        fams = {}
        for it in base["results"]:
            fams.setdefault(it["family"], []).append(it["question"])
        fam_rows = {}
        for fam, qs in fams.items():
            qs = [q for q in qs if q in common]
            fb = sum(1 for q in qs if not base_i[q][f"correct_{a.metric}"]
                     and arm_i[q][f"correct_{a.metric}"])
            fc = sum(1 for q in qs if base_i[q][f"correct_{a.metric}"]
                     and not arm_i[q][f"correct_{a.metric}"])
            fam_rows[fam] = {"n": len(qs), "fixes": fb, "breaks": fc,
                             "p": round(exact_mcnemar(fb, fc), 5)}

        row = {"arm": os.path.basename(p), "n_paired": len(common),
               "fixes": b, "breaks": c, "net": b - c,
               "p_exact_mcnemar": round(p_val, 5),
               "significant_p05": p_val < 0.05,
               "net_wilson95": [round(lo, 4), round(hi, 4)],
               "n_fired": len(fired),
               "fired_and_wrong_before": len(fired_wrong_before),
               "repairs_per_fired_wrong": round(repairs_per_fire, 4),
               "still_wrong_total": len(still_wrong),
               "still_wrong_that_fired": len(landed),
               "still_wrong_quiet": len(quiet),
               "per_family": fam_rows}
        out_rows.append(row)

        print(f"\n--- {os.path.basename(p)} ---")
        print(f"paired n={len(common)}  fixes={b}  breaks={c}  net={b - c}")
        print(f"exact McNemar p={p_val:.5f}  significant@0.05={p_val < 0.05}")
        print(f"net 95% CI (Wilson) [{lo:.4f}, {hi:.4f}]")
        print(f"fired={len(fired)} (of which wrong-before={len(fired_wrong_before)})"
              f"  repairs/fired-wrong={repairs_per_fire:.3f}")
        print(f"still wrong: {len(still_wrong)} total | fired {len(landed)} "
              f"(candidate drifts) | quiet {len(quiet)} (candidate voids)")
        print("per family:")
        for fam, fr in sorted(fam_rows.items()):
            print(f"  {fam:<9} n={fr['n']:<4} fixes={fr['fixes']:<3} "
                  f"breaks={fr['breaks']:<3} p={fr['p']}")

    payload = {"metric": a.metric, "baseline": os.path.basename(base_name),
               "baseline_n": base["n"], "arms": out_rows}
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=1)
    print(f"\nsaved {a.out}")
    print("\nReading: 'significant' is the honest bar (exact McNemar on the paired")
    print("strict metric). repairs_per_fired_wrong is the quantity that must hold")
    print("on a new domain for the method to generalise. The quiet remainder is")
    print("the Static Memory Void population that NTW explains.")


if __name__ == "__main__":
    main()
