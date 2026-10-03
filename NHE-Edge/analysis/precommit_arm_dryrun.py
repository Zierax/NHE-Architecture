"""Before spending an hour of CPU on a live arm, check whether it can succeed.

WHY THIS EXISTS
    `reachability_precommit.py` found the never-fired-wrong population is separable
    at pre-commit AUROC 0.6925 (L11), above the 0.65 gate. That justifies a live
    L11 arm. But AUROC is a population statistic; the arm only succeeds if, among
    the specific items L11 would fire on, a useful number are actually wrong AND
    actually repairable. This script computes that, offline, from data already
    collected:

      - which items a correct-calibrated p90 threshold at L11 (pre-commit) fires on
      - how many of those were wrong (precision), how many errors it reaches (recall)
      - the lead available for each fired-wrong item (the timing law's requirement)
      - the EXPECTED fixes at the observed per-fired repair rate (0.333), and
        whether that clears the 6-fix significance threshold

    If the expected fixes are far below 6, the live run is not worth the CPU and
    the honest recommendation is recorded instead.

METHOD
    The threshold is calibrated the way the deployed detector calibrates: a
    percentile of the CORRECT items' feature values (never the wrong ones), so the
    same false-positive budget is used. The feature is the per-item pre-commit
    maximum at L11. A fired item is "intervenable" if its commit_pos exceeds the
    token at which the threshold is crossed by at least 1 (lead >= 1).

Usage:
  python analysis/precommit_arm_dryrun.py
"""
import argparse
import json
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE = os.path.dirname(os.path.abspath(__file__))
EDGE_ROOT = os.path.dirname(BASE)
RES = os.path.join(EDGE_ROOT, "results")
sys.path.insert(0, os.path.join(EDGE_ROOT, "core"))


def first_cross(jumps, thr):
    """First token index where the jump exceeds thr, else None."""
    for t, v in enumerate(jumps):
        if v > thr:
            return t
    return None


def precommit_max(jumps, commit_pos):
    if commit_pos is None or commit_pos <= 0:
        return None, None
    seg = jumps[:commit_pos]
    if not seg:
        return None, None
    t = first_cross(seg, float("-inf"))
    return float(max(seg)), (first_cross(seg, 1e18))  # cross index within seg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=11)
    ap.add_argument("--pct", type=int, default=90)
    ap.add_argument("--repair-rate", type=float, default=0.333,
                    help="observed per-fired-wrong repair rate from the L19 arm")
    ap.add_argument("--need", type=int, default=6,
                    help="fixes required for significance (power analysis)")
    a = ap.parse_args()

    flows_p = os.path.join(RES, "greedy_flows_single_task.json")
    arm_p = os.path.join(RES, "eval_single_task_mask_L19_w5.json")
    for p in (flows_p, arm_p):
        if not os.path.exists(p):
            sys.exit(f"missing {p}")
    flows = {r["id"]: r for r in json.load(open(flows_p, encoding="utf-8"))["items"]}
    arm = {r["id"]: r for r in json.load(open(arm_p, encoding="utf-8"))["results"]}

    rows = []
    for i, f in flows.items():
        r = arm.get(i)
        if r is None:
            continue
        cp = r.get("commit_pos")
        jumps = f["jumps"].get(str(a.layer), [])
        if cp is None or cp <= 0 or not jumps:
            continue
        seg = jumps[:cp]
        rows.append({"id": i, "q": f["question"], "feat": float(max(seg)),
                     "jumps_pre": seg, "commit_pos": cp,
                     "wrong_before": not r["correct_strict"]})

    if not rows:
        sys.exit("no usable rows")

    correct = [x["feat"] for x in rows if not x["wrong_before"]]
    thr = float(np.percentile(correct, a.pct))
    print(f"layer L{a.layer} pre-commit feature, threshold p{a.pct} on correct "
          f"items = {thr:.2f}  (n_correct={len(correct)})")

    fired, fired_wrong, intervenable_wrong = [], [], []
    for x in rows:
        t = first_cross(x["jumps_pre"], thr)
        if t is None:
            continue
        x["fire_at"] = t
        x["lead"] = x["commit_pos"] - t
        fired.append(x)
        if x["wrong_before"]:
            fired_wrong.append(x)
            if x["lead"] >= 1:
                intervenable_wrong.append(x)

    n_wrong = sum(1 for x in rows if x["wrong_before"])
    prec = len(fired_wrong) / len(fired) if fired else 0.0
    rec = len(fired_wrong) / n_wrong if n_wrong else 0.0
    expected = len(intervenable_wrong) * a.repair_rate

    print(f"\ndry-run L{a.layer}/p{a.pct}:")
    print(f"  fires on {len(fired)}/{len(rows)} items")
    print(f"  of which wrong-before: {len(fired_wrong)}  (precision {prec:.3f})")
    print(f"  errors reached: {len(fired_wrong)}/{n_wrong}  (recall {rec:.3f})")
    print(f"  of those, lead>=1 (intervenable): {len(intervenable_wrong)}")
    print(f"  expected fixes at repair rate {a.repair_rate}: {expected:.1f}")
    print(f"  significance needs {a.need} fixes: "
          f"{'CLEARS' if expected >= a.need else 'DOES NOT CLEAR'}")

    print(f"\nintervenable wrong items the arm would target:")
    for x in sorted(intervenable_wrong, key=lambda r: r["lead"]):
        print(f"  lead={x['lead']:>2} {x['q'][:52]}")

    payload = {"layer": a.layer, "pct": a.pct, "threshold": thr,
               "n_rows": len(rows), "n_fired": len(fired),
               "n_fired_wrong": len(fired_wrong), "precision": round(prec, 4),
               "n_errors": n_wrong, "recall": round(rec, 4),
               "n_intervenable_wrong": len(intervenable_wrong),
               "expected_fixes": round(expected, 2), "need": a.need,
               "clears_significance": bool(expected >= a.need),
               "target_items": [{"id": x["id"], "question": x["q"], "lead": x["lead"]}
                                for x in sorted(intervenable_wrong,
                                                key=lambda r: r["lead"])]}
    out = os.path.join(RES, "precommit_arm_dryrun.json")
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=1, ensure_ascii=False)
    print(f"\nsaved {out}")
    if not payload["clears_significance"]:
        print("\nThe dry run does NOT clear significance. A live arm is still the")
        print("decisive test, but the prior now says the residual is mostly")
        print("unreachable even under the corrected pre-commit feature.")


if __name__ == "__main__":
    main()