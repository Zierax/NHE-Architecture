"""Re-test the reachability claim under the timing law's own constraint.

THE DEFECT THIS FIXES
    `reachability_of_quiet.py` searched fixed windows up to w15 and reported the
    never-fired-wrong population as separable at AUROC 0.713 (L15/w15), concluding
    "reachable, trigger problem". That measurement was taken BEFORE checking the
    timing law against the same data, and it violates the law:

        the commit position over the 463 items is 6 for 180 items, 7 for 200, and
        at most 23; the never-fired-wrong items commit at 5-12. A w15 feature reads
        jumps at tokens 6-15, which is AT OR AFTER the commit for every one of those
        items. Edge's own proven rule (lead >= 1 or the cut is too late) says a
        detector must fire strictly before the commit token.

    So the 0.713 may be measuring post-commit magnitude, i.e. the same post-commit
    transients the cross-arch work already showed are useless to intervene on. A
    conclusion that contradicts the project's own established law and was not
    checked against it should be treated as unproven until it is.

THE CORRECT FEATURE
    For each item, the feature must be the maximum jump over tokens STRICTLY
    BEFORE that item's own commit position - never a fixed window that can extend
    past it. That is the only quantity the timing law permits.

WHAT THIS SCRIPT DOES
    Recomputes the separation using the per-item pre-commit maximum at every layer,
    with NO window parameter, and reports:
      - AUROC of never-fired-wrong vs correct on the pre-commit feature
      - the best layer, and the median lead actually available for those items
      - whether any layer clears the same 0.65 gate the earlier version used

    If the corrected feature does NOT clear 0.65, the earlier "reachable" verdict
    is retracted and the honest conclusion is that the quiet residual is a void
    population after all - which is a different, and stronger, claim.

Usage:
  python analysis/reachability_precommit.py
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


def auc(y, s):
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    ok = ~np.isnan(s)
    if ok.sum() < 8 or len(np.unique(y[ok])) < 2:
        return float("nan")
    order = np.argsort(s[ok])
    ranks = np.empty(len(order), dtype=float)
    sorted_s = s[ok][order]
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    pos = y[ok][order] == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def precommit_feature(row, commit_pos, layer):
    """Max jump strictly before the item's own commit token.

    This is the only feature the timing law allows. It is per-item adaptive and
    can never read at or after the commit, unlike a fixed window.
    """
    jumps = row["jumps"].get(str(layer), [])
    if commit_pos is None or commit_pos <= 0 or not jumps:
        return None
    seg = jumps[:commit_pos]  # strictly before commit_pos
    if not seg:
        return None
    return float(max(seg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", type=float, default=0.65)
    ap.add_argument("--out", default=os.path.join(RES, "reachability_precommit.json"))
    a = ap.parse_args()

    flows_p = os.path.join(RES, "greedy_flows_single_task.json")
    arm_p = os.path.join(RES, "eval_single_task_mask_L19_w5.json")
    if not os.path.exists(flows_p):
        sys.exit("missing greedy_flows_single_task.json; run core/collect_single_task.py")
    if not os.path.exists(arm_p):
        sys.exit("missing eval_single_task_mask_L19_w5.json; run core/run_single_task.py --arm mask")

    flows = {r["id"]: r for r in json.load(open(flows_p, encoding="utf-8"))["items"]}
    arm = json.load(open(arm_p, encoding="utf-8"))["results"]

    groups = {"fired_wrong": [], "never_fired_wrong": [], "correct": []}
    for r in arm:
        f = flows.get(r["id"])
        if f is None:
            continue
        if not r["correct_strict"] and r["fired_at"] is not None:
            g = "fired_wrong"
        elif not r["correct_strict"]:
            g = "never_fired_wrong"
        else:
            g = "correct"
        groups[g].append((f, r.get("commit_pos")))
    print("groups:", {k: len(v) for k, v in groups.items()})
    if len(groups["never_fired_wrong"]) < 8:
        sys.exit("not enough never-fired-wrong items")

    print("\ncommit positions of never-fired-wrong items:")
    cps = sorted(cp for _, cp in groups["never_fired_wrong"] if cp is not None)
    print(f"  min={cps[0]} median={cps[len(cps)//2]} max={cps[-1]}")

    # sanity: how much pre-commit room do they actually have?
    usable = sum(1 for cp in cps if cp and cp > 0)
    print(f"  items with >0 pre-commit tokens: {usable}/{len(cps)}")

    rows = []
    n_layers = max(int(k) for k in flows[next(iter(flows))]["jumps"]) + 1
    for L in range(1, n_layers):
        tv = [precommit_feature(f, cp, L) for f, cp in groups["never_fired_wrong"]]
        cv = [precommit_feature(f, cp, L) for f, cp in groups["correct"]]
        tv_ok = [v for v in tv if v is not None]
        cv_ok = [v for v in cv if v is not None]
        if len(tv_ok) < 8 or len(cv_ok) < 8:
            continue
        y = np.array([1] * len(tv_ok) + [0] * len(cv_ok))
        s = tv_ok + cv_ok
        av = auc(y, s)
        rows.append({"layer": L,
                     "auc": None if np.isnan(av) else round(av, 4),
                     "n_never_with_signal": len(tv_ok)})

    scored = [r for r in rows if r["auc"] is not None]
    scored.sort(key=lambda r: -r["auc"])
    print("\n=== PRE-COMMIT ONLY (the feature the timing law allows) ===")
    print(f"{'layer':>5} {'AUROC':>8} {'n':>4}")
    for r in scored[:10]:
        print(f"{r['layer']:>5} {r['auc']:>8.4f} {r['n_never_with_signal']:>4}")

    best = scored[0]["auc"] if scored else float("nan")
    clears = best >= a.gate

    # direct comparison against the retracted claim
    print("\n=== comparison with the fixed-window claim ===")
    print("  fixed window L15/w15 (previous claim):  AUROC 0.7129 -- but w15 reads")
    print(f"  tokens 6-15, at/after the commit for all but {usable} never-fired items.")
    print(f"  pre-commit-only best layer L{scored[0]['layer']}: AUROC {best}")
    delta = best - 0.7129
    print(f"  delta = {delta:+.4f}")

    print("\n=== VERDICT ===")
    if clears:
        print(f"REACHABLE under the timing law too (best pre-commit AUROC {best} "
              f">= gate {a.gate}). The earlier claim was wrong in its reasoning but")
        print("right in its conclusion; a live pre-commit arm is still worth running.")
        verdict = "REACHABLE_PRECOMMIT"
    else:
        print(f"NOT REACHABLE under the timing law. Best pre-commit AUROC {best} "
              f"< gate {a.gate}.")
        print("The earlier 0.7129 was measuring POST-COMMIT magnitude, which Edge's")
        print("own proven rule says cannot be intervened on. The correct conclusion")
        print("is the opposite of the previous one: the quiet residual IS a void")
        print("population, and no trigger of this family can reach it. Direction")
        print("reverts to fact-level repair (NTW), not detector work.")
        verdict = "VOID_PRECOMMIT"

    payload = {"verdict": verdict, "gate": a.gate,
               "best_precommit_auc": best,
               "best_layer": scored[0]["layer"] if scored else None,
               "retracted_fixed_window_auc": 0.7129,
               "delta_vs_fixed_window": round(delta, 4),
               "n_never_with_signal": scored[0]["n_never_with_signal"] if scored else 0,
               "median_commit_pos_never": cps[len(cps) // 2] if cps else None,
               "top_layers": scored[:12]}
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=1)
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()