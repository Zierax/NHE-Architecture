"""The direction-setting experiment: is the quiet population REACHABLE or not?

THE QUESTION THIS ANSWERS
    The 463-item benchmark left the action arm at fixes=3, breaks=0, p=0.25. The
    measured cause is that the detector fires on only 15.3% of the wrong items
    (recall 0.153). That splits the residual error into two populations:

        fired-but-wrong  9 items  - the detector saw them; repair rate 0.333
        never-fired     50 items  - the detector did not fire on them at all

    The whole next direction depends on which population the 50 belong to, and
    that has NOT been measured on this benchmark:

        HYPOTHESIS R (reachable ceiling)
            The 50 items DO carry a pre-commit spike, but at a lower magnitude
            than the current threshold admits. Then a better trigger (different
            layer, lower threshold, richer feature) can raise recall, and the
            method's ceiling is a DETECTOR problem that is worth attacking.

        HYPOTHESIS V (void population)
            The 50 items carry NO pre-commit excess at all, matching NTW's
            Static Memory Void prediction. Then no trigger can reach them, more
            detector work is wasted effort, and the honest direction is to stop
            extending the runtime arm and publish the taxonomy instead.

    This script measures the pre-commit feature distribution of the never-fired
    population against both the correct population and the fired-wrong
    population, on the committed flows. It answers R vs V with data.

METHOD
    For each item, take the deployed feature family (max hidden-state jump at the
    detector layer over the pre-commit window) from the saved greedy flows, and
    compare distributions:
      - never-fired wrong  vs  correct   (is there any separation at all?)
      - never-fired wrong  vs  fired-wrong (does the fired subset look different?)
    It reports AUROC for never-fired-wrong vs correct at EVERY layer x window, so
    a better (layer, window) is not assumed away. If some cell separates them at
    usable AUROC, the ceiling is reachable and the direction is a better trigger.
    If no cell does, the void population is confirmed and the direction is the
    taxonomy write-up, not more detector work.

    Flows are required: results/greedy_flows_africa.npz covers only the 54 Africa
    items, so this script requires a single-task flow file. If it is absent the
    script says so and the answer must come from a collect run; it never guesses.

Usage:
  python analysis/reachability_of_quiet.py
  python analysis/reachability_of_quiet.py --flows <npz>
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
        avg = (i + j) / 2.0 + 1.0
        ranks[i:j + 1] = avg
        i = j + 1
    pos = y[ok][order] == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flows", default=None)
    ap.add_argument("--layer", type=int, default=19, help="deployed detector layer")
    ap.add_argument("--max-layer", type=int, default=25)
    ap.add_argument("--max-window", type=int, default=15)
    a = ap.parse_args()

    bench = os.path.join(RES, "bench_single_task.json")
    score_p = os.path.join(RES, "single_task_score.json")
    mask_p = os.path.join(RES, "eval_single_task_mask_L19_w5.json")
    for p in (bench, score_p, mask_p):
        if not os.path.exists(p):
            sys.exit(f"missing {p}; run build/score/run first")

    flows_p = a.flows or os.path.join(RES, "greedy_flows_single_task.json")
    if not os.path.exists(flows_p):
        print("NO SINGLE-TASK FLOWS on disk.")
        print("This experiment needs hidden states for the 463 items.")
        print("Collect them with:")
        print("  python core/collect_single_task.py")
        print("Without them the reachability question cannot be answered from data,")
        print("and it will NOT be guessed. Rerun with --flows <path> when collected.")
        return 2

    flows = json.load(open(flows_p, encoding="utf-8"))
    mask = json.load(open(mask_p, encoding="utf-8"))
    arm = {r["id"]: r for r in mask["results"]}

    def group(r):
        if not r["correct_strict"] and r["fired_at"] is not None:
            return "fired_wrong"
        if not r["correct_strict"]:
            return "never_fired_wrong"
        return "correct"

    by_id = {r["id"]: r for r in flows["items"]}
    groups = {"fired_wrong": [], "never_fired_wrong": [], "correct": []}
    missing = 0
    for r in mask["results"]:
        f = by_id.get(r["id"])
        if f is None:
            missing += 1
            continue
        g = group(r)
        groups[g].append(f)
    if missing:
        print(f"note: {missing} arm items had no flow row and were skipped")

    for k, v in groups.items():
        print(f"{k}: n={len(v)}")
    if len(groups["never_fired_wrong"]) < 8:
        print("not enough never-fired-wrong items to test separation; collect flows first")
        return 2

    target = groups["never_fired_wrong"]
    correct = groups["correct"]
    fired = groups["fired_wrong"]
    y_q = np.array([1] * len(target) + [0] * len(correct))
    y_f = np.array([1] * len(target) + [0] * len(fired))

    def feat(rows, L, W):
        """Max jump over the first W generated tokens at layer L."""
        out = []
        for r in rows:
            arr = r["jumps"].get(str(L), [])
            seg = arr[:W] if W <= len(arr) else arr
            out.append(float(max(seg)) if len(seg) else 0.0)
        return out

    best = []
    for L in range(1, a.max_layer + 1):
        for W in range(1, a.max_window + 1):
            q = feat(target, L, W)
            c = feat(correct, L, W)
            f = feat(fired, L, W)
            a_qc = auc(y_q, q + c)
            a_qf = auc(y_f, q + f)
            best.append({"layer": L, "window": W,
                         "auc_vs_correct": None if np.isnan(a_qc) else round(a_qc, 4),
                         "auc_vs_fired": None if np.isnan(a_qf) else round(a_qf, 4)})

    scored = [b for b in best if b["auc_vs_correct"] is not None]
    scored.sort(key=lambda b: -b["auc_vs_correct"])
    print("\n=== can the NEVER-FIRED wrong population be separated from correct? ===")
    print("top cells by AUROC (never-fired-wrong vs correct):")
    for b in scored[:10]:
        print(f"  L{b['layer']:<2} w{b['window']:<3} auc_vs_correct={b['auc_vs_correct']:<8} "
              f"auc_vs_fired={b['auc_vs_fired']}")

    best_auc = scored[0]["auc_vs_correct"] if scored else float("nan")
    reachable = best_auc >= 0.65
    print(f"\nbest AUROC ever separating the never-fired population = {best_auc}")
    if reachable:
        print("VERDICT: R (reachable ceiling) - a pre-commit excess exists in the")
        print("never-fired wrong items. The bottleneck is the TRIGGER, not the")
        print("mechanism. Next direction: raise detector recall (feature, layer,")
        print("threshold) without losing the zero-break property.")
    else:
        print("VERDICT: V (void population) - no layer/window separates the")
        print("never-fired wrong items from correct ones at usable AUROC. This")
        print("matches NTW's Static Memory Void prediction on real data. Next")
        print("direction: stop extending the runtime arm; publish the taxonomy")
        print("and route these items to fact-level repair.")

    out = os.path.join(RES, "reachability_of_quiet.json")
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"n_groups": {k: len(v) for k, v in groups.items()},
                   "best_auc_vs_correct": best_auc,
                   "verdict": "REACHABLE" if reachable else "VOID",
                   "top_cells": scored[:20]}, fh, indent=1)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    raise SystemExit(main())
