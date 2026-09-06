"""Offline sweeps over saved greedy flows (no model needed).
Replaces sweep_thresholds.py and sweep_windows.py (moved to legacy/).

Usage:
  python sweep.py thresholds   # p70-p95 threshold sweep + 1:1 validation vs live run
  python sweep.py windows       # window x threshold grid (w2-w6 x p80-p95)
"""
import argparse
import json
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, "results")
LAYER = 19


def load_flows():
    d = np.load(os.path.join(RES, "greedy_flows_africa.npz"), allow_pickle=True)
    flows = list(d["flows"])
    texts = d["texts"]
    labels = np.array(d["labels"]).astype(int)
    y = 1 - labels
    jumps_all = []
    for f in flows:
        h = f[:, LAYER + 1, :].astype(np.float32)
        dlt = h[1:] - h[:-1]
        jumps_all.append(np.linalg.norm(dlt, axis=1))
    return flows, texts, y, jumps_all


def cmd_thresholds(_args):
    flows, texts, y, jumps_all = load_flows()
    hall_idx = [i for i in range(len(y)) if y[i] == 1]
    baseline = json.load(open(os.path.join(RES, "eval_africa_baseline.json"), encoding="utf-8"))
    base_correct = {r["question"]: r["correct"] for r in baseline["results"]}
    peak_t = [int(np.argmax(j)) for j in jumps_all]
    truth_mask = y == 0
    early_max = np.array([j[: min(10, len(j))].max() for j in jumps_all])

    print(f"hall items: {[texts[i][:30] for i in hall_idx]}")
    print("per-hall peak (commit) token:", [(i, peak_t[i], texts[i][:40]) for i in hall_idx])

    reported = json.load(open(os.path.join(RES, "eval_runtime_africa_jump_gt_L19_t90_mask.json"), encoding="utf-8"))
    rep_fire = {r["question"]: r["fired_at"] for r in reported["results"]}

    for p in (70, 75, 80, 85, 88, 90, 92, 95):
        thr = np.percentile(early_max[truth_mask], p)
        eff, any_fire, fp = 0, 0, 0
        fired_questions = {}
        for i in range(len(y)):
            j = jumps_all[i]
            cand = np.where(j > thr)[0]
            if len(cand) == 0:
                continue
            ft = int(cand[0])
            any_fire += 1
            if y[i] == 0:
                fp += 1
                fired_questions[texts[i][:24]] = ft
            else:
                eff += 1 if ft < peak_t[i] else 0
        hall_eff = [(i, int(np.where(jumps_all[i] > thr)[0][0])) for i in hall_idx
                    if len(np.where(jumps_all[i] > thr)[0]) and int(np.where(jumps_all[i] > thr)[0][0]) < peak_t[i]]
        print(f"p{p}: thr={thr:.0f} fires={any_fire} (fp={fp}, hall_precommit={len(hall_eff)}/{len(hall_idx)}) "
              f"hall_precommit={[texts[i][:18] for i, _ in hall_eff]}")

    print()
    print("validation: simulated fired_at vs recorded (early t90):")
    sim_fire = {}
    for i in range(len(y)):
        j = jumps_all[i]
        thr = np.percentile(early_max[truth_mask], 90)
        cand = np.where(j > thr)[0]
        sim_fire[texts[i]] = int(cand[0]) if len(cand) else None
    mismatch = 0
    for q, r in rep_fire.items():
        s = sim_fire.get(q)
        if s != r:
            mismatch += 1
            print(f"  MISMATCH {q}: sim={s} recorded={r}")
    print(f"mismatches: {mismatch}/{len(rep_fire)}")


def first_cross(j, thr, window):
    c = np.where(j[:window] > thr)[0]
    return int(c[0]) + 1 if len(c) else None


def cmd_windows(_args):
    flows, texts, y, jumps_all = load_flows()
    truth = y == 0
    hall_idx = [i for i in range(len(y)) if y[i] == 1]
    early_max = np.array([j[: min(10, len(j))].max() for j in jumps_all])

    print(f"{'w':>3} {'p':>3} {'fires':>5} {'hall_fires':>10} {'fixable':>8} {'fp':>4}  fixable_fired")
    for w in (2, 3, 4, 5, 6):
        for p in (80, 85, 90, 95):
            thr = np.percentile(early_max[truth], p)
            fires = 0
            hall_fires = 0
            fixable_hits = []
            fp = 0
            for i in range(len(y)):
                ft = first_cross(jumps_all[i], thr, w)
                if ft is None:
                    continue
                fires += 1
                if y[i] == 1:
                    hall_fires += 1
                    if i in (17, 20, 41):  # Eswatini/Gambia/Senegal by known greedy index
                        fixable_hits.append((i, ft))
                else:
                    fp += 1
            print(f"{w:>3} {p:>3} {fires:>5} {hall_fires:>10} {len(fixable_hits):>8} {fp:>4}  {[(i, ft) for i, ft in fixable_hits]}")

    print("\n7 greedy hallucinations:")
    for i in hall_idx:
        print(f"  idx={i} {texts[i][:55]}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sweep.py", description="Offline threshold/window sweeps (no model).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("thresholds", help="p70-p95 sweep + live-run validation")
    sub.add_parser("windows", help="window x threshold grid")
    args = ap.parse_args(argv)
    if args.cmd == "thresholds":
        cmd_thresholds(args)
    elif args.cmd == "windows":
        cmd_windows(args)


if __name__ == "__main__":
    main()
