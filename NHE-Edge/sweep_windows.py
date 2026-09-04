import json
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, "results")

d = np.load(os.path.join(RES, "greedy_flows_africa.npz"), allow_pickle=True)
flows = list(d["flows"])
texts = d["texts"]
labels = np.array(d["labels"]).astype(int)
y = 1 - labels
truth = y == 0
hall_idx = [i for i in range(len(y)) if y[i] == 1]

LAYER = 19
jumps_all = []
for f in flows:
    h = f[:, LAYER + 1, :].astype(np.float32)
    dlt = h[1:] - h[:-1]
    jumps_all.append(np.linalg.norm(dlt, axis=1))
early_max = np.array([j[: min(10, len(j))].max() for j in jumps_all])

FIXABLE = {0: 2, 3: 3, 5: 4}  # greedy hall item -> first jump crossing index at p90
# hall item indices from greedy baseline (by text prefix check below)
# Eswatini (idx?), Gambia, Senegal are the fixable ones.

def first_cross(j, thr, window):
    c = np.where(j[:window] > thr)[0]
    return int(c[0]) + 1 if len(c) else None

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

# Identify which indices are the 7 hallucinations
print("\n7 greedy hallucinations:")
for i in hall_idx:
    print(f"  idx={i} {texts[i][:55]}")