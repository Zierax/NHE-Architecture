import json
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import runtime_rollback as rr

RES = rr.RES_DIR

# greedy runtime arms on the error-rich topics
# africa_largest: full 54; world_cap_traps: full 134; world_largest: hard 30 only
bench = json.load(open(os.path.join(RES, "bench_hard.json"), encoding="utf-8"))["items"]
hard_lg = [i for t, i in bench if t == "world_largest"]
SUBSETS = {
    "africa_largest": list(range(54)),
    "world_cap_traps": list(range(134)),
    "world_largest": hard_lg,
}

ARMS = ["none", "mask", "abstain"]
WINDOW = 5
SCALE = 0.3


def tag(arm):
    t = f"jump_gt_L19_t90_{arm}"
    if arm == "mask":
        t += f"_sft{SCALE}"
    return t


def make_det(arm):
    dg = json.load(open(os.path.join(RES, "detector_greedy.json"), encoding="utf-8"))
    det = dict(dg["detector_early"])
    det["threshold"] = det["threshold_t90"]
    det["threshold_key"] = "t90"
    det["mode"] = arm
    det["window"] = WINDOW
    det["scale"] = SCALE if arm == "mask" else 0.0
    det["sample"] = False
    det["seed"] = 0
    return det


def main():
    t0 = time.time()
    for topic in SUBSETS:
        for arm in ARMS:
            out_file = os.path.join(RES, f"eval_runtime_{topic}_{tag(arm)}.json")
            if os.path.exists(out_file):
                print(f"skip existing {out_file}", flush=True)
                continue
            det = make_det(arm)
            det["subset"] = SUBSETS[topic]
            print(f"\n[{time.time()-t0:.0f}s] RUN topic={topic} arm={arm} items={len(SUBSETS[topic])}", flush=True)
            rr.run_topic(topic, det)
    print(f"\nDONE elapsed={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()