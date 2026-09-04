import json
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import runtime_rollback as rr

RES = rr.RES_DIR

# frozen error-rich bench: [[topic, item_index], ...]
bench = json.load(open(os.path.join(RES, "bench_hard.json"), encoding="utf-8"))["items"]

# group bench by topic, preserving per-topic item order
SUBSETS = {}
for t, i in bench:
    SUBSETS.setdefault(t, []).append(i)

SEEDS = [int(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else [1000, 1001, 1002, 1003, 1004, 1006]
ARMS = ["none", "mask", "abstain"]
WINDOW = 5
SCALE = 0.3


def tag(arm, seed):
    t = f"jump_gt_L19_t90_{arm}"
    if arm == "mask":
        t += f"_sft{SCALE}"
    if WINDOW != 5:
        t += f"_w{WINDOW}"
    return f"{t}_s{seed}"


def out_exists(topic, arm, seed):
    return os.path.exists(os.path.join(RES, f"eval_runtime_{topic}_{tag(arm, seed)}.json"))


def make_det(arm, seed):
    dg = json.load(open(os.path.join(RES, "detector_greedy.json"), encoding="utf-8"))
    det = dict(dg["detector_early"])
    det["threshold"] = det["threshold_t90"]
    det["threshold_key"] = "t90"
    det["mode"] = arm
    det["window"] = WINDOW
    det["scale"] = SCALE if arm == "mask" else 0.0
    det["sample"] = True
    det["seed"] = seed
    return det


def main():
    t0 = time.time()
    total, done, skipped = 0, 0, 0
    for topic in SUBSETS:
        for arm in ARMS:
            for seed in SEEDS:
                total += 1
                if out_exists(topic, arm, seed):
                    skipped += 1
                    continue
                det = make_det(arm, seed)
                det["subset"] = SUBSETS[topic]
                print(f"\n[{time.time()-t0:.0f}s] RUN topic={topic} arm={arm} seed={seed} "
                      f"items={len(SUBSETS[topic])}", flush=True)
                rr.run_topic(topic, det)
                done += 1
    print(f"\nDONE total={total} done={done} skipped={skipped} elapsed={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()