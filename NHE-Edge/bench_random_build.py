"""Build bench_random.json: 99 random items from the 361-item pool, seed 42.

Pool: africa_largest (54) + world_cap_traps (134) + world_largest (173).
Procedure (frozen, reproduces the committed file byte-identically):
    random.seed(42); shuffle(pool); take first 99; sort by (topic, idx).
Overlap with bench_hard.json is 19/99 (verified).
"""
import json
import os
import random
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, "results")

POOL = (
    [["africa_largest", i] for i in range(54)]
    + [["world_cap_traps", i] for i in range(134)]
    + [["world_largest", i] for i in range(173)]
)


def build(seed=42, n=99):
    pool = [list(p) for p in POOL]
    random.seed(seed)
    random.shuffle(pool)
    return sorted([tuple(p) for p in pool[:n]])


def main():
    items = [[t, i] for (t, i) in build()]
    out = os.path.join(RES, "bench_random.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"items": items}, fh, indent=1)
    hard = set(tuple(x) for x in json.load(open(os.path.join(RES, "bench_hard.json"), encoding="utf-8"))["items"])
    print(f"bench_random.json: {len(items)} items, overlap hard: {len(set(map(tuple, items)) & hard)}")


if __name__ == "__main__":
    main()
