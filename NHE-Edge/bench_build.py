import json
import os
import re
import sys
import unicodedata

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import topics

RES = os.path.join(BASE, "results")

ALTS = {}
for name in ["AFRICA_LARGEST", "WORLD_CAP_TRAPS", "WORLD_LARGEST"]:
    for tup in getattr(topics, name):
        ALTS[tup[0]] = list(tup[1:])


def norm(s):
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def fs(gen):
    m = re.split(r"[.\n]", gen.strip())
    return norm(m[0]) if m and m[0].strip() else None


def strict_wrong(path, topic_name):
    ev = json.load(open(path, encoding="utf-8"))
    idx = []
    for r in ev["results"]:
        a = [r["answer"]] + list(ALTS.get(r["question"], []))
        f = fs(r["generated"])
        if not (f and any(norm(x) in f for x in a)):
            idx.append(r["id"])
    print(f"{topic_name}: {len(idx)}/{len(ev['results'])} strict-wrong")
    return idx


def _rp(name):
    return os.path.join(RES, name)

HARD = {
    "africa_largest": list(range(54)),
    "world_cap_traps": strict_wrong(_rp("eval_world_cap_traps_baseline.json"), "world_cap_traps"),
    "world_largest": strict_wrong(_rp("eval_world_largest_baseline.json"), "world_largest"),
}

bench = {"items": [[t, i] for t, ids in HARD.items() for i in ids]}
with open(_rp("bench_hard.json"), "w", encoding="utf-8") as fh:
    json.dump(bench, fh, indent=1)
print(f"bench_hard.json: {len(bench['items'])} items across {list(HARD.keys())}")