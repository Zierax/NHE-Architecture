import json
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

for topic in ["asia", "us_states"]:
    for tag in ["k32_midwrong", "k128_wrong"]:
        b = {r["question"]: r for r in json.load(open(f"results/eval_{topic}_baseline.json", encoding="utf-8"))["results"]}
        ev = {r["question"]: r for r in json.load(open(f"results/eval_{topic}_{tag}.json", encoding="utf-8"))["results"]}
        fl = [(q, r) for q, r in ev.items() if r["correct"] != b[q]["correct"]]
        print(f"{topic} {tag}: {len(fl)} flips")
        for q, r in fl:
            print(f"   {q}: base={b[q]['generated'][:50]!r} -> mask={r['generated'][:50]!r}")