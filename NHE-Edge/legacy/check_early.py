import json
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

base = {r["question"]: r for r in json.load(open("results/eval_africa_baseline.json", encoding="utf-8"))["results"]}
ev = json.load(open("results/eval_runtime_africa_jump_gt_L19_t90_mask.json", encoding="utf-8"))
print(f"=== early t90: hall={ev['hallucination_rate']} fired={ev['n_fired']}/{ev['n']}")
for r in ev["results"]:
    b = base[r["question"]]
    fired = r["fired_at"] is not None
    flip = r["correct"] != b["correct"]
    if fired or flip:
        kind = ""
        if r["correct"] != b["correct"]:
            kind = "W2C" if r["correct"] else "C2W"
        mark = f"FIRE@{r['fired_at']:<2}" if fired else "     "
        print(f"  {mark} {kind:<3} {r['question']}")
        if flip:
            print(f"      base: {b['generated'][:60]!r}")
            print(f"      rt  : {r['generated'][:60]!r}")