import json
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

base = {r["question"]: r for r in json.load(open("results/eval_africa_largest_baseline.json", encoding="utf-8"))["results"]}
rt = {r["question"]: r for r in json.load(open("results/eval_runtime_africa_largest_jump_gt_L19_t90_mask_sft0.3.json", encoding="utf-8"))["results"]}
print("africa_largest baseline hall:", [q for q, b in base.items() if not b["correct"]])
for q, b in base.items():
    r = rt[q]
    if r["correct"] != b["correct"] or (r["fired_at"] is not None):
        kind = "W2C" if r["correct"] else "C2W"
        print(f"  {kind} fire@{r['fired_at']} {q}")
        if r["correct"] != b["correct"]:
            print(f"    base: {b['generated'][:60]!r}")
            print(f"    run : {r['generated'][:60]!r}")