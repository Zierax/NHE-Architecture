import json
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

for tag in ["jump_gt_L18_t95_mask", "jump_gt_L18_t90_mask", "jump_gt_L18_t90_none"]:
    ev = json.load(open(f"results/eval_runtime_africa_{tag}.json", encoding="utf-8"))
    print(f"=== {tag}: hall={ev['hallucination_rate']} fired={ev['n_fired']}/{ev['n']}")
    for r in ev["results"]:
        if r["fired_at"] is not None:
            print(f"  at={r['fired_at']:>2} correct={r['correct']} {r['question']} -> {r['generated'][:45]!r}")