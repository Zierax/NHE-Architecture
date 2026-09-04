import json
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

base = {r["question"]: r for r in json.load(open("results/eval_runtime_africa_jump_gt_L19_t90_none_s1000.json", encoding="utf-8"))["results"]}
for tag in ["mask_sft0.3_s1000", "mask_w4_s1000", "mask_s1000"]:
    ev = json.load(open(f"results/eval_runtime_africa_jump_gt_L19_t90_{tag}.json", encoding="utf-8"))
    mask = {r["question"]: r for r in ev["results"]}
    print(f"=== {tag}: hall={ev['hallucination_rate']} fired={ev['n_fired']}")
    for q, rn in base.items():
        rm = mask[q]
        if rn["correct"] != rm["correct"]:
            kind = "W2C" if rm["correct"] else "C2W"
            print(f"  {kind} {q}")
            print(f"    base: {rn['generated'][:60]!r}")
            print(f"    run : {rm['generated'][:60]!r}")