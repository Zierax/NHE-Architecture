import json
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

none = {r["question"]: r for r in json.load(open("results/eval_runtime_africa_jump_gt_L19_t90_none_s1000.json", encoding="utf-8"))["results"]}
mask = {r["question"]: r for r in json.load(open("results/eval_runtime_africa_jump_gt_L19_t90_mask_s1000.json", encoding="utf-8"))["results"]}

print("== sampled africa (seed 1000): none hall 0.130 vs mask 0.111")
for q, rn in none.items():
    rm = mask[q]
    if rn["correct"] != rm["correct"]:
        print(f"  {'W2C' if rm['correct'] else 'C2W'} {q}")
        print(f"    none: {rn['generated'][:70]!r}")
        print(f"    mask: {rm['generated'][:70]!r}")
    elif rm["fired_at"] is not None and rm["correct"]:
        print(f"  fire@{rm['fired_at']} (no flip, correct) {q}")

print()
print("== fires with window (greedy africa t90):")
g = json.load(open("results/eval_runtime_africa_jump_gt_L19_t90_mask.json", encoding="utf-8"))
print(f"  greedy: fired={g['n_fired']}/{g['n']} hall={g['hallucination_rate']}")
for r in g["results"]:
    if r["fired_at"] is not None:
        print(f"    at={r['fired_at']} correct={r['correct']} {r['question']}")