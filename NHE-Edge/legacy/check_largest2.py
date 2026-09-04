import json
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

base = {r["question"]: r for r in json.load(open("results/eval_africa_largest_baseline.json", encoding="utf-8"))["results"]}
rt = {r["question"]: r for r in json.load(open("results/eval_runtime_africa_largest_jump_gt_L19_t90_mask_sft0.3.json", encoding="utf-8"))["results"]}

w2c, c2w = [], []
for q, b in base.items():
    r = rt[q]
    if b["correct"] != r["correct"]:
        (w2c if r["correct"] else c2w).append((q, b["generated"][:50], r["generated"][:50]))

w2c_qs = {t[0] for t in w2c}
print(f"real flips on africa_largest: W2C={len(w2c)} C2W={len(c2w)}")
for q, bg, rg in w2c + c2w:
    print(f"  {'W2C' if q in w2c_qs else 'C2W'} {q}\n    base: {bg!r}\n    run : {rg!r}")

print("\nall fired items on africa_largest:")
for q, r in rt.items():
    if r["fired_at"] is not None:
        print(f"  fire@{r['fired_at']} base={base[q]['correct']} run={r['correct']} {q}")