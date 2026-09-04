import json
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

base = {r["question"]: r for r in json.load(open("results/eval_africa_baseline.json", encoding="utf-8"))["results"]}
static = {r["question"]: r for r in json.load(open("results/eval_africa_k32_midwrong.json", encoding="utf-8"))["results"]}
rt = {r["question"]: r for r in json.load(open("results/eval_runtime_africa_jump_gt_L19_t90_mask.json", encoding="utf-8"))["results"]}

print("== C2W (correct->wrong) under static:")
for q, b in base.items():
    if b["correct"] and not static[q]["correct"]:
        print(f"  {q}")
        print(f"    base  : {b['generated'][:70]!r}")
        print(f"    static: {static[q]['generated'][:70]!r}")
print("== W2C (wrong->correct) under static:")
for q, b in base.items():
    if not b["correct"] and static[q]["correct"]:
        print(f"  {q} -> {static[q]['generated'][:70]!r}")
print("== C2W under runtime early t90:")
for q, b in base.items():
    if b["correct"] and not rt[q]["correct"]:
        print(f"  {q} -> {rt[q]['generated'][:70]!r}")