import json
import os
import re
import sys
import unicodedata

import topics

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

ALTS = {}
for name in ["AFRICA", "EUROPE", "AFRICA_LARGEST", "WORLD_TRICKY",
             "WORLD_CAP_TRAPS", "WORLD_LARGEST"]:
    for tup in getattr(topics, name):
        ALTS[tup[0]] = list(tup[1:])


def norm(s):
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def fs(gen):
    m = re.split(r"[.\n]", gen.strip())
    return norm(m[0]) if m and m[0].strip() else None


def answers(q, r):
    return [r["answer"]] + list(ALTS.get(q, []))


def score(path, label):
    path = path if os.path.isabs(path) else os.path.join(BASE, path)
    ev = json.load(open(path, encoding="utf-8"))
    n = len(ev["results"])
    subj = 0
    strict = 0
    flips = []
    for r in ev["results"]:
        a = answers(r["question"], r)
        g = r["generated"] if "generated" in r else r["samples"][0]
        if any(norm(x) in norm(g) for x in a):
            subj += 1
        if fs(g) and any(norm(x) in fs(g) for x in a):
            strict += 1
    print(f"{label:<18} n={n:>3}  substring_hall={1-subj/n:>7.3f}  strict_firstsent_hall={1-strict/n:>7.3f}")
    return n, subj, strict


print("== AFRICA greedy, strict (first-sentence) metric with alternatives ==")
for f, label in [
    ("results/eval_africa_baseline.json", "baseline"),
    ("results/eval_africa_k32_midwrong.json", "static k32"),
    ("results/eval_africa_k128_wrong.json", "static k128"),
    ("results/eval_runtime_africa_jump_gt_L19_t90_mask.json", "rt w5 hard"),
    ("results/eval_runtime_africa_jump_gt_L19_t90_mask_sft0.3.json", "rt w5 soft"),
    ("results/eval_runtime_africa_jump_gt_L19_t90_mask_w4.json", "rt w4 hard"),
    ("results/eval_runtime_africa_jump_gt_L19_t90_mask_w3.json", "rt w3 hard"),
]:
    score(f, label)

print("\n== AFRICA_LARGEST / WORLD_TRICKY greedy, strict ==")
for f, label in [
    ("results/eval_africa_largest_baseline.json", "al baseline"),
    ("results/eval_runtime_africa_largest_jump_gt_L19_t90_mask_sft0.3.json", "al rt soft"),
    ("results/eval_world_tricky_baseline.json", "wt baseline"),
    ("results/eval_world_tricky_k32_midwrong.json", "wt k32"),
    ("results/eval_world_tricky_k128_wrong.json", "wt k128"),
    ("results/eval_runtime_world_tricky_jump_gt_L19_t90_mask_sft0.3.json", "wt rt soft"),
]:
    score(f, label)

print("\n== EUROPE greedy, strict ==")
for f, label in [
    ("results/eval_europe_baseline.json", "europe baseline"),
    ("results/eval_europe_k32_midwrong.json", "europe k32"),
    ("results/eval_europe_k128_wrong.json", "europe k128"),
    ("results/eval_runtime_europe_jump_gt_L19_t90_mask_sft0.3.json", "europe rt soft"),
]:
    score(f, label)

print("\n== NEW error-rich bench (greedy, strict) ==")
for f, label in [
    ("results/eval_runtime_world_cap_traps_jump_gt_L19_t90_none.json", "cap_traps baseline"),
    ("results/eval_runtime_world_cap_traps_jump_gt_L19_t90_mask_sft0.3.json", "cap_traps rt mask"),
    ("results/eval_runtime_world_cap_traps_jump_gt_L19_t90_abstain.json", "cap_traps rt abstain"),
    ("results/eval_runtime_world_largest_jump_gt_L19_t90_none.json", "world_largest baseline"),
    ("results/eval_runtime_world_largest_jump_gt_L19_t90_mask_sft0.3.json", "world_largest rt mask"),
    ("results/eval_runtime_world_largest_jump_gt_L19_t90_abstain.json", "world_largest rt abstain"),
]:
    score(f, label)