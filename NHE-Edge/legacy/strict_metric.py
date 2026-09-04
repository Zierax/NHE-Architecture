import json
import os
import re
import sys
import unicodedata

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def norm(s):
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def first_sentence(gen):
    m = re.split(r"[.\n]", gen.strip())
    return norm(m[0]) if m and m[0].strip() else None


def first_sentence_correct(gen, answers):
    fs = first_sentence(gen)
    if not fs:
        return None
    return 1 if any(norm(a) in fs for a in answers) else 0


def score(path):
    ev = json.load(open(path, encoding="utf-8"))
    n = len(ev["results"])
    subj = 0
    fs = 0
    unmeas = 0
    for r in ev["results"]:
        a = [r["answer"]] + list(r.get("alt", []) or [])
        gens = [r["generated"]] if "generated" in r else r["samples"]
        g0 = gens[0]
        if any(norm(x) in norm(g0) for x in a):
            subj += 1
        sc = first_sentence_correct(g0, a)
        if sc == 1:
            fs += 1
        elif sc == 0:
            pass
        else:
            unmeas += 1
    return n, subj, fs, unmeas


FILES = [
    "results/eval_africa_baseline.json",
    "results/eval_africa_k32_midwrong.json",
    "results/eval_africa_k128_wrong.json",
    "results/eval_runtime_africa_jump_gt_L19_t90_mask.json",
    "results/eval_runtime_africa_jump_gt_L19_t90_mask_sft0.3.json",
    "results/eval_runtime_africa_jump_gt_L19_t90_mask_w4.json",
    "results/eval_runtime_africa_jump_gt_L19_t90_mask_w3.json",
    "results/eval_africa_largest_baseline.json",
    "results/eval_runtime_africa_largest_jump_gt_L19_t90_mask_sft0.3.json",
    "results/eval_europe_baseline.json",
    "results/eval_europe_k32_midwrong.json",
    "results/eval_europe_k128_wrong.json",
]

print(f"{'file':<72} {'n':>3} {'subj%':>6} {'firstSent%':>10} {'unmeas':>6}")
for f in FILES:
    if not os.path.exists(f):
        continue
    n, subj, fs, unmeas = score(f)
    print(f"{os.path.basename(f):<72} {n:>3} {subj/n*100:>6.1f} {fs/n*100:>10.1f} {unmeas:>6}")