import json
import re
import sys
import unicodedata

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def norm(s):
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def fs(gen):
    m = re.split(r"[.\n]", gen.strip())
    return norm(m[0]) if m and m[0].strip() else None


def load(path):
    ev = json.load(open(path, encoding="utf-8"))
    out = {}
    for r in ev["results"]:
        a = [r["answer"]] + list(r.get("alt", []) or [])
        out[r["question"]] = (a, r["generated"])
    return out


base = load("results/eval_africa_baseline.json")
files = {
    "k32": "results/eval_africa_k32_midwrong.json",
    "k128": "results/eval_africa_k128_wrong.json",
    "rt_w5": "results/eval_runtime_africa_jump_gt_L19_t90_mask.json",
    "rt_w5_soft": "results/eval_runtime_africa_jump_gt_L19_t90_mask_sft0.3.json",
    "rt_w4": "results/eval_runtime_africa_jump_gt_L19_t90_mask_w4.json",
    "rt_w3": "results/eval_runtime_africa_jump_gt_L19_t90_mask_w3.json",
}
sc = {k: load(v) for k, v in files.items()}

strict_base = {q: (1 if any(norm(a) in fs(b[1]) for a in b[0]) else 0) for q, b in base.items()}
print("strict per-run diff vs strict baseline (africa greedy):")
for name, d in sc.items():
    wrong = []
    for q, (a, g) in d.items():
        st = 1 if any(norm(x) in fs(g) for x in a) else 0
        if st != strict_base[q]:
            wrong.append((q, strict_base[q], st, base[q][1][:40], g[:40]))
    print(f"\n  {name}: strict hall = {sum(1 for q,(a,g) in d.items() if not any(norm(x) in fs(g) for x in a))}/54")
    for q, sb, st, bg, rg in wrong:
        print(f"    {sb}->{st} {q}\n      base: {bg!r}\n      run : {rg!r}")