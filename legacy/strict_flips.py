import json
import re
import sys
import unicodedata

sys.path.insert(0, ".")
import topics

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ALTS = {}
for name in ["AFRICA", "EUROPE", "AFRICA_LARGEST", "WORLD_TRICKY"]:
    for tup in getattr(topics, name):
        ALTS[tup[0]] = list(tup[1:])


def norm(s):
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def fs(gen):
    m = re.split(r"[.\n]", gen.strip())
    return norm(m[0]) if m and m[0].strip() else None


def load(path):
    ev = json.load(open(path, encoding="utf-8"))
    out = {}
    for r in ev["results"]:
        a = [r["answer"]] + list(ALTS.get(r["question"], []))
        out[r["question"]] = (a, r["generated"])
    return out


def strict(a, g):
    f = fs(g)
    return 1 if f and any(norm(x) in f for x in a) else 0


base = load("results/eval_africa_baseline.json")
sb = {q: strict(a, g) for q, (a, g) in base.items()}
print(f"strict baseline wrong = {sum(1 for v in sb.values() if not v)}/54")

for name, p in [
    ("k32", "results/eval_africa_k32_midwrong.json"),
    ("k128", "results/eval_africa_k128_wrong.json"),
    ("rt_w5", "results/eval_runtime_africa_jump_gt_L19_t90_mask.json"),
    ("rt_w4", "results/eval_runtime_africa_jump_gt_L19_t90_mask_w4.json"),
    ("rt_w3", "results/eval_runtime_africa_jump_gt_L19_t90_mask_w3.json"),
]:
    d = load(p)
    fixes, breaks = [], []
    for q, (a, g) in d.items():
        st = strict(a, g)
        if st == 1 and sb[q] == 0:
            fixes.append(q)
        elif st == 0 and sb[q] == 1:
            breaks.append(q)
    wrong = sum(1 for (a, g) in d.values() if not strict(a, g))
    print(f"{name}: wrong={wrong}/54  fixes={fixes}  breaks={breaks}")