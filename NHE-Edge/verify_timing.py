"""Verify the format-causality 2x2 + lead-time distribution from raw files.

For each cell (model x format): strict W2C/C2W none-vs-mask, plus per fired
item the city-token index vs fired_at. Convention: city_idx = 0-indexed position
of the answer-span start in generated tokens; lead = city_idx - fired_at.
lead >= 1 means the spike was measured strictly before the city token.
Template-aware search per format family, with strict scoring from topics.py alts.
"""
import json
import os
import re
import sys
import unicodedata

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
REPO_ROOT = os.path.dirname(BASE) if os.path.basename(BASE) == "NHE-Edge" else BASE
import topics

RES = os.path.join(BASE, "results")
GEM_TOK = os.path.join(REPO_ROOT, "models", "gemma3-1b-tokenizer")

ALTS = {}
for _n in ["AFRICA"]:
    for _t in getattr(topics, _n):
        ALTS[_t[0]] = list(_t[1:])


def norm(s):
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def fs(gen):
    m = re.split(r"[.\n]", gen.strip())
    return norm(m[0]) if m and m[0].strip() else None


def strict_of(question, answer, gen):
    a = [answer] + list(ALTS.get(question, []))
    f = fs(gen)
    return 1 if (f and any(norm(x) in f for x in a)) else 0


def city_index(tok, gen, family):
    """0-indexed position of answer-span start in generated tokens."""
    ids = tok(gen, add_special_tokens=False)["input_ids"]
    toks = [tok.decode([t]) for t in ids]
    if not toks:
        return None
    # bold marker: city starts at or right after "**"
    for k, t in enumerate(toks):
        s = t.strip()
        if s == "**":
            return k + 1
        if s.startswith("**") and len(s) > 2:
            return k
    # plain "The capital of X is Y": first content token after "is"
    for k, t in enumerate(toks):
        if t.strip().lower() == "is":
            j = k + 1
            while j < len(toks) and toks[j].strip() in ("", "**"):
                j += 1
            return j if j < len(toks) else None
    # bare answer ("Algiers"): city is token 0
    return 0


CELLS = [
    ("gemma-native", "eval_africa_baseline.json",
     "eval_runtime_africa_jump_gt_L19_t90_mask_sft0.3.json", "gemma", "bold"),
    ("gemma-plain", "eval_runtime_africa_jump_gt_L19_t90_none_fmtplain.json",
     "eval_runtime_africa_jump_gt_L19_t90_mask_sft0.3_fmtplain.json", "gemma", "plain"),
    ("qwen-plain", "eval_runtime_africa_qwen2.5-0.5b_jump_gt_L22_t90_none_w10.json",
     "eval_runtime_africa_qwen2.5-0.5b_jump_gt_L22_t90_mask_w10.json", "qwen", "plain"),
    ("qwen-bold", "eval_runtime_africa_qwen2.5-0.5b_jump_gt_L22_t90_none_w10_fmtbold.json",
     "eval_runtime_africa_qwen2.5-0.5b_jump_gt_L22_t90_mask_w10_fmtbold.json", "qwen", "bold"),
    ("qwen-long", "eval_runtime_africa_qwen2.5-0.5b_jump_gt_L22_t90_none_w10_fmtlong.json",
     "eval_runtime_africa_qwen2.5-0.5b_jump_gt_L22_t90_mask_w10_fmtlong.json", "qwen", "plain"),
]


def main():
    from transformers import AutoTokenizer
    gt = AutoTokenizer.from_pretrained(GEM_TOK)
    import runtime_rollback_qwen as qq
    _, qt = qq.model_and_tok("qwen2.5-0.5b")
    toks = {"gemma": gt, "qwen": qt}
    print(f"{'cell':<13} {'W2C':>4} {'C2W':>4}  leads(fired items: city_idx-fired_at)")
    for name, fn, fm, model, family in CELLS:
        n = json.load(open(os.path.join(RES, fn), encoding="utf-8"))
        m = json.load(open(os.path.join(RES, fm), encoding="utf-8"))
        tok = toks[model]
        nb = {r["id"]: r for r in n["results"]}
        w2c = cw = 0
        leads = []
        for rm in m["results"]:
            rn = nb[rm["id"]]
            s0 = strict_of(rn["question"], rn["answer"], rn["generated"])
            s1 = strict_of(rm["question"], rm["answer"], rm["generated"])
            if (not s0) and s1:
                w2c += 1
            if s0 and (not s1):
                cw += 1
            # fired_at from mask run; pre-fire trajectory identical to none run
            if rm.get("fired_at") is not None:
                ci = city_index(tok, rm["generated"], family)
                if ci is not None:
                    leads.append(ci - rm["fired_at"])
        print(f"{name:<13} {w2c:>4} {cw:>4}  {sorted(leads)}")


if __name__ == "__main__":
    main()
