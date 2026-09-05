"""Prompt baseline: same bench items, greedy, constrained prompt.

Compares the default open prompt vs "Answer with only the city name."
Same-budget baseline the reviewers asked for: if a prompt fix matches NHE,
the gain is trivial. Frozen benches, greedy, strict metric.
"""
import json
import os
import re
import sys
import time
import unicodedata

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
REPO_ROOT = os.path.dirname(BASE) if os.path.basename(BASE) == "NHE-Edge" else BASE
import topics

SAVE_DIR = os.path.join(REPO_ROOT, "models", "gemma3-1b-fp16")
TOK_DIR = os.path.join(REPO_ROOT, "models", "gemma3-1b-tokenizer")
RES = os.path.join(BASE, "results")
MAX_NEW = 48

ALTS = {}
for _n in ["AFRICA_LARGEST", "WORLD_CAP_TRAPS", "WORLD_LARGEST"]:
    for _t in getattr(topics, _n):
        ALTS[_t[0]] = list(_t[1:])


def norm(s):
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def fs(gen):
    m = re.split(r"[.\n]", gen.strip())
    return norm(m[0]) if m and m[0].strip() else None


def run_bench(bench_name, constraint):
    import torch

    bench = json.load(open(os.path.join(RES, f"bench_{bench_name}.json"), encoding="utf-8"))["items"]
    by_topic = {}
    for t, i in bench:
        by_topic.setdefault(t, []).append(i)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(SAVE_DIR, dtype=torch.float16, attn_implementation="eager")
    tok = AutoTokenizer.from_pretrained(TOK_DIR)
    end_id = tok.convert_tokens_to_ids("<end_of_turn>")
    out_items = []
    t0 = time.time()
    done = 0
    with torch.no_grad():
        for topic, idxs in by_topic.items():
            items = getattr(topics, {"africa_largest": "AFRICA_LARGEST", "world_cap_traps": "WORLD_CAP_TRAPS", "world_largest": "WORLD_LARGEST"}[topic])
            for i in idxs:
                q, ans = items[i][0], items[i][1]
                alts = list(items[i][2:])
                prompt_q = q + (" Answer with only the city name." if constraint else "")
                text = "<start_of_turn>user\n" + prompt_q + "<end_of_turn>\n<start_of_turn>model\n"
                ids = tok(text, return_tensors="pt")["input_ids"]
                out = model.generate(ids, max_new_tokens=MAX_NEW, do_sample=False, use_cache=True, return_dict_in_generate=True)
                gen_ids = out.sequences[0][ids.shape[1]:]
                if end_id in gen_ids:
                    gen_ids = gen_ids[:gen_ids.tolist().index(end_id)]
                gen = tok.decode(gen_ids, skip_special_tokens=True)
                a = [ans] + alts + list(ALTS.get(q, []))
                f = fs(gen)
                sc = 1 if (f and any(norm(x) in f for x in a)) else 0
                out_items.append({"topic": topic, "id": i, "question": q, "answer": ans, "generated": gen, "strict_correct": sc})
                done += 1
                if done % 20 == 0:
                    print(f"  {bench_name} constrained={constraint} {done}/{len(bench)} ({time.time()-t0:.0f}s)", flush=True)
    n = len(out_items)
    w = sum(1 for r in out_items if not r["strict_correct"])
    tag = "prompt_constrained" if constraint else "prompt_open"
    path = os.path.join(RES, f"prompt_baseline_{bench_name}_{tag}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"bench": bench_name, "constraint": constraint, "n": n, "wrong": w, "hall": round(w / n, 4), "results": out_items}, fh, indent=1, ensure_ascii=False)
    print(f"[{bench_name} constrained={constraint}] strict hall={w/n:.3f} ({w}/{n}) -> {path}", flush=True)


if __name__ == "__main__":
    for bench in ["hard", "random"]:
        for constraint in [False, True]:
            run_bench(bench, constraint)
