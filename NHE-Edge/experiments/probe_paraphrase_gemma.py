"""Paraphrase persistence: do the 4 quiet lies survive rephrasing?

If a wrong answer is a hardcoded parametric lie, it should persist across
paraphrases (same wrong city, high confidence). If it is a fragile guess, it
should flip. Tests the 4 quiet (7,15,19,22) + 3 fired-wrong (17,20,41) with
3 paraphrase templates, greedy. Appends to results/entropy_audit.json.

Usage: python probe_paraphrase_gemma.py  (Gemma 3 1B CPU, minutes)
"""
import json
import os
import re
import sys
import unicodedata

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE = os.path.dirname(os.path.abspath(__file__))
EDGE_ROOT = os.path.dirname(BASE)
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(EDGE_ROOT, "core"))
REPO_ROOT = os.path.dirname(EDGE_ROOT)
import topics

RES = os.path.join(EDGE_ROOT, "results")
SAVE_DIR = os.path.join(REPO_ROOT, "models", "gemma3-1b-fp16")
TOK_DIR = os.path.join(REPO_ROOT, "models", "gemma3-1b-tokenizer")

ALTS = {}
for _t in topics.AFRICA:
    ALTS[_t[0]] = list(_t[1:])


def norm(s):
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


IDS = [7, 15, 19, 22, 17, 20, 41]
TEMPLATES = [
    "What is the capital of {c}?",
    "Name the capital city of {c}.",
    "Which city is the capital of {c}?",
    "The capital of {c} is the city of",
]


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(SAVE_DIR, dtype=torch.float16, attn_implementation="eager")
    tok = AutoTokenizer.from_pretrained(TOK_DIR)
    end_id = tok.convert_tokens_to_ids("<end_of_turn>")
    items = {i: (q, a) for i, (q, a, *al) in enumerate(topics.AFRICA)}
    out = []
    model.eval()
    with torch.no_grad():
        for i in IDS:
            q, ans = items[i]
            country = q.replace("What is the capital of ", "").rstrip("?")
            gens = []
            for t in TEMPLATES:
                prompt = t.format(c=country)
                text = "<start_of_turn>user\n" + prompt + "<end_of_turn>\n<start_of_turn>model\n"
                ids = tok(text, return_tensors="pt")["input_ids"]
                cur, past, out_ids = ids, None, []
                for _ in range(48):
                    o = model(input_ids=cur, past_key_values=past, use_cache=True)
                    past = o.past_key_values
                    nx = int(o.logits[0, -1].argmax())
                    if nx == end_id:
                        break
                    out_ids.append(nx)
                    cur = torch.tensor([[nx]])
                g = tok.decode(out_ids, skip_special_tokens=True)
                m = re.split(r"[.\n]", g.strip())
                f = norm(m[0]) if m and m[0].strip() else None
                a = [ans] + list(ALTS.get(q, []))
                gens.append({"prompt": prompt, "generated": g[:100],
                             "strict_correct": 1 if (f and any(norm(x) in f for x in a)) else 0})
            persists = sum(1 for g in gens[1:] if g["generated"][:40] == gens[0]["generated"][:40])
            out.append({"id": i, "question": q, "answer": ans, "gens": gens,
                        "n_same_as_p0": persists})
            print(f"id={i} {q[24:38]:<16} " + " | ".join(
                f"T{k}={'OK' if g['strict_correct'] else '--'}:{g['generated'][:28]!r}" for k, g in enumerate(gens)), flush=True)
    path = os.path.join(RES, "entropy_audit.json")
    prev = json.load(open(path, encoding="utf-8"))
    prev["paraphrase"] = out
    json.dump(prev, open(path, "w", encoding="utf-8"), indent=1)
    print("appended paraphrase block to results/entropy_audit.json", flush=True)


if __name__ == "__main__":
    main()
