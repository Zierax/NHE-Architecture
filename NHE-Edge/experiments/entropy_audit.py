"""Entropy audit: for each greedy-wrong Africa case (+correct controls), capture
the FULL output distribution at the commit (city) token: top-1 prob, entropy,
top1-top2 margin. Distinguishes confident lies (peaked, low entropy) from
uncertain guesses (flat, argmax won by a hair) - the alternative explanation
for quiet+wrong that must be ruled out before claiming training-data error.

Usage: python entropy_audit.py  (Gemma 3 1B CPU, minutes)
Writes: results/entropy_audit.json
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


# ids from STATUS: quiet = 7,15,19,22 ; fired-wrong = 17,20,41 ; controls = correct
QUIET = [7, 15, 19, 22]
FIRED_WRONG = [17, 20, 41]
CONTROLS = [0, 1, 2, 3]


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(SAVE_DIR, dtype=torch.float16, attn_implementation="eager")
    tok = AutoTokenizer.from_pretrained(TOK_DIR)
    end_id = tok.convert_tokens_to_ids("<end_of_turn>")
    items = {r["id"]: r for r in
             [{"id": i, "q": q, "a": a, "alts": list(al)} for i, (q, a, *al) in enumerate(topics.AFRICA)]}
    fired = {}
    try:
        n = json.load(open(os.path.join(RES, "eval_runtime_africa_jump_gt_L19_t90_none_s1000.json"), encoding="utf-8"))
        for r in n["results"]:
            fired[r["id"]] = r["fired_at"]
    except FileNotFoundError:
        pass

    out = []
    model.eval()
    with torch.no_grad():
        for i in QUIET + FIRED_WRONG + CONTROLS:
            it = items[i]
            q, ans = it["q"], it["a"]
            text = "<start_of_turn>user\n" + q + "<end_of_turn>\n<start_of_turn>model\n"
            ids = tok(text, return_tensors="pt")["input_ids"]
            out_ids, probs_trace = [], []
            cur, past = ids, None
            for _ in range(48):
                o = model(input_ids=cur, past_key_values=past, use_cache=True)
                past = o.past_key_values
                probs = torch.softmax(o.logits[0, -1].float(), dim=-1)
                probs_trace.append(probs)
                nx = int(probs.argmax())
                if nx == end_id:
                    break
                out_ids.append(nx)
                cur = torch.tensor([[nx]])
            gen = tok.decode(out_ids, skip_special_tokens=True)
            # commit token = first content token after the last "is"
            # (city span start: "**Viana" or "Viana")
            gids = tok(gen, add_special_tokens=False)["input_ids"]
            strs = [tok.decode([t]) for t in gids]
            is_pos = [k for k, s in enumerate(strs) if s.strip().lower() == "is"]
            ci = 0
            if is_pos:
                # FIRST "is": the template commit point ("The capital of X is Y").
                # (last "is" would land in trailing sentences.)
                j = is_pos[0] + 1
                while j < len(strs) and not strs[j].strip():
                    j += 1
                ci = j if j < len(strs) else len(strs) - 1
            probs = probs_trace[min(ci, len(probs_trace) - 1)]
            top2 = torch.topk(probs, 2)
            p1, p2 = float(top2.values[0]), float(top2.values[1])
            ent = float(-(probs * (probs + 1e-12).log()).sum())
            tok_str = tok.decode([gids[ci]]) if ci < len(gids) else ""
            a = [ans] + list(ALTS.get(q, []))
            m = re.split(r"[.\n]", gen.strip())
            f = norm(m[0]) if m and m[0].strip() else None
            correct = 1 if (f and any(norm(x) in f for x in a)) else 0
            row = {"id": i, "question": q, "answer": ans, "generated": gen[:120],
                   "commit_token": tok_str, "commit_pos": ci,
                   "top1_prob": round(p1, 4), "entropy": round(ent, 3),
                   "margin_top1_top2": round(p1 - p2, 4),
                   "strict_correct": correct, "fired_at": fired.get(i)}
            out.append(row)
            print(f"id={i} commit_tok={tok_str!r}@{ci} p1={p1:.4f} ent={ent:.3f} margin={p1-p2:.4f} correct={correct} fired={fired.get(i)} :: {gen[:50]!r}", flush=True)
    with open(os.path.join(RES, "entropy_audit.json"), "w", encoding="utf-8") as fh:
        json.dump({"rows": out}, fh, indent=1)
    print("saved results/entropy_audit.json", flush=True)


if __name__ == "__main__":
    main()
