import os
import sys
import time
import json
import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE, "models", "gemma3-1b-fp16")
TOK_DIR = os.path.join(BASE, "models", "gemma3-1b-tokenizer")
RES_DIR = os.path.join(BASE, "results")

N_TARGETS = 6
TOP_K = 4000
PROGRESS = os.path.join(RES_DIR, "attribution_causal2_progress.npz")

def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    with open(os.path.join(RES_DIR, "eval_africa_baseline.json"), "r", encoding="utf-8") as fh:
        baseline = json.load(fh)
    hall = [r for r in baseline["results"] if r["correct"] == 0]
    print(f"greedy-hallucinated examples: {len(hall)}", flush=True)

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(SAVE_DIR, dtype=torch.float16, attn_implementation="eager")
    model = model.float()
    tok = AutoTokenizer.from_pretrained(TOK_DIR)
    print(f"model loaded+converted in {time.time()-t0:.1f}s", flush=True)

    layers = model.model.layers
    n_layers = len(layers)
    n_units = layers[0].mlp.down_proj.in_features

    c_wrong = np.zeros((n_layers, n_units), dtype=np.float64)
    c_correct = np.zeros((n_layers, n_units), dtype=np.float64)
    n_done = 0
    done_qs = []

    if os.path.exists(PROGRESS):
        ckpt = np.load(PROGRESS, allow_pickle=True)
        c_wrong = ckpt["c_wrong"].astype(np.float64)
        c_correct = ckpt["c_correct"].astype(np.float64)
        n_done = int(ckpt["n_done"])
        done_qs = list(ckpt["done_qs"])
        print(f"resumed: {n_done} examples done", flush=True)

    acts = {}
    handles = []

    def make_hook(l):
        def hook(module, args, output):
            args[0].retain_grad()
            acts[l] = args[0]
        return hook

    for l in range(n_layers):
        handles.append(layers[l].mlp.down_proj.register_forward_hook(make_hook(l)))

    model.eval()

    for r in hall:
        if r["question"] in done_qs:
            continue
        q = r["question"]
        ans = r["answer"]
        text = "<start_of_turn>user\n" + q + "<end_of_turn>\n<start_of_turn>model\n"
        ids = tok(text, return_tensors="pt")["input_ids"]
        gen_ids = tok(r["generated"], add_special_tokens=False)["input_ids"]
        ans_ids = tok.encode(ans, add_special_tokens=False)
        if not gen_ids or not ans_ids:
            done_qs.append(q); n_done += 1; continue
        correct_tok = ans_ids[0]

        prompt_len = ids.shape[1]
        n_gen = min(len(gen_ids), 48)
        full = torch.cat([ids[0], torch.tensor(gen_ids[:n_gen], dtype=torch.long)]).unsqueeze(0)

        ex_t0 = time.time()
        with torch.enable_grad():
            acts.clear()
            logits = model(full, use_cache=False).logits[0]
            n_t = min(n_gen, N_TARGETS)
            positions = []
            for p in range(n_t):
                pos = prompt_len + p
                gt = full[0, pos].item()
                if gt in (tok.bos_token_id, tok.eos_token_id, tok.pad_token_id):
                    continue
                positions.append(pos)
            if not positions:
                done_qs.append(q); n_done += 1; continue
            target = None
            for pos in positions:
                term = logits[pos, full[0, pos].item()]
                target = term if target is None else target + term
            model.zero_grad(set_to_none=True)
            target.backward(retain_graph=True)
            for l in range(n_layers):
                g = acts[l].grad
                if g is not None:
                    a = acts[l].float()
                    c_wrong[l] += (g[0].float() * a[0]).detach().numpy()[positions, :].sum(axis=0)
            model.zero_grad(set_to_none=True)
            target = None
            for pos in positions:
                term = logits[pos, correct_tok]
                target = term if target is None else target + term
            target.backward()
            for l in range(n_layers):
                g = acts[l].grad
                if g is not None:
                    a = acts[l].float()
                    c_correct[l] += (g[0].float() * a[0]).detach().numpy()[positions, :].sum(axis=0)
        n_done += 1
        done_qs.append(q)
        np.savez(PROGRESS, c_wrong=c_wrong, c_correct=c_correct, n_done=n_done, done_qs=np.array(done_qs))
        print(f"[{n_done}/{len(hall)}] {time.time()-ex_t0:.1f}s: {q[:50]}", flush=True)

    for h in handles:
        h.remove()

    c_wrong /= max(n_done, 1)
    c_correct /= max(n_done, 1)
    score = c_wrong - c_correct

    flat = []
    for l in range(n_layers):
        for u in range(n_units):
            flat.append((l, u, float(score[l, u]), float(c_wrong[l, u]), float(c_correct[l, u])))
    flat.sort(key=lambda e: -e[2])
    top_all = [{"layer": l, "unit": u, "causal": s, "c_wrong": cw, "c_correct": cc} for l, u, s, cw, cc in flat[:TOP_K]]

    def top_where(name, cond, k=TOP_K):
        sub = [e for e in flat if cond(e)]
        return [{"layer": l, "unit": u, "causal": s, "c_wrong": cw, "c_correct": cc} for l, u, s, cw, cc in sub[:k]]

    mid = top_where("mid", lambda e: 8 <= e[0] <= 17)
    wrong_only = top_where("wrong_only", lambda e: e[3] > 0 and e[4] <= 0.0)
    mid_wrong_only = top_where("mid_wrong_only", lambda e: 8 <= e[0] <= 17 and e[3] > 0 and e[4] <= 0.0)

    out = {
        "method": "atp_two_targets_fp32",
        "n_examples": n_done,
        "top_by_causal": top_all,
        "top_mid_l8_17": mid,
        "top_wrong_only": wrong_only,
        "top_mid_wrong_only": mid_wrong_only,
    }
    with open(os.path.join(RES_DIR, "attribution_causal2_africa.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)

    for name, lst in [("ALL", top_all), ("MID 8-17", mid), ("WRONG_ONLY", wrong_only), ("MID+WRONG", mid_wrong_only)]:
        print(f"\n== {name} top 10 ==", flush=True)
        for e in lst[:10]:
            print(f"  L{e['layer']:>2} u {e['unit']:>5}  c={e['causal']:+.4f}  cw={e['c_wrong']:+.4f}  cc={e['c_correct']:+.4f}", flush=True)
    print("\nsaved results/attribution_causal2_africa.json", flush=True)

if __name__ == "__main__":
    main()