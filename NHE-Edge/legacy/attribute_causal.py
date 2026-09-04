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
TOP_K = 2000
PROGRESS = os.path.join(RES_DIR, "attribution_causal_progress.npz")

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
    print(f"layers={n_layers} ffn_units={n_units}", flush=True)

    score_sum = np.zeros((n_layers, n_units), dtype=np.float64)
    n_done = 0
    done_qs = []

    ckpt = {}
    if os.path.exists(PROGRESS):
        ckpt = np.load(PROGRESS, allow_pickle=True)
        score_sum = ckpt["score_sum"].astype(np.float64)
        n_done = int(ckpt["n_done"])
        done_qs = list(ckpt["done_qs"])
        print(f"resumed: {n_done} examples already done", flush=True)

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
        if not gen_ids:
            done_qs.append(q)
            n_done += 1
            continue
        ans_ids = tok.encode(ans, add_special_tokens=False)
        if not ans_ids:
            done_qs.append(q)
            n_done += 1
            continue
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
                done_qs.append(q)
                n_done += 1
                continue
            target = None
            for pos in positions:
                gt = full[0, pos].item()
                term = logits[pos, gt] - logits[pos, correct_tok]
                target = term if target is None else target + term
            model.zero_grad(set_to_none=True)
            target.backward()
            ex_score = np.zeros((n_layers, n_units), dtype=np.float64)
            for l in range(n_layers):
                g = acts[l].grad
                if g is not None:
                    a = acts[l].float()
                    ex_score[l] += (g[0].float() * a[0]).detach().numpy()[positions, :].sum(axis=0)
            ex_score /= len(positions)
            score_sum += ex_score
        n_done += 1
        done_qs.append(q)
        np.savez(PROGRESS, score_sum=score_sum, n_done=n_done, done_qs=np.array(done_qs))
        print(f"[{n_done}/{len(hall)}] done in {time.time()-ex_t0:.1f}s: {q[:50]}", flush=True)

    for h in handles:
        h.remove()

    mean_scale = np.mean(np.abs(score_sum)) / max(n_done, 1)
    print(f"examples done: {n_done}, mean |score|: {mean_scale:.6f}", flush=True)

    flat = []
    for l in range(n_layers):
        for u in range(n_units):
            flat.append((l, u, float(score_sum[l, u])))
    flat.sort(key=lambda e: -e[2])
    top = [{"layer": l, "unit": u, "causal": s} for l, u, s in flat[:TOP_K]]

    out = {
        "method": "atp_combined_backward",
        "n_examples": n_done,
        "target": "sum over first N_TARGETS positions of (logit[generated_wrong] - logit[correct_first_token])",
        "score": "per-neuron grad*act, averaged per example",
        "top_by_causal": top,
    }
    with open(os.path.join(RES_DIR, "attribution_causal_africa.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)

    print("top 15 neurons by causal score:", flush=True)
    for e in top[:15]:
        print(f"  L{e['layer']:>2} u {e['unit']:>5}  causal={e['causal']:+.4f}", flush=True)
    print("saved results/attribution_causal_africa.json", flush=True)

if __name__ == "__main__":
    main()