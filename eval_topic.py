import os
import sys
import time
import json
import unicodedata
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE, "models", "gemma3-1b-fp16")
TOK_DIR = os.path.join(BASE, "models", "gemma3-1b-tokenizer")
RES_DIR = os.path.join(BASE, "results")

TOPICS = {"africa": "topics.AFRICA", "europe": "topics.EUROPE", "elements": "topics.ELEMENTS",
          "asia": "topics.ASIA", "us_states": "topics.US_STATES",
          "africa_largest": "topics.AFRICA_LARGEST", "world_tricky": "topics.WORLD_TRICKY",
          "world_cap_traps": "topics.WORLD_CAP_TRAPS", "world_largest": "topics.WORLD_LARGEST"}

def norm(s):
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()

def apply_mask(model, mask_path):
    with open(mask_path, "r", encoding="utf-8") as fh:
        mask = json.load(fh)
    items = mask["items"]
    layers = model.model.layers
    for entry in items:
        l, u = entry["layer"], entry["unit"]
        mlp = layers[l].mlp
        with torch.no_grad():
            mlp.down_proj.weight[:, u] = 0.0
            mlp.up_proj.weight[u, :] = 0.0
            mlp.gate_proj.weight[u, :] = 0.0
    return len(items)

def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = [a for a in sys.argv[1:] if a.startswith("--")]
    samples = 1
    for o in opts:
        if o.startswith("--samples="):
            samples = int(o.split("=")[1])
    if len(args) not in (1, 2) or args[0] not in TOPICS:
        sys.exit("usage: eval_topic.py <africa|europe|elements> [mask.json] [--samples=N]")
    topic = args[0]
    mask_path = args[1] if len(args) == 2 else None

    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(SAVE_DIR, dtype=torch.float16, attn_implementation="eager")
    tok = AutoTokenizer.from_pretrained(TOK_DIR)
    n_masked = apply_mask(model, mask_path) if mask_path else 0
    print(f"model loaded in {time.time()-t0:.1f}s masked={n_masked}", flush=True)

    mod = __import__("topics")
    items = getattr(mod, TOPICS[topic].split(".")[1])
    end_id = tok.convert_tokens_to_ids("<end_of_turn>")

    results = []
    t0 = time.time()
    for i, item in enumerate(items):
        q, ans = item[0], item[1]
        alt = item[2:]
        text = "<start_of_turn>user\n" + q + "<end_of_turn>\n<start_of_turn>model\n"
        ids = tok(text, return_tensors="pt")["input_ids"]
        gen_texts = []
        with torch.no_grad():
            if samples == 1:
                out = model.generate(ids, max_new_tokens=48, do_sample=False, use_cache=True, return_dict_in_generate=True)
                gen_ids = out.sequences[0][ids.shape[1]:]
                if end_id in gen_ids:
                    gen_ids = gen_ids[:gen_ids.tolist().index(end_id)]
                gen_texts = [tok.decode(gen_ids, skip_special_tokens=True)]
            else:
                for s in range(samples):
                    torch.manual_seed(1000 + i * 100 + s)
                    out = model.generate(ids, max_new_tokens=48, do_sample=True, temperature=0.9,
                                         top_p=0.9, use_cache=True, return_dict_in_generate=True)
                    gen_ids = out.sequences[0][ids.shape[1]:]
                    if end_id in gen_ids:
                        gen_ids = gen_ids[:gen_ids.tolist().index(end_id)]
                    gen_texts.append(tok.decode(gen_ids, skip_special_tokens=True))
        corr = [1 if any(norm(a) in norm(g) for a in (ans,) + alt) else 0 for g in gen_texts]
        results.append({"id": i, "question": q, "answer": ans, "generated": gen_texts[0],
                        "samples": gen_texts, "correct": corr if samples > 1 else corr[0],
                        "majority_correct": int(sum(corr) > samples / 2)})
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)

    n = len(results)
    if samples == 1:
        n_correct = sum(r["correct"] for r in results)
    else:
        n_correct = sum(r["majority_correct"] for r in results)
        n_correct_sampled = sum(sum(r["correct"]) for r in results) / samples
    tag = "baseline"
    if mask_path:
        with open(mask_path, "r", encoding="utf-8") as fh:
            mask = json.load(fh)
        tag = f"k{len(mask['items'])}_{mask['score']}"
    if samples > 1:
        tag += f"_s{samples}"
    out_file = os.path.join(RES_DIR, f"eval_{topic}_{tag}.json")
    summary = {
        "topic": topic, "mask": mask_path, "n": n, "samples": samples,
        "n_correct": n_correct, "hallucination_rate": round((n - n_correct) / n, 4),
        "correct_rate": round(n_correct / n, 4), "results": results,
    }
    if samples > 1:
        summary["hallucination_rate_sampled"] = round(1 - n_correct_sampled / n, 4)
    with open(out_file, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(f"topic={topic} mask={mask_path} n={n} samples={samples} "
          f"majority_hallucination_rate={(n-n_correct)/n:.3f} -> {out_file}", flush=True)

if __name__ == "__main__":
    main()