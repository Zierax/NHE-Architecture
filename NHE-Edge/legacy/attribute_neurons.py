import os
import sys
import time
import json
import pickle
import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE, "models", "gemma3-1b-fp16")
TOK_DIR = os.path.join(BASE, "models", "gemma3-1b-tokenizer")
DATA_DIR = os.path.join(BASE, "data", "africa")
RES_DIR = os.path.join(BASE, "results")
os.makedirs(RES_DIR, exist_ok=True)

SEED = 0
torch.manual_seed(SEED)

def norm(s):
    import unicodedata
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()

def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(SAVE_DIR, dtype=torch.float16, attn_implementation="eager")
    tok = AutoTokenizer.from_pretrained(TOK_DIR)
    print(f"model loaded in {time.time()-t0:.1f}s", flush=True)

    layers = model.model.layers
    n_layers = len(layers)
    n_units = layers[0].mlp.down_proj.in_features
    print(f"layers={n_layers} ffn_units={n_units}", flush=True)

    activations = {}
    handles = []

    def make_hook(l):
        def hook(module, args, output):
            x = args[0]
            if x.dim() == 3 and x.shape[0] == 1:
                activations.setdefault(l, []).append(x[0, -1, :].float().cpu())
        return hook

    for l in range(n_layers):
        handles.append(layers[l].mlp.down_proj.register_forward_hook(make_hook(l)))

    end_id = tok.convert_tokens_to_ids("<end_of_turn>")
    files = sorted(f for f in os.listdir(DATA_DIR) if f.startswith("ex_"))
    per_example = []
    labels = []
    for f in files:
        with open(os.path.join(DATA_DIR, f), "rb") as fh:
            rec = pickle.load(fh)
        activations.clear()
        text = "<start_of_turn>user\n" + rec["question"] + "<end_of_turn>\n<start_of_turn>model\n"
        ids = tok(text, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            model.generate(
                ids, max_new_tokens=48, do_sample=True, temperature=0.9, top_p=0.9,
                use_cache=True,
            )
        vecs = []
        for l in range(n_layers):
            acts = activations.get(l, [])
            if not acts:
                raise RuntimeError(f"no activations captured for layer {l}")
            mat = torch.stack(acts).numpy() if len(acts) > 1 else acts[0].numpy()[None, :]
            vecs.append(mat)
        per_example.append(vecs)
        labels.append(rec["label"])
        print(f"{f} tokens={len(acts)} label={rec['label']}", flush=True)

    for h in handles:
        h.remove()

    labels = np.array(labels, dtype=int)
    n = len(per_example)
    T = [per_example[i][0].shape[0] for i in range(n)]

    means = np.zeros((n, n_layers, n_units), dtype=np.float32)
    vars_ = np.zeros((n, n_layers, n_units), dtype=np.float32)
    for i in range(n):
        for l in range(n_layers):
            means[i, l] = per_example[i][l].mean(axis=0)
            vars_[i, l] = per_example[i][l].var(axis=0)

    np.savez_compressed(
        os.path.join(RES_DIR, "attribution_africa.npz"),
        labels=labels, means=means, vars_=vars_,
    )

    m0 = means[labels == 0].mean(axis=0)
    m1 = means[labels == 1].mean(axis=0)
    v0 = vars_[labels == 0].mean(axis=0)
    v1 = vars_[labels == 1].mean(axis=0)

    def dscore(a, b):
        denom = np.sqrt((np.var(means[labels == 0], axis=0) + np.var(means[labels == 1], axis=0)) / 2 + 1e-9)
        return (a - b) / denom

    d_mean = dscore(m0, m1)
    denom_v = np.sqrt((np.var(vars_[labels == 0], axis=0) + np.var(vars_[labels == 1], axis=0)) / 2 + 1e-9)
    d_var = (v0 - v1) / denom_v

    flat_mean = d_mean.reshape(-1)
    flat_var = d_var.reshape(-1)
    order_mean = np.argsort(-flat_mean)
    order_var = np.argsort(-flat_var)

    top_mean = [{"layer": int(idx // n_units), "unit": int(idx % n_units), "d_mean": float(flat_mean[idx]), "d_var": float(flat_var[idx])} for idx in order_mean[:2000]]
    top_var = [{"layer": int(idx // n_units), "unit": int(idx % n_units), "d_mean": float(flat_mean[idx]), "d_var": float(flat_var[idx])} for idx in order_var[:2000]]

    with open(os.path.join(RES_DIR, "attribution_africa.json"), "w", encoding="utf-8") as fh:
        json.dump({"n_examples": n, "n_hallucinated": int((1 - labels).sum()),
                   "top_by_mean": top_mean, "top_by_var": top_var}, fh, indent=2)

    print("\ntop 15 neurons by d_mean (high activation on hallucinated):")
    for item in top_mean[:15]:
        print(f"  L{item['layer']:>2} u{item['unit']:>5}  d_mean={item['d_mean']:+.3f}  d_var={item['d_var']:+.3f}")
    print("\ntop 15 neurons by d_var (high instability on hallucinated):")
    for item in top_var[:15]:
        print(f"  L{item['layer']:>2} u{item['unit']:>5}  d_mean={item['d_mean']:+.3f}  d_var={item['d_var']:+.3f}")
    print("\nsaved results/attribution_africa.npz + .json")

if __name__ == "__main__":
    main()