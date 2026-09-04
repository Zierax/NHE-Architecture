import os
import sys
import time
import pickle
import unicodedata
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE) if os.path.basename(BASE) == "NHE-Edge" else BASE
SAVE_DIR = os.path.join(REPO_ROOT, "models", "gemma3-1b-fp16")
TOK_DIR = os.path.join(REPO_ROOT, "models", "gemma3-1b-tokenizer")

TOPICS = {
    "africa": "topics.AFRICA",
    "europe": "topics.EUROPE",
    "elements": "topics.ELEMENTS",
}

def norm(s):
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()

def main():
    if len(sys.argv) != 2 or sys.argv[1] not in TOPICS:
        sys.exit(f"usage: collect_topic.py <{ '/'.join(TOPICS) }>")
    topic = sys.argv[1]
    mod = __import__("topics")
    items = getattr(mod, TOPICS[topic].split(".")[1])

    out_dir = os.path.join(REPO_ROOT, "data", topic)
    os.makedirs(out_dir, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(SAVE_DIR, dtype=torch.float16, attn_implementation="eager")
    tok = AutoTokenizer.from_pretrained(TOK_DIR)
    print(f"model loaded in {time.time()-t0:.1f}s", flush=True)
    end_id = tok.convert_tokens_to_ids("<end_of_turn>")

    for i, item in enumerate(items):
        cache_file = os.path.join(out_dir, f"ex_{i:03d}.pkl")
        if os.path.exists(cache_file):
            continue
        q, ans = item[0], item[1]
        alt = item[2:]
        text = "<start_of_turn>user\n" + q + "<end_of_turn>\n<start_of_turn>model\n"
        ids = tok(text, return_tensors="pt")["input_ids"]
        t1 = time.time()
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=48, do_sample=True, temperature=0.9, top_p=0.9,
                return_dict_in_generate=True, output_hidden_states=True, use_cache=True,
            )
        gen_ids = out.sequences[0][ids.shape[1]:]
        if end_id in gen_ids:
            gen_ids = gen_ids[:gen_ids.tolist().index(end_id)]
        gen_text = tok.decode(gen_ids, skip_special_tokens=True)
        steps = out.hidden_states
        n_steps = len(steps)
        L = len(steps[0]) - 1
        flow = torch.zeros(n_steps, L + 1, 1152, dtype=torch.float16)
        for t in range(n_steps):
            for l in range(L + 1):
                flow[t, l] = steps[t][l][0, -1, :]
        gen_n = norm(gen_text)
        label = 1 if any(norm(a) in gen_n for a in (ans,) + alt) else 0
        record = {
            "id": i, "question": q, "answer": ans, "alt_answers": list(alt),
            "generated": gen_text, "label": label, "flow": flow.cpu(), "n_steps": n_steps,
        }
        with open(cache_file, "wb") as f:
            pickle.dump(record, f)
        print(f"ex {i:03d} label={label} steps={n_steps} t={time.time()-t1:.1f}s | {gen_text[:70]!r}", flush=True)

    count = sum(1 for f in os.listdir(out_dir) if f.startswith("ex_"))
    labels = []
    for f in os.listdir(out_dir):
        if not f.startswith("ex_"):
            continue
        with open(os.path.join(out_dir, f), "rb") as fh:
            labels.append(pickle.load(fh)["label"])
    print(f"topic={topic} total={count} truthful={sum(labels)} hallucinated={len(labels)-sum(labels)}", flush=True)

if __name__ == "__main__":
    main()