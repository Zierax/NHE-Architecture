import time
import os
import torch

GGUF_NAME = "gemma-3-1b-it-Q4_K_M.gguf"
GGUF_DIR = r"C:\Users\DELL\AppData\Roaming\MstyStudio\models\hub\models--unsloth--gemma-3-1b-it-GGUF\snapshots\f0b45be0aac41bd6a100a4b5734cad5f67255bfb"
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "gemma3-1b-fp16")

from transformers import AutoModelForCausalLM, AutoTokenizer

if os.path.isdir(SAVE_DIR):
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(SAVE_DIR, dtype=torch.float16, attn_implementation="eager")
    tok = AutoTokenizer.from_pretrained(SAVE_DIR)
    print(f"loaded from {SAVE_DIR} in {time.time()-t0:.1f}s dtype={model.dtype}")
else:
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(GGUF_DIR, gguf_file=GGUF_NAME, dtype=torch.float16, attn_implementation="eager")
    tok = AutoTokenizer.from_pretrained(GGUF_DIR, gguf_file=GGUF_NAME)
    print(f"loaded from GGUF in {time.time()-t0:.1f}s dtype={model.dtype}")
    os.makedirs(SAVE_DIR, exist_ok=True)
    t0 = time.time()
    model.save_pretrained(SAVE_DIR)
    tok.save_pretrained(SAVE_DIR)
    print(f"saved fp16 copy in {time.time()-t0:.1f}s")

prompt = "The capital of France is"
ids = tok(prompt, return_tensors="pt")["input_ids"]

t0 = time.time()
with torch.no_grad():
    out = model.generate(ids, max_new_tokens=20, do_sample=True, temperature=0.7, return_dict_in_generate=True, output_hidden_states=True, use_cache=True)
dt = time.time() - t0
gen = tok.decode(out.sequences[0][ids.shape[1]:], skip_special_tokens=True)
print(f"generated {out.sequences.shape[1] - ids.shape[1]} tokens in {dt:.1f}s = { (out.sequences.shape[1]-ids.shape[1])/dt:.2f} tok/s")
print("output:", gen)
hs = out.hidden_states
print("hidden_states type:", type(hs), "len:", len(hs))
first_step = hs[0]
print("step0 type:", type(first_step), "len:", len(first_step))
print("step0 last layer shape:", tuple(first_step[-1].shape))