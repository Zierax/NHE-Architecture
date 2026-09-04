import time
import sys
import torch

GGUF = r"C:\Users\DELL\AppData\Roaming\MstyStudio\models\hub\models--unsloth--gemma-3-1b-it-GGUF\snapshots\f0b45be0aac41bd6a100a4b5734cad5f67255bfb\gemma-3-1b-it-Q4_K_M.gguf"
gguf_dir = GGUF.rsplit("\\", 1)[0]
gguf_name = GGUF.rsplit("\\", 1)[1]

from transformers import AutoModelForCausalLM, AutoTokenizer

t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(gguf_dir, gguf_file=gguf_name, dtype=torch.float32, attn_implementation="eager")
print(f"[ok] model loaded in {time.time()-t0:.1f}s dtype={model.dtype}")

cfg = model.config
print(f"[cfg] layers={cfg.num_hidden_layers} hidden={cfg.hidden_size} heads={cfg.num_attention_heads} kv={getattr(cfg,'num_key_value_heads',None)} vocab={cfg.vocab_size}")

try:
    tok = AutoTokenizer.from_pretrained(gguf_dir, gguf_file=gguf_name)
    print("[ok] tokenizer from GGUF:", type(tok).__name__)
except Exception as e:
    print("[warn] GGUF tokenizer failed, falling back to HF:", e)
    tok = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")

text = "The capital of France is"
inputs = tok(text, return_tensors="pt")
print(f"[tok] input_ids={inputs['input_ids'].tolist()}")

with torch.no_grad():
    t1 = time.time()
    out = model(**inputs, output_hidden_states=True, use_cache=False)
    print(f"[fwd] forward took {time.time()-t1:.1f}s")

hs = out.hidden_states
print(f"[hs] num tensors={len(hs)} layers={len(hs)-1}")
last = hs[-1]
print(f"[hs] last layer shape={tuple(last.shape)} (batch, seq, hidden)")

with torch.no_grad():
    att = model(**inputs, output_attentions=True, use_cache=False).attentions
print(f"[att] num={len(att)} shape={tuple(att[0].shape)} (batch, heads, seq, seq)")

logits = out.logits
print(f"[logits] shape={tuple(logits.shape)}")