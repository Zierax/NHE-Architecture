import time
import math
import torch

GGUF_NAME = "gemma-3-1b-it-Q4_K_M.gguf"
GGUF_DIR = r"C:\Users\DELL\AppData\Roaming\MstyStudio\models\hub\models--unsloth--gemma-3-1b-it-GGUF\snapshots\f0b45be0aac41bd6a100a4b5734cad5f67255bfb"

from transformers import AutoModelForCausalLM, AutoTokenizer

STATEMENTS = [
    ("The capital of France is Paris.", "The capital of France is Tokyo."),
    ("Water boils at 100 degrees Celsius at sea level.", "Water boils at 50 degrees Celsius at sea level."),
    ("The Earth orbits the Sun.", "The Earth orbits the Moon."),
    ("A human being has two lungs.", "A human being has four lungs."),
]

def entropy(p):
    p = p[p > 0]
    return -(p * p.log2()).sum().item()

t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(GGUF_DIR, gguf_file=GGUF_NAME, dtype=torch.float32, attn_implementation="eager")
tok = AutoTokenizer.from_pretrained(GGUF_DIR, gguf_file=GGUF_NAME)
print(f"loaded in {time.time()-t0:.1f}s")

def run(text):
    ids = tok(text, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        out = model(ids, output_hidden_states=True, output_attentions=True, use_cache=False)
    hs = [h[0].float() for h in out.hidden_states]
    atts = [a[0].float() if a.dim() == 4 else a.float() for a in out.attentions]
    n_layers = len(hs) - 1
    L = hs[0].shape[0]
    norm = [h[-1].norm().item() for h in hs]
    jump = [ (hs[l+1][-1] - hs[l][-1]).norm().item() for l in range(n_layers) ]
    def attn_entropy(att):
        if att.dim() == 3:
            att = att.unsqueeze(0)
        heads = att[0]
        return sum(entropy(heads[h, -1, :]) for h in range(heads.shape[0])) / heads.shape[0]
    attn_ent = [attn_entropy(atts[l]) for l in range(n_layers)]
    tok_sim = [ torch.cosine_similarity(hs[l][-2].unsqueeze(0), hs[l][-1].unsqueeze(0)).item() for l in range(n_layers+1) ]
    return dict(norm=norm, jump=jump, attn_ent=attn_ent, tok_sim=tok_sim, n_tokens=L)

print(f"{'pair':<6}{'tokens':>6}{'metric':<12}{'truth':>12}{'false':>12}{'sep(|d|/pooled_std)':>6}")
print("-" * 60)
for i, (t, f) in enumerate(STATEMENTS):
    rt, rf = run(t), run(f)
    for name in ["jump", "attn_ent", "tok_sim"]:
        vt = rt[name]; vf = rf[name]
        last_t = vt[-1]; last_f = vf[-1]
        pooled = math.sqrt(( (last_t - sum(vt)/len(vt))**2 + (last_f - sum(vf)/len(vf))**2 ) / 2 + 1e-9)
        sep = abs(last_t - last_f) / pooled
        print(f"p{i+1:<6}{rt['n_tokens']:>6}{name:<12}{last_t:>12.4f}{last_f:>12.4f}{sep:>18.2f}")
    print()

print("layer-by-layer: jump (||h_{l+1}-h_l|| at last token)  [T=truth, F=false]")
js = {}
for i, (t, f) in enumerate(STATEMENTS):
    rt, rf = run(t), run(f)
    js[f"p{i+1}T"] = rt["jump"]; js[f"p{i+1}F"] = rf["jump"]
L = len(js["p1T"])
for key in ["p1T", "p1F", "p2T", "p2F"]:
    m = sum(js[key])/L
    std = (sum((x-m)**2 for x in js[key])/L)**0.5
    print(f"  {key}: mean={m:.4f} std={std:.4f} min={min(js[key]):.3f} max={max(js[key]):.3f}")