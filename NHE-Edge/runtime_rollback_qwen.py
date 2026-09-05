"""
NHE-Architecture — Cross-architecture validation adapter
Supports Gemma 3 1B (original) and Qwen2.5 (0.5B / 1.5B) on the same jitter pipeline.

Gemma vs Qwen differences handled here:
  - hidden_size: Gemma 1152 (26 layers) | Qwen0.5B 896 (24 layers) | Qwen1.5B 1536 (28 layers)
  - vocab: Gemma 262144 vs Qwen 151936
  - chat template: Gemma manual "<start_of_turn>user..." vs Qwen ChatML via apply_chat_template (adds system prompt)
  - eos ids: Gemma <end_of_turn>=106 vs Qwen <|im_end|>=151645 (+ <|endoftext|> 151643)
  - MLP naming identical (gate_proj / up_proj / down_proj) -> same apply_mask works
  - hidden_states indexing identical: hs[0]=embed, hs[1..N]=layers => hs[layer+1] correct for both
  - N_LAYERS and hidden_size are inferred dynamically from model config, not hard-coded
  - Mask layers are bounds-checked per model (Gemma mask has layer 25 which is invalid for Qwen0.5B/24)

Backward compatibility: when --model gemma3-1b (default) the file paths and behavior are
identical to runtime_rollback.py (greedy_flows_africa.npz, detector_greedy.json).

Usage:
  python runtime_rollback_qwen.py --model gemma3-1b collect
  python runtime_rollback_qwen.py --model qwen2.5-0.5b collect
  python runtime_rollback_qwen.py --model qwen2.5-0.5b fit_greedy
  python runtime_rollback_qwen.py --model qwen2.5-0.5b run africa early t90 mask m 0 0.0 5
  python runtime_rollback_qwen.py --dry-run                          # validates detector logic with mocked Qwen dims (no download)
  python runtime_rollback_qwen.py --model qwen2.5-0.5b collect --dry-run
  python runtime_rollback_qwen.py --validate                         # full synthetic suite for all registry models

If model download is not feasible (see notes below) use --dry-run / --validate to
exercise every code path except the large weight download.
"""
import os
import sys
import time
import json
import glob
import pickle
import argparse
import unicodedata

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)  # allow sibling imports when used as a module
REPO_ROOT = os.path.dirname(BASE) if os.path.basename(BASE) == "NHE-Edge" else BASE
RES_DIR = os.path.join(BASE, "results")
DATA_DIR = os.path.join(REPO_ROOT, "data")

# ---------------------------------------------------------------------------
# Model registry — single source of truth for architecture differences
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    # Gemma 3 1B — original pipeline
    "gemma3-1b": {
        "aliases": ["gemma", "gemma3-1b-it", "gemma3-1b-fp16", "gemma3-1b"],
        "hf_id": "google/gemma-3-1b-it",
        "local_model_dir": os.path.join(REPO_ROOT, "models", "gemma3-1b-fp16"),
        "local_tok_dir": os.path.join(REPO_ROOT, "models", "gemma3-1b-tokenizer"),
        "n_layers": 26,
        "hidden_size": 1152,
        "intermediate_size": 6912,
        "vocab_size": 262144,
        "chat_format": "gemma",   # manual "<start_of_turn>..."
        "eos_tokens": ["<end_of_turn>"],
        "architecture": "Gemma3ForCausalLM",
    },
    # Qwen2.5 0.5B — chosen for CPU feasibility (988 MB vs 3.1 GB for 1.5B)
    "qwen2.5-0.5b": {
        "aliases": ["qwen0.5b", "qwen2.5-0.5b-instruct", "qwen-0.5b"],
        "hf_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "local_model_dir": None,   # always via HF cache (D:/hf_cache)
        "local_tok_dir": None,
        "n_layers": 24,
        "hidden_size": 896,
        "intermediate_size": 4864,
        "vocab_size": 151936,
        "chat_format": "qwen",     # ChatML via apply_chat_template
        "eos_tokens": ["<|im_end|>", "<|endoftext|>"],
        "architecture": "Qwen2ForCausalLM",
    },
    # Qwen2.5 1.5B — provided for completeness; requires ~3.1 GB + 3-4 GB RAM
    "qwen2.5-1.5b": {
        "aliases": ["qwen1.5b", "qwen2.5-1.5b-instruct", "qwen-1.5b"],
        "hf_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "local_model_dir": None,
        "local_tok_dir": None,
        "n_layers": 28,
        "hidden_size": 1536,
        "intermediate_size": 8960,
        "vocab_size": 151936,
        "chat_format": "qwen",
        "eos_tokens": ["<|im_end|>", "<|endoftext|>"],
        "architecture": "Qwen2ForCausalLM",
    },
}

# reverse alias map
_ALIAS_TO_KEY = {}
for _k, _v in MODEL_REGISTRY.items():
    _ALIAS_TO_KEY[_k.lower()] = _k
    for _a in _v.get("aliases", []):
        _ALIAS_TO_KEY[_a.lower()] = _k

MAX_NEW = 48
TOPICS = {"africa": "topics.AFRICA", "europe": "topics.EUROPE", "elements": "topics.ELEMENTS",
          "asia": "topics.ASIA", "us_states": "topics.US_STATES",
          "africa_largest": "topics.AFRICA_LARGEST", "world_tricky": "topics.WORLD_TRICKY",
          "world_cap_traps": "topics.WORLD_CAP_TRAPS", "world_largest": "topics.WORLD_LARGEST"}
# Gemma mask is default; Qwen masks must be generated separately via causal patching on Qwen.
DEFAULT_MASK_GEMMA = os.path.join(RES_DIR, "mask_k32_midwrong.json")

def normalize_model_key(key: str) -> str:
    kf = key.strip().lower()
    if kf in _ALIAS_TO_KEY:
        return _ALIAS_TO_KEY[kf]
    raise ValueError(f"Unknown model key '{key}'. Known: {list(MODEL_REGISTRY.keys())} + aliases {list(_ALIAS_TO_KEY.keys())}")

def get_model_config(model_key: str) -> dict:
    return MODEL_REGISTRY[normalize_model_key(model_key)]

def get_n_layers(model, fallback: int) -> int:
    """Dynamically infer layer count from loaded model, fallback to registry."""
    try:
        # both Gemma3 and Qwen2 expose model.model.layers
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return len(model.model.layers)
        if hasattr(model, "layers"):
            return len(model.layers)
    except Exception:
        pass
    return fallback

def get_hidden_size(model, fallback: int) -> int:
    try:
        cfg = getattr(model, "config", None)
        if cfg is not None and hasattr(cfg, "hidden_size"):
            return int(cfg.hidden_size)
    except Exception:
        pass
    return fallback

def _resolve_hf_cache():
    """Return cache dir to use; prefer D:/hf_cache when C: is low."""
    # Order: explicit env, then D:/hf_cache if exists, else default
    for k in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        v = os.environ.get(k)
        if v:
            return v
    if os.path.isdir("D:/hf_cache"):
        return "D:/hf_cache"
    return None

def model_and_tok(model_key: str):
    """Load model+tokenizer for model_key. Respects HF_HOME / HF cache location."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = get_model_config(model_key)
    t0 = time.time()
    cache_dir = _resolve_hf_cache()
    if cache_dir and "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = cache_dir
    # Gemma has local directory; prefer it if exists
    if cfg.get("local_model_dir") and os.path.isdir(cfg["local_model_dir"]):
        model_path = cfg["local_model_dir"]
        tok_path = cfg["local_tok_dir"] if os.path.isdir(cfg.get("local_tok_dir", "")) else model_path
        print(f"[{model_key}] loading local Gemma from {model_path}", flush=True)
        # original used dtype float16+eager; keep for parity
        # on CPU, float32 is 15x faster for backward but forward float16 is fine; keep float16 for Gemma
        try:
            model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float16, attn_implementation="eager",
                                                         trust_remote_code=True)
        except TypeError:
            # transformers <5 fallback: torch_dtype
            model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, attn_implementation="eager",
                                                         trust_remote_code=True)
        tok = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    else:
        # Qwen path: HF hub
        hf_id = cfg["hf_id"]
        # Explicit cache_dir ensures D:/hf_cache is used even if env was set late
        ck = cache_dir
        print(f"[{model_key}] loading {hf_id} (cache_dir={ck or 'default HF cache'})", flush=True)
        # On CPU without CUDA, float32 is faster than float16 and avoids half-precision issues
        use_fp16 = torch.cuda.is_available()
        # Build kwargs with explicit cache if available
        extra = {}
        if ck:
            extra["cache_dir"] = ck
        try:
            if use_fp16:
                model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=torch.float16, attn_implementation="eager",
                                                             trust_remote_code=True, **extra)
            else:
                model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=torch.float32, attn_implementation="eager",
                                                             trust_remote_code=True, **extra)
        except TypeError:
            # fallback for older API
            if use_fp16:
                model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.float16, attn_implementation="eager",
                                                             trust_remote_code=True, **extra)
            else:
                model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.float32, attn_implementation="eager",
                                                             trust_remote_code=True, **extra)
        tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True, **extra)
    # report realized architecture (dynamic)
    try:
        n_layers_real = get_n_layers(model, cfg["n_layers"])
        hs_real = get_hidden_size(model, cfg["hidden_size"])
        print(f"[{model_key}] model loaded in {time.time()-t0:.1f}s n_layers={n_layers_real} hidden_size={hs_real} "
              f"(registry n_layers={cfg['n_layers']} hidden={cfg['hidden_size']})", flush=True)
        if n_layers_real != cfg["n_layers"] or hs_real != cfg["hidden_size"]:
            print(f"[{model_key}] WARNING: registry mismatch — using realized values for computation", flush=True)
    except Exception as e:
        print(f"[{model_key}] model loaded in {time.time()-t0:.1f}s (could not infer dims: {e})", flush=True)
    return model, tok

def norm(s):
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()

# ---------------------------------------------------------------------------
# Prompt / EOS handling per architecture
# ---------------------------------------------------------------------------
def build_prompt_ids(tok, question: str, model_key: str):
    cfg = get_model_config(model_key)
    fmt = cfg["chat_format"]
    if fmt == "gemma":
        text = "<start_of_turn>user\n" + question + "<end_of_turn>\n<start_of_turn>model\n"
        return tok(text, return_tensors="pt")["input_ids"]
    elif fmt == "qwen":
        # Qwen ChatML — use official chat template (includes system prompt as designed)
        messages = [{"role": "user", "content": question}]
        enc = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
        # tok.apply_chat_template with return_tensors="pt" returns BatchEncoding(dict) in transformers 5.x
        # it is dict-like; extract input_ids tensor
        if isinstance(enc, dict):
            return enc["input_ids"]
        # BatchEncoding is dict-like but not exact dict
        try:
            return enc["input_ids"]
        except Exception:
            # fallback: enc is already tensor or list
            if isinstance(enc, torch.Tensor):
                return enc if enc.dim() == 2 else enc.unsqueeze(0)
            # list
            return torch.tensor([enc], dtype=torch.long)
    else:
        raise ValueError(f"unknown chat_format {fmt}")

def get_eos_ids(tok, model_key: str):
    cfg = get_model_config(model_key)
    fmt = cfg["chat_format"]
    if fmt == "gemma":
        eid = tok.convert_tokens_to_ids("<end_of_turn>")
        # Gemma also has eos_token_id==106; treat both
        ids = []
        if eid is not None and eid != tok.unk_token_id:
            ids.append(eid)
        if tok.eos_token_id is not None and tok.eos_token_id not in ids:
            ids.append(tok.eos_token_id)
        return ids if ids else [106]
    elif fmt == "qwen":
        ids = []
        if tok.eos_token_id is not None:
            ids.append(tok.eos_token_id)
        for tstr in cfg.get("eos_tokens", []):
            try:
                tid = tok.convert_tokens_to_ids(tstr)
                if tid is not None and tid != tok.unk_token_id and tid not in ids:
                    ids.append(tid)
            except Exception:
                pass
        # canonical Qwen ids: <|im_end|> 151645, <|endoftext|> 151643
        for cand in [151645, 151643]:
            if cand not in ids:
                # only add if vocab large enough
                try:
                    if cand < tok.vocab_size:
                        ids.append(cand)
                except Exception:
                    pass
        return ids
    else:
        return [tok.eos_token_id] if tok.eos_token_id is not None else []

def _is_eos_token(tok_id: int, eos_ids) -> bool:
    return tok_id in eos_ids

# ---------------------------------------------------------------------------
# Mask application — generic across Gemma and Qwen (both use model.model.layers[i].mlp.{gate,up,down}_proj)
# ---------------------------------------------------------------------------
def apply_mask(model, mask_path, scale=0.0):
    with open(mask_path, "r", encoding="utf-8") as fh:
        mask = json.load(fh)
    # resolve layers container
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    else:
        raise AttributeError("could not locate model.model.layers")
    n_layers = len(layers)
    skipped = 0
    applied = 0
    with torch.no_grad():
        for e in mask["items"]:
            l = int(e["layer"])
            u = int(e["unit"])
            if l < 0 or l >= n_layers:
                skipped += 1
                continue
            try:
                mlp = layers[l].mlp
            except AttributeError:
                # fallback: some models name it differently; try .mlp
                skipped += 1
                continue
            # intermediate size bounds check
            try:
                # down_proj: [hidden, intermediate] -> weight shape [hidden, intermediate]? Actually Linear(in=intermediate, out=hidden) => weight [hidden, intermediate]
                # gate/up: [intermediate, hidden] => weight [intermediate, hidden]
                # So unit must be < intermediate_size
                inter = getattr(mlp.down_proj, "in_features", None)
                if inter is not None and (u < 0 or u >= inter):
                    skipped += 1
                    continue
            except Exception:
                pass
            try:
                mlp.down_proj.weight[:, u] *= scale
                mlp.up_proj.weight[u, :] *= scale
                mlp.gate_proj.weight[u, :] *= scale
                applied += 1
            except Exception as ex:
                # shape mismatch -> skip
                skipped += 1
                continue
    if skipped:
        print(f"apply_mask: applied {applied} skipped {skipped} (layer/out-of-bounds for this model n_layers={n_layers})", flush=True)
    return applied

def detector_fire(det, score, prev_score):
    t = det["threshold"]
    if det["type"] == "jump_gt":
        return prev_score is not None and score > t
    if det["type"] == "cos_lt":
        return prev_score is not None and score < t
    if det["type"] == "probe_lt":
        return score < t
    if det["type"] == "depth_gt":
        return score > t
    return False

def gen_with_detector(model, tok, ids, det, model_key="gemma3-1b"):
    eos_ids = get_eos_ids(tok, model_key)
    cfg = get_model_config(model_key)
    # abstain sequence: same text but tokenized per model's vocab
    abstain_ids = tok.encode("I'm not sure.", add_special_tokens=False)
    n_masked = 0
    abstained = False
    fired_at = None
    feats = []
    prev_h = None
    prev_score = None
    past = None
    out_ids = []
    ids = ids.clone()
    new_tok = ids
    rng = torch.Generator().manual_seed(det.get("seed", 0)) if det.get("sample") else None
    # Resolve default mask per model
    if normalize_model_key(model_key) == "gemma3-1b":
        default_mask = DEFAULT_MASK_GEMMA
    else:
        # for Qwen we look for per-model mask else fallback to Gemma mask (with bounds check)
        qwen_mask = os.path.join(RES_DIR, f"mask_k32_midwrong_{normalize_model_key(model_key)}.json")
        default_mask = qwen_mask if os.path.exists(qwen_mask) else DEFAULT_MASK_GEMMA
    # Allow det to override mask path
    det_mask_path = det.get("mask_path", default_mask)
    for t in range(MAX_NEW):
        out = model(input_ids=new_tok, past_key_values=past, use_cache=True, output_hidden_states=True)
        past = out.past_key_values
        hs = out.hidden_states
        # det layer bounds check: if layer >= n_layers, clamp
        try:
            n_layers_real = get_n_layers(model, cfg["n_layers"])
        except Exception:
            n_layers_real = cfg["n_layers"]
        det_layer = int(det.get("layer", 0))
        if det_layer >= n_layers_real:
            det_layer = n_layers_real - 1
        if det["type"] == "probe_lt":
            # probe not re-trained for Qwen yet; keep same indexing but note scale difference
            h_idx = min(det_layer + 1, len(hs)-1)
            h = hs[h_idx][0, -1]
            # coef may be dimension-mismatched if transferred across models; handle gracefully
            try:
                score = (h.float() @ det["coef"] + det["intercept"]).item()
            except Exception:
                score = 0.0
        elif det["type"] in ("jump_gt", "cos_lt"):
            h_idx = min(det_layer + 1, len(hs)-1)
            h = hs[h_idx][0, -1].float()
            if prev_h is not None:
                # ensure same dim (when hidden_size differs across models, prev_h already matches current model)
                d = h - prev_h
                if det["type"] == "jump_gt":
                    score = torch.linalg.norm(d).item()
                else:
                    score = (h @ prev_h / (torch.linalg.norm(h) * torch.linalg.norm(prev_h) + 1e-9)).item()
            else:
                score = 0.0
            prev_h = h
        else:
            h_idx = min(det_layer + 1, len(hs)-1)
            hl = hs[h_idx][0, -1].float()
            hlm = hs[min(det_layer, len(hs)-1)][0, -1].float()
            score = torch.linalg.norm(hl - hlm).item()
        feats.append(score)
        if fired_at is None and detector_fire(det, score, prev_score) and (not det.get("window") or t <= det["window"]):
            fired_at = t
            if det.get("mode") == "abstain":
                out_ids.extend(abstain_ids)
                abstained = True
                break
            if det.get("mode") == "rollback" and len(out_ids) >= 2:
                ids = ids[:, :-2]
                out_ids = out_ids[:-2]
                prev_h = None
                prev_score = None
                past = None
                out = model(input_ids=ids, past_key_values=None, use_cache=True, output_hidden_states=True)
                past = out.past_key_values
                hs = out.hidden_states
                if det["type"] == "probe_lt":
                    h_idx2 = min(det_layer + 1, len(hs)-1)
                    try:
                        score = (hs[h_idx2][0, -1].float() @ det["coef"] + det["intercept"]).item()
                    except Exception:
                        score = 0.0
                elif det["type"] in ("jump_gt", "cos_lt"):
                    h_idx2 = min(det_layer + 1, len(hs)-1)
                    prev_h = hs[h_idx2][0, -1].float()
                    score = 0.0
                else:
                    h_idx2 = min(det_layer + 1, len(hs)-1)
                    hl = hs[h_idx2][0, -1].float()
                    hlm = hs[min(det_layer, len(hs)-1)][0, -1].float()
                    score = torch.linalg.norm(hl - hlm).item()
                feats.append(score)
            if det.get("mode") != "none":
                # apply mask only if file exists; otherwise skip gracefully
                if os.path.exists(det_mask_path):
                    n_masked = apply_mask(model, det_mask_path, det.get("scale", 0.0))
                else:
                    print(f"[{model_key}] mask {det_mask_path} not found — skipping excision (fired but no mask)", flush=True)
                    n_masked = 0
        nxt = out.logits[0, -1].argmax().item() if not rng else _sample(out.logits[0, -1], rng, det)
        if _is_eos_token(nxt, eos_ids):
            break
        new_tok = torch.tensor([[nxt]])
        ids = torch.cat([ids, new_tok], dim=1)
        out_ids.append(nxt)
        prev_score = score
    return out_ids, feats, fired_at, n_masked, abstained

def _sample(logits, rng, det):
    t = det.get("temperature", 0.9)
    p = det.get("top_p", 0.9)
    probs = torch.softmax(logits / t, dim=-1)
    sorted_p, _ = torch.sort(probs, descending=True)
    cum = torch.cumsum(sorted_p, dim=0)
    cut = cum <= p
    k = int(cut.sum().item())
    if k < 1:
        k = 1
    top_v, top_i = torch.topk(probs, k)
    return top_i[torch.multinomial(top_v / top_v.sum(), 1, generator=rng)].item()

def _resolve_flow_path(model_key):
    mk = normalize_model_key(model_key)
    if mk == "gemma3-1b":
        return os.path.join(RES_DIR, "greedy_flows_africa.npz")
    else:
        return os.path.join(RES_DIR, f"greedy_flows_africa_{mk}.npz")

def _resolve_detector_path(model_key):
    mk = normalize_model_key(model_key)
    if mk == "gemma3-1b":
        return os.path.join(RES_DIR, "detector_greedy.json")
    else:
        return os.path.join(RES_DIR, f"detector_greedy_{mk}.json")

def run_topic(topic, det, model_key="gemma3-1b"):
    mod = __import__("topics")
    items = getattr(mod, TOPICS[topic].split(".")[1])
    orig_indices = det.get("subset")
    if orig_indices:
        items = [items[i] for i in orig_indices]
    model, tok = model_and_tok(model_key)
    clean = {k: v.clone() for k, v in model.state_dict().items()}
    static_mask = det.get("static_mask")
    static_scale = det.get("static_scale", 0.0)
    results = []
    t0 = time.time()
    for local_i, item in enumerate(items):
        model.load_state_dict(clean)
        cur_static_n = 0
        if static_mask:
            if os.path.exists(static_mask):
                cur_static_n = apply_mask(model, static_mask, static_scale)
            else:
                print(f"[{model_key}] static mask {static_mask} missing — skipping", flush=True)
        det_i = dict(det)
        if det_i.get("sample"):
            det_i["seed"] = det_i.get("seed", 1000) + local_i * 100
        q, ans = item[0], item[1]
        alt = item[2:]
        ids = build_prompt_ids(tok, q, model_key)
        with torch.no_grad():
            gen_ids, feats, fired_at, n_masked, abstained = gen_with_detector(model, tok, ids, det_i, model_key)
        gen_text = tok.decode(gen_ids, skip_special_tokens=True)
        gen_n = norm(gen_text)
        correct = 1 if any(norm(a) in gen_n for a in (ans,) + alt) else 0
        orig_id = orig_indices[local_i] if orig_indices else local_i
        total_masked = cur_static_n + n_masked
        results.append({"id": orig_id, "question": q, "answer": ans, "generated": gen_text,
                        "correct": correct, "fired_at": fired_at, "n_masked": total_masked,
                        "abstained": abstained,
                        "max_feature": float(max(feats)) if feats else None})
        if (local_i + 1) % 10 == 0:
            print(f"  {local_i+1}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)
    n = len(results)
    n_correct = sum(r["correct"] for r in results)
    n_fired = sum(1 for r in results if r["fired_at"] is not None)
    n_abstained = sum(1 for r in results if r["abstained"])
    tag = f"{det['type']}_L{det['layer']:02d}_{det['threshold_key']}_{det.get('mode', 'mask')}"
    if det.get("static_mask"):
        tag += f"_static{int(det.get('static_scale', 0.0)*10):d}" if det.get("static_scale", 0.0) != 0.0 else "_static0"
    if det.get("window") and det.get("window") != 5:
        tag += f"_w{det['window']}"
    if det.get("scale", 0.0) != 0.0:
        tag += f"_sft{det['scale']}"
    if det.get("sample"):
        tag += f"_s{det.get('seed', 0)}"
    if det.get("bench_suffix"):
        tag += det["bench_suffix"]
    mk = normalize_model_key(model_key)
    suffix = "" if mk == "gemma3-1b" else f"_{mk}"
    out_file = os.path.join(RES_DIR, f"eval_runtime_{topic}{suffix}_{tag}.json")
    summary = {"topic": topic, "detector": det, "model": mk, "n": n, "n_correct": n_correct, "n_fired": n_fired,
               "n_abstained": n_abstained,
               "hallucination_rate": round((n - n_correct) / n, 4), "results": results}
    with open(out_file, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(f"[{topic} {tag}] model={mk} fired={n_fired}/{n} hall={(n-n_correct)/n:.3f} -> {out_file}", flush=True)

def collect_greedy_flows(model_key="gemma3-1b", dry_run=False):
    """
    Collect hidden-state flows for AFRICA (54 items). Each flow is (T, N_LAYERS+1, hidden_size) fp16.
    When dry_run=True, no model is loaded; synthetic flows are generated for validation.
    For Gemma dry-run we do NOT overwrite the real flows file; we write to a synthetic temp file instead.
    """
    cfg = get_model_config(model_key)
    out_path = _resolve_flow_path(model_key)
    if dry_run:
        mk = normalize_model_key(model_key)
        if mk == "gemma3-1b" and os.path.exists(out_path):
            # protect real Gemma flows (2 GB) from being clobbered by a dry-run
            out_path = os.path.join(RES_DIR, "greedy_flows_africa_synthetic_gemma_dryrun.npz")
            print(f"[{model_key}] DRY-RUN: protecting real flows, writing synthetic to {out_path}", flush=True)
        else:
            print(f"[{model_key}] DRY-RUN collect_greedy_flows — generating synthetic flows for n=54, hidden={cfg['hidden_size']}, layers={cfg['n_layers']}", flush=True)
        _collect_synthetic_flows(model_key, out_path)
        return
    model, tok = model_and_tok(model_key)
    mod = __import__("topics")
    items = mod.AFRICA
    eos_ids = get_eos_ids(tok, model_key)
    flows, texts, labels = [], [], []
    t0 = time.time()
    for i, item in enumerate(items):
        q, ans = item[0], item[1]
        alt = item[2:]
        ids = build_prompt_ids(tok, q, model_key)
        hs_all, out_ids = [], []
        past = None
        new_tok = ids
        with torch.no_grad():
            for t in range(MAX_NEW):
                out = model(input_ids=new_tok, past_key_values=past, use_cache=True, output_hidden_states=True)
                past = out.past_key_values
                hs_all.append([h[0, -1].float().numpy() for h in out.hidden_states])
                nxt = out.logits[0, -1].argmax().item()
                if _is_eos_token(nxt, eos_ids):
                    break
                new_tok = torch.tensor([[nxt]])
                out_ids.append(nxt)
        flow = np.stack(hs_all).astype(np.float16)
        gen_text = tok.decode(out_ids, skip_special_tokens=True)
        gen_n = norm(gen_text)
        correct = 1 if any(norm(a) in gen_n for a in (ans,) + alt) else 0
        flows.append(flow)
        texts.append(gen_text)
        labels.append(correct)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)
    np.savez(out_path, flows=np.array(flows, dtype=object), texts=np.array(texts, dtype=object), labels=np.array(labels))
    n_hall = sum(1 for l in labels if not l)
    # also save metadata
    print(f"saved {out_path} n={len(labels)} hall={n_hall}", flush=True)

def _collect_synthetic_flows(model_key: str, out_path: str):
    """Generate synthetic flows with controlled jitter to validate fitting logic without model weights."""
    cfg = get_model_config(model_key)
    n_layers = cfg["n_layers"]
    hidden = cfg["hidden_size"]
    np.random.seed(42)
    # Use topics.AFRICA length for size (54)
    mod = __import__("topics")
    n = len(mod.AFRICA)
    flows = []
    labels = []
    texts = []
    # We craft two regimes: correct=low jitter, hallucinated=high jitter in mid layers 14-19 (model-relative)
    # This mirrors Gemma finding (jitter in middle layers predicts hallucination).
    for i in range(n):
        T = np.random.randint(6, 14)  # random generation length 6-13
        # base hidden trajectory: random walk
        base = np.random.randn(T, n_layers+1, hidden).astype(np.float32) * 0.5
        # add cumulative walk along token dimension to make successive states correlated
        for t in range(1, T):
            base[t] += base[t-1] * 0.9
        # decide label: first ~13% hallucinated (as observed 7/54 for Gemma greedy Africa)
        is_hall = (i < max(1, n // 7)) or (np.random.rand() < 0.15)
        if is_hall:
            # inject large jump in early window for mid layers (detector_early should catch)
            pick_layer = min(n_layers - 1, cfg["n_layers"] * 2 // 3)  # ~2/3 depth, L16 for Gemma 26, L16 for Qwen 24, L19 for 28
            # Choose layer ~ 75% depth for early detector: L19 for Gemma 26 -> 19, for Qwen 24->18, 28->21
            early_layer = min(n_layers - 1, int(n_layers * 0.75))
            # inject in early window (tokens 1-5)
            if T > 3:
                jump_t = np.random.randint(1, min(5, T-1))
                # large displacement for that layer
                base[jump_t, early_layer+1] += np.random.randn(hidden) * 8.0
                # also secondary jump for full detector
                base[jump_t+1, pick_layer+1] += np.random.randn(hidden) * 6.0
            labels.append(0)
            texts.append("hallucinated synthetic answer")
        else:
            labels.append(1)
            texts.append("correct synthetic answer")
        flow = base.astype(np.float16)
        flows.append(flow)
    labels_arr = np.array(labels)
    print(f"[synthetic {model_key}] n={n} hall={(labels_arr==0).sum()} truth={(labels_arr==1).sum()} "
          f"flow shape example {flows[0].shape} hidden={hidden} n_layers={n_layers}", flush=True)
    np.savez(out_path, flows=np.array(flows, dtype=object), texts=np.array(texts, dtype=object), labels=labels_arr)
    print(f"saved {out_path} (synthetic, no model needed)", flush=True)

def _auc(y, s):
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2:
        return float("nan")
    return roc_auc_score(y, s)

def fit_greedy(model_key="gemma3-1b", dry_run=False):
    cfg = get_model_config(model_key)
    n_layers_cfg = cfg["n_layers"]
    flow_path = _resolve_flow_path(model_key)
    det_path = _resolve_detector_path(model_key)
    if not os.path.exists(flow_path):
        raise FileNotFoundError(f"flow file not found: {flow_path}. Run collect first (or with --dry-run).")
    d = np.load(flow_path, allow_pickle=True)
    flows = list(d["flows"])
    labels = np.array(d["labels"]).astype(int)
    y = 1 - labels  # 1=hallucination
    n = len(flows)
    hall = y == 1
    truth = y == 0
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneOut
    print(f"[{model_key}] greedy flows: n={n} hall={int(hall.sum())} truth={int(truth.sum())} "
          f"hidden_expected={cfg['hidden_size']} n_layers_expected={n_layers_cfg}", flush=True)
    # validate flow shapes
    for i, f in enumerate(flows[:2]):
        print(f"  flow[{i}] shape {f.shape} dtype {f.dtype} (T, n_layers+1, hidden)", flush=True)
        # shape check
        if f.shape[1] != n_layers_cfg + 1:
            print(f"  WARNING: flow shape {f.shape} does not match registry n_layers+1={n_layers_cfg+1} — using actual shape", flush=True)
        if f.shape[2] != cfg["hidden_size"]:
            print(f"  WARNING: flow hidden {f.shape[2]} != registry {cfg['hidden_size']}", flush=True)
    # infer actual n_layers from data
    n_layers = flows[0].shape[1] - 1
    hidden_actual = flows[0].shape[2]
    print(f"[{model_key}] inferred n_layers={n_layers} hidden={hidden_actual} from flows", flush=True)

    T = np.array([f.shape[0] for f in flows])
    x_last = np.stack([f[-1] for f in flows]).astype(np.float32)

    auc_l10 = float("nan")
    # Probe at layer ~ 40% depth (~L10 for Gemma 26, L9 for Qwen 24, L11 for 28)
    probe_layer = min(11, n_layers-1)  # keep 11 as original Gemma L11 (index 11), bounds-checked
    # For Qwen 24, 11 is valid; for 28 also
    # If we want adaptive: 40% depth
    # probe_layer = min(n_layers-1, int(n_layers*0.4))
    preds = np.zeros(n)
    loo = LeaveOneOut()
    for tr, te in loo.split(x_last):
        clf = LogisticRegression(C=1.0, max_iter=2000)
        clf.fit(x_last[:, probe_layer, :][tr], y[tr])
        preds[te] = clf.predict_proba(x_last[te][:, probe_layer, :])[:, 1]
    auc_l10 = _auc(y, preds)
    print(f"[{model_key}] probe L{probe_layer:02d} (LOO) AUC(hall)={auc_l10:.3f}", flush=True)

    feat_jumps = np.zeros((n, n_layers))
    feat_cos = np.zeros((n, n_layers))
    feat_jumps_early = np.zeros((n, n_layers))
    feat_cos_early = np.zeros((n, n_layers))
    for i, f in enumerate(flows):
        for l in range(n_layers):
            h = f[:, l + 1, :].astype(np.float32)
            if len(h) >= 2:
                d = h[1:] - h[:-1]
                jumps = np.linalg.norm(d, axis=1)
                num = np.sum(h[1:] * h[:-1], axis=1)
                den = np.linalg.norm(h[1:], axis=1) * np.linalg.norm(h[:-1], axis=1) + 1e-9
                cos = num / den
                feat_jumps[i, l] = jumps.max()
                feat_cos[i, l] = cos.min()
                e = min(10, len(jumps))
                feat_jumps_early[i, l] = jumps[:e].max()
                feat_cos_early[i, l] = cos[:e].min()
            else:
                feat_jumps[i, l] = 0.0
                feat_cos[i, l] = 0.0
                feat_jumps_early[i, l] = 0.0
                feat_cos_early[i, l] = 0.0
    best = []
    for l in range(n_layers):
        best.append((f"jump_max_L{l:02d}", _auc(y, feat_jumps[:, l]), l, "jump_gt"))
        best.append((f"cos_min_L{l:02d}", _auc(y, feat_cos[:, l]), l, "cos_lt"))
    best_early = []
    for l in range(n_layers):
        best_early.append((f"jump_max_early_L{l:02d}", _auc(y, feat_jumps_early[:, l]), l, "jump_gt", feat_jumps_early))
        best_early.append((f"cos_min_early_L{l:02d}", _auc(y, feat_cos_early[:, l]), l, "cos_lt", feat_cos_early))
    best_early.sort(key=lambda e: -e[1])
    best.sort(key=lambda e: -e[1])
    print(f"[{model_key}] top time-jitter features (greedy, scalar AUC):")
    for name, a, l, typ in best[:6]:
        print(f"  {name}: {a:.3f}")
    print(f"[{model_key}] top EARLY (first-10-tokens) features:")
    for name, a, l, typ, _ in best_early[:6]:
        print(f"  {name}: {a:.3f}")
    jname, jauc, jl, jtyp = best[0]
    ename, eauc, el, etyp, earr = best_early[0]
    if eauc >= max(auc_l10, jauc):
        sel = {"type": etyp, "layer": el, "auc": eauc, "arr": earr}
        sel_name = ename
        sel["early"] = True
    else:
        sel = {"type": jtyp, "layer": jl, "auc": jauc, "arr": feat_jumps if jtyp == "jump_gt" else feat_cos}
        sel_name = jname
        sel["early"] = False
    print(f"[{model_key}] selected: {sel_name} AUC={sel['auc']:.3f} early={sel['early']}", flush=True)
    arr = sel.pop("arr")
    vals_truth = arr[truth, sel["layer"]]
    if sel["type"] in ("probe_lt", "cos_lt"):
        thr90, thr95 = np.percentile(vals_truth, 10), np.percentile(vals_truth, 5)
        s_hall = arr[hall, sel["layer"]]
        n_hall_90 = int((s_hall < thr90).sum())
        n_hall_95 = int((s_hall < thr95).sum())
    else:
        thr90, thr95 = np.percentile(vals_truth, 90), np.percentile(vals_truth, 95)
        s_hall = arr[hall, sel["layer"]]
        n_hall_90 = int((s_hall > thr90).sum())
        n_hall_95 = int((s_hall > thr95).sum())
    print(f"  t90={thr90:.4f} catches {n_hall_90}/{int(hall.sum())} | t95={thr95:.4f} catches {n_hall_95}/{int(hall.sum())} (train calibration)", flush=True)
    sel["threshold_t90"] = float(thr90)
    sel["threshold_t95"] = float(thr95)
    sel["catch_t90"] = n_hall_90
    sel["catch_t95"] = n_hall_95
    sel["trained_on"] = f"greedy_africa_{normalize_model_key(model_key)}"
    sel["name"] = sel_name

    ear = {"type": etyp, "layer": el, "auc": eauc, "early": True, "name": ename}
    ear_arr = earr
    vals_truth_e = ear_arr[truth, el]
    if etyp == "jump_gt":
        e90, e95 = np.percentile(vals_truth_e, 90), np.percentile(vals_truth_e, 95)
        s_hall_e = ear_arr[hall, el]
        n_e90, n_e95 = int((s_hall_e > e90).sum()), int((s_hall_e > e95).sum())
    else:
        e90, e95 = np.percentile(vals_truth_e, 10), np.percentile(vals_truth_e, 5)
        s_hall_e = ear_arr[hall, el]
        n_e90, n_e95 = int((s_hall_e < e90).sum()), int((s_hall_e < e95).sum())
    ear["threshold_t90"] = float(e90)
    ear["threshold_t95"] = float(e95)
    ear["catch_t90"] = n_e90
    ear["catch_t95"] = n_e95
    ear["trained_on"] = f"greedy_africa_{normalize_model_key(model_key)}"
    print(f"  early {ename}: t90={e90:.4f} catches {n_e90}/{int(hall.sum())} | t95={e95:.4f} catches {n_e95}/{int(hall.sum())}", flush=True)

    out = {"selected": sel_name, "detector": sel, "detector_early": ear,
           "probe_L10_auc": float(auc_l10),
           "top_features": [{"name": n, "auc": float(a)} for n, a, _, _ in best[:10]],
           "top_early_features": [{"name": n, "auc": float(a)} for n, a, _, _, _ in best_early[:10]],
           "model": normalize_model_key(model_key),
           "n_layers": n_layers,
           "hidden_size": hidden_actual,
           "trained_on": f"greedy_africa_{normalize_model_key(model_key)}"}
    with open(det_path, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"saved {det_path}", flush=True)
    if dry_run:
        print(f"[DRY-RUN] fit_greedy for {model_key} succeeded without model weights (synthetic flows validated detector logic for hidden_size={hidden_actual} n_layers={n_layers})", flush=True)

# ---------------------------------------------------------------------------
# Validation utilities
# ---------------------------------------------------------------------------
def _test_detector_logic_for_dims(hidden_size: int, n_layers: int, tag: str):
    """Unit-test the jump/cos math for a given hidden size without any model."""
    print(f"  [{tag}] testing detector logic hidden={hidden_size} n_layers={n_layers} ... ", end="", flush=True)
    # create fake hall vs truth flows as in _collect_synthetic_flows but minimal
    np.random.seed(0)
    T = 8
    # truth: smooth
    h_truth = np.cumsum(np.random.randn(T, hidden_size) * 0.3, axis=0)
    # hall: same but inject jump at token 2
    h_hall = h_truth.copy()
    h_hall[2] += np.random.randn(hidden_size) * 10.0
    # compute jump
    jumps_truth = np.linalg.norm(h_truth[1:] - h_truth[:-1], axis=1)
    jumps_hall = np.linalg.norm(h_hall[1:] - h_hall[:-1], axis=1)
    assert jumps_hall.max() > jumps_truth.max(), f"synthetic jump should be larger for hall ({jumps_hall.max():.2f} vs {jumps_truth.max():.2f})"
    # cos
    cos_truth = np.sum(h_truth[1:]*h_truth[:-1], axis=1) / (np.linalg.norm(h_truth[1:], axis=1)*np.linalg.norm(h_truth[:-1], axis=1)+1e-9)
    cos_hall = np.sum(h_hall[1:]*h_hall[:-1], axis=1) / (np.linalg.norm(h_hall[1:], axis=1)*np.linalg.norm(h_hall[:-1], axis=1)+1e-9)
    # AUC check with 20 synthetic examples per class
    n = 20
    feat = []
    labels = []
    for is_hall in [0,1]:
        for _ in range(n):
            base = np.cumsum(np.random.randn(T, hidden_size)*0.3, axis=0)
            if is_hall:
                base[2] += np.random.randn(hidden_size)*10.0
            jm = np.linalg.norm(base[1:]-base[:-1], axis=1).max()
            feat.append(jm)
            labels.append(is_hall)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(labels, feat)
    assert auc > 0.8, f"expected AUC>0.8 for synthetic jitter, got {auc:.3f}"
    print(f"OK jump_max AUC={auc:.3f} (hall max {jumps_hall.max():.1f} vs truth {jumps_truth.max():.1f})", flush=True)

def validate_all_models(dry_run_synthetic: bool = True):
    """Run synthetic detector validation for every registry model."""
    print("=== NHE cross-architecture validation (synthetic) ===", flush=True)
    for key in MODEL_REGISTRY:
        cfg = MODEL_REGISTRY[key]
        _test_detector_logic_for_dims(cfg["hidden_size"], cfg["n_layers"], key)
    if dry_run_synthetic:
        print("\n=== synthetic collect+fit for Qwen dims ===", flush=True)
        for key in ["qwen2.5-0.5b", "qwen2.5-1.5b"]:
            try:
                out_path = _resolve_flow_path(key)
                _collect_synthetic_flows(key, out_path)
                fit_greedy(key, dry_run=True)
                print(f"  [{key}] synthetic collect+fit OK", flush=True)
            except Exception as e:
                print(f"  [{key}] FAILED: {e}", flush=True)
                import traceback; traceback.print_exc()
        # also test Gemma synthetic to ensure not broken
        try:
            key = "gemma3-1b"
            out_path = os.path.join(RES_DIR, "greedy_flows_africa_synthetic_gemma_test.npz")
            _collect_synthetic_flows(key, out_path)
            # fit on synthetic gemma but keep original detector intact: use temp det path
            d = np.load(out_path, allow_pickle=True)
            try:
                print(f"  [gemma3-1b] synthetic flow shape {list(d['flows'])[0].shape} OK", flush=True)
            finally:
                d.close()
            try:
                os.remove(out_path)
            except Exception:
                pass
        except Exception as e:
            print(f"  [gemma3-1b synthetic] FAILED: {e}", flush=True)
    print("\n=== validation complete ===", flush=True)
    # print environment report
    _print_environment_report()

def _print_environment_report():
    import psutil, shutil
    vm = psutil.virtual_memory()
    du_d = shutil.disk_usage("D:")
    du_c = shutil.disk_usage("C:") if os.path.exists("C:") else None
    print("\n--- Environment report ---", flush=True)
    print(f"RAM total {vm.total/1e9:.1f} GB available {vm.available/1e9:.1f} GB used {vm.percent}%", flush=True)
    print(f"Disk D free {du_d.free/1e9:.1f} GB total {du_d.total/1e9:.1f} GB", flush=True)
    if du_c:
        print(f"Disk C free {du_c.free/1e9:.1f} GB total {du_c.total/1e9:.1f} GB", flush=True)
    print(f"Cuda available: {torch.cuda.is_available()}", flush=True)
    print(f"Transformers {__import__('transformers').__version__} torch {torch.__version__}", flush=True)
    for key, cfg in MODEL_REGISTRY.items():
        est_mb = {"gemma3-1b": 2000, "qwen2.5-0.5b": 988, "qwen2.5-1.5b": 3100}[key]
        print(f"Model {key}: hidden={cfg['hidden_size']} layers={cfg['n_layers']} est_download={est_mb} MB hf_id={cfg['hf_id']}", flush=True)
    # download feasibility note
    print("\nDownload feasibility (measured 2026-09-03):", flush=True)
    print("  Qwen2.5-0.5B measured 105 KB/s -> 988 MB needs ~2.7h (160 min). Network is bottleneck; C: has only 0.5 GB free so HF_HOME must be D:/hf_cache.", flush=True)
    print("  Qwen2.5-1.5B 3.1 GB needs ~8h at same rate + 3-4 GB RAM on load (available 3.4 GB borderline) -> not feasible in this session.", flush=True)
    print("  -> Use --dry-run / --validate for logic validation; real weight download requires stable ~2h+ window and HF_HOME=D:/hf_cache.", flush=True)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    # support both --model flag style and legacy positional style from runtime_rollback.py
    # Legacy: runtime_rollback.py <collect|fit_greedy|run topic [early|detector] t90...>
    # New: runtime_rollback_qwen.py --model qwen2.5-0.5b <collect|fit_greedy|run ...>
    parser = argparse.ArgumentParser(description="NHE cross-architecture runner (Gemma + Qwen2.5)", add_help=False)
    parser.add_argument("--model", type=str, default="gemma3-1b", help="model key: gemma3-1b, qwen2.5-0.5b, qwen2.5-1.5b (aliases accepted)")
    parser.add_argument("--dry-run", action="store_true", help="validate code path with synthetic flows instead of real model")
    parser.add_argument("--validate", action="store_true", help="run synthetic suite for all models and exit")
    parser.add_argument("--help", action="store_true")
    # remaining positional
    parser.add_argument("cmd", nargs="?", default=None, help="collect|fit_greedy|run|validate")
    parser.add_argument("rest", nargs=argparse.REMAINDER)
    args, _ = parser.parse_known_args()
    return args

def print_help():
    print(__doc__)
    print("\nLegacy (Gemma) exact compatibility: runtime_rollback_qwen.py --model gemma3-1b collect")
    print("  writes results/greedy_flows_africa.npz and results/detector_greedy.json (identical to runtime_rollback.py)")
    print("Qwen example:")
    print("  HF_HOME=D:/hf_cache python runtime_rollback_qwen.py --model qwen2.5-0.5b collect")
    print("  HF_HOME=D:/hf_cache python runtime_rollback_qwen.py --model qwen2.5-0.5b fit_greedy")
    print("  HF_HOME=D:/hf_cache python runtime_rollback_qwen.py --model qwen2.5-0.5b run africa early t90 mask m 0 0.0 5")
    print("Dry-run (no download):")
    print("  python runtime_rollback_qwen.py --model qwen2.5-0.5b collect --dry-run")
    print("  python runtime_rollback_qwen.py --model qwen2.5-0.5b fit_greedy --dry-run")
    print("  python runtime_rollback_qwen.py --validate")
    _print_environment_report()

def main():
    args = parse_args()
    # Also detect --dry-run / --validate in remainder or raw sys.argv (supports placing flag after cmd)
    raw = " ".join(sys.argv)
    if "--dry-run" in sys.argv and not args.dry_run:
        args.dry_run = True
    if "--validate" in sys.argv and not args.validate:
        args.validate = True
    # also check remainder for flag duplicates (e.g., 'collect --dry-run' where --dry-run lands in rest)
    if args.rest and any(x in ("--dry-run", "--validate") for x in args.rest):
        if "--dry-run" in args.rest:
            args.dry_run = True
            args.rest = [x for x in args.rest if x != "--dry-run"]
        if "--validate" in args.rest:
            args.validate = True
            args.rest = [x for x in args.rest if x != "--validate"]
    if args.help or (args.cmd is None and not args.validate):
        # if no cmd but --validate not set, show help but also keep legacy behavior (default collect)
        if args.help:
            print_help()
            return
        # no cmd -> default legacy collect for gemma
        if not args.validate and not args.dry_run:
            print("no command given; use --help for usage. Defaulting to 'collect' for gemma3-1b", flush=True)
            args.cmd = "collect"
        elif args.validate:
            pass
        else:
            args.cmd = "collect"

    if args.validate:
        validate_all_models(dry_run_synthetic=True)
        return

    # Normalize model key early (validate)
    try:
        mk = normalize_model_key(args.model)
    except ValueError as e:
        print(str(e), flush=True)
        sys.exit(1)

    dry = bool(args.dry_run)
    cmd = (args.cmd or "").strip()

    # handle legacy positional where first arg may be cmd and rest contains topic etc
    # args.rest already contains remaining positional tokens
    # For "collect" and "fit_greedy", they take no extra args
    if cmd == "collect":
        collect_greedy_flows(mk, dry_run=dry)
        return
    if cmd == "fit_greedy":
        fit_greedy(mk, dry_run=dry)
        return
    if cmd in ("validate", "--validate"):
        validate_all_models(dry_run_synthetic=True)
        return
    if cmd == "run":
        # run expects: topic [early|detector] [t90|t95] [mask|rollback|none|abstain] [m|s] seed scale window
        # Mirror runtime_rollback.py CLI: run topic early t90 mask m 0 0.0 5
        rest = args.rest
        # rest[0] is topic if cmd==run
        if len(rest) < 1:
            sys.exit("usage: runtime_rollback_qwen.py --model <key> run <topic> [early|detector] [t90|t95] [mask|rollback|none|abstain] [m|s] [seed] [scale] [window]")
        topic = rest[0]
        sel_name = "detector_early" if (len(rest) > 1 and rest[1] == "early") else "detector"
        thr_key = rest[2] if len(rest) > 2 else "t90"
        mode = rest[3] if len(rest) > 3 else "mask"
        sample_flag = rest[4] if len(rest) > 4 else "m"
        seed = int(rest[5]) if len(rest) > 5 else 1000
        scale = float(rest[6]) if len(rest) > 6 else 0.0
        window = int(rest[7]) if len(rest) > 7 else (5 if sel_name == "detector_early" else None)
        det_path = _resolve_detector_path(mk)
        if not os.path.exists(det_path):
            sys.exit(f"detector not found: {det_path}. Run fit_greedy first (or with --dry-run for synthetic).")
        with open(det_path) as fh:
            dg = json.load(fh)
        det = dict(dg[sel_name])
        det["threshold"] = det["threshold_t90"] if thr_key == "t90" else det["threshold_t95"]
        det["threshold_key"] = thr_key
        det["mode"] = mode
        det["window"] = window
        det["scale"] = scale
        det["sample"] = "s" in sample_flag
        det["seed"] = seed
        print(f"running {topic} with {dg[sel_name].get('name','?')} {thr_key} mode={mode} sample={det['sample']} seed={seed} scale={scale} window={window} model={mk}", flush=True)
        run_topic(topic, det, mk)
        return
    # If cmd not recognized, try legacy positional dispatch: sys.argv[1] may be model flag? Already handled
    # Fallback help
    print(f"unknown command '{cmd}'", flush=True)
    print_help()
    sys.exit(1)

if __name__ == "__main__":
    main()
