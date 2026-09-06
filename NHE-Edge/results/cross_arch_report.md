# Cross-Architecture Validation — Qwen2.5 (REAL RUNS, 0.5B)

**Date:** 2026-09-03 adapter (synthetic) → 2026-09-05 real Qwen2.5-0.5B weights
**Task:** Replicate NHE jitter/excision pipeline on Qwen2.5 to test generalization beyond Gemma 3 1B.
**Script:** `runtime_rollback_qwen.py` + `attribute_causal2_qwen.py`
**Bottom line:** signal family exists on Qwen (early AUC 0.778 > Gemma 0.742), but its
timing sits post-commit (Qwen answers have no bold markers; city lands ~token 6,
spike at 7–9) → intervention inert (0 flips), 0 breaks. Timing thesis confirmed
by its failure mode.

---

## 1. What was done

### 1.1 Inspected Gemma-specific assumptions in `runtime_rollback.py:1-405`

| Assumption | Gemma hard-code | Qwen reality | Fix in adapter |
|---|---|---|---|
| `N_LAYERS = 26` (`runtime_rollback.py:20`) | 26 layers × hidden 1152 × vocab 262k | Qwen0.5B 24×896×151936, Qwen1.5B 28×1536×151936 | `MODEL_REGISTRY` dict + dynamic inference `get_n_layers(model, fallback)` and `get_hidden_size()` from `model.config` after load (`runtime_rollback_qwen.py:70-90`) |
| `hidden_states` indexing `hs[det["layer"]+1]` (`runtime_rollback.py:82,85,97`) | hs[0]=embed, len=27 | Same tuple layout for Qwen (`model.model.layers` output), len=25/29 | Identical indexing kept; added bounds clamp `min(det_layer+1, len(hs)-1)` (`runtime_rollback_qwen.py:165-183`) |
| `model.model.layers[i].mlp.{gate,up,down}_proj` (`runtime_rollback.py:43-48`) | Gemma3MLP identical structure | Qwen2MLP identical (`gate_proj`, `up_proj`, `down_proj` all `nn.Linear`) — verified via `inspect.getsource(Qwen2MLP)` | Same `apply_mask` kept; added layer/out-of-bounds and `in_features` checks so Gemma mask (layer 25) gracefully skips on Qwen 24 (`runtime_rollback_qwen.py:130-158`) |
| `MAX_NEW = 48`, `TOPICS` 9 topics | hard-coded | same values | kept identical |
| Chat template (`runtime_rollback.py:65,172`) | Manual `"<start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n"` and `tok.convert_tokens_to_ids("<end_of_turn>")==106` | Qwen ChatML via `apply_chat_template` (adds system `"You are Qwen..."`), eos `<|im_end|>` 151645 + `<|endoftext|>` 151643 | `build_prompt_ids()` branches on `chat_format=="gemma"/"qwen"` (`runtime_rollback_qwen.py:100-125`), `get_eos_ids()` returns list per model (`runtime_rollback_qwen.py:127-148`) |
| `SAVE_DIR`/`TOK_DIR` (`runtime_rollback.py:14-15`) | Local `models/gemma3-1b-fp16` | Qwen via HF Hub (no local copy) | `MODEL_REGISTRY` with `local_model_dir` vs `hf_id`; `model_and_tok()` picks local if exists else HF (`runtime_rollback_qwen.py:92-135`) |
| `MASK` global (`runtime_rollback.py:26`) | `results/mask_k32_midwrong.json` (Gemma L8-17 etc.) | Must be re-generated per model via causal patching; Gemma mask invalid for Qwen dims | `apply_mask` now bounds-checks; per-model mask path `mask_k32_midwrong_{model}.json` supported; dead code `h0 = hs[0]...if False` preserved for parity but not relied on |
| `dtype=torch.float16` (`runtime_rollback.py:34`) | Works for Gemma | Qwen config dtype `bfloat16`; on CPU float16 slower than float32 (STATUS notes 15×) | `model_and_tok` uses float16 only if CUDA available else float32, with `cache_dir` explicitly set to `D:/hf_cache` (`runtime_rollback_qwen.py:98-115`) |
| Flow/detector file names | `greedy_flows_africa.npz`, `detector_greedy.json` | Need per-model separation | `*_qwen2.5-0.5b.npz / .json` suffix when `model != gemma3-1b` (`runtime_rollback_qwen.py:210-222`) |
| Detector fitting loops over `N_LAYERS` (`runtime_rollback.py:281-311`) | Fixed 26 | Must infer from actual flows shape | `fit_greedy` now infers `n_layers = flows[0].shape[1]-1` and `hidden_actual = flows[0].shape[2]` and validates vs registry (`runtime_rollback_qwen.py:667-671`) |

### 1.2 Environment audit (measured, not guessed)

```
RAM:  total 16.9 GB, available 3.4 GB at start → 1.8 GB after synthetic run (87-89% used)
Disk D: free 39.3 GB / 302.4 GB  (target for HF cache)
Disk C: free  0.55 GB / 208.6 GB (default HF cache location — would OOM)
CUDA: False (CPU only, torch 2.13.0+cpu, transformers 5.15.1)
Gemma safetensor: 1 999 MB on disk (D:/Axioms/No-Hallucinations-Ever/models/gemma3-1b-fp16/model.safetensors)
```

HF download probe:
```
hf download Qwen/Qwen2.5-0.5B-Instruct --dry-run  -> 999.6 MB (10 files), model.safetensors 988.1 MB
hf download Qwen/Qwen2.5-1.5B-Instruct --dry-run  -> 3.1 GB

Direct Range GET to model.safetensors: 105 KB/s measured (single-stream)
  -> Qwen0.5B 988 MB needs ~2.7 h (160 min) at that rate
  -> Qwen1.5B 3.1 GB needs ~8 h
C: only 0.5 GB free, so default cache fails; HF_HOME must be D:/hf_cache.
hf download without HF_HOME hits C:\Users\DELL\.cache... "Not enough free disk space" warning (554 MB free).
Even with HF_HOME=D:/hf_cache, single-file streaming at 105 KB/s times out our 2-5 min session windows.
```

**Conclusion:** Qwen0.5B *disk- and RAM-feasible* (988 MB download, ~0.9 GB model in RAM float32 ~1.8 GB peak, fits 16.9 GB), Qwen1.5B *borderline RAM* (3.1 GB) and both *network-infeasible* in this session. Real weight download was attempted twice (hf download with HF_HOME=D:/hf_cache, also `snapshot_download`), but each timed out after 120-300 s with 0 MB progress on the blob.

### 1.3 Adapter script `runtime_rollback_qwen.py`

Created `runtime_rollback_qwen.py:1-1008` — single-file, zero dependencies beyond original (torch, transformers, sklearn, numpy).  
Key design choices (Zero-Compromise):

* **Backward compatible:** `python runtime_rollback_qwen.py --model gemma3-1b collect` writes *exactly* the same paths as `runtime_rollback.py` (`greedy_flows_africa.npz`, `detector_greedy.json`). No existing Gemma results were clobbered (verified `detector_greedy.json` mtime unchanged 2026-08-23).
* **Registry:** `MODEL_REGISTRY` with aliases (`qwen0.5b`, `qwen1.5b`, `gemma`), canonical configs (layers/hidden/vocab/hf_id/architecture).
* **Dry-run vs real:** `--dry-run` generates synthetic flows with controlled jitter (hall = large jump in early window for mid layers) and runs the *identical* fitting logic (jump_max / cos_min, early vs full, LOO probe, AUC, thresholds). `--validate` runs full suite for all three models.
* **Cache handling:** `_resolve_hf_cache()` + explicit `cache_dir` arg to `from_pretrained` so `HF_HOME=D:/hf_cache` is respected even when env is set late.
* **CLI paranoia:** `parse_args` handles both `collect --dry-run` and `collect` with posterior `--dry-run` in `rest`; protects Gemma real flows from dry-run overwrite by writing to `greedy_flows_africa_synthetic_gemma_dryrun.npz`.
* **No TODOs:** every path implemented, error-checked (FileNotFound, layer bounds, FileNotFound for mask, eos id list, probe coef dimension mismatch).

### 1.4 Synthetic validation (actual run, not mock doc)

```
python runtime_rollback_qwen.py --validate
  [gemma3-1b]       hidden=1152 layers=26 -> OK jump_max AUC 1.0
  [qwen2.5-0.5b]    hidden=896  layers=24 -> OK jump_max AUC 1.0
  [qwen2.5-1.5b]    hidden=1536 layers=28 -> OK jump_max AUC 1.0

  synthetic collect+fit
  [qwen2.5-0.5b] flow shape (12,25,896) -> probe L11 AUC 0.349 (synthetic random) -> top jump_max_L16 early 1.0 -> t90 15.82 catches 16/16 -> saved detector_greedy_qwen2.5-0.5b.json
  [qwen2.5-1.5b] flow shape (12,29,1536) -> top jump_max_L18 early 1.0 -> t90 20.68 catches 13/13 -> saved detector_greedy_qwen2.5-1.5b.json
```

Also validated:
* Prompt building: Qwen `apply_chat_template` → ids shape (1,35) vs Gemma (1,15); decode correctly shows system prompt.
* EOS handling: Qwen `[151645,151643]`, Gemma `[106]`.
* `apply_mask` on fake models: Gemma mask (32 entries, includes L25) on Qwen fake 24-layer -> `applied 25 skipped 7` correctly; on Gemma fake 26-layer -> `applied 32 skipped 0`.
* Hidden-state indexing clamp for out-of-range layer.

All code paths exercised without downloading weights.

---

## 2. Whether Qwen download/run succeeded

* **Download:** *Not completed* — see environment audit. The 988 MB blob streamed at 105 KB/s; our 2-5 min timeouts yielded 0 MB of progress (incomplete files 0 bytes). The HF tokenizer+config (11.5 MB) *did* download to `D:/hf_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct` (verified), so tokenizer-only paths work.
* **Run:** *Not run on real Qwen weights* — no `model.safetensors` present, so `collect`/`run` with real weights cannot be executed. The adapter is *ready* to run; `model_and_tok("qwen2.5-0.5b")` will succeed as soon as the blob is present (see instructions).
* **What *did* run:** Full synthetic `collect`+`fit` for both Qwen dims, including saving `results/greedy_flows_africa_qwen2.5-0.5b.npz`, `detector_greedy_qwen2.5-0.5b.json` (synthetic), same for 1.5B, plus per-dimension logic tests. All passed.

---

## 3. What the adapted code looks like (architecture notes)

Excerpt of registry (`runtime_rollback_qwen.py:22-62`):

```python
MODEL_REGISTRY = {
  "gemma3-1b":   {hf_id:"google/gemma-3-1b-it",  n_layers:26, hidden:1152, chat_format:"gemma", eos:["<end_of_turn>"]},
  "qwen2.5-0.5b":{hf_id:"Qwen/Qwen2.5-0.5B-Instruct", n_layers:24, hidden:896,  chat_format:"qwen",  eos:["<|im_end|>"]},
  "qwen2.5-1.5b":{hf_id:"Qwen/Qwen2.5-1.5B-Instruct", n_layers:28, hidden:1536, chat_format:"qwen",  eos:["<|im_end|>"]},
}
```

Prompt branch (`runtime_rollback_qwen.py:100-125`) — Gemma keeps the original string concatenation, Qwen uses `tok.apply_chat_template(..., return_tensors="pt")["input_ids"]`.

Detector fitting now infers dimensions (`runtime_rollback_qwen.py:667-671`):

```python
n_layers = flows[0].shape[1] - 1
hidden_actual = flows[0].shape[2]  # 1152 vs 896 vs 1536
```

Mask application (`runtime_rollback_qwen.py:130-158`) bounds-checks `layer < len(layers)` and `unit < mlp.down_proj.in_features`.

File suffixing (`runtime_rollback_qwen.py:210-222`) preserves Gemma paths; Qwen gets `_{model_key}` suffix.

---

## 4. What remains to fully validate (for NeurIPS)

1. **Real download:** In a stable network window (~3 h uninterrupted), run:
   ```powershell
   $env:HF_HOME="D:/hf_cache"
   hf download Qwen/Qwen2.5-0.5B-Instruct   # 988 MB, expect ~160 min at measured 105 KB/s; faster mirrors may be ~10 min
   # or via python:
   # python -c "import os; os.environ['HF_HOME']='D:/hf_cache'; from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-0.5B-Instruct')"
   ```
   Verify `D:/hf_cache/hub/models--Qwen--Qwen2.5-0.5B-Instruct/blobs/*` > 900 MB. If 1.5B is desired, ensure ~4 GB RAM free before load (close other apps); otherwise prefer 0.5B.

2. **Regenerate real flows & detector (overwrites synthetic placeholders):**
   ```powershell
   $env:HF_HOME="D:/hf_cache"
   python runtime_rollback_qwen.py --model qwen2.5-0.5b collect   # writes results/greedy_flows_africa_qwen2.5-0.5b.npz (real)
   python runtime_rollback_qwen.py --model qwen2.5-0.5b fit_greedy # writes results/detector_greedy_qwen2.5-0.5b.json (real, threshold ~not 15 but scale ~ hidden)
   ```
   Compare `probe_L11 AUC` and `jump_max_early` layer/AUC to Gemma (Gemma: jump_early_L19 AUC 0.742 vs jump_L18 0.860). Hypothesis: Qwen will also show middle-layer jitter (layers ~0.6-0.75 depth) with AUC >0.6, but exact layer will shift (expect L16 for 24-layer, L19-21 for 28-layer).

3. **Causal patching per model:** The current Gemma mask (`mask_k32_midwrong.json`) is Gemma-specific. For Qwen, re-run the ported `attribute_causal2` (same patching logic but with Qwen layers/hidden and Qwen prompt formatting) to generate `results/mask_k32_midwrong_qwen2.5-0.5b.json`. The adapter already supports per-model mask via `det.get("mask_path")` or auto-looking for that file; until then it falls back to Gemma mask with bounds-skip (25/32 survive on Qwen0.5B).

4. **Temporal runtime evaluation:**
   ```powershell
   python runtime_rollback_qwen.py --model qwen2.5-0.5b run africa early t90 mask m 0 0.0 5
   python runtime_rollback_qwen.py --model qwen2.5-0.5b run europe early t90 mask m 0 0.0 5
   # and bench:
   # python bench_driver.py  -> needs porting to use build_prompt_ids/get_eos_ids per model
   ```
   Check fired/hallucination rates vs Gemma baseline (Gemma Africa greedy 7/54 → 5/54 with runtime soft). Expect similar direction; quantify with `results/eval_runtime_africa_qwen2.5-0.5b_*.json`.

5. **Report real numbers:** Replace synthetic `detector_greedy_qwen*.json` (thresholds 15-20) with real thresholds (expect ~few thousand, scaled by hidden size and RMSNorm), and fill `results/NUMBERS.md` cross-arch table. Mention that `probe_L10` is weak (0.67 Gemma) and likely also weak on Qwen.

6. **Optional 1.5B check:** Same steps with `--model qwen2.5-1.5b` if RAM allows (needs `available > 4 GB`, so close Chrome etc. before load, or use `dtype float16` + `device_map="auto"` on CPU offload).

---

## 5. Production readiness checklist

* [x] No existing Gemma files clobbered (verified mtime, dry-run protection).
* [x] `runtime_rollback_qwen.py` compiles (`py_compile`), `--help` and `--validate` pass.
* [x] Synthetic flows for all three dims pass detector math with AUC 1.0 on injected signal.
* [x] Bounds checks for mask and detector layer prevent IndexError on 24 vs 26 vs 28 layers.
* [x] Chat/EOS abstraction tested with real tokenizers (Qwen system prompt, Gemma markers).
* [x] HF cache explicitly on D: to avoid C: OOM; documented.
* [x] Script is drop-in: `python runtime_rollback_qwen.py --model gemma3-1b fit_greedy` would re-fit Gemma identically if pointed at real flows (inferred dims match registry).

Limitations labelled: synthetic detectors are **not** real; real Qwen excision still requires causal mask re-derivation.

---

## 6. Quick start when resources are available

```bash
# 1. set cache (critical: C: has 0.5 GB free)
$env:HF_HOME="D:/hf_cache"   # PowerShell, or export HF_HOME=D:/hf_cache (bash)

# 2. download (choose one; prefer 0.5B for CPU)
hf download Qwen/Qwen2.5-0.5B-Instruct

# 3. flows + detector (real, overwrites synthetic placeholder)
python runtime_rollback_qwen.py --model qwen2.5-0.5b collect
python runtime_rollback_qwen.py --model qwen2.5-0.5b fit_greedy

# 4. temporal eval (same window/scale as Gemma best: first 5 tokens, scale 0.0 mask, soft 0.3 variant)
python runtime_rollback_qwen.py --model qwen2.5-0.5b run africa early t90 mask m 0 0.0 5

# 5. inspect
cat results/detector_greedy_qwen2.5-0.5b.json
cat results/eval_runtime_africa_qwen2.5-0.5b_*.json
```

For 1.5B replace `0.5b` with `1.5b` and ensure `available RAM > 4 GB`.

---

*Code:* `runtime_rollback_qwen.py` • *Synthetic artifacts (1.5B only now):* `results/greedy_flows_africa_qwen2.5-1.5b.npz`, `detector_greedy_qwen2.5-1.5b.json` (labelled synthetic until re-run).

---

## 7. REAL Qwen2.5-0.5B results (2026-09-05, supersedes §1.4/§2 for 0.5B)

**Download:** `download_qwen.py` (`HF_HOME=D:/hf_cache`, `HF_HUB_DISABLE_XET=1` after
xet CAS reconstruction failure on this network; plain HTTPS with resume). 10/10
files, 38 min, `model.safetensors` 942 MB. Verified on disk.

**OOM fix:** first live runs died silently after model load — fp32 0.5B (~2 GB) +
per-item `state_dict` clone (~2 GB) exceeded 1.1 GB free RAM. Fixed two ways:
`wsl --shutdown` (free 1.1→4.4 GB) + fp16 default for Qwen path (`NHE_FP16=1`
default in `model_and_tok`; Gemma path untouched).

**Collect (real):** 54 Africa items, 9 greedy-wrong (0.167), ~2 min.
`greedy_flows_africa_qwen2.5-0.5b.npz` is now REAL (overwrote synthetic).

**Fit (real):** full `jump_max_L20` AUC 0.783 (Gemma L18 0.860); **early
`jump_max_early_L22` AUC 0.778** (Gemma L19 0.742 — Qwen early is *stronger*);
probe L11 0.607 (Gemma 0.672). t90 catches 4/9. `detector_greedy_qwen2.5-0.5b.json`
is now REAL (threshold t90 = 81.57, scale ~hundreds not thousands — smaller hidden).

**Live timing (the key finding):** w≤5 fires **0/54** although fit predicted 4/9.
Offline check on saved flows: exceeding jumps sit at jumps[6..8] = live tokens
7–9, outside w≤5. w≤10 fires **9/54 — exactly the offline prediction (1:1)**.
Qwen's spikes come later than Gemma's (ChatML prefix + no bold markers).

**Masks:** Gemma-mask fallback (25/32 applied, 7 skipped OOB) → 0 flips, 0 breaks.
Qwen's own mask via `attribute_causal2_qwen.py` (AtP on Qwen's 9 hallucinations,
ChatML prompts, mid band 10–20): top-32 wrong-commit neurons in **L13–20**
(`mask_k32_midwrong_qwen2.5-0.5b.json`; Gemma's are L10–17 — same relative band,
shifted with the signal peak).

**Intervention:** Qwen's own mask, hard, w≤10 → fired 9/54, hall 9/54 → 9/54,
**0 flips, 0 breaks**. Token-position check (`qwen_timing` probe): Qwen commits
the city at token ~6 (`The capital of Benin is |Con|akry`), spike measured at
7–9 → **post-commit, inert** — the Gemma-full-detector trap replayed on Qwen.

**Pre-commit check (offline, no model):** max jump over live window t≤5 gives
AUC **0.353** (worse than chance); over 10 tokens 0.778. Spike positions:
hall t∈{5,7,7,8,8,8,9,9,10}, truth t∈{5..10} overlapping. **There is no
pre-commit signal on Qwen in this answer format** — the spike *is* the commit
transient, not a precursor.

**Files (all real):** `greedy_flows_africa_qwen2.5-0.5b.npz`,
`detector_greedy_qwen2.5-0.5b.json`, `attribution_causal2_africa_qwen2.5-0.5b.json`,
`mask_k32_midwrong_qwen2.5-0.5b.json`,
`eval_runtime_africa_qwen2.5-0.5b_jump_gt_L22_t90_{none,mask}[_w10].json`.

**Interpretation:** the jitter *family* generalizes across architectures (same
`jump_max` shape, same mid-upper band, offline→live 1:1, own wrong-commit neurons
in the shifted band). But NHE-temporal only works when the spike precedes the
commit by ≥1 token — a *format* property (Gemma's bold markers delay the city),
not a model property. On Qwen-format the method is provably inert, harmlessly
(0 breaks). This boundary condition is reported as a finding, not a failure.

