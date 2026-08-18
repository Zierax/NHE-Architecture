# Canonical, protocol-labelled numbers — NHE-Architecture

Last updated: 2026-08-18. **This file is the single source of truth for every number
cited by this project.** Any other document/presentation that quotes a result MUST read
from this table and MUST label the protocol — greedy and sampled-majority are different
experiments and must never be compared without the protocol qualifier.

## Protocols

| Protocol | Decode | n | Evidence file |
|---|---|---|---|
| **greedy** | deterministic argmax, one generation per question | 54 Africa / 44 Europe / 50 US states / 46 Asia / 41 elements | `results/eval_{topic}_{mask}.json` (no `_s5`) |
| **sampled (majority)** | temp=0.9, top_p=0.9, 5 seeds (1000+item*100+seed), majority vote per question | 54 Africa | `results/eval_{topic}_{mask}_s5.json` |

> Conflict resolved: `12.96%->5.56%` is the **greedy** row for `k128_wrong` (7/54 -> 3/54).
> `0.093` is the **sampled-majority** row for the same mask (5/54). Same mask, two
> protocols — both values are correct once the protocol is stated.

## Headline results — Africa (54 questions)

| Intervention | greedy | sampled (majority) | Greedy collateral | Evidence |
|---|---|---|---|---|
| baseline | 0.130 (7/54) | 0.167 (9/54) | — | `eval_africa_baseline.json`, `eval_africa_baseline_s5.json` |
| statistical d_mean/d_var | 0.130 (null) | — | none | `eval_africa_k32_mean.json` |
| causal `k32_midwrong` (layers 8-17, wrong-only) | 0.093 (5/54) — fixes Eswatini/Gambia/Senegal, **breaks South Sudan (Juba->Bor)** | **0.111 (6/54), McNemar p=0.017, net +15 [5,24]** | Europe 0.000 -> 0.000 | `eval_africa_k32_midwrong.json`, `_s5.json`, `eval_europe_k32_midwrong.json` |
| causal `k128_wrong` | **0.056 (3/54)** | **0.093 (5/54), p=0.003, net +20 [8,33]** | **Europe 0.000 -> 0.091 (4/44 broken), p<0.001** | `eval_africa_k128_wrong.json`, `eval_europe_k128_wrong.json`, `_s5` |
| **runtime early excision** (jump_max_early_L19, t90, window<=5, k32_midwrong mask) | **0.074 (4/54)** — 3 fixes, **0 breaks**, 7/54 fires | seed 1000: 0.130 -> 0.111 (2 fixes + **1 documented break: Burundi Gitega->Bujumbura**) | Europe 0 fires, 0.000; Asia/US/elements = baseline | `eval_runtime_africa_jump_gt_L19_t90_mask.json`, `_s1000.json`, `eval_runtime_europe_*.json` |

## Detectors (Africa, greedy, leave-one-out AUC)

| Detector | AUC (LOSO) | Notes |
|---|---|---|
| jump_max_L18 (full generation) | 0.860 | fires post-commit -> inert for intervention |
| jump_max_early_L19 (first 10 tokens) | 0.742 | fires pre-commit on 3/7 -> effective for intervention |
| probe L10 (logistic) | 0.672 | weakest; does NOT transfer across decode protocols |

## Confirmed ceiling

4/7 greedy hallucinations (Cape Verde, Equatorial Guinea, Gabon, Guinea) commit
"quietly" with no pre-commit jitter spike and are not fixable by any threshold of this
detector family nor by the k32_midwrong mask. No threshold/mask in the current family
exceeds this ceiling.

## Honest caveats

- The textual matcher is intentionally permissive (e.g. "Salaffaire...and Praia" counts as
  correct, and Senegal counts as a "fix" but is a hedge: "Diou... While Dakar is the largest
  city"). Every flip above was hand-inspected.
- Single model (Gemma 3 1B, fp16), single domain (Africa capitals), CPU-only runs.
- The detector is protocol-bound: does not transfer from sampled to greedy (probe L10
  AUC ~ 0.05).
- Runtime excision is safe under greedy decoding; under sampling one documented break
  (Burundi). A softer/scaled intervention is the open follow-up.

## Model provenance (verified)

- `models/gemma3-1b-fp16/model.safetensors`: single consolidated fp16 checkpoint,
  **999.9M params, dtype=torch.float16**, config `Gemma3ForCausalLM` (26 layers, d=1152).
  This is the Gemma 3 1B text model; obtain the original from
  https://huggingface.co/google/gemma-3-1b-it (gated) and cast to fp16.
- `models/gemma3-1b-tokenizer`: Gemma 3 SentencePiece tokenizer (vocab 32768).