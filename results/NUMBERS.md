# Canonical, protocol-labelled numbers — NHE-Architecture

Last updated: 2026-08-18 (evening — strict-metric re-scoring + new topics + full sampled battery).
**This file is the single source of truth for every number cited by this project.** Any
other document/presentation that quotes a result MUST read from this table and MUST label
both the **protocol** (greedy / sampled) and the **metric** (substring / strict
first-sentence). Comparing values from different protocols or metrics is invalid.

## Protocols and metrics

| Protocol | Decode | n |
|---|---|---|
| **greedy** | deterministic argmax | 54 africa / 44 europe / 54 africa_largest / 49 world_tricky |
| **sampled (majority)** | temp=0.9, top_p=0.9, seeds 1000+item*100+s, s=0..4, majority per item | 54 africa |
| **sampled (per-seed mean)** | same draws, mean over the 270 samples | 54 africa |

| Metric | Rule | Notes |
|---|---|---|
| **substring** | any answer (incl. alternatives) is a substring of the full generation | permissive — misses hedges |
| **strict (first sentence)** | any answer is a substring of the text up to the first `.` or newline | catches commit/hedge; format-independent |

> Conflict resolved: `12.96%->5.56%` = **greedy** row for `k128_wrong` under the **substring**
> metric. `0.093` = **sampled-majority** row for the same mask. Same mask, two protocols —
> both correct once labelled.

## Headline — Africa (54), greedy

| Intervention | substring hall | strict hall | Strict flips | Evidence |
|---|---|---|---|---|
| baseline | 0.130 (7/54) | 0.130 (7/54) | — | `eval_africa_baseline.json` |
| static `k32_midwrong` | 0.093 (5/54) | 0.093 (5/54) | fixes Eswatini, Gambia, Senegal; **breaks South Sudan (Juba->Bor)** | `eval_africa_k32_midwrong.json` |
| static `k128_wrong` | 0.056 (3/54) | 0.074 (4/54) | fixes 5 (incl. Eq.Guinea, Gabon, Guinea); **breaks Benin, South Africa, South Sudan** | `eval_africa_k128_wrong.json` |
| **runtime w5 soft** (early t90, window<=5, scale 0.3) | 0.074 (4/54) | **0.093 (5/54)** | fixes Eswatini, Gambia (Senegal stays wrong: hedge); **0 breaks** | `eval_runtime_africa_jump_gt_L19_t90_mask_sft0.3.json` |
| runtime w5 hard | 0.074 | 0.093 | same | `..._mask.json` |
| runtime w4 hard | 0.093 | 0.093 | fixes Eswatini, Gambia | `..._mask_w4.json` |
| runtime w3 hard | 0.111 | 0.111 | fixes Eswatini only | `..._mask_w3.json` |

## Headline — Africa (54), sampled (temp 0.9, 5 seeds)

| | baseline | runtime w5 soft |
|---|---|---|
| per-seed mean hall (270 samples) | 0.122 (33/270) | **0.104 (28/270)** — net -5; McNemar p=0.18, CI [-0.041, 0.000] |
| majority-of-5 hall (54 items) | 0.111 (6/54) | **0.074 (4/54)** — fixes Eswatini, Senegal; 0 breaks; p=0.50 |
| individual flips | — | 7 W2C, 2 C2W across seeds (Burundi break occurs only under hard mask; soft avoids it) |

Static reference (majority, substring): baseline 0.148 -> `k32_midwrong` 0.111 (p=0.017), `k128_wrong` 0.093 (p=0.003).

## Transfer / new topics (greedy, strict first-sentence metric)

| Topic | baseline | runtime w5 soft | static k32 | static k128 |
|---|---|---|---|---|
| `africa_largest` (n=54) | 0.296 (16/54) | **0.278 (15/54)** — 1 fix, 0 breaks | 0.296 (1 fix) | 0.315 (damage) |
| `world_tricky` (n=49) | 0.020 (1/49) | 0.020 (0 fires, 0 damage) | 0.041 (1 break) | 0.122 (3 breaks) |
| `europe` (n=44) | 0.000 | 0.000 (0 fires) | 0.000 | **0.091 (4 breaks)** |

## Detectors (Africa, greedy, leave-one-out AUC)

| Detector | AUC (LOSO) | Notes |
|---|---|---|
| jump_max_L18 (full generation) | 0.860 | fires post-commit -> inert for intervention |
| jump_max_early_L19 (first 10 tokens) | 0.742 | fires pre-commit on Eswatini/Gambia/Senegal -> effective |
| probe L10 (logistic) | 0.672 | weakest; does NOT transfer across decode protocols |

Window/threshold tradeoff (offline-validated vs live runs 1:1):
- window<=5, p90: fires 7/54 (3 fixable + 4 FP), 3 catches
- window<=4, p90: fires 6/54 (2 catches)
- window<=3, p90: fires 2/54 (1 catch), safest
- window<=2: nothing fires (floor is 3)

## Confirmed ceiling

4/7 greedy hallucinations (Cape Verde, Equatorial Guinea, Gabon, Guinea) commit
"quietly" with no pre-commit jitter spike and are not fixable by any threshold of this
detector family nor by the k32_midwrong mask. No threshold/mask in the current family
exceeds this ceiling (the runtime never fires on them even at p80).

## Honest caveats

- The permissive substring matcher inflated improvements by up to 2 points (Senegal hedge
  counts as a "fix" in substring but is wrong under the strict first-sentence metric).
  All headline numbers are reported under both metrics; use strict.
- Runtime soft intervention is consistently direction-improving and never breaks a control,
  but its improvement is NOT statistically significant at n=270 (p=0.18, CI includes 0).
  Static excision carries significance (p=0.017/0.003) at the cost of documented breaks.
- Single model (Gemma 3 1B, fp16), CPU-only. Sampled runs use a manual sampler; draws are
  not bit-identical to `model.generate`, so runtime sampled rows and static `_s5` rows are
  independent samples (both valid; baselines differ by sampling noise: 0.111 vs 0.148).
- Detector is protocol-bound: does not transfer from sampled to greedy (probe L10 ~0.05).

## Model provenance (verified)

- `models/gemma3-1b-fp16/model.safetensors`: single consolidated fp16 checkpoint,
  **999.9M params, dtype=torch.float16**, config `Gemma3ForCausalLM` (26 layers, d=1152).
  This is the Gemma 3 1B text model; obtain the original from
  https://huggingface.co/google/gemma-3-1b-it (gated) and cast to fp16.
- `models/gemma3-1b-tokenizer`: Gemma 3 SentencePiece tokenizer (vocab 32768).