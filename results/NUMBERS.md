# Canonical, protocol-labelled numbers — NHE-Architecture

Last updated: 2026-08-20 — full battery complete, bug-fixed bench.
**This file is the single source of truth for every number cited by this project.** Any
other document/presentation that quotes a result MUST read from this table and MUST label
both the **protocol** (greedy / sampled) and the **metric** (substring / strict
first-sentence). Comparing values from different protocols or metrics is invalid.

## Protocols and metrics

| Protocol | Decode | n |
|---|---|---|
| **greedy** | deterministic argmax | 54 africa / 44 europe / 54 africa_largest / 49 world_tricky / 134 cap_traps / 173 world_largest |
| **sampled (majority)** | temp=0.9, top_p=0.9, seeds 1000+item*100+s, s=0..4, majority per item | 54 africa / 99 bench |
| **sampled (per-seed mean)** | same draws, mean over the 270/594 samples | 54 africa / 99 bench |

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
| static `k128_wrong` | 0.056 (3/54) | 0.074 (4/54) | fixes 5 (incl. Eq.Guinea, Gabon, Guinea); **breaks Benin, South Sudan** (South Africa's Cape Town is valid via 3-capitals alternative) | `eval_africa_k128_wrong.json` |
| **runtime w5 soft** (early t90, window≤5, scale 0.3) | 0.074 (4/54) | **0.093 (5/54)** | fixes Eswatini, Gambia (Senegal stays wrong: hedge); **0 breaks** | `eval_runtime_africa_jump_gt_L19_t90_mask_sft0.3.json` |
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

## Error-rich bench — 99 items (africa_largest 54 + cap_traps 15 + world_largest 30), sampled (6 seeds, 594 samples)

The bench is constructed from greedy-wrong items on africa_largest (all 54) plus the greedy-wrong items from two new hard topics: `world_cap_traps` (15/134 strict-wrong) and `world_largest` (30/173 strict-wrong). Baseline hallucination rate on this bench is **0.596** (strict).

| | baseline (none) | runtime w5 soft (mask) | runtime w5 soft (abstain) |
|---|---|---|---|
| per-seed mean hall (594 samples) | 0.596 (354/594) | **0.562 (334/594)** | 0.471 (280/594) + 90 abstained |
| flips across seeds | — | 20 W2C, 0 C2W | 74 W2C, 0 C2W (abstained=not-wrong) |
| significance (McNemar, paired n=594) | — | **p < 0.001** (W2C=20, net=+20) | **p < 0.001** (W2C=74) |
| bootstrap 95% CI (mask-none hall diff) | — | **[-0.049, -0.020]** (significant) | — |
| majority-of-5 hall (99 items) | 0.596 (59/99) | **0.566 (56/99)** | 0.475 (47/99) |
| fired items (90) | 74 wrong | 54 wrong, **36 correct (40% fixed)** | 90 refused (0 wrong, 0 correct) |

## Transfer — new topics (greedy, strict metric)

| Topic | baseline | runtime w5 soft (mask) | static k32 | static k128 |
|---|---|---|---|---|
| `africa_largest` (54) | 0.296 | **0.278** (1 fix, 0 breaks) | 0.296 | 0.315 (damage) |
| `world_cap_traps` (134) | 0.112 | **0.097** (improvement) | — | — |
| `world_largest` (173) | 0.173 | **0.168** (small improvement) | — | — |
| `world_tricky` (49) | 0.020 | 0.020 (0 fires) | 0.041 (1 break) | 0.122 (3 breaks) |
| `europe` (44) | 0.000 | 0.000 (0 fires) | 0.000 | **0.091 (4 breaks)** |

## Detectors (Africa, greedy, leave-one-out AUC)

| Detector | AUC (LOSO) | Notes |
|---|---|---|
| jump_max_L18 (full generation) | 0.860 | fires post-commit -> inert for intervention |
| jump_max_early_L19 (first 10 tokens) | 0.742 | fires pre-commit on Eswatini/Gambia/Senegal -> effective |
| probe L10 (logistic) | 0.672 | weakest; does NOT transfer across decode protocols |

Window/threshold tradeoff (offline-validated vs live runs 1:1):
- window≤5, p90: fires 7/54 (3 fixable + 4 FP), 3 catches
- window≤4, p90: fires 6/54 (2 catches)
- window≤3, p90: fires 2/54 (1 catch), safest
- window≤2: nothing fires (floor is 3)

## Confirmed ceiling

4/7 greedy hallucinations (Cape Verde, Equatorial Guinea, Gabon, Guinea) commit
"quietly" with no pre-commit jitter spike and are not fixable by any threshold of this
detector family nor by the k32_midwrong mask. No threshold/mask in the current family
exceeds this ceiling (the runtime never fires on them even at p80).

## Honest caveats

- The permissive substring matcher inflated improvements by up to 2 points (Senegal hedge
  counts as a "fix" in substring but is wrong under the strict first-sentence metric).
  All headline numbers are reported under both metrics; use strict.
- The **runtime soft excision is statistically significant on the error-rich bench** (p < 0.001, bootstrap CI entirely negative). On the original Africa set it was direction-improving but not significant (p=0.18) — the error-rich bench provides the power needed.
- Static excision carries significance at the cost of documented breaks (k32 breaks South Sudan; k128 breaks 3+).
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