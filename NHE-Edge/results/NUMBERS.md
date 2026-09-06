# Canonical, protocol-labelled numbers — NHE-Architecture

Last updated: 2026-09-04 — corrected after independent review (see caveats).
**This file is the single source of truth for every number cited by this project.** Any
other document/presentation that quotes a result MUST read from this table and MUST label
both the **protocol** (greedy / sampled) and the **metric** (substring / strict
first-sentence). Comparing values from different protocols or metrics is invalid.
**Primary inference is item-level (majority vote); per-draw McNemar/CI are reported
but overstate power (6 correlated draws per item) — see caveats.**

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
> metric. `0.093` = **sampled per-draw mean** for the same mask (majority = 0.074).
> Same mask, two protocols — both correct once labelled.

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

## Headline — Africa (54), sampled (temp 0.9, 5 seeds, substring metric via file `correct`)

| | baseline | runtime w5 soft |
|---|---|---|
| per-seed mean hall (270 samples) | 0.122 (33/270) | **0.104 (28/270)** — net -5; McNemar p=0.18, CI [-0.041, 0.000] |
| majority-of-5 hall (54 items) | 0.111 (6/54) | **0.074 (4/54)** — fixes Eswatini, Senegal; 0 breaks; p=0.50 |
| individual flips | — | 7 W2C, 2 C2W across seeds (Burundi break occurs only under hard mask; soft avoids it) |

Static reference (sampled, substring): baseline majority 0.148 (8/54) -> `k32_midwrong` majority 0.111 (6/54, p=0.017), `k128_wrong` majority 0.074 (4/54, p=0.003); per-draw means 0.111/0.093. (`eval_africa_*_s5.json`: `hallucination_rate` = majority, `hallucination_rate_sampled` = per-draw.)

## Error-rich bench — 99 items, sampled strict metric (6 seeds, 594 draws)

Construction (frozen `bench_hard.json`, built by `bench_build.py`): **all 54**
`africa_largest` (of which 16/54 strict-wrong at greedy baseline) + the 15
strict-wrong `world_cap_traps` + the 30 strict-wrong `world_largest`. Baseline
hallucination rate on this bench is **0.596** (strict).

| | baseline (none) | runtime w5 soft (mask) | runtime w5 soft (abstain) |
|---|---|---|---|
| per-draw mean hall (594 draws) | 0.596 (354/594) | **0.562 (334/594)** | 0.471 (280/594) + 90 abstained |
| flips across draws | — | 20 W2C, 0 C2W | 74 W2C, 0 C2W (abstained=not-wrong) |
| per-draw McNemar (n=594, exploratory) | — | p < 0.001 | p < 0.001 |
| per-draw bootstrap 95% CI (mask-none) | — | [-0.049, -0.020] | — |
| **majority-of-6 hall (99 items, PRIMARY)** | 0.596 (59/99) | **0.566 (56/99)** — W2C=3, C2W=0, **p=0.25 (not significant)** | 0.475 (47/99) |
| cluster bootstrap 95% CI by item (mask-none) | — | **[-0.066, -0.007] (excludes 0; uses per-item magnitudes, more powerful than majority vote)** | — |
| fired samples (90/594) | 74 wrong | 54 wrong, 36 correct post-hoc (**20/90 = 22% true wrong→correct fixes**; 27% of wrong-fired) | 90 refused (0 wrong, 0 correct) |

Reading: per-draw tests overstate power (6 correlated draws per item). The honest
primary — item majority — is direction-consistent but not significant (3 vs 0).
The effect is real in sign, small in size.

## Random bench — 99 items (random sample, seed 42), sampled strict (6 seeds, 594 draws)

Random sample from same pool (africa_largest 11 + cap_traps 40 + world_largest 48, overlap 19 with hard bench). Baseline **0.099** (vs hard 0.596).

| | baseline (none) | runtime w5 soft (mask) | runtime w5 soft (abstain) |
|---|---|---|---|
| per-draw mean hall (594) | 0.099 (59/594) | **0.089 (53/594)** | 0.079 (47/594) + 42 abstained |
| per-draw McNemar (exploratory) | — | p=0.031 (W2C=6, C2W=0) | p<0.001 |
| per-draw bootstrap 95% CI | — | [-0.0185, -0.0034] | — |
| **majority-of-6 (99 items, PRIMARY)** | 0.101 (10/99) | 0.091 (9/99) — 1 discordant, n.s. | 0.081 (8/99) |

Same direction as hard bench at unbiased baseline; absolute effect 6/594 (1pp).
Does not survive multiple-comparison correction — reported as suggestive, not conclusive.

## Merged static+temporal — hard bench (99 items, sampled strict, 6 seeds, 594 draws)

Static `k32_midwrong` hard (scale 0.0) always on + temporal soft/abstain on fire. Fires 348/594 (58%) vs temporal alone 90/594 (15%) — static changes dynamics, detector no longer selective.

| | baseline (none) | temporal mask | temporal abstain | **merged mask** (static+temporal mask) | **merged abstain** (static+abstain) |
|---|---|---|---|---|---|
| per-draw mean hall (594) | 0.596 (354/594) | 0.562 (334/594) | 0.471 | **0.389 (231/594)** | **0.088 (52/594)** + 348 abstained |
| per-draw flips vs none | — | 20/0 | 74/0 | **132 W2C / 9 C2W (9 breaks disclosed)** | **305 W2C / 3 C2W** |
| vs temporal mask | — | — | — | 112 W2C / 9 C2W | — |

Merged mask repairs substantially but breaks 9 draws (not "without refusal" — without *abstention*, with 9 breaks). Merged abstain nearly eliminates hallucination at cost of 348 abstentions (58% refusal); abstain-coded-as-correct mechanically suppresses the rate. No per-item proof yet that the 4 Africa quiet cases are among the fixed — "breaks the ceiling" is bench-aggregate only.

## Transfer — new topics (greedy, strict metric)

| Topic | baseline | runtime w5 soft (mask) | static k32 | static k128 |
|---|---|---|---|---|
| `africa_largest` (54) | 0.296 (16/54) | **0.278 (15/54: 1 fix Eswatini, 0 breaks)** | **0.185 (10/54: 7 fixes, breaks South Sudan)** | **0.259 (14/54: 7 fixes, breaks Ghana/Libya/S.Africa/S.Sudan/Zimbabwe)** |
| `world_cap_traps` (134) | 0.112 | **0.097** (improvement) | — | — |
| `world_largest` (173) | 0.173 | **0.168** (small improvement) | — | — |
| `world_tricky` (49) | 0.020 (1/49) | 0.020 (0 fires) | 0.041 (2/49: fixes 0, breaks 1) | 0.122 (6/49: fixes UAE, **breaks Benin, Kazakhstan, Lithuania, Myanmar, Romania, Vietnam**) |
| `europe` (44) | 0.000 | 0.000 (0 fires) | 0.000 | **0.091 (4 breaks)** |

## Detectors (Africa, greedy, leave-one-out AUC)

| Detector | AUC (LOSO) | Notes |
|---|---|---|
| jitter last-token L10 (any token) | 0.968 | `jitter_report_africa.json:17` (`per_layer_auroc_last_token[10]`); pooled over tokens, not window-constrained — shows signal exists, NOT used for intervention. Note: that file labels 44 truthful/10 hallucinated vs headline 47/7 (different labeling run) |
| jump_max_L18 (full generation) | 0.860 | `detector_greedy.json:6` fires post-commit -> inert for intervention |
| jump_max_early_L19 (first 10 tokens) | 0.742 | `detector_greedy.json:18` fires pre-commit on Eswatini/Gambia/Senegal -> effective |
| probe L10 (logistic) | 0.672 | `detector_greedy.json:27` weakest; sampled→greedy transfer ~0.05 is a probe-L10-only estimate (no transfer file tracked) |

Window/threshold tradeoff (offline-validated vs live runs 1:1):
- window≤5, p90: fires 7/54 (3 fixable + 4 FP), 3 catches
- window≤4, p90: fires 6/54 (2 catches)
- window≤3, p90: fires 2/54 (1 catch), safest
- window≤2: nothing fires (floor is 3)

## Prompt baseline — same benches, greedy strict (`eval_prompt_baseline.py`)

Constrained prompt appends "Answer with only the city name." Same items, same greedy decode.

| Bench | open prompt | constrained prompt |
|---|---|---|
| hard (99) | 0.616 (61/99; matches bench construction) | **0.525 (52/99)** — 9 fixes, free |
| random (99) | 0.091 (9/99) | 0.111 (11/99) — 2 worse (noise) |

Reading: prompting is a real competitor on the hard bench (greedy 61→52 beats
NHE-mask greedy 61→57). NHE's distinct value is sampled-significance-tested repair
(20/594, cluster CI excludes 0), no-refusal fixes, and transfer without prompt
engineering — not raw greedy rate. (`prompt_baseline_{hard,random}_prompt_{open,constrained}.json`)

## Cross-architecture — Qwen2.5-0.5B, real weights (greedy Africa 54)

Same pipeline via `runtime_rollback_qwen.py` + `attribute_causal2_qwen.py`
(24 layers, hidden 896, ChatML prompts, fp16). Full report: `cross_arch_report.md` §7.

| | Gemma 3 1B | Qwen2.5-0.5B |
|---|---|---|
| greedy baseline hall | 0.130 (7/54) | 0.167 (9/54) |
| full detector | jump L18, AUC 0.860 | jump L20, AUC 0.783 |
| early detector | jump L19, AUC 0.742 | jump L22, AUC 0.778 |
| own wrong-commit mask | k32 L10–17 | k32 L13–20 (same relative band) |
| live fires (w≤5) | 7/54 | **0/54** (spikes at live t 7–9) |
| live fires (w≤10) | — | 9/54 (= offline prediction, 1:1) |
| intervention flips | 2 fixes / 0 breaks (strict) | **0 flips / 0 breaks** (post-commit: city at token ~6, spike at 7–9) |
| pre-commit AUC (t≤5) | catches 3/7 | **0.353** (no pre-commit signal) |

Reading: signal family generalizes; timing doesn't. NHE-temporal needs spike-before-commit (a format property — Gemma's bold markers delay the city); on Qwen-format it is provably inert and harmless. Qwen1.5B remains synthetic-only.

## Side effects — general knowledge (greedy)

| Dataset | baseline | static k32 soft (×0.3) |
|---|---|---|
| real MMLU 200 (`cais/mmlu:all/test` streaming) | 0.925 substr (185/200), 0.610 strict (122/200) | **0.925 substr (185/200), 0.620 strict (124/200)** — preserved, +0.01 strict (`mmlu_side_effect.json:159-172`) |
| control proxy 181 (Europe+Asia+elements+US) | 0.934 substr, 0.917 strict | 0.917 / 0.901 (-0.017, <2%) — superseded by real MMLU row |

Note: only the *static* mask was tested on MMLU, not the temporal method.

## Temporal on MMLU — 200 real MMLU, greedy strict (`eval_mmlu_temporal.py`)

Same 200 streamed items, paired baseline (greedy, no mask) vs temporal (early L19 t90, w≤5, soft ×0.3 — the headline config). **Different 200 than the static-MMLU run** (streaming order not pinned — baselines differ: 0.390 here vs 0.610 there; paired comparison within this run is valid).

| | baseline (greedy) | temporal mask |
|---|---|---|
| strict hall (200) | 0.390 (78/200) | **0.395 (79/200)** — fired 155/200 (78%), W2C=0, C2W=1, p=1.0 |

Reading: the Africa-calibrated detector fires constantly on MMLU (out-of-distribution threshold) and the mask changes nothing. Temporal does **not** transfer to MMLU-style questions — honest negative. (`mmlu_temporal.json`)

## Confirmed ceiling

4/7 greedy hallucinations (Cape Verde, Equatorial Guinea, Gabon, Guinea) commit
"quietly" with no pre-commit jitter spike and are not fixable by any threshold of this
detector family nor by the k32_midwrong mask. No threshold/mask in the current family
exceeds this ceiling (the runtime never fires on them even at p80).
Logit-lens probe (`probe_quiet.py`, `quiet_diagnostic.json`): 0/4 classified
parametric, 4/4 dynamic — the "training-data error" hypothesis was refuted; quiet
cases look like dynamic errors with subtler jitter. No second signal (attention
entropy/drift) has been tested yet.

## Honest caveats

- The permissive substring matcher inflated improvements by up to 2 points (Senegal hedge
  counts as a "fix" in substring but is wrong under the strict first-sentence metric).
  All headline numbers are reported under both metrics; use strict.
- **Inference level matters more than the p-value.** Per-draw McNemar/CI treat 6
  correlated draws per item as independent and overstate power ~6x. The honest
  primary is item-majority: hard bench 59/99→56/99 (W2C=3, C2W=0, p=0.25, n.s.),
  random bench 10/99→9/99 (1 discordant, n.s.). Direction is consistent everywhere;
  conclusive significance is not claimed.
- **No multiple-comparison correction** is applied across the ~10 reported McNemars
  (masks × windows × arms × benches). Random-bench p=0.031 would not survive one.
  Treat it as suggestive.
- **Abstain "wins" are mechanical.** Coding refusal as not-wrong guarantees a lower
  rate; abstain arms must be read with refusal cost (90/594, 348/594) and correct-rate,
  not hallucination p-values.
- **Hard bench is enriched by construction** (all 54 africa_largest + greedy-wrong
  15 + 30), so 0.596 is not a population rate and gains are conditional, subject to
  regression to the mean. Random bench (0.099) is the unbiased estimate: 1pp absolute.
- Static excision carries significance at the cost of documented breaks (k32 breaks South Sudan; k128 breaks Benin/S.Sudan + 6 on world_tricky).
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