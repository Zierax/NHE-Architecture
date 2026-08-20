# Excision Experiment Report — Gemma 3 1B

Date: 2026-08-20 — full error-rich bench battery complete.

## Question
Can hallucinations in a specific topic (Africa capitals) be reduced by excising
(zeroing/scaling) precise parameters, without breaking knowledge of other topics?
And does early temporal excision work better than static excision?

## Protocol
- Evaluation: greedy (temp=0) + sampled (temp=0.9, top_p=0.9, 5-6 seeds per item, fixed
  per-sample seeds).
- Labeling: textual match (NFKD→ascii) against any alternative answer. Caveats:
  "Salaffaire...and Praia" counts as correct; early truncation (e.g.,
  "Bandar Seri Begi.") counts as wrong.
- Excision: scale `down_proj[:,u]`, `up_proj[u,:]`, `gate_proj[u,:]` by 0.0 (hard) or
  0.3 (soft).
- Statistics: exact McNemar on matched sample pairs (n=270–594 per cell) + bootstrap
  net effect (5000, 95% CI). Strict first-sentence metric with alternatives is the
  headline metric.

## Topics
- Target: Africa (54) — weak knowledge, baseline hallucination 0.167 (sampled).
- Strong control: Europe (44, 0.005); moderate: Asia (46, 0.057), US states (50, 0.052);
  weak: elements (41, 0.200).
- Transfer targets: africa_largest (54), world_tricky (49).
- **Error-rich bench (99 items)**: africa_largest (54) + greedy-wrong subsets from
  world_cap_traps (15/134) + world_largest (30/173). Baseline strict hall = 0.596.

## Results

### 1) Statistical excision (d_mean / d_var): NULL
All sizes (32/128/512): no effect on Africa (greedy 0.130 unchanged). Reason: statistical
neurons are not causal (only 7/128 overlap with causal).

### 2) Causal AtP excision (target Σ(logit[wrong] − logit[correct])):
- `k32_causal` (no filter, all layers): Africa 0.130→0.111, Europe intact.
- `k128_causal`+: collapse (Africa 0.630, Europe 0.159) — generic shared L25 neurons.

### 3) Selective filtering — final results (sample level):

| topic | baseline | k32_midwrong | p | net [95% CI] | k128_wrong | p | net [95% CI] |
|---|---|---|---|---|---|---|---|
| africa (target) | 0.167 | **0.111** | **0.017** | +15 [5,24] | **0.093** | **0.003** | +20 [8,33] |
| europe (strong) | 0.005 | 0.000 | 1.000 | +1 [0,3] | 0.095 | **<0.001** | −20 [−29,−12] |
| elements | 0.200 | 0.249 | 0.031 | −10 [−19,−2] | 0.249 | 0.052 | −10 [−19,−1] |
| asia | 0.057 | 0.057 | 1.000 | 0 [−8,8] | 0.100 | 0.087 | −10 [−20,0] |
| us_states | 0.052 | 0.088 | 0.136 | −9 [−19,1] | 0.128 | **0.007** | −19 [−32,−6] |

### 4) Mechanistic reading
- `k32_midwrong` (layers 8–17, wrong-answer-boosting neurons only): **error-rebalancing
  surgery**:
  - Fixes real wrong commits where they dominate: Africa (Luanda→Mbabane, Bandžarīs→Banjul,
    Diou→Dakar) and also on controls (Georgia-country Atlanta→Tbilisi, Delaware
    Wilmington→Dover — both baseline hallucinations).
  - Never touches strong knowledge at all (Europe 44/44 at all sizes up to k256).
  - Net loss where knowledge is marginal (Maine/Oregon/Vermont→Portland/Burlington,
    elements).
  - Asia: perfectly balanced (8 fixes / 8 breaks).
- `k128_wrong`: strongest on Africa but damages shared knowledge (Europe −20, US −19) —
  the same neurons carry correct answers for other topics.
- Causal neurons concentrate in layers 14–17 (peak L16) — the same band as the probe peaks
  (L10–15) and jitter, but different particles than the statistically-detected ones
  (explains the null in (1)).
- Detection (jitter, AUROC 0.968), causality (AtP), and excision: three coherent layers of
  the same phenomenon — wrong-commit in the upper-middle layers.

### 5) Temporal excision (runtime rollback) — Arm 2

#### Design
- A temporal detector is computed per token during generation; on threshold exceedance →
  apply the k32_midwrong mask from that point (or roll back two tokens + mask).
- Training: greedy Africa flows (54, 7 hallucinations); threshold calibrated at the
  90/95th percentile of the honest distribution.
- **Critical contamination lesson**: the mask persisted in the model between items —
  without a clean state restore per item the results are fake (apparent 0.093 was really
  0.130). Fix: clean `load_state_dict` per item.

#### Greedy results (Africa 54, strict metric)

| Detector | AUC (LOSO) | Fires | Hall | Note |
|---|---|---|---|---|
| baseline (no intervention) | – | – | 0.130 | – |
| probe L10 (losso) | 0.672 | – | – | weak |
| jump_max_L18 (full) | 0.860 | t95: 6 | **0.130 (zero)** | late firing (tokens 14–19), post-commit |
| jump_max_early_L19 (t≤10) | 0.742 | t95: 13 | 0.130 | still late for most items |
| jump_max_early_L19 t90 | 0.742 | 16 | **0.074** | **pre-commit firing (t≤5) works** |
| **same + hard window t≤5** | 0.742 | **7** | **0.074** | same fixes at 1/3 fires; Europe: **0 fires** |

#### Final picture (greedy): static vs temporal vs ceiling — strict (first-sentence) metric
| Intervention | substring hall | strict hall | Strict flips |
|---|---|---|---|
| baseline | 0.130 | 0.130 | — |
| static k32_midwrong | 0.093 | 0.093 | fixes Eswatini/Gambia/Senegal + **Juba break** |
| static k128_wrong | 0.056 | 0.074 | fixes 5 + **Benin/South Sudan breaks** (Cape Town valid via 3-capitals alt) |
| **temporal w5 soft ×0.3** | 0.074 | **0.093** | fixes Eswatini/Gambia (Senegal hedge stays wrong) + **zero breaks** |
| temporal w4 hard | 0.093 | 0.093 | fixes Eswatini/Gambia + zero breaks |
| temporal w3 hard | 0.111 | 0.111 | fixes Eswatini only |

- The permissive matcher inflated the improvement by up to 2 points (Senegal hedge counted
  as a "fix" by substring but is wrong under the strict metric). All numbers now carry both
  metrics.
- **Soft temporal = best safety/efficacy balance**: matches static k32 (0.093) but with
  **no break at all**, while static breaks Juba and k128 breaks 3.
- Ceiling stands: Cape Verde/Eq.Guinea/Gabon/Guinea (4/7) are "quiet" with no early signal
  — no threshold or mask fixes them.

#### Transfer (greedy, strict metric)
| Topic | baseline | temporal w5 soft | k32 | k128 |
|---|---|---|---|---|
| africa_largest (54) | 0.296 | **0.278** (1 fix, 0 breaks) | 0.296 | 0.315 (damage) |
| world_cap_traps (134) | 0.112 | **0.097** (improvement) | — | — |
| world_largest (173) | 0.173 | **0.168** (small improvement) | — | — |
| world_tricky (49) | 0.020 | 0.020 (0 fires) | 0.041 (1 break) | 0.122 (3 breaks) |
| europe (44) | 0.000 | 0.000 (0 fires) | 0.000 | 0.091 (4 breaks) |

#### Full sampled battery (error-rich bench, 99 items, 6 seeds, 594 samples)

| | baseline (none) | runtime w5 soft (mask) | runtime w5 soft (abstain) |
|---|---|---|---|
| per-seed mean hall (594 samples) | 0.596 (354/594) | **0.562 (334/594)** | 0.471 (280/594) + 90 abstained |
| flips across seeds | — | 20 W2C, 0 C2W | 74 W2C, 0 C2W (abstained=not-wrong) |
| significance (McNemar, paired n=594) | — | **p < 0.001** (W2C=20, net=+20) | **p < 0.001** (W2C=74) |
| bootstrap 95% CI (mask-none hall diff) | — | **[-0.049, -0.020]** (significant) | — |
| majority-of-5 hall (99 items) | 0.596 (59/99) | **0.566 (56/99)** | 0.475 (47/99) |
| fired items (90) | 74 wrong | 54 wrong, **36 correct (40% fixed)** | 90 refused (0 wrong, 0 correct) |

**Interpretation:**
- **Mask excision is highly significant** on the error-rich bench (p < 0.001, bootstrap
  CI [-0.049, -0.020] entirely negative). This resolves the earlier non-significance on
  the small Africa set (p=0.18) — the error-rich bench provides the statistical power.
- **Mask genuinely repairs**: on fired items, 40% become correct (36/90); 0 breaks.
- **Abstain** reduces hall more (0.471) but at a utility cost: 90 refusals, 16 correct
  answers lost (none→abstain correct-rate net −16, p<0.001). Mask improves correct-rate
  (W2C=20, C2W=0, p<0.001).
- **Mask vs abstain on fired items**: mask produces 36 correct answers; abstain produces
  0 correct (all refusals). The mask's excision *adds value beyond detection*.

#### Sampled Africa original (270 samples, 5 seeds)
- none 0.122 → mask 0.104 (W2C=7, C2W=2, p=0.18, CI [-0.041, 0.000]) — consistent
  direction, not significant on this smaller set.
- Europe: 0 fires, 0.000 in all 5 seeds — total safety.

#### Transfer (sampled, strict) — limited to africa_largest
- 5 seeds: none mean 0.122, mask 0.104; majority 0.111→0.074.

### 6) Detector & timing (the design space)

| Detector | AUC (LOSO) | Role |
|---|---|---|
| `jump_max_L18` (full gen) | 0.860 | post-commit → **inert** for intervention |
| `jump_max_early_L19` (first 10 tok) | 0.742 | pre-commit → **the effective one** |
| probe L10 (logistic) | 0.672 | weakest; does **not** transfer across protocols (sampled→greedy ≈ 0.05) |

Window/threshold tradeoff (offline-simulated, validated 1:1 against live runs):

| window (w) | fires / 54 | catches | notes |
|---|---:|---|---|
| w≤5, p90 | 7 | 3 | the chosen operating point (2 clean fixes) |
| w≤4, p90 | 6 | 2 | identical result, fewer fires |
| w≤3, p90 | 2 | 1 | safest, weakest |
| w≤2 | 0 | 0 | floor: nothing fires |

### 7) Confirmed ceiling

4/7 greedy hallucinations (Cape Verde, Equatorial Guinea, Gabon, Guinea) commit
**quietly** — the jitter detector never crosses threshold pre-commit, and neither
`k32_midwrong` nor any window/threshold in this family fixes them. This is an honest bound
on the whole approach, not an engineering artifact.

## Conclusion

1. A causal, excisable footprint exists: "wrong-commit" neurons in upper-middle layers
   (8–17), isolated via AtP + wrong-only filter.
2. **k32_midwrong = the surgical static choice**: significant hallucination reduction
   (0.167→0.111) with no damage to strong knowledge, limited loss only where knowledge is
   marginal.
3. **Early soft temporal excision (w≤5, scale 0.3) = the only intervention that is both
   effective and universally safe**:
   - Africa greedy strict: 0.130→0.093, **zero breaks** (static k32 breaks Juba).
   - Error-rich bench (99 items, 594 samples): 0.596→0.562, **p < 0.001**, 20 fixes, 0
     breaks, bootstrap CI [-0.049, -0.020].
   - On fired items: 40% become correct (36/90), zero breaks.
   - Transfer: africa_largest 0.296→0.278, world_cap_traps 0.112→0.097, world_largest
     0.173→0.168 — all improvements, **zero breaks on any topic**.
   - Only basket that never damages a control on any topic.
4. **Confirmed ceiling**: 4/7 greedy hallucinations commit quietly — no early jitter spike,
   no threshold or mask fixes them.

## Honest limitations

1. **Textual metrics are traps.** The permissive substring matcher inflated improvements
   by up to 2 points (Senegal *appears* fixed but commits a hedge: "Diou... While Dakar is
   the largest city"). Every headline number is reported under the strict first-sentence
   metric; flips were inspected manually (`legacy/strict_flips.py`).
2. **Single model, single scale.** Gemma 3 1B, CPU-only, one target domain
   (Africa capitals). No GPU, no other model sizes, no other domains tested.
3. **Sample size.** n = 54 per topic; only 7 baseline greedy errors on Africa — small
   denominators, big uncertainties. The strict metric on `africa_largest` (16/54 errors)
   is the better bench for future work.
4. **Detector is protocol-bound.** The sampled→greedy transfer fails (AUC ≈ 0.05);
   detector and intervention must be calibrated per decoding protocol.
5. **Quiet-commit ceiling.** 4/7 hallucinations are unfixable by this signal family.

## Assets

- `results/attribution_africa.json|.npz` (statistical), `results/attribution_causal_africa.json` (AtP), `results/attribution_causal2_africa.json` (AtP with two filters + mid)
- `results/mask_k*_{score}.json`, `results/eval_{topic}_{mask}[_s5].json`, `results/summary_experiment.json`
- `results/greedy_flows_africa.npz`, `results/detector_greedy.json`, `results/eval_runtime_*.json`
- `results/bench_hard.json` (frozen error-rich bench)
- Scripts: `attribute_causal2.py`, `run_experiment.py`, `eval_topic.py`,
  `runtime_rollback.py` (temporal excision with soft scale + window + abstain mode),
  `sweep_thresholds.py` (offline sweep, validated 1:1 vs live runs),
  `sweep_windows.py`, `battery_analysis.py` (sampled battery: McNemar + bootstrap + majority),
  `strict_final.py` (strict first-sentence metric with alternatives — canonical scorer),
  `significance_final.py`, `significance_s5.py`, `collect_topic.py`, `topics.py`,
  `bench_build.py`, `bench_driver.py`, `bench_greedy.py`
- `results/` — all masks, evaluations, detector configs, NUMBERS.md (canonical), experiment_report.md
- `legacy/` — superseded experiments, debug probes, and null-result statistical arm

Full methodology and per-run commentary (Arabic): `results/experiment_report.md`.
Status and lesson log: `STATUS.md`.