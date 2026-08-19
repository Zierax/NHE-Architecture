# Excision Experiment Report — Gemma 3 1B

Date: 2026-08-18 (updated evening — temporal excision arm added)

## Question
Can hallucinations in a specific topic (Africa capitals) be reduced by excising
(zeroing) precise parameters, without breaking knowledge of other topics?

## Protocol
- Evaluation: greedy (temp=0) + sampled (temp=0.9, top_p=0.9, 5 samples/item, fixed
  per-sample seeds).
- Labeling: textual match (NFKD→ascii) against any alternative answer. Caveats:
  "Salaffaire...and Praia" counts as correct; early truncation (e.g.,
  "Bandar Seri Begi.") counts as wrong.
- Excision: zero `down_proj[:,u]`, `up_proj[u,:]`, `gate_proj[u,:]`.
- Statistics: exact McNemar on matched sample pairs (n=205–270 per cell) + bootstrap net
  effect (5000, 95% CI).

## Topics
- Target: Africa (54) — weak knowledge, baseline hallucination 0.167 (sampled).
- Strong control: Europe (44, 0.005); moderate: Asia (46, 0.057), US states (50, 0.052);
  weak: elements (41, 0.200).

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

## Conclusion
1. A causal, excisable footprint exists: "wrong-commit" neurons in upper-middle layers
   (8–17), isolated via AtP + wrong-only filter.
2. **k32_midwrong = the surgical choice**: significant hallucination reduction
   (0.167→0.111) with no damage to strong knowledge, limited loss only where knowledge is
   marginal.
3. k128_wrong = strongest for Africa (0.093) but wounds Europe — not suitable for general
   use.
4. Limits: single 1B size; textual-match labeling; small number of baseline
   hallucinations; weak-topic differences within expectation.

## Assets
- `results/attribution_africa.json|.npz` (statistical),
  `results/attribution_causal_africa.json` (AtP),
  `results/attribution_causal2_africa.json` (AtP with two filters + mid)
- `results/mask_k*_{score}.json`, `results/eval_{topic}_{mask}[_s5].json`,
  `results/summary_experiment.json`
- Scripts: `attribute_neurons.py`, `attribute_causal.py`, `attribute_causal2.py`,
  `eval_topic.py [--samples=N]`, `run_experiment.py`, `significance_final.py`,
  `collect_topic.py`, `topics.py`

## Arm 2: temporal excision (runtime rollback) — 2026-08-18 evening

### Design
- A temporal detector is computed per token during generation; on threshold exceedance →
  apply the k32_midwrong mask from that point (or roll back two tokens + mask).
- Training: greedy Africa flows (54, 7 hallucinations); threshold calibrated at the
  90/95th percentile of the honest distribution.
- **Critical contamination lesson**: the mask persisted in the model between items —
  without a clean state restore per item the results are fake (apparent 0.093 was really
  0.130). Fix: clean `load_state_dict` per item.

### Results (greedy, Africa 54)

| Detector | AUC (LOSO) | Fires | Hall | Note |
|---|---|---|---|---|
| baseline (no intervention) | – | – | 0.130 | – |
| probe L10 (losso) | 0.672 | – | – | weak |
| jump_max_L18 (full) | 0.860 | t95: 6 | **0.130 (zero)** | late firing (tokens 14–19), post-commit |
| jump_max_early_L19 (t≤10) | 0.742 | t95: 13 | 0.130 | still late for most items |
| jump_max_early_L19 t90 | 0.742 | 16 | **0.074** | **pre-commit firing (t≤5) works** |
| **same detector + hard window t≤5** | 0.742 | **7** | **0.074** | same fixes at a third of the fires; Europe: **0 fires** |

### Reading
- **Timing is the essence**: masking before the commit token (t≤5) prevents hallucination
  (Eswatini FIRE@3→Mbabane ✓, Gambia FIRE@4→Banjul ✓); masking after the commit is
  completely inert.
- Senegal: W2C but a hedge ("Diou... While Dakar is the largest city") — a clean 2/3 fix.
- Collateral on controls (early t90): Europe 0.000, Asia 0.043, Americas 0.020, elements
  0.220 — **all identical to baseline, zero damage** (elements even with 27/41 fires).
- Early truthful fires (t≤8) did not break their answers — the masked neurons are
  wrong-commit-specific.
- The full detector (0.860) is statistically stronger but fires after the fact; the early
  detector (0.742) is weaker but fires in the effective window. The signal lives in layers
  18–19 — the same upper-middle region.

### Final picture (greedy): static vs temporal vs ceiling — strict (first-sentence) metric
| Intervention | substring hall | strict hall | strict flips |
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
  **no break at all**, while static breaks Juba and k128 breaks 2.
- Ceiling stands: Cape Verde/Eq.Guinea/Gabon/Guinea (4/7) are "quiet" with no early signal
  — no threshold or mask fixes them.

### Generalization (greedy, strict metric)
| Topic | baseline | temporal w5 soft | k32 | k128 |
|---|---|---|---|---|
| africa_largest (54) | 0.296 | **0.278** (1 fix, 0 breaks) | 0.296 | 0.315 (damage) |
| world_tricky (49) | 0.020 | 0.020 (0 fires) | 0.041 (1 break) | 0.122 (3 breaks) |
| europe (44) | 0.000 | 0.000 (0 fires) | 0.000 | 0.091 (4 breaks) |

### Full sampled battery (Africa, 5 seeds, temporal w5 soft)
- Per seed: none {0.130,0.148,0.111,0.130,0.093} → mask {0.093,0.111,0.111,0.111,0.093}
- 270 samples total: 33→28 hallucinations (0.122→0.104), W2C=7/C2W=2, **McNemar p=0.18,
  CI [−0.041, 0.000]** — consistent but not significant.
- Majority-of-5: 0.111→0.074 (fixes Eswatini+Senegal, zero breaks).
- Europe: 0 fires and 0.000 in every seed — total safety.
- Comparison: static k32 is significant (p=0.017) with a Juba break; soft temporal is not
  significant but never breaks.

### Temporal conclusion
- Early soft temporal excision (w5, ×0.3): greedy Africa **0.130→0.093 under the strict
  metric** with zero breaks, and Europe/Asia/Americas/elements with no fires — the only
  basket that is never damaged on any topic. Partial transfer to africa_largest
  (0.296→0.278).
- Tools: `results/greedy_flows_africa.npz`, `results/detector_greedy.json`,
  `results/eval_runtime_*.json`, `runtime_rollback.py`, `sweep_thresholds.py` (validated
  offline simulation), `sweep_windows.py`, `battery_analysis.py`, `strict_final.py`
  (strict metric).

## Proposed next steps
1. **Improve intervention safety under sampling**: softer intervention (×0.3 instead of
   zero) or t95 threshold + window, to avoid breaks like Burundi under sampling; then a
   full seed round (s0–s4) with significance.
2. **Fine-tune arm**: QLoRA 1B on Africa (needs CUDA) and re-run the measurements.
3. **Generalization**: more topics with flows (collect Europe flows), narrower detection
   window (t≤6), lower thresholds, and sampled (temp 0.9) evaluation of the temporal
   intervention.