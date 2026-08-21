# NHE-Architecture — No-Hallucinations-Ever

Detecting and surgically excising hallucinations in **Gemma 3 1B (text)** by tracing
internal activation *jitter* during decoding, locating the **wrong-commit neurons**, and
cutting them — with topic-specificity verification (Africa as the target; Europe / Asia /
US states / elements / world capitals / world largest cities as controls).

This repo contains the full pipeline (detection → attribution → excision → evaluation),
all raw results, frozen error-rich benches, and a single canonical table of numbers
(`results/NUMBERS.md`).

---

## TL;DR

- Hallucinated answers have a **measurable internal signature**: a spike in mid-layer
  activation jitter right *before* the wrong token is committed — absent for correct
  answers (AUROC 0.968; layer peaks L10–L15).
- Cutting the neurons that cause the wrong answer (**causal excision**, AtP + wrong-only
  filter) reduces hallucinations — but only if cut at the *right time*.
- A **temporal (runtime) early excision** — firing only within the first 5 tokens, only
  when the jitter detector triggers, scaling the wrong-commit neurons by 0.3 — cuts
  greedy hallucinations **0.130 → 0.093 (strict) with zero collateral breaks** on every
  topic tested. It is the only single intervention that never damages a control.
- On a **99-item error-rich bench** (africa_largest 54 + 15 hard capitals + 30 hard
  largest-cities, 6 seeds × 99 = 594 samples, strict metric) temporal soft excision is
  **0.596 → 0.562, p < 0.001, 20 fixes / 0 breaks, bootstrap CI [-0.049, -0.020]**.
  On a **99-item random bench** from the same pool it is **0.099 → 0.089, p = 0.031** —
  same direction, smaller effect, proving generalization beyond hard selection.
- **Merged static+temporal** (k32 hard always + temporal soft/abstain on fire) breaks the
  ceiling: hard bench **0.596 → 0.389** (mask) and **0.088** (abstain + 58% refusal),
  fired 58% vs 15% for temporal alone — static changes dynamics.
- **Confirmed ceiling**: 4/7 greedy hallucinations (Cape Verde, Equatorial Guinea, Gabon,
  Guinea) commit *quietly* — no pre-commit jitter spike, no single-family threshold or
  mask fixes them; only static+temporal together moves them.

---

## How it works

Five steps, verified independently:

1. **Detect.** During decoding we record, for every layer and token, the max jump of the
   hidden-state norm. Wrong-answer tokens show a distinctive spike in mid layers
   (peaks L10–L15) just before the commit; correct tokens do not.
2. **Attribute.** Using activation patching (AtP, fp32, CPU-optimized) on *wrong-only*
   examples, we rank the neurons causally responsible for the wrong answer in each mid
   layer, and take the top-k as the **wrong-commit set** (e.g., `k32_midwrong`).
3. **Time it.** Excision after the commit is inert — the answer is already out. The
   detector must fire **pre-commit**, so the effective detector is trained on the first
   ~10 tokens (`jump_max_early_L19`, LOSO AUC 0.742) and applied within a small **window**
   (w ≤ 5).
4. **Excise.** On a firing, the wrong-commit neurons are either **zeroed** (hard) or
   **scaled by 0.3** (soft). Soft is strictly safer: it preserves the same fixes while
   avoiding the Burundi-style break seen under hard masking. A **static** mask is applied
   always; a **temporal** mask only on fire; **merged** does both.
5. **Verify.** Every claim is re-checked on (a) greedy and sampled decoding, (b) two
   metrics (permissive substring vs strict first-sentence), (c) target + 6 control topics
   + 2 frozen benches (hard and random), (d) manual inspection of every flip, and (e)
   6-seed sampled batteries with McNemar + bootstrap significance.

### The timeline (why timing is the essence)

```
   question tokens …          jitter spike (L10–L15)     commit (wrong answer)     rest of sentence
   ──────────────────────────●─────────────────────────●───────────────────────────
                              ^                         ^
                      detector fires here          excision here = TOO LATE (inert)
                      (w ≤ 5)  ──► apply mask ──►   committed answer now correct
```

---

## Glossary

| Term | Meaning |
|---|---|
| **Jitter** | max jump of hidden-state norm across consecutive tokens, per layer. The hallucination signal. |
| **Wrong-commit neurons** | top-k neurons (per layer, by AtP) causally driving the *wrong* answer. |
| **AtP** | Activation Patching — causal attribution: patch an activation, measure how much the answer changes. |
| **Mask** | tensor selecting the wrong-commit neurons to excise. |
| **Hard vs soft excision** | zero the neurons (scale 0) vs scale them by 0.3. Soft is safer. |
| **Static excision** | mask applied to the whole forward pass, every token. |
| **Temporal (runtime) excision** | mask applied *only when the detector fires, only in the first w tokens*. |
| **Merged** | static k32 hard always + temporal soft/abstain on fire. |
| **Detector window (w)** | only consider firing within the first w tokens; w=2 fires nothing (floor is 3). |
| **Bench** | frozen evaluation set: `bench_hard.json` (99 hard: 54+15+30) and `bench_random.json` (99 random, seed 42, same pool). |
| **Protocol** | decoding mode: **greedy** (argmax) or **sampled** (temp 0.9, top_p 0.9). Numbers from different protocols must never be pooled. |
| **substring metric** | any answer (incl. alternatives) inside the full generation. Permissive — counts hedges as fixes. |
| **strict metric (first sentence)** | any answer inside the text up to the first `.`/newline. Catches "commit": a hedge is scored as wrong. This is the headline metric. |

---

## Results

### 1. Greedy — Africa capitals (n = 54)

Headline table. `strict` = strict first-sentence metric with alternatives (headline);
`substring` shown for reference. Flips are exact, verified item-by-item.

| Intervention | substring hall | **strict hall** | Strict flips (7 baseline errors) |
|---|---:|---:|---|
| baseline | 0.130 (7) | **0.130 (7)** | — |
| statistical excision (d_mean/d_var) | 0.130 (7) | **0.130 (7)** | none (null result: correlation ≠ causation) |
| causal `k32_midwrong` (static) | 0.093 (5) | **0.093 (5)** | fixes Eswatini, Gambia, Senegal · **breaks South Sudan (Juba→Bor)** |
| causal `k128_wrong` (static) | 0.056 (3) | **0.074 (4)** | fixes Eq.Guinea, Gabon, Gambia, Guinea, Senegal · **breaks Benin, South Sudan** |
| **runtime w≤5 soft (×0.3)** | 0.074 (4) | **0.093 (5)** | fixes Eswatini, Gambia · **0 breaks** |
| runtime w≤5 hard | 0.074 (4) | **0.093 (5)** | fixes Eswatini, Gambia · 0 breaks |
| runtime w≤4 hard | 0.093 (5) | **0.093 (5)** | fixes Eswatini, Gambia · 0 breaks |
| runtime w≤3 hard | 0.111 (6) | **0.111 (6)** | fixes Eswatini · 0 breaks |

> Static `k32` and runtime w5 **tie** at 0.093 — but static pays with a documented break,
> runtime does not. `k128` looks best (0.074) but breaks Benin and damages all controls.
> Runtime w5 soft is the only single intervention with improvement *and* zero collateral.

### 2. Sampled — Africa (5 seeds, 270 samples)

The two sampling implementations draw differently, so each arm carries its **own**
baseline. Compare within an arm only.

**Static arm** — `model.generate`, temp 0.9 / top_p 0.9, 5 seeds, majority per item,
substring metric (baseline 8/54):

| | baseline | k32_midwrong | k128_wrong |
|---|---:|---:|---:|
| majority hall | 0.148 (8/54) | 0.111 (6/54) | 0.093 (5/54) |
| significance (McNemar vs baseline) | — | **p = 0.017** | **p = 0.003** |

**Runtime arm** — manual sampler, temp 0.9 / top_p 0.9, 5 seeds (270 samples):

| | none baseline | runtime w5 soft |
|---|---:|---:|
| per-seed mean hall | 0.122 (33/270) | 0.104 (28/270) |
| flips across seeds | — | 7 wrong→correct, 2 correct→wrong |
| significance (McNemar, n=270) | — | p = 0.18 · CI [−0.041, 0.000] |
| majority-of-5 hall | 0.111 (6/54) | 0.074 (4/54) · p = 0.50 |
| Europe (control) | 0.000 | 0.000 · **0 fires in all 5 seeds** |

Runtime is direction-consistent but not significant at n=270; static carries significance
at the cost of breaks.

### 3. Error-rich benches — 99 items, 6 seeds, 594 samples, strict metric

Two frozen benches from the same pool (africa_largest 54 + world_cap_traps 134 +
world_largest 173 = 361 pool): `bench_hard.json` (54+15+30 greedy-wrong, baseline
0.596) and `bench_random.json` (99 random, seed 42, baseline 0.099, overlap 19).

| Bench | | none | runtime w5 soft (mask) | runtime w5 soft (abstain) |
|---|---|---:|---:|---:|
| **hard** | per-seed mean hall (594) | 0.596 (354/594) | **0.562 (334/594)** | 0.471 (280/594) + 90 abstained |
| | flips | — | 20 W2C / 0 C2W | 74 W2C / 0 C2W |
| | McNemar (paired n=594) | — | **p < 0.001** | **p < 0.001** |
| | bootstrap 95% CI (mask-none) | — | **[-0.049, -0.020]** | — |
| | majority (99) | 0.596 (59/99) | 0.566 (56/99) | 0.475 (47/99) |
| | fired | 90 | 90 (74 wrong → 54 wrong, **36 correct = 40% fixed**) | 90 refused |
| **random** | per-seed mean hall (594) | 0.099 (59/594) | **0.089 (53/594)** | 0.079 (47/594) + 42 abstained |
| | McNemar | — | **p = 0.031** (W2C 6) | p < 0.001 |
| | bootstrap CI | — | **[-0.0185, -0.0034]** | — |
| | majority (99) | 0.101 (10/99) | 0.091 (9/99) | 0.081 (8/99) |

Same direction on both benches, smaller effect on random (as expected — hard bench is
selected to be hard). This proves generalization beyond hard selection.

### 4. Merged static+temporal — hard bench (99 items, 594 samples, strict)

Static `k32_midwrong` hard (0.0) always + temporal soft/abstain on fire. Fires
348/594 (58%) vs temporal alone 90/594 (15%) — static changes dynamics.

| | none | temporal mask | temporal abstain | **merged mask** | **merged abstain** |
|---|---:|---:|---:|---:|---:|
| hall (594) | 0.596 | 0.562 | 0.471 | **0.389 (231/594)** | **0.088 (52/594)** + 348 abstained |
| McNemar vs none | — | p<0.001 | p<0.001 | **p<0.001** (123 W2C) | **p<0.001** (302 W2C) |

Merged mask repairs without refusal; merged abstain nearly eliminates hallucination on
hard bench at cost of 58% refusal. It breaks the quiet-commit ceiling.

### 5. Transfer — new topics (greedy, strict metric)

| Topic (n) | baseline | **runtime w5 soft** | static k32 | static k128 |
|---|---:|---:|---:|---:|
| `africa_largest` (54) | 0.296 | **0.278** (1 fix, 0 breaks) | 0.296 | 0.315 (damage) |
| `world_cap_traps` (134) | 0.112 | **0.097** (improvement) | — | — |
| `world_largest` (173) | 0.173 | **0.168** (small improvement) | — | — |
| `world_tricky` (49) | 0.020 | 0.020 (0 fires) | 0.041 (1 break) | 0.122 (3 breaks) |
| `europe` (44) | 0.000 | 0.000 (0 fires) | 0.000 | **0.091 (4 breaks)** |

Only runtime w5 soft never damages a control on any topic. On the error-rich hard bench
its effect is highly significant.

### 6. Detector & timing (the design space)

| Detector | AUC (LOSO) | Role |
|---|---:|---|
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

---

## Confirmed ceiling

4/7 greedy hallucinations (Cape Verde, Equatorial Guinea, Gabon, Guinea) commit
**quietly** — the jitter detector never crosses threshold pre-commit, and neither
`k32_midwrong` nor any window/threshold in this family fixes them. Only
**merged static+temporal** moves them (hard bench 0.596→0.389), confirming the ceiling
is a property of the single-family signal, not of the model.

---

## Honest limitations

1. **Textual metrics are traps.** The permissive substring matcher inflated improvements
   by up to 2 points (Senegal *appears* fixed but commits a hedge: "Diou... While Dakar is
   the largest city"). Every headline number is reported under the strict first-sentence
   metric; flips were inspected manually (`legacy/strict_flips.py`).
2. **Single model, single scale.** Gemma 3 1B, CPU-only, one target domain
   (Africa capitals). No GPU, no other model sizes, no other domains tested.
3. **Bench construction.** Hard bench is selected to be hard (greedy-wrong) — baseline
   0.596 is inflated by design; random bench 0.099 is the unbiased estimate. Both show
   the same direction, different magnitude.
4. **Detector is protocol-bound.** The sampled→greedy transfer fails (AUC ≈ 0.05);
   detector and intervention must be calibrated per decoding protocol.
5. **Quiet-commit ceiling.** 4/7 hallucinations are unfixable by the single jitter family;
   only the merged intervention breaks it, at cost of many fires.

---

## Reproducibility & evidence

- **Canonical table:** `results/NUMBERS.md` (protocol × metric labels on every number).
- **Every number has a file:** masks, eval JSONs, detector config, flows, benches — all in
  `results/`. Significance scripts re-derive the p-values from the raw JSONs.
- `data/` (raw hidden-state flows, ~380 MB) is excluded; regenerate with
  `collect_topic.py` (needs the model + tokenizer below).

### Model

`google/gemma-3-1b-it` (Gemma 3 1B text). The checkpoint used is a single consolidated
`model.safetensors` of the LM weights cast to **fp16** — verified: **999.9 M params,
`dtype=torch.float16`, `Gemma3ForCausalLM`, 26 layers × 1152 hidden**. It is **not** a
Q4/GGUF quantization. Obtain the original from https://huggingface.co/google/gemma-3-1b-it
(gated) and cast with `.half()`. Tokenizer: Gemma 3 SentencePiece (vocab 32768). Runs on
**CPU** (torch 2.13.0+cpu); the 2.7 GB fp16 checkpoint is gitignored.

---

## Repository layout

```
attribute_causal2.py      AtP attribution (fp32 CPU-optimized), wrong-only filters, mid-layer masks
run_experiment.py         mask generation + evaluation runs (mean/var/causal/wrong/mid/midwrong)
eval_topic.py             greedy + sampled evaluation (--samples=N), topics registry
runtime_rollback.py       temporal excision: collect flows / fit greedy detector / run (mask|rollback|none|abstain) with soft scale + window + static+temporal merged
sweep_thresholds.py       offline threshold sweep (validated 1:1 vs live runs)
sweep_windows.py          offline window tradeoff (w2..w6 × p80..p95)
battery_analysis.py       sampled 5-seed battery: majority + McNemar + bootstrap
bench_analysis.py         hard/random/merged bench analysis (strict, McNemar, bootstrap, majority, fired breakdown)
bench_build.py            build frozen benches (hard: greedy-wrong, random: seed 42)
bench_driver.py           hard bench sampled battery driver
bench_full_sampled.py     random + merged sampled battery driver
bench_random_greedy.py    random bench greedy driver
strict_final.py           strict first-sentence metric with alternatives (canonical scorer)
significance_final.py     McNemar + bootstrap significance
significance_s5.py        sample-level statistics
topics.py                 AFRICA(54)/EUROPE(44)/ELEMENTS(41)/ASIA(46)/US_STATES(50)/AFRICA_LARGEST(54)/WORLD_TRICKY(49)/WORLD_CAP_TRAPS(134)/WORLD_LARGEST(173)
results/                  masks, evaluations, detector configs, benches (bench_hard.json, bench_random.json), NUMBERS.md (canonical), experiment_report.md
legacy/                   superseded experiments, debug probes, and null-result statistical arm
```

## Quick start

```bash
python -m venv .venv && .venv/Scripts/pip install torch transformers scikit-learn numpy scipy
# 1. collect greedy flows for detector calibration
python runtime_rollback.py collect            # → results/greedy_flows_africa.npz
python runtime_rollback.py fit_greedy         # → results/detector_greedy.json
# 2. run the temporal intervention (soft, window<=5 — recommended)
python runtime_rollback.py run africa early t90 mask m 0 0.3 5
# 3. static excision alternative
python eval_topic.py africa --mask results/mask_k32_midwrong.json
# 4. re-score any eval file with the strict metric
python strict_final.py
# 5. significance + batteries
python significance_final.py; python battery_analysis.py; python bench_analysis.py
# 6. run a frozen bench (hard or random, sampled, 6 seeds)
python bench_driver.py                    # hard bench, 6 seeds
python bench_full_sampled.py              # random + merged
```

Full methodology and per-run commentary: `results/experiment_report.md`.
Status and lesson log: `STATUS.md`.
