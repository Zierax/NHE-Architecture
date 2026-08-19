# NHE-Architecture — No-Hallucinations-Ever

Detecting and surgically excising hallucinations in Gemma 3 1B (text) by tracing internal
activation "jitter" during decoding, locating wrong-commit neurons, and cutting them —
with topic-specificity verification (Africa as target; Europe / Asia / US states / elements
as controls).

**Model used:** `google/gemma-3-1b-it` (Gemma 3 1B text model). The checkpoint we
run is a single consolidated `model.safetensors` of the language-model weights cast to
**fp16** (verified: 999.9M params, `dtype=torch.float16`, `Gemma3ForCausalLM`,
26 layers × 1152 hidden) — NOT a Q4/GGUF quantization. Obtain the original from
https://huggingface.co/google/gemma-3-1b-it (gated) and cast with `.half()`. Tokenizer:
the Gemma 3 SentencePiece tokenizer (vocab 32768; unsloth/gemma-3-1b-it).

Everything runs on **CPU** (torch 2.13.0+cpu) — no CUDA required. The 2.7 GB fp16
checkpoint is excluded from this repo (see `STATUS.md` / `.gitignore`) and reconstructed
locally.

## Headline results (Africa capitals, n=54; strict first-sentence metric unless noted)

| Intervention | greedy (strict) | sampled (5 seeds, majority) | Collateral (greedy, strict) |
|---|---|---|---|
| baseline | 0.130 | 0.148 | — |
| statistical excision (d_mean/d_var) | 0.130 (null) | — | none |
| causal k32_midwrong (layers 8–17, wrong-only) | 0.093 | **0.111 (p=0.017)** | **breaks South Sudan (Juba→Bor)** |
| causal k128_wrong | 0.074 | **0.093 (p=0.003)** | **breaks Benin, South Africa, South Sudan + Europe 0.000→0.091** |
| **runtime early soft excision** (w≤5, scale 0.3) | **0.093** | 0.074 (majority; per-seed mean 0.122→0.104, p=0.18) | **0 breaks anywhere; Europe 0 fires, world_tricky 0 fires** |

**The canonical, protocol-labelled table (greedy/sampled × substring/strict metrics)
with evidence files: `results/NUMBERS.md`.** Read numbers from there — greedy and
sampled-majority are different protocols and metrics must never be mixed in a comparison.

Runtime excision details: detector `jump_max_early_L19` (AUC 0.742 LOSO on greedy flows)
fires only within the first 5 tokens; a soft mask (neurons scaled by 0.3, not zeroed) of
wrong-commit neurons is applied before the answer commit. Post-commit firing is inert
(timing is the essence). Fires 7/54 with 2 clean fixes (Eswatini, Gambia) and zero breaks;
the Senegal fix seen under the permissive matcher is a hedge ("Diou... While Dakar is the
largest city") and counts as wrong under the strict metric.

**Transfer (new topics, greedy, strict):** `africa_largest` baseline 0.296 → runtime 0.278
(1 fix, 0 breaks); `world_tricky` 0.020 unchanged (0 fires); Europe 0.000 unchanged (0
fires). Static k128 damages all three controls (0.091 / 0.122 / breaks).

**Confirmed ceiling:** 4/7 greedy hallucinations (Cape Verde, Equatorial Guinea, Gabon,
Guinea) commit "quietly" with no pre-commit jitter spike and are not fixable by any
threshold of this detector family or by the k32_midwrong mask.

## Repository layout

```
attribute_causal2.py      AtP attribution (fp32 CPU-optimized), wrong-only filters, mid-layer masks
run_experiment.py         mask generation + evaluation runs (mean/var/causal/wrong/mid/midwrong)
eval_topic.py             greedy + sampled evaluation (--samples=N), topics registry
runtime_rollback.py       temporal excision: collect flows / fit greedy detector / run (mask|rollback|none), soft scale, window
sweep_thresholds.py       offline threshold sweep (validated 1:1 against live runs)
sweep_windows.py          offline window tradeoff (w2..w6 × p80..p95)
battery_analysis.py       sampled 5-seed battery: majority + McNemar + bootstrap
strict_final.py           strict first-sentence metric with alternatives
significance_final.py     McNemar + bootstrap significance
significance_s5.py        sample-level statistics
topics.py                 AFRICA(54)/EUROPE(44)/ELEMENTS(41)/ASIA(46)/US_STATES(50)/AFRICA_LARGEST(54)/WORLD_TRICKY(49)
results/                  all masks, evaluations, detector configs, NUMBERS.md, experiment_report.md
legacy/                   superseded experiments and debug probes (statistical arm = null result)
```

`data/` (raw hidden-state flows, ~380 MB) is intentionally **not** in this repository;
reconstruct it with `collect_topic.py` (needs the model + tokenizer above).

## Quick start

```bash
python -m venv .venv && .venv/Scripts/pip install torch transformers scikit-learn numpy scipy
# 1. collect greedy flows for detector calibration
python runtime_rollback.py collect            # saves results/greedy_flows_africa.npz
python runtime_rollback.py fit_greedy         # fits detector, saves results/detector_greedy.json
# 2. run the temporal intervention (soft, window<=5, recommended)
python runtime_rollback.py run africa early t90 mask m 0 0.3 5
# 3. static excision alternative
python eval_topic.py africa --mask results/mask_k32_midwrong.json
# 4. strict re-scoring of any eval file
python strict_final.py
```

## Honest limitations

- Textual matcher is permissive; every flip in the final results was manually inspected
  (e.g., Senegal counts as a fix but is a hedge: "Diou... While Dakar is the largest city").
- Single model (Gemma 3 1B), single target domain (Africa capitals), CPU-only runs.
- Detector does not transfer across decoding protocols (temp-0.9 probe → greedy AUC ≈ 0.05).
- Runtime intervention is safe under greedy decoding; under sampling 1/54 documented break
  (Burundi Gitega→Bujumbura) — a softer intervention is the open follow-up.

Full report: `results/experiment_report.md`. Status: `STATUS.md`.