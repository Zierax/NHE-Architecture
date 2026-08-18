# NHE-Architecture — No-Hallucinations-Ever

Detecting and surgically excising hallucinations in Gemma 3 1B (text) by tracing internal
activation "jitter" during decoding, locating wrong-commit neurons, and cutting them —
with topic-specificity verification (Africa as target; Europe / Asia / US states / elements
as controls).

**Model used:** [google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it)
(official weights; we use a local fp16 cast for CPU inference — cast any revision with
`model.half()` after download). Tokenizer: [unsloth/gemma-3-1b-it](https://huggingface.co/unsloth/gemma-3-1b-it).
Everything runs on CPU (torch 2.13.0+cpu), no CUDA required.

## Headline results (Africa capitals, n=54)

| Intervention | greedy | sampled (temp 0.9, majority of 5) | Collateral (greedy) |
|---|---|---|---|
| baseline | 0.130 | 0.167 | — |
| statistical excision (d_mean/d_var) | 0.130 (null) | — | none |
| causal k32_midwrong (layers 8–17, wrong-only) | 0.093 | **0.111 (p=0.017)** | Europe 0.000→0.000 |
| causal k128_wrong | **0.056** | **0.093 (p=0.003)** | **Europe 0.000→0.091 (damage)** |
| **runtime early excision** (jitter detector, window ≤5) | **0.074** | 0.111 (seed 1000; 1 documented break) | Europe 0 fires, 0.000 |

**The canonical, protocol-labelled table with evidence files: `results/NUMBERS.md`.**
Read numbers from there — greedy and sampled-majority are different protocols and must
never be mixed in a comparison.

Runtime excision details: detector `jump_max_early_L19` (AUC 0.742 LOSO on greedy flows)
fires only within the first 5 tokens; a mask of wrong-commit neurons is applied before the
answer commit. Post-commit firing is inert (timing is the essence). Fires: 7/54 with
3 clean fixes (Eswatini, Gambia, Senegal-hedge) and zero breaks — strictly better than the
static excision, which fixes the same 3 items but breaks South Sudan (Juba→Bor).

**Confirmed ceiling:** 4/7 greedy hallucinations (Cape Verde, Equatorial Guinea, Gabon,
Guinea) commit "quietly" with no pre-commit jitter spike and are not fixable by any
threshold of this detector family or by the k32_midwrong mask.

## Repository layout

```
attribute_causal2.py      AtP attribution (fp32 CPU-optimized), wrong-only filters, mid-layer masks
run_experiment.py         mask generation + evaluation runs (mean/var/causal/wrong/mid/midwrong)
eval_topic.py             greedy + sampled evaluation (--samples=N), topics registry
runtime_rollback.py       temporal excision: collect flows / fit greedy detector / run with mask
sweep_thresholds.py       offline threshold sweep (validated 1:1 against live runs)
significance_final.py     McNemar + bootstrap significance
significance_s5.py        sample-level statistics
topics.py                 AFRICA(54)/EUROPE(44)/ELEMENTS(41)/ASIA(46)/US_STATES(50)
results/                  all masks, evaluations, detector configs, NUMBERS.md, experiment_report.md
legacy/                   superseded experiments and debug probes (statistical arm = null result)
```

`data/` (raw hidden-state flows, ~380 MB) is intentionally **not** in this repository;
reconstruct it with `collect_topic.py` (needs the model + tokenizer above).

## Quick start

```bash
python -m venv .venv && .venv/Scripts/pip install torch transformers scikit-learn numpy
# 1. collect greedy flows for detector calibration
python runtime_rollback.py collect            # saves results/greedy_flows_africa.npz
python runtime_rollback.py fit_greedy         # fits detector, saves results/detector_greedy.json
# 2. run the temporal intervention
python runtime_rollback.py run africa early t90 mask
# 3. static excision alternative
python eval_topic.py africa --mask results/mask_k32_midwrong.json
```

## Honest limitations

- Textual matcher is permissive; every flip in the final results was manually inspected
  (e.g., Senegal counts as a fix but is a hedge: "Diou... While Dakar is the largest city").
- Single model (Gemma 3 1B), single target domain (Africa capitals), CPU-only runs.
- Detector does not transfer across decoding protocols (temp-0.9 probe → greedy AUC ≈ 0.05).
- Runtime intervention is safe under greedy decoding; under sampling 1/54 documented break
  (Burundi Gitega→Bujumbura) — a softer intervention is the open follow-up.

Full report: `results/experiment_report.md`. Status: `STATUS.md`.