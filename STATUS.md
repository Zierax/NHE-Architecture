# Status — No-Hallucinations-Ever

Updated: 2026-08-21

We try to catch hallucinations in Gemma 3 1B by watching activations while it
writes, finding the neurons that push the wrong answer, and turning them down.
We test on African capitals and make sure other topics still work.

## Setup

- Windows, Python 3.11.15, `.venv`, torch 2.13.0+cpu (no CUDA), transformers 5.15.0
- Model: `models/gemma3-1b-fp16` (fp16, 2.7 GB, gitignored), tokenizer from unsloth
- 3050 4GB present but not used — fp16 backward on CPU is very slow, fp32 is 15× faster

## Where we are

All numbers below are strict (first sentence) and labeled. Full table: `NHE-Edge/results/NUMBERS.md`. All code and results live in `NHE-Edge/` (root holds only overview docs, roadmap, env).

1. **Signal.** Jitter (hidden-state jump) separates wrong vs correct well on Africa
   (AUROC 0.968, peak layers 10–15). It doesn't transfer across greedy ↔ sampled
   (AUC 0.05).

2. **Statistics don't work.** d_mean / d_var: nothing (only 7/128 overlap with causal).

3. **Causal does.** k32 (layers 8–17, wrong-only): Africa greedy 7/54→5/54 (strict), sampled substring majority 8/54→6/54 p=0.017, but breaks South Sudan. k128: greedy 7/54→4/54 strict, sampled 8/54→5/54 p=0.003, breaks two plus Europe.

4. **Timing matters.** Early detector (first 10 tokens, AUC 0.742) firing in first 5
   tokens with soft scaling (0.3) gives 7/54→5/54 greedy with 0 breaks. It's the
   only single fix that never hurts a control.

5. **Benches.** Hard bench (99 picked to be hard: 54+15+30, 594 draws): 354/594→
   334/594 p<0.001, 20 fixes, CI [-0.049, -0.020]. Random bench (99 random, same pool,
   594 draws): 59/594→53/594 p=0.031, CI [-0.0185, -0.0034]. Same direction — not
   just cherry-picking.

6. **Merged.** Static always + runtime when it fires (hard bench): 354/594→231/594
   (repair) and 52/594 with 348 refusals (abstain). Fires jump 15%→58%. Only merged
   moves the quiet cases.

7. **Ceiling.** 4/7 Africa errors never spike (Cape Verde, Eq Guinea, Gabon, Guinea).
   No single jitter threshold or k32 mask catches them. Only merged does.

8. **Quiet check.** Logit lens on those 4 shows they are *not* early high-confidence
   parametric errors — they look dynamic with subtler jitter (late prob, high
   entropy like the fixable ones). So a second signal may catch them
   (`probe_quiet.py`, `results/quiet_diagnostic.json`).

9. **Cross-model.** Qwen2.5-0.5B/1.5B adapter ready, synthetic validated
   (`runtime_rollback_qwen.py`, `results/cross_arch_report.md`); real download
   pending.

10. **Side effects.** Soft k32 on 200 real MMLU: 0.925→0.925 substr (0.0), 0.610→0.620 strict (+0.01) — preserved, even slight gain; proxy 181 controls also preserved (-0.016) (`eval_mmlu.py --use-real`, `results/mmlu_side_effect.json:142`).

Lessons: clean state per item (or you fake results), strict scoring (loose counts
hedges), offline simulation matches live 1:1, manual sampler ≠ `model.generate`
(both valid, different draws).

Full details: `NHE-Edge/results/experiment_report.md`. Raw files: `NHE-Edge/results/*.json`.

## Data

- `data/` 378 MB (repo root, gitignored): flows (T,27,1152) fp16
- 9 topics in `NHE-Edge/topics.py` (54+44+41+46+50+54+49+134+173)
- `NHE-Edge/results/greedy_flows_africa.npz` + `bench_hard.json` + `bench_random.json` (99 each)

## What's next

- Try the same on a bigger model (Gemma 3 4B or Qwen 1.5B) — does the signal hold?
- Train detector on sampled flows — fix the greedy↔sampled gap.
- QLoRA fine-tune on Africa as a baseline to compare against excision.

## Repo

Public: https://github.com/Zierax/NHE-Architecture branch `NHE-Architecture`
Code + results + benches. `data/` and `models/` are gitignored, rebuild with
`collect_topic.py`. Old history in local `local-history`.
