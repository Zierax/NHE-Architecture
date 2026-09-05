# Status — No-Hallucinations-Ever

Updated: 2026-09-04

We try to catch hallucinations in Gemma 3 1B by watching activations while it
writes, finding the neurons that push the wrong answer, and turning them down.
We test on African capitals and make sure other topics still work.

## Setup

- Windows, Python 3.11.15, `.venv`, torch 2.13.0+cpu (no CUDA), transformers 5.15.0
- Model: `models/gemma3-1b-fp16` (fp16, 2.7 GB, gitignored), tokenizer from unsloth
- 3050 4GB present but not used — fp16 backward on CPU is very slow, fp32 is 15× faster

## Where we are

All numbers below are strict (first sentence) and labeled. Full table: `NHE-Edge/results/NUMBERS.md`. All code and results live in `NHE-Edge/` (root holds only overview docs, roadmap, env).

1. **Signal.** Jitter separates wrong vs correct on Africa (last-token L10 AUROC
   0.968, pooled feature — signal exists but not the deployed one; deployed early
   detector is 0.742). Sampled→greedy transfer fails for the probe (AUC ~0.05).

2. **Statistics don't work.** d_mean / d_var: nothing (only 7/128 overlap with causal).

3. **Causal does.** k32 (layers 8–17, wrong-only): Africa greedy 7/54→5/54 (strict), sampled substring majority 8/54→6/54 p=0.017 (per-draw mean 0.111), but breaks South Sudan. k128: greedy 7/54→4/54 strict, sampled majority 8/54→4/54 p=0.003 (per-draw mean 0.093), breaks Benin + South Sudan plus Europe.

4. **Timing matters.** Early detector (first 10 tokens, AUC 0.742) firing in first 5
   tokens with soft scaling (0.3) gives 7/54→5/54 greedy with 0 breaks. It's the
   only single fix that never hurts a control.

5. **Benches (sampled, strict; per-draw p overstates — item majority is primary).**
   Hard bench (all 54 africa_largest + greedy-wrong 15+30, 594 draws): 354/594→
   334/594 per-draw p<0.001, 20 fixes / 0 new errors per-draw; majority 59/99→56/99
   (W2C=3, C2W=0, p=0.25, n.s.). Random bench (99 random, 594 draws): 59/594→53/594
   per-draw p=0.031; majority 10/99→9/99 (n.s.). Same direction — small effect.

6. **Merged.** Static always + runtime when it fires (hard bench): 354/594→231/594
   (mask; 132 fixes but 9 new breaks) and 52/594 with 348 refusals (abstain; 305/3).
   Fires jump 15%→58%. Bench-aggregate improvement; no per-item proof the four
   Africa quiet cases are among the fixed.

7. **Ceiling.** 4/7 Africa errors never spike (Cape Verde, Eq Guinea, Gabon, Guinea).
   No single jitter threshold or k32 mask catches them (greedy Africa).

8. **Quiet check.** Logit lens on those 4 shows they are *not* early high-confidence
   parametric errors — all 7 cases classify dynamic (`NHE-Edge/probe_quiet.py`,
   `NHE-Edge/results/quiet_diagnostic.json`). So a second signal may catch them,
   but none has been tested.

9. **Cross-model.** Qwen2.5-0.5B/1.5B adapter ready, synthetic validated
   (`runtime_rollback_qwen.py`, `results/cross_arch_report.md`); real download
   pending.

10. **Side effects.** Soft k32 static on 200 real MMLU: 0.925→0.925 substr, 0.610→0.620 strict — preserved. Temporal on 200 MMLU (different 200): 78/200→79/200 strict, fires 155/200, W2C=0/C2W=1 p=1.0 — **no effect, honest negative** (threshold miscalibrated off-distribution). (`NHE-Edge/eval_mmlu_temporal.py`, `NHE-Edge/results/mmlu_temporal.json`).

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
