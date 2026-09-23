# Status — No-Hallucinations-Ever

Updated: 2026-09-23

**Paper:** *NHE-Edge: Sub-Second Hallucination Suppression for Sub-1B Edge LLMs*
— fix a hallucination in 24 hours, on a CPU, without retraining, and test the
result for regressions.

**Companion:** *NHE-NTW* — a tiny GPT trained from scratch with four planted
false facts causally demonstrates the Static Memory Void category. In that
controlled setting, false facts are confident and jitter-quiet, while
zero-training countries remain uncertain. This provides a plausible mechanism
for Gemma's quiet cases, not a direct proof about Gemma. See
`NHE-NTW/README.md`.

We watch Gemma 3 1B while it writes. When it's about to make up a capital, the
hidden state jumps. We find the neurons that push the wrong answer and turn them
down. We test on African capitals and make sure other topics still work.

## Setup

- Windows, Python 3.11.15, `.venv`, torch 2.13.0+cpu, transformers 5.15.0
- Model: `models/gemma3-1b-fp16` (fp16, 2.7 GB, gitignored), tokenizer from unsloth
- 3050 4GB present but not used — fp16 backward is 15× slower than fp32 on CPU

## Where we are

All numbers are strict (first sentence) and labeled. Full table:
`NHE-Edge/results/NUMBERS.md`. Code and results live in `NHE-Edge/`.

1. **Signal.** Jitter separates wrong vs correct (pooled AUROC 0.968, but the
   deployed early detector is 0.742). It doesn't transfer greedy ↔ sampled.

2. **Statistics don't work.** Means and variances: nothing (only 7/128 overlap
   with causal).

3. **Causal does.** k32 (layers 8–17, wrong-only): Africa 7→5 greedy, 8→6 sampled
   (p=0.017), but breaks South Sudan. k128: 7→4 greedy, 8→4 sampled (p=0.003),
   breaks two plus Europe.

4. **Timing matters.** Early detector in the first 5 tokens + soft scaling (0.3)
   gives 7→5 greedy with 0 breaks. The only fix that never hurts a control.

5. **Benches.** Hard bench (99 hard: 54+15+30, 594 draws): 354→334, p<0.001
   per-draw, but majority 59→56 (p=0.25, not significant). Random bench (99 random):
   59→53, p=0.031 per-draw, majority 10→9 (not significant). Same direction,
   small effect — honest.

6. **Merged.** Static always + runtime when it fires (hard bench): 354→231 (mask,
   but 9 new breaks) and 52 with 348 refusals (abstain). Fires jump 15%→58%.

7. **Ceiling and NTW.** Four Africa errors never spike (Cape Verde, Eq Guinea,
   Gabon, Guinea). The NTW controlled experiment demonstrates the complementary
   Static Memory Void: a false fact can be learned confidently and quietly. This
   gives the ceiling a plausible mechanism and a triage path: check provenance
   or factual data first; use runtime repair when a pre-commit drift is visible.

8. **Direct Gemma audit.** Three of the four quiet cases are highly confident at
   the commit token (Cape Verde p=1.0, Guinea p=0.9986; Eq Guinea p=0.956), while
   Gabon is a separate uncertainty/truncation case. All four remain unanswered by
   the current jitter detector.

9. **Cross-model.** Qwen2.5-0.5B on real weights: signal exists (AUC 0.778), but
   the spike lands after the city, so 0 flips. Timing is format-dependent.

10. **Side effects.** Soft k32 on 200 real MMLU: 0.925→0.925, 0.610→0.620 — preserved.
    Temporal on MMLU: 78→79, fires on 155/200 — no effect (threshold doesn't transfer).

Lessons: clean state per item (or you fake it), strict scoring (loose counts hedges),
offline simulation matches live 1:1, manual sampler ≠ `model.generate`.

Full story: `NHE-Edge/results/experiment_report.md`. Files: `NHE-Edge/results/*.json`.

## Data

- `data/` 378 MB (gitignored): flows (T,27,1152) fp16
- 9 topics in `NHE-Edge/core/topics.py` (54+44+41+46+50+54+49+134+173)
- `NHE-Edge/results/greedy_flows_africa.npz` + `bench_hard.json` + `bench_random.json`

## What's next

- Test the Static Memory Void signature directly on Gemma/Claude-scale models.
- Qwen sampled battery (hard/random) with its own mask.
- Train detector on sampled flows — fix the greedy↔sampled gap.
- QLoRA fine-tune on Africa as a baseline.

## Repo

Public: https://github.com/Zierax/NHE-Architecture branch `NHE-Architecture`
Code + results + benches. `data/` and `models/` gitignored, rebuild with
`NHE-Edge/core/collect_topic.py`.
