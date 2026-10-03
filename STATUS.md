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

6. **Single-task power test (463 items).** One merged task (244 capital, 178
   largest, 41 element), audited truth layer (11 cross-topic truth conflicts
   resolved by explicit union), token-boundary scoring (26 substring hazards
   closed). Baseline 59/463 wrong (0.127). Runtime L19/w5: **56/463, fixes=3,
   breaks=0, p=0.25 (still not significant)**. What it does establish at n=463:
   zero collateral damage across 463 items and 3 frames; the timing law (40/41
   fired items lead≥1); the per-fired repair rate generalises (0.333 vs 0.286
   Africa); and 89.3% of the residual error never fires under this trigger.
   The shortfall is measured: firing precision 0.220, detector reaches 15.3% of
   errors, so significance needs a higher-recall trigger, not a bigger benchmark
   (see item 13 for why the trigger is the right target).

7. **Merged.** Static always + runtime when it fires (hard bench): 354→231 (mask,
   but 9 new breaks) and 52 with 348 refusals (abstain). Fires jump 15%→58%.

8. **Ceiling and NTW.** Four Africa errors never spike (Cape Verde, Eq Guinea,
   Gabon, Guinea). This is a scope boundary of the current runtime signal, not a
   framework failure: NHE-Edge targets transient pre-commit drift, and a stored
   false fact can be committed without one. NTW's controlled experiment shows a
   false fact can be learned confidently and quietly, which motivates checking
   provenance/factual data first and using runtime repair when a drift is visible.
   On the 463-item benchmark, 89.3% of the residual error never fires under the
   deployed trigger — consistent with the void reading, and see item 13 for the
   caveat that part of that population is in fact trigger-reachable.

9. **Direct Gemma audit.** Three of the four quiet cases are highly confident at
   the commit token (Cape Verde p=1.0, Guinea p=0.9986; Eq Guinea p=0.956), while
   Gabon is a separate uncertainty/truncation case (p=0.6925). All four remain
   unanswered by the current jitter detector; direct provenance testing is next.

10. **Cross-model + applicability gate.** Qwen2.5-0.5B has the signal family but
    no *pre-commit* signal in its current format. The gate searches 192
    (layer x window) cells and controls for the commit-position artifact: Gemma
    0.781 fixed-slice = ACTIVE, Qwen 0.691 = INERT. AUROC gates applicability but
    does not select the intervention (L20 0.781 → net −1; L19 0.772 → net +2).

11. **L19 justification (partial).** The detector reads L19, two layers ABOVE the
    patching-derived causal band (L10-17, zero neurons at L19) — causally
    informed, not arbitrary. But proximity does not uniquely select it (L18-25 are
    equally adjacent), so choosing it without a live run would be config luck.
    Recorded honestly in `results/detector_layer_justification.json`.

12. **Side effects.** Soft k32 on 200 real MMLU: 0.925→0.925, 0.610→0.620 — preserved.
    Temporal on MMLU: 78→79, fires on 155/200 — no effect (different task; NHE-Edge
    is single-task by design, and MMLU is not the deployment task).

13. **Reachability of the quiet set — the direction (2026-09-23).** Hidden-state
    jumps collected for all 463 items; the never-fired-wrong population (n=50) is
    separated from correct (n=407) at AUROC **0.713** (L15/w15), above the 0.65
    gate. So the quiet set is **reachable**: the bottleneck is the trigger, not
    the mechanism. Caveat: that cell separates never-fired-wrong from
    *fired-wrong* only 0.34, so quiet-wrong and detected-wrong look different and
    the 0.713 is a hypothesis until validated live.

Lessons: clean state per item (or you fake it), strict scoring (loose counts hedges),
offline simulation matches live 1:1, manual sampler ≠ `model.generate`.

## Direction from here

The runtime arm is safe and its per-fired repair rate generalises; it is not yet
significant because recall on errors is 0.153 and precision 0.220. The next
experiment is fixed and falsifiable, and it runs on data already collected:

1. **Live L15/w15 arm on the same 463 items**, same mask, same scoring, same exact
   McNemar test. It either produces >= 6 fixes at 0 breaks (significance reached)
   or it does not. This is the only experiment that can move the headline.
2. If it fails, the fallback is already determined: the never-fired residual is
   fact-level (NTW), so the remaining work is the diagnostic write-up, not more
   detector tuning.

Full story: `NHE-Edge/results/experiment_report.md`. Files: `NHE-Edge/results/*.json`.

## Data

- `data/` 378 MB (gitignored): flows (T,27,1152) fp16
- 9 topics in `NHE-Edge/core/topics.py` (54+44+41+46+50+54+49+134+173)
- `NHE-Edge/results/greedy_flows_africa.npz` + `bench_hard.json` + `bench_random.json`

## What's next

- **Live L15/w15 arm on the 463-item benchmark** — the direction-setting
  experiment. Either >= 6 fixes at 0 breaks (p<0.05) or it does not.
- Direct NTW signature test on Gemma/Claude-scale models; fact/provenance check
  for any residual that stays unreachable.
- Qwen sampled battery (hard/random) with its own mask.
- QLoRA fine-tune on Africa as a baseline.

## Repo

Public: https://github.com/Zierax/NHE-Architecture branch `NHE-Architecture`
Code + results + benches. `data/` and `models/` gitignored, rebuild with
`NHE-Edge/core/collect_topic.py`.
