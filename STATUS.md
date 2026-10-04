# Status — No-Hallucinations-Ever

Updated: 2026-09-23

**Paper:** *NHE-Edge: Sub-Second Dynamic Hallucination Suppression for Sub-1B Edge LLMs*

**Mission:** a deployed sub-1B model produces a wrong answer in the field. Fix it
within 24 hours, on CPU, without retraining, and verify nothing else changed.

**Companion:** *NHE-NTW* — a controlled causal proof that a second, distinct failure
mode exists: a false fact stored confidently in the weights, which produces no
usable runtime signal. See `NHE-NTW/README.md`.

---

## How to read this file

Results are separated by the standard of evidence that supports them.

- **[DEPLOYMENT-VALID]** — verified by direct measurement of the deployed
  configuration. Sufficient to ship.
- **[MECHANISM]** — verified by controlled causal experiment. Sufficient to
  explain behaviour; not a deployment result.
- **[OPEN]** — a scientific question that the current evidence does not settle.

Nothing in the deployment-valid category depends on a significance test. Nothing
in the open category is presented as a working capability.

## Deployment status — the deliverable

**[DEPLOYMENT-VALID]** Runtime suppression on Gemma 3 1B (fp16, CPU):

| | value |
|---|---:|
| Paired benchmark | 463 items, one task, three question frames |
| Baseline errors | 59 (0.1274) |
| Errors after intervention | **56 (0.1210)** |
| Fixes / regressions | **3 / 0** |
| Latency overhead | **+20.5% per token**, 160 ms per firing |
| Retraining | none — 32 neurons scaled at inference |
| Wall-clock to deploy | ~6–10 h (collect 60 min, fit/attribute hours, verify 80 min) |

**[DEPLOYMENT-VALID]** Supporting properties, each measured rather than asserted:

- **Zero collateral damage.** 0 regressions across 463 items and three question
  frames. The merged static+temporal arm regresses 9 in the same setting, so the
  guarantee is a property of the runtime arm specifically, not of the method in
  general.
- **The timing law.** Of 41 fired items, 40 spiked at least one token before the
  answer commit; the single late firing changed nothing. Every pre-commit firing
  on a wrong item either repaired it or left it unharmed.
- **The repair mechanism generalises.** Repairs per fired-wrong item: 0.286 on
  Africa, 0.333 on the 463-item benchmark. This is the quantity that must survive
  a change of domain, and it did.
- **The intervention is causal.** Selecting neurons by activation statistics
  (mean/variance) yields no effect — only 7 of 128 overlap with the causal set.
  Selecting by activation patching on wrong-only examples yields the neurons that
  demonstrably drive the error.
- **The scoping is automatic.** `analysis/nhe_onboard.py` decides per model whether
  the method applies before any intervention runs, and refuses to proceed when it
  does not.

## Mechanism status — what the residual error is

**[MECHANISM]** Not all wrong answers are the same failure. The project separates
two and measures the size of each on the 463-item benchmark:

| class | signature | response | share of residual error |
|---|---|---|---:|
| Transient execution drift | pre-commit spike, lower confidence | runtime suppression — in scope | 6 / 56 |
| Static memory void | quiet, confident, no usable signal | fact-level repair (provenance, data, RAG) | **50 / 56 (89.3%)** |

**[MECHANISM]** The void class exists as a construct. `NHE-NTW/parametric_proof.py`
trains a tiny GPT from scratch with four planted false facts. All four are answered
at confidence 1.000, P(truth)=0.0000, entropy 0.000, with preamble jitter
indistinguishable from correct answers (669.3 vs 668.0). Countries with zero
training examples instead answer at confidence 0.548 — uncertainty, not confident
error. Ignorance alone does not reproduce the signature.

**[DEPLOYMENT-VALID]** The same split is visible in Gemma itself. Three of the four
Africa items that never spike commit with high confidence (Cape Verde p=1.000,
Guinea p=0.9986, Equatorial Guinea p=0.956); Gabon is a separate
uncertainty/truncation case (p=0.6925).

**[MECHANISM]** The operational consequence is a triage rule: a confident, quiet,
wrong answer is a fact-provenance problem and should be routed to data or an
external knowledge source; a wrong answer preceded by a pre-commit spike is a
runtime problem and is in scope. Checking provenance first costs minutes and
avoids spending a day tuning a threshold that cannot fire.

## Scientific status — open questions

**[OPEN] Statistical significance.** 3 fixes with 0 regressions gives exact
McNemar p = 0.25 at n = 463. The power analysis, computed before the run,
determined that 6 fixes at 0 regressions is the minimum for p < 0.05. The
benchmark has the headroom — 59 error items — but the deployed trigger reaches only
15.3% of them at precision 0.220, so three discordant pairs result where six are
needed. The direction of the shortfall is known and is not sample size.

**[OPEN] Population-level generality.** All Edge results come from one model
(Gemma 3 1B), one task family, one machine class (CPU), and greedy decoding. The
cross-model evidence is a scoping result, not a performance result: Qwen2.5-0.5B
was tested and the method was found inapplicable to it in its current format.

**[OPEN] Whether the quiet residual is fully unreachable.** 89.3% of residual error
never fires, which is consistent with the void reading. But a higher-recall
pre-commit trigger was built and tested: it reaches 0.286 of errors versus 0.153
and produces 4 fixes instead of 3 — and it introduces 2 regressions, both in the
`element` family, because the mask was fitted on wrong-only capital answers and is
applied to a family it never saw. Since zero collateral damage is the deployment
requirement, that arm is rejected on evidence. The remaining open question is
whether a family-aware trigger can raise recall without regressing.

**[OPEN] Detector layer selection.** The deployed layer (L19) is two layers above
the patching-derived causal band (L10–17, zero neurons at L19), so it is causally
informed rather than arbitrary. But proximity does not uniquely select it — L18
through L25 are equally adjacent. Selecting it offline would be configuration
luck; it was selected and then validated by a live run. Recorded in
`results/detector_layer_justification.json`.

**[MECHANISM] The trigger does not select the intervention.** Pre-commit AUROC
gates whether a model is applicable but does not rank interventions within it: the
highest-AUROC cell on Gemma (L20, 0.781) is worse live (net −1) than a lower-AUROC
cell (L19, 0.772, net +2). Cut-alignment controls on a single layer swing the
outcome from +1 to −1. This is a genuine negative result about offline model
selection and is reported as such.

## Corrections made during this work

Recorded rather than quietly amended, because each one changed a conclusion:

1. A reachability measurement using a fixed 15-token window read tokens at and
   after the answer commit, violating the project's own timing rule. Re-measured
   with a per-item pre-commit maximum: the conclusion held, the responsible layer
   changed from L15 (0.713 → 0.577) to L11 (0.6925).
2. "Qwen is provably inert" rested on three sampled configurations. A 192-cell
   search returned the same verdict, so it stands — as a searched result rather
   than a proof.
3. "Intervenability is a format property, not a model property" is no longer
   asserted; the reachability result shows part of the Gemma quiet set is
   trigger-reachable.
4. The substring scoring metric was unsafe on this task — 26 rows contained a
   shorter accepted answer inside a longer one. Replaced with unit-tested
   whole-token matching; the residual exposure is measured, not assumed.

## Environment

- Windows, Python 3.11.15, `.venv`, torch 2.13.0+cpu, transformers 5.15.0
- Model: `models/gemma3-1b-fp16` (fp16, 2.7 GB, gitignored)
- 3050 4GB present but unused — fp16 backward is 15× slower than fp32 on CPU

## Data

- `data/` 378 MB (gitignored): flows (T,27,1152) fp16
- 9 topics in `NHE-Edge/core/topics.py`, merged into one 463-item benchmark with an
  audited truth layer (`results/bench_single_task_provenance.json`)
- `NHE-Edge/results/`: all outputs, benches, and per-item generations

## Next

- **Family-aware trigger or per-family mask** — the one untried idea that could
  raise recall while preserving the zero-regression guarantee, since both rejected
  breaks were in the one family the mask never saw. Small and decidable.
- Direct provenance test for any residual confirmed as a static void.
- A larger model, to test whether the deployment result survives a change of size.

If the family-aware trigger fails, the honest conclusion is that the ceiling is
real for this task, and the remaining contribution is the taxonomy plus the
measured ceiling.

## Repo

Public: https://github.com/Zierax/NHE-Architecture branch `NHE-Architecture`.
`data/` and `models/` are gitignored and rebuild from `NHE-Edge/core/collect_topic.py`.