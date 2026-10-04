# NHE-Architecture

**No-Hallucinations-Ever — suppressing hallucinations in sub-1B models at runtime,
without retraining, on CPU, inside a 24-hour field window.**

A deployed model produces a wrong answer. You have 24 hours, a CPU, no GPU farm,
and no permission to retrain. NHE-Edge locates the internal features that produce
the error, suppresses them during generation, and leaves the rest of the model's
behaviour verified unchanged.

**Paper:** *NHE-Edge: Sub-Second Dynamic Hallucination Suppression for Sub-1B Edge LLMs*

---

## The deliverable

| Requirement of the mission | Measured result | Evidence |
|---|---|---|
| Runs on-device, sub-1B, CPU | Gemma 3 1B, fp16, no CUDA | `results/latency.json` |
| No retraining | weights on disk unchanged; 32 neurons scaled at inference | `core/runtime_rollback.py` |
| Latency acceptable in the field | **+20.5% per token**, 160 ms once per firing | `results/latency.json` |
| Removes errors | **59 → 56** wrong of 463 items | `results/single_task_score.json` |
| No collateral damage | **0 regressions** across 463 items, 3 question frames | same |
| Fits the 24-hour window | collect 60 min + fit/mask hours + verify 80 min = **6–10 h** | measured wall-clock, this repo |
| Generalises across architectures | gated by an automatic applicability test; Qwen correctly refused | `results/applicability_gate_*.json` |

## Two claims, held to two different standards

This distinction governs how every number in this repository should be read, and
it is the reason the project is not in the state a single headline number would
suggest.

**Claim 1 — Deployment validity.** *For a given model on a given task, NHE-Edge
removes errors at runtime, without retraining, without regression, within the
operational budget.* This is an engineering claim. It is verified by **direct
measurement on the deployed configuration**: a paired 463-item run with zero
regressions, measured latency, and measured wall-clock. It does not require, and
does not benefit from, a significance test.

**Claim 2 — Population-level statistical inference.** *The measured reduction
generalises beyond this model, task, seed and sample.* This is a scientific claim
about an underlying effect size. It is tested by exact McNemar on the paired
463-item run and it is **not established**: 3 fixes, 0 breaks, p = 0.25.

These claims are independent. Claim 1 is complete. Claim 2 is open. A field fix
that removes three verified errors from a specific device with zero regressions
is a finished deliverable even when the population effect is not separable from
zero at n = 463 — and a large, significant population effect would still not by
itself constitute a deployable fix.

Everywhere results appear in this repository, they are labelled as one or the
other. The canonical table is
[`NHE-Edge/results/NUMBERS.md`](NHE-Edge/results/NUMBERS.md).

## What is established, and at what strength

**Established by direct measurement (deployment-valid):**
- Zero collateral damage. 0 regressions over 463 items spanning three question
  frames, against 9 regressions in the merged static+temporal arm.
- The timing law. 40 of 41 fired items spiked at least one token before the answer
  commit; the single late firing never changed an answer.
- Generalisation of the repair mechanism. Repairs per fired-wrong item: 0.286 on
  Africa (n=54), 0.333 on the 463-item benchmark.
- Measured cost and measured wall-clock, on the target hardware class.
- A causal, not correlational, intervention: statistical neuron selection
  (mean/variance) yields nothing; activation patching on wrong-only examples
  yields the neurons that actually drive the error.

**Established as controlled causal evidence (mechanism, not deployment):**
- The Static Memory Void exists as a construct. A tiny GPT trained from scratch
  with four planted false facts answers them at confidence 1.000, P(truth)=0,
  entropy 0, with jitter indistinguishable from correct answers; zero-training
  countries instead answer uncertain. See [`NHE-NTW/README.md`](NHE-NTW/README.md).

**Not established (open scientific questions):**
- Statistical significance of the error reduction (p = 0.25 at n = 463).
- Population-level generality beyond Gemma 3 1B and this task family.
- Whether the residual quiet error is fully unreachable. 89.3% of it never fires
  under the deployed trigger, which is consistent with the void reading, but a
  higher-recall trigger at L11 reaches 0.286 of errors versus 0.153 — and fails,
  because it introduces regressions. See `ROADMAP.md`.

## Three tracks

Hallucination is not one phenomenon. The project separates two and addresses each
on its own terms.

- **NHE-Edge/** — the deliverable. Runtime suppression of transient execution
  drift on Gemma 3 1B. [`NHE-Edge/README.md`](NHE-Edge/README.md)
- **NHE-NTW/** — companion causal proof and triage framework for static confident
  errors. [`NHE-NTW/README.md`](NHE-NTW/README.md)
- **NHE-GenPM/** — planned. Feature-space intervention (SAE / steering) for larger
  models where raw neuron scaling is unacceptable.
  [`NHE-GenPM/plan.md`](NHE-GenPM/plan.md)

## Reproduce

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
huggingface-cli login   # gated Gemma, once
# fetch weights per NHE-Edge/README into models/, then:
cd NHE-Edge
python core/runtime_rollback.py collect        # flows
python core/runtime_rollback.py fit_greedy     # detector
python core/attribute_causal2.py               # causal attribution (hours, CPU)
python core/run_experiment.py                  # masks
python core/runtime_rollback.py run africa early t90 mask m 0 0.3 5
python experiments/bench.py build
python experiments/bench.py run --bench hard --mode sampled
python experiments/bench.py analyze --bench hard
python analysis/stats.py strict
```

`models/` and `data/` are gitignored and regenerable. `paper/` holds local LaTeX
drafts. `requirements.txt` pins the CPU environment.

Status: [`STATUS.md`](STATUS.md) · Plan: [`ROADMAP.md`](ROADMAP.md) ·
Full numbers: [`NHE-Edge/results/NUMBERS.md`](NHE-Edge/results/NUMBERS.md) ·
Narrative: [`NHE-Edge/results/experiment_report.md`](NHE-Edge/results/experiment_report.md)