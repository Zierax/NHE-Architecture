# NHE-Architecture

**No-Hallucinations-Ever — fixing hallucinations on small models without retraining.**

A medical device hallucinates a capital in the field. You have 24 hours, a CPU,
and no way to fine-tune. NHE-Edge finds the neurons that cause the wrong answer
and turns them down — in seconds, on-device, with no regressions on the controls
we tested.

**Paper:** *NHE-Edge: Sub-Second Dynamic Hallucination Suppression for Sub-1B Edge LLMs*

## Three tracks, one taxonomy

Hallucination is not one thing. Some errors are late wrong-commits: a middle-layer
spike appears before the bad token, and a timed intervention can reduce them. Other
errors are quiet facts stored confidently in the weights. NHE-Edge addresses the
first; NHE-NTW gives a controlled causal proof and triage framework for the second.

- **NHE-Edge/** — **working.** Gemma 3 1B, jitter detector + soft scaling (0.3) in the
  first 5 tokens. On the single-task 463-item benchmark: 59→56 wrong, **3 fixes,
  0 breaks, p=0.25** — safety, the timing law and the per-fired repair rate
  (0.333) are confirmed at scale, significance is still open. Start here:
  [`NHE-Edge/README.md`](NHE-Edge/README.md).

- **NHE-GenPM/** — **planned.** Same detector, but steering / SAEs for larger models
  where you must keep MMLU. See [`NHE-GenPM/plan.md`](NHE-GenPM/plan.md).

- **NHE-NTW/** — **companion proof and triage framework.** A tiny GPT trained from
  scratch with four planted false facts causally demonstrates the Static Memory Void:
  quiet, confident, wrong answers can be encoded in the weights. Zero-training
  countries instead show uncertainty. This gives Gemma's quiet ceiling a plausible
  mechanism and a clear split: check provenance/facts first, then use runtime repair
  when a pre-commit drift is present. The runtime signal's silence is therefore a
  scope boundary, not a framework failure. See [`NHE-NTW/README.md`](NHE-NTW/README.md).

## Where this is going

The action arm is safe and its repair rate generalises across three question
frames; it is not yet statistically significant because the trigger reaches only
15.3% of errors. We measured whether the quiet residual is reachable rather than
assuming it: **AUROC 0.713 (L15/w15)** separates the 50 never-fired wrong items
from correct, so the ceiling is a **trigger** problem, not a mechanism limit. The
next experiment is one live L15/w15 arm on already-collected data. Full decision
tree in [`ROADMAP.md`](ROADMAP.md).

`paper/` holds local LaTeX drafts (gitignored). `models/` and `data/` are gitignored
and rebuilt — see `NHE-Edge/README.md` for the exact recipe. `requirements.txt`
pins the CPU env.

## 30-second start (Edge)

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
huggingface-cli login   # gated Gemma, once
# fetch weights per NHE-Edge/README into models/, then:
cd NHE-Edge
python core/runtime_rollback.py collect            # flows -> results/greedy_flows_africa.npz
python core/runtime_rollback.py fit_greedy         # detector -> results/detector_greedy.json
python core/attribute_causal2.py                   # attributions (hours, CPU)
python core/run_experiment.py                      # masks -> results/mask_k32_midwrong.json
python core/runtime_rollback.py run africa early t90 mask m 0 0.3 5
python experiments/bench.py build
python experiments/bench.py run --bench hard --mode sampled  # 6 seeds, hours
python experiments/bench.py analyze --bench hard
python analysis/stats.py strict
```

**One table to quote from:** [`NHE-Edge/results/NUMBERS.md`](NHE-Edge/results/NUMBERS.md)
— every number labeled by protocol and metric. Full story:
[`NHE-Edge/results/experiment_report.md`](NHE-Edge/results/experiment_report.md).
Status: [`STATUS.md`](STATUS.md). Plan: [`ROADMAP.md`](ROADMAP.md).
