# NHE-Architecture

**No-Hallucinations-Ever — fixing hallucinations on small models without retraining.**

A medical device hallucinates a capital in the field. You have 24 hours, a CPU,
and no way to fine-tune. NHE-Edge finds the neurons that cause the wrong answer
and turns them down — in seconds, on-device, with proof it broke nothing.

**Paper:** *NHE-Edge: Sub-Second Dynamic Hallucination Suppression for Sub-1B Edge LLMs*

## Two tracks, one idea

Hallucination is a late wrong-commit in the middle layers. The spike comes just
before the bad token. Cut there, and you fix it. Cut after, and it's too late.

- **NHE-Edge/** — **working.** Gemma 3 1B, jitter detector + soft scaling (0.3) in the
  first 5 tokens. Hard bench 20 fixes (p<0.001 per-draw), random bench 6 fixes
  (p=0.031), zero breaks on all tested controls. Cross-model (Qwen) and MMLU done.
  Start here: [`NHE-Edge/README.md`](NHE-Edge/README.md).

- **NHE-GenPM/** — **planned.** Same detector, but steering / SAEs for larger models
  where you must keep MMLU. See [`NHE-GenPM/plan.md`](NHE-GenPM/plan.md).

- **NHE-NTW/** — **companion proof.** Tiny GPT trained from scratch with 4 planted
  lies proves the quiet cases are a real category: confident lies baked into weights,
  not just missed detections. See [`NHE-NTW/README.md`](NHE-NTW/README.md).

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
