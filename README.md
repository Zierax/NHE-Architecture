# NHE-Architecture - No-Hallucinations-Ever

Paper title: **NHE-Edge: Sub-Second Dynamic Hallucination Suppression for
Sub-1B Edge LLMs in Mission-Critical Scenarios.**

One mission: a medical device or defense system hallucinates in the field.
There are 24 hours, no GPU farm, no fine-tuning option. NHE-Edge suppresses
the hallucination on sub-1B models, on-device, without retraining - and proves
it broke nothing.

Two tracks, one core idea (hallucination is a late wrong-commit in middle layers):

- **NHE-Edge/** - the working track. Gemma 3 1B, jitter detector + soft scaling
  (0.3) in the first 5 tokens. Hard bench per-draw p<0.001 (item majority n.s.),
  random bench per-draw p=0.031 (majority n.s.), zero breaks on tested controls.
  All code, results, and benches. Start here: `NHE-Edge/README.md`.
- **NHE-GenPM/** - planned. Same detector, but steering / SAEs instead of raw
  scaling, to keep MMLU on bigger models. See `NHE-GenPM/plan.md`.
- **paper/** - local LaTeX drafts (gitignored, not pushed).

Shared, gitignored: `models/` (weights), `data/` (flows). Pinned env:
`requirements.txt`.

Quick start (Edge, PowerShell 5.1, CPU hours for model runs):

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
huggingface-cli login   # gated model, once
# fetch weights per NHE-Edge/README "How to reproduce" into models/, then:
cd NHE-Edge
python core/runtime_rollback.py collect            # flows -> results/greedy_flows_africa.npz
python core/runtime_rollback.py fit_greedy         # detector -> results/detector_greedy.json
python core/attribute_causal2.py                   # attributions (hours)
python core/run_experiment.py                      # masks incl. results/mask_k32_midwrong.json
python core/runtime_rollback.py run africa early t90 mask m 0 0.3 5
#  early=detector set | t90 | mode mask | m=greedy | seed 0 | scale 0.3 | window 5
python experiments/bench.py build                           # freeze benches
python experiments/bench.py run --bench hard --mode sampled  # battery, 6 seeds (hours)
python experiments/bench.py analyze --bench hard            # stats over committed runs
python analysis/stats.py strict                             # re-score committed evals
```

Numbers live in `NHE-Edge/results/NUMBERS.md` - the only table we quote from.
Full walkthrough: `NHE-Edge/results/experiment_report.md`. Short status:
`STATUS.md`. Plan: `ROADMAP.md`.
