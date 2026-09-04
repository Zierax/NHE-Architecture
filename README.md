# NHE-Architecture — No-Hallucinations-Ever

We catch hallucinations in small language models by watching activations while
they write, finding the neurons that push the wrong answer, and turning them
down — without retraining.

Two tracks, one core idea (hallucination is a late wrong-commit in middle layers):

- **NHE-Edge/** — the working track. Gemma 3 1B, jitter detector + soft scaling
  (0.3) in the first 5 tokens. Hard bench 0.596→0.562 p<0.001, random bench
  0.099→0.089 p=0.031, zero breaks on any control. All code, results, and benches.
  Start here: `NHE-Edge/README.md`.
- **NHE-GenPM/** — planned. Same detector, but steering / SAEs instead of raw
  scaling, to keep MMLU on bigger models. See `NHE-GenPM/plan.md`.
- **paper/** — local LaTeX drafts (gitignored, not pushed).

Shared, gitignored: `models/` (weights), `data/` (flows). Pinned env:
`requirements.txt`.

Quick start (Edge):

```bash
python -m venv .venv && .\.venv\Scripts\pip install -r requirements.txt
cd NHE-Edge
python runtime_rollback.py collect            # needs models/ (see NHE-Edge README)
python runtime_rollback.py fit_greedy
python runtime_rollback.py run africa early t90 mask m 0 0.3 5
python strict_final.py                        # re-score, no model needed
python bench_analysis.py                      # battery stats, no model needed
```

Numbers live in `NHE-Edge/results/NUMBERS.md` — the only table we quote from.
Full walkthrough: `NHE-Edge/results/experiment_report.md`. Short status:
`STATUS.md`. Plan: `ROADMAP.md`.
