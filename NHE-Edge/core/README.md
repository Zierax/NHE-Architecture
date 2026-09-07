# core/ — the NHE-Edge engine

Importable + runnable modules. Everything here resolves paths from its own
location (`EDGE_ROOT` = parent dir), so run from anywhere:

- `topics.py` — frozen question sets (no paths, pure data)
- `runtime_rollback.py` — Gemma collect / fit / run (temporal + merged + abstain)
- `runtime_rollback_qwen.py` — same pipeline for Qwen2.5 (registry + ChatML)
- `attribute_causal2.py` — AtP patching, Gemma k32 mask
- `attribute_causal2_qwen.py` — same for Qwen (mid band 10-20)
- `collect_topic.py` — rebuild `data/` flows
- `run_experiment.py` — masks from attributions + static evals
- `eval_topic.py` — greedy / sampled eval, positional mask path

Usage: `python core/runtime_rollback.py run africa early t90 mask m 0 0.3 5`
(weights in repo-root `models/`, outputs in `../results/`).
