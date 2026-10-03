# legacy/ - superseded scripts, kept for history

Every script here has been replaced by a consolidated tool. They are kept so the
history of the experiments stays inspectable, and they are NOT imported by any
live code path (verified: nothing under `core/`, `analysis/` or `experiments/`
imports from this directory).

**Do not use these for new results.** If a number in the project depends on one of
these scripts' behaviour, it is now produced by the replacement listed below.

## Replacements

| legacy script | replaced by | command |
|---|---|---|
| `bench_build.py`, `bench_random_build.py` | `experiments/bench.py` | `python experiments/bench.py build` |
| `bench_driver.py`, `bench_greedy.py`, `bench_random_greedy.py`, `bench_full_sampled.py` | `experiments/bench.py` | `python experiments/bench.py run --bench hard --mode sampled` |
| `bench_analysis.py` | `experiments/bench.py` + `analysis/stats.py` | `python experiments/bench.py analyze --bench hard` |
| `sweep_thresholds.py`, `sweep_windows.py` | `analysis/sweep.py` | `python analysis/sweep.py thresholds` |
| `verify_timing.py` | `experiments/bench.py` | `python experiments/bench.py analyze --formats` |
| `eval_mmlu_temporal.py` | `experiments/eval_mmlu.py` | `python experiments/eval_mmlu.py --temporal` |

## Diagnostic one-offs

These were point probes run once to answer a specific question. They still run and
their output files are still committed, but there is no consolidated driver and no
re-run protocol for them.

State inspection (`check_*`, `debug_*`): `check_fired.py`, `check_early.py`,
`check_largest.py`, `check_largest2.py`, `check_runtime.py`, `check_sampled.py`,
`check_sampled2.py`, `check_breaks.py`, `check_complement.py`,
`check_new_controls.py`, `debug_fire.py`, `debug_fire2.py`, `debug_gen.py`.

Scoring and significance (`strict_*`, `flip_report.py`, `significance*.py`): all
superseded by `analysis/stats.py` (strict|battery|significance) and, for the
single-task benchmark, `analysis/scoring.py` + `analysis/score_single_task.py`.

Attribution: `attribute_causal.py` and `attribute_neurons.py` are superseded by
`core/attribute_causal2.py`.

Data collection: `collect_data.py` superseded by `core/collect_topic.py`.

Other: `analyze_jitter.py` (superseded by `analysis/sweep.py` and
`analysis/gate.py`), `bench_gen.py`, `relabel_data.py`, `extract_demo.py`,
`probe_load_gguf.py`, `flip_report.py`.

There is no `eval_prompt_baseline.py` here; that experiment lives only in
`experiments/eval_prompt_baseline.py` and was never duplicated into legacy.

## Important caveat about these numbers

Scripts in this directory predate several corrections that the current numbers
depend on. In particular they use plain-substring scoring, which
`analysis/audit_single_task.py` showed is unsafe on this task (26 rows where a
shorter accepted answer is a substring of a longer one, e.g. 'bern' inside
'berne'), and they have no clean-state discipline on the model weights. Any file
whose provenance is one of these scripts should be re-scored with
`analysis/scoring.py` before it is quoted. The canonical table is
`../results/NUMBERS.md`; it is the only place numbers should be read from.
