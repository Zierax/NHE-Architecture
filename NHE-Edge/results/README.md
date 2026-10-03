# results/ - what is what

Canonical table: `NUMBERS.md` (every quoted number lives here with protocol/metric labels).
Full walkthrough: `experiment_report.md`.

## Benches (frozen inputs)
- `bench_hard.json` - 99 items: all 54 `africa_largest` (16 greedy-wrong) + 15
  greedy-wrong `world_cap_traps` + 30 greedy-wrong `world_largest`. Built by
  `../bench.py build --bench hard`. Baseline strict 0.596.
- `bench_random.json` - 99 random items from the same 361 pool, seed 42, overlap
  19 with hard. Built by `../bench.py build --bench random` (byte-identical). Baseline 0.099.

## Detector, masks, flows
- `detector_greedy.json` - L19-early t90/t95 thresholds + full/Early AUCs (real, Gemma).
- `detector_greedy_qwen2.5-0.5b.json` - REAL Qwen detector (early L22 AUC 0.778); `detector_greedy_qwen2.5-1.5b.json` - **synthetic placeholder** (do not quote).
- `mask_k32_midwrong.json` - the 32-neuron mask used by every runtime run.
- `mask_k*.json` - other static masks (k32/k64/k128/k256/k512 x scores).
- `greedy_flows_africa.npz` (~58 MB, local-only, gitignored) - real Gemma flows for the detector; rebuild with `python runtime_rollback.py collect`.
- `greedy_flows_africa_qwen2.5-0.5b.npz` - REAL Qwen flows, local-only (22 MB exceeds usable uplink; rebuild in ~2 min via `runtime_rollback_qwen.py --model qwen2.5-0.5b collect`).
- `greedy_flows_africa_qwen2.5-1.5b.npz`, `*synthetic*.npz` - synthetic, gitignored, on disk only.
- `attribution_*.json/.npz` - AtP/statistical attributions (+ progress checkpoints).

## Evals (all recomputable via scripts)
- `eval_{topic}_baseline.json`, `eval_{topic}_{mask}.json` - greedy static (`eval_topic.py`).
- `eval_{topic}_baseline_s5.json`, `eval_{topic}_{mask}_s5.json` - sampled static, 5 draws.
`abstain` = refuse on fire. `_s{seed}` = sampled seed. No suffix = greedy.
- `..._static0_...` - merged static+temporal arms. `..._rand.json` - random bench.
- `mmlu_side_effect.json` - 200 real MMLU, static soft k32 (`eval_mmlu.py --use-real`).
- `quiet_diagnostic.json` - logit-lens on 4 quiet + 3 dynamic (`probe_quiet.py`).
- `entropy_audit.json` - answer-commit confidence/entropy for the quiet cases;
  three are highly confident at commit, while Gabon is a separate uncertain case.
- `applicability_gate_{model}.json` - per-model ACTIVE/INERT verdict from a joint
  (layer x window) pre-commit search with an edge-effect control
  (`analysis/gate.py`). One file per model so a single-model rerun cannot
  truncate a shared report.
- `applicability_gate_sweep.json` - both models in one report
  (`gate.py --sweep-models`), for reading them side by side.
- `gate_live_{model}.json` - result of running the gate's winning cell live,
  used to show that pre-commit AUROC does not pick the best intervention.
- `onboarding_{model}.json` - full onboarding state-machine record: stages,
  timings, gate verdict, live run, strict W2C/C2W, terminal verdict
  (`analysis/nhe_onboard.py`).
- `bench_single_task.json` - the merged 463-item single-task benchmark (one task,
  three question frames) with the union truth layer
  (`analysis/build_single_task.py`, deterministic, `--verify`).
- `bench_single_task_provenance.json` - every cross-topic duplicate and every
  answer-set widening, with the reason. 11 questions had disjoint truths across
  topics; this file is why that is defensible rather than accidental.
- `eval_single_task_{none_full,mask_L19_w5}.json` - the paired single-task runs
  (463 items each). Runtime L19/w5: 59->56 wrong, fixes=3 breaks=0, p=0.25.
- `single_task_score.json` - paired exact McNemar, Wilson CI, per-family, and the
  landing split (89.3% of residual error never fires).
- `power_single_task.json` - what significance requires (6 fixes / 0 breaks),
  computed before the run.
- `single_task_audit.json` - truth-layer audit of topics.py (duplicates,
  contradictions, substring hazards).
- `answer_set_gaps.json` - measured residual exposure of the token-boundary metric
  (19 structurally exposed items, 9 where the model emits the longer form).
- `detector_layer_justification.json` - why L19 is causally adjacent to the
  attribution band, and why that is not a unique selection.
- `reachability_of_quiet.json` - stage-1 reachability (fixed window, AUROC 0.7129
  at L15/w15). SUPERSEDED: that window reads at/after the commit.
- `reachability_precommit.json` - stage-2 reachability using the per-item
  pre-commit maximum, the only feature the timing law permits: best L11 at
  0.6925; L15 falls to 0.577. This is the correct measurement.
- `precommit_arm_dryrun.json` - offline estimate for the L11 arm before spending
  CPU on it: recall 0.286, ~5 expected fixes.
- `eval_single_task_mask_L11_w10.json` + `single_task_score_L11.json` - the live
  higher-recall arm. fixes=4, breaks=2, p=0.6875: more fixes than the deployed arm
  but it breaks the zero-break guarantee, both breaks in the `element` family.
  Rejected on evidence.
- `greedy_flows_single_task.json` - per-layer hidden-state jumps for all 463
  single-task items (`core/collect_single_task.py`, resumable). Independent
  collection reproduces the baseline's 59 wrong, which cross-validates the run.
  Gitignored (regenerable); the derived verdicts are committed.
- `jitter_report*.json`, `summary_experiment.json` - early detector/feature reports.

## Naming
`none` = detector records but never cuts. `mask` = soft x0.3 on fire (w<=5, t90).
`abstain` = refuse on fire. `_s{seed}` = sampled seed. No suffix = greedy.
`_static0` = static k32-hard always on (merged). `_rand` = random-bench subset.
