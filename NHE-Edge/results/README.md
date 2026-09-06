# results/ - what is what

Canonical table: `NUMBERS.md` (every quoted number lives here with protocolxmetric labels).
Full walkthrough: `experiment_report.md`.

## Benches (frozen inputs)
- `bench_hard.json` - 99 items: all 54 `africa_largest` (16 greedy-wrong) + 15
  greedy-wrong `world_cap_traps` + 30 greedy-wrong `world_largest`. Built by
  `../bench.py build --bench hard`. Baseline strict 0.596.
- `bench_random.json` - 99 random items from the same 361 pool, seed 42, overlap
  19 with hard. Built by `../bench.py build --bench random` (byte-identical). Baseline 0.099.

## Detector, masks, flows
- `detector_greedy.json` - L19-early t90/t95 thresholds + full/Early AUCs (real, Gemma).
- `detector_greedy_qwen*.json` - **synthetic placeholders** (do not quote).
- `mask_k32_midwrong.json` - the 32-neuron mask used by every runtime run.
- `mask_k*.json` - other static masks (k32/k64/k128/k256/k512 x scores).
- `greedy_flows_africa.npz` (~58 MB, local-only, gitignored) - real Gemma flows for the detector; rebuild with `python runtime_rollback.py collect`.
- `greedy_flows_africa_qwen2.5-0.5b.npz` - REAL Qwen flows, local-only (22 MB exceeds usable uplink; rebuild in ~2 min via `runtime_rollback_qwen.py --model qwen2.5-0.5b collect`).
- `greedy_flows_africa_qwen2.5-1.5b.npz`, `*synthetic*.npz` - synthetic, gitignored, on disk only.
- `attribution_*.json/.npz` - AtP/statistical attributions (+ progress checkpoints).

## Evals (all recomputable via scripts)
- `eval_{topic}_baseline.json`, `eval_{topic}_{mask}.json` - greedy static (`eval_topic.py`).
- `eval_{topic}_baseline_s5.json`, `eval_{topic}_{mask}_s5.json` - sampled static, 5 draws.
- `eval_runtime_{topic}_jump_gt_L19_t90_{none,mask,abstain}[_sft0.3][_sSEED].json` - temporal arms.
- `..._static0_...` - merged static+temporal arms. `..._rand.json` - random bench.
- `mmlu_side_effect.json` - 200 real MMLU, static soft k32 (`eval_mmlu.py --use-real`).
- `quiet_diagnostic.json` - logit-lens on 4 quiet + 3 dynamic (`probe_quiet.py`).
- `jitter_report*.json`, `summary_experiment.json` - early detector/feature reports.

## Naming
`none` = detector records but never cuts. `mask` = soft x0.3 on fire (w<=5, t90).
\bstain\ = refuse on fire. \_s{seed}\ = sampled seed. No suffix = greedy.
`_static0` = static k32-hard always on (merged). `_rand` = random-bench subset.
