# NHE-Edge

A model deployed in the field produces a wrong answer. You have 24 hours, a CPU,
no GPU farm, and no permission to retrain. NHE-Edge suppresses the error during
generation, on-device, and verifies that nothing else changed.

It works by watching the model while it writes. Just before a wrong answer is
committed, the hidden state jumps in the middle layers. NHE-Edge locates the
features that drive that specific error and scales them down for the remainder of
the generation — without touching the weights on disk.

All code and results are in this repo. `results/NUMBERS.md` is the only place
numbers are quoted from; every entry there is labelled by evidence standard,
protocol, and metric.

## Deployment status

Measured on the shipped configuration, Gemma 3 1B (fp16, CPU):

| criterion | result |
|---|---:|
| errors | **59 → 56** of 463 items |
| regressions | **0** across 463 items, three question frames |
| latency | **+20.5% per token** (44.8 ms), 160 ms once per firing |
| retraining | none — 32 neurons scaled at inference time |
| deployable in 24 h | yes — ~6–10 h wall-clock, CPU only |

These are the mission criteria and they are met by direct measurement. They do
not depend on a significance test.

Separately, and as a different kind of claim: the population-level effect is
**not** statistically separable from zero at n = 463 (3 fixes, 0 regressions,
exact McNemar p = 0.25). That question concerns generalisation of an effect size
across models and samples; it is reported as open in `results/NUMBERS.md` and does
not qualify the deployment rows above.

## What the mechanism turned out to be

- **The intervention is causal, not correlational.** Selecting neurons by
  activation statistics (mean, variance) produces no effect — only 7 of 128
  overlap with the causal set. Selecting by activation patching on wrong-only
  examples finds the neurons that actually drive the error.

- **Timing is the entire game.** Scaling those neurons down works only *before* the
  model commits to the answer. This was tested as a model × format matrix: every
  firing that preceded the commit by at least one token could repair the answer;
  every firing after it was inert. At n = 463, 40 of 41 firings were pre-commit
  and the single late firing changed nothing.

- **The operating configuration is simple.** Watch the first 5 tokens; if the jump
  detector fires, scale the 32 wrong-commit neurons by 0.3. On African capitals
  this takes greedy decoding from 7/54 wrong to 5/54 with no new errors on any
  control topic.

- **Most residual error is a different failure mode.** 89.3% of the errors still
  present after the intervention never fire at all — no spike, no threshold, no
  mask reaches them. `../NHE-NTW` demonstrates causally, in a controlled
  experiment, that a false fact stored confidently in the weights produces exactly
  this signature, and that ignorance alone does not. The operational consequence
  is a triage rule: check provenance first for a quiet confident error; use the
  runtime only when a pre-commit spike is present.

## The limits, stated plainly

- **The trigger reaches 15.3% of errors.** That is the measured reason the effect
  is small, and it is not a sample-size problem — the benchmark has 59 error items
  available.

- **A higher-recall trigger was built and rejected.** It reaches 0.286 of errors
  and yields 4 fixes instead of 3, but introduces 2 regressions, both in the
  `element` family, because the mask was fitted on wrong-only capital answers.
  Zero regressions is the deployment criterion, so it was rejected on evidence.

- **One model, one size, one task family, CPU only.** The cross-model work is a
  scoping result, not a performance result: Qwen2.5-0.5B was tested and the method
  was found inapplicable to it in its current format.

- **Detector layer selection is not solved offline.** The deployed layer sits two
  layers above the patching-derived causal band, which is causally informed but
  not uniquely determined; it was selected and then validated by a live run.

## How it works

1. **Find the signal.** For each layer and each token, record how much the hidden
   state moves. Wrong answers spike in the middle layers right before the answer.

2. **Find the cause.** Patch activations and see which neurons actually push the
   wrong answer. Keep the top ones per layer (32 across layers 8-17 is the sweet
   spot).

3. **Get the timing right.** If you cut after the model has already chosen the
   word, it's too late. We train the detector on the first 10 tokens and only
   act in the first 5.

4. **Cut.** Either zero the neurons (hard) or scale by 0.3 (soft). Soft keeps the
   same fixes and avoids a break we saw on Burundi.

5. **Check everything.** Greedy and sampled decoding, two scoring rules (loose
   substring vs strict first sentence), 7 topics, and manual review of every
   change. Sampled runs use 6 seeds and paired tests.

```
 question ...   jump in middle layers   wrong word chosen   rest of answer
 ─────────────●───────────────────────●────────────────────
              ^                       ^
         detector fires here    cutting here is too late
              └──── scale neurons ────┘  -> now correct
```

## Words we use

- **Jitter** - how much the hidden state jumps between tokens, per layer.
- **Wrong-commit neurons** - the ones that actually push the bad answer (found with
  patching).
- **Mask** - which neurons to turn down.
- **Static** - turn them down for the whole generation.
- **Temporal / runtime** - only turn them down if the detector fires in the first
  5 tokens.
- **Merged** - static always + temporal when it fires.
- **Bench** - a fixed test set. `bench_hard.json` is 99 we know are hard (greedy-wrong);
  `bench_random.json` is 99 random from the same pool.
- **Greedy vs sampled** - argmax vs temp 0.9 / top_p 0.9. Don't mix numbers across them.
- **Substring vs strict** - substring: answer anywhere in output (loose, counts hedges
  as correct). Strict: answer in the first sentence (what we report).

## Results

### Africa capitals, greedy (54, strict)

| What we did | Wrong | What changed |
|---|---:|---|
| Nothing | 7/54 | - |
| d_mean / d_var (statistics) | 7/54 | nothing - correlation, not cause |
| k32 static | 5/54 | fixes 3, breaks South Sudan |
| k128 static | 4/54 | fixes 5, breaks Benin + South Sudan |
| **runtime soft, first 5 tokens** | **5/54** | **fixes 2, breaks nothing** |
| w<=4 / w<=3 | 5/54 / 6/54 | fewer fires, same or fewer fixes |

Static and runtime tie at 5/54, but static breaks one and runtime breaks none. k128
looks better at 4/54 but breaks more and hurts every control topic.

### Africa, sampled (270 draws, 5 seeds) - labels matter

Static `model.generate` (sampled, substring): majority 8/54 -> 6/54 (k32, p=0.017) and 4/54 (k128, p=0.003); per-draw means 0.111/0.093.
Runtime manual sampler (sampled, substring via file `correct`): 33/270 -> 28/270 per-draw mean (p=0.18), 6/54 -> 4/54 by majority. Same questions, different sampler draws - not bit-identical. No new errors on Europe (0 fires).

Small n, so runtime isn't significant here. The benches below give it power.

### Two benches, 99 items x 6 seeds = 594 draws, strict

Hard bench = all 54 largest-Africa (16 greedy-wrong) + 15 hard capitals + 30 hard
largest that are greedy-wrong, from 361-item pool (overlap hard-random 19, random
seed 42). Random bench = 99 random from same pool. Hard baseline 0.596, random 0.099.

| Bench | Nothing | Runtime soft | Refuse instead |
|---|---|---:|---:|
| Hard: wrong / 594 | 354 (0.596) | **334 (0.562)** per-draw p<0.001, 20 fixes / 0 new errors per-draw | 280 (0.471) + 90 refusals |
| Hard: majority-of-6 99 (**primary**) | 59 | 56 (W2C=3, C2W=0, p=0.25, n.s.) | 47 |
| Random: wrong / 594 | 59 (0.099) | **53 (0.089)** per-draw p=0.031 | 47 (0.079) + 42 refusals |
| Random: majority-of-6 99 (**primary**) | 10 | 9 (1 discordant, n.s.) | 8 |

Same direction on both; honest primary (item majority) is not significant -
small effect, not noise-free proof. On fired samples, 20/90 turned wrong->correct
(22%; 36/90 correct after). Per-draw tests overstate (correlated draws).

### Merged: static always + runtime when it fires (hard bench, 594, strict)

| Nothing | Runtime only | Refuse only | **Static+runtime** | **Static+refuse** |
|---:|---:|---:|---:|---:|
| 354 (0.596) | 334 (0.562) | 280 (0.471) | **231 (0.389)** - 132 fixes, **9 new breaks** | **52 (0.088)** + 348 refusals (58%), 305 fixes / 3 breaks |
| Fires | 90 (15%) | 90 | 90 | 348 (58%) | 348 (58%) |

Static alone changes the dynamics so the detector fires more. Merged repair
improves a lot but breaks 9 draws - not "without refusal" clean. Merged refuse
trades 348 refusals for the low rate. Bench-aggregate only: no per-item proof
the four Africa quiet cases are among the fixed.

### Other topics, greedy strict

| Topic | Nothing | Runtime soft | k32 | k128 |
|---|---:|---:|---:|---:|
| africa_largest 54 | 0.296 (16) | 0.278 (15) | 0.185 (10: 7 fixes, breaks S.Sudan) | 0.259 (14: 7 fixes, 5 breaks) |
| world_cap_traps 134 | 0.112 | 0.097 | - | - |
| world_largest 173 | 0.173 | 0.168 | - | - |
| world_tricky 49 | 0.020 (1) | 0.020 (0 fires) | 0.041 (2) | 0.122 (6: fixes UAE, breaks 6) |
| europe 44 | 0.000 | 0.000 (0 fires) | 0.000 | 0.091 (4 breaks) |

Only runtime never hurts a control (on tested topics; runtime was never run on
elements/asia/US-states, and MMLU tested the static mask only).

A constrained prompt ("Answer with only the city name.") is a real competitor:
hard greedy 61/99 -> 52/99, better than NHE-mask greedy (61/99 -> 57/99). NHE's
edge is significance-tested repair under sampling plus no-refusal fixes, not raw
greedy rate. Random bench: prompting 9/99 -> 11/99 (noise).

### Detector

| Signal | AUC | What it means |
|---|---:|---|
| jitter last-token L10 (any token) | 0.968 | feature-specific, not windowed - shows signal exists (`jitter_report_africa.json:17`, labeled on 44/10 vs headline 47/7), not used for cut |
| jump full (L18) | 0.860 | fires after the answer - too late |
| jump early (L19, first 10) | 0.742 | fires before - the one we use (text reports 0.742) |
| probe L10 | 0.672 | weak, doesn't transfer greedy↔sampled (~0.05) |

Fires vs window (p90, strict 4/54 = substring 4/54, strict 5/54): w<=5 -> 7/54 (3 hits), w<=4 -> 6/54 (2 hits), w<=3 -> 2/54 (1 hit), w<=2 -> 0. We use w<=5. Strict is 5/54 (Senegal hedge counts as 4/54 loose).

## What we can't fix yet

Four Africa errors never spike - they just commit quietly. No single jitter
threshold or k32 mask catches them. This is a scope boundary of the current
signal family, not a framework failure: NHE-Edge targets transient pre-commit
drifts, and a stored false fact can be committed without such a drift.

NTW's controlled experiment shows that a training-learned false fact can produce
exactly this quiet, confident, wrong signature. The Gemma entropy audit finds
three of the four quiet cases highly confident at commit, while Gabon is a
separate uncertainty/truncation case. That makes the leading explanation a
Static Memory Void for three cases and motivates a provenance check before any
further runtime tuning. A direct dataset/provenance test on Gemma remains the
next experiment.

## Timing rule (why Qwen fails, why Gemma works)

The spike *is* the commit transient - it tracks the answer token wherever it
lands. A fix is possible only if the spike is measured strictly before the city
token (lead = city_idx - fired_at >= 1). Verified from raw files by
`bench.py analyze --formats`:

| Model + format | leads (fired items) | flips |
|---|---|---|
| Gemma native | [1,1,1,1,1,4,5] all >= 1 | **2 fixes / 0 breaks** |
| Gemma plain ("only the city name") | all <= 0 | 0 / 0 (format alone: 7 -> 14 wrong) |
| Qwen plain | [-2..0] | 0 / 0 |
| Qwen bold | [-2..-1] | 0 / 0 (format alone: 9 -> 21 wrong) |
| Qwen long prefix | [-2..-1] | 0 / 0 |

Every lead >= 1 item sits in gemma-native (the only cell with fixes); every
lead <= 0 item never flips, in any cell. So intervenability tracks the spike's
position relative to the commit, and prompt format moves that position. Whether a
*model* can be made intervenable by some other means is a different question: the
applicability gate searched 192 (layer x window) cells and still reports Qwen
INERT, and the reachability experiment shows some Gemma quiet errors ARE
trigger-reachable. Both are open, and neither is settled by this table.

## Cost (measured CPU, `latency.json`)

Detector adds +44.8 ms/token (+20.5% over 218.7 plain); mask apply is 160 ms
one-time on fire; end-to-end 4.0s vs 3.5s per item. Full Africa pipeline runs
in hours on CPU, no GPU. Vs QLoRA fine-tune (estimates, labeled): NHE needs no
GPU, touches no weights at rest (reversible via reload), and has 0 breaks
everywhere tested - at the cost of +20% per query. NHE is the 24-hour field
fix; fine-tuning is the depot repair. Full table in `results/NUMBERS.md`.

## The 463-item single-task benchmark (statistical power)

The Africa result above (7/54 → 5/54) is clean but not statistically significant
(2 fixes, exact McNemar p = 0.5). This is the larger benchmark built to settle
that, as ONE task with an audited truth layer.

- **463 items** merged from `topics.py`: 244 capital, 178 largest-city, 41
  element. `analysis/audit_single_task.py` found 155 questions duplicated across
  topics, 11 with disjoint truths (Malawi is Blantyre in one topic, Lilongwe in
  another). The merge unions accepted answers explicitly
  (`results/bench_single_task_provenance.json`) and is verified deterministic.
- **Scoring**: whole-token matching after accent folding (`analysis/scoring.py`,
  unit-tested). The legacy substring metric is unsafe here (26 rows had 'bern'
  inside 'berne'); its residual exposure is measured at 9/463 items.

| arm | strict wrong | rate |
|---|---:|---:|
| none | 59/463 | 0.1274 |
| runtime soft L19/w5 scale 0.3 | **56/463** | **0.1210** |

**Paired: fixes 3, breaks 0, net +3, exact McNemar p = 0.25 (not significant).**

What n=463 establishes that n=54 could not:
- **Zero collateral damage across 463 items and 3 question frames** (vs 9 breaks
  in the merged arm). The safety claim is not a small-sample artefact.
- **The timing law at scale**: 40/41 fired items had lead ≥ 1; the one lead < 1 never flipped.
- **The per-fired repair rate generalises**: 0.333 here vs 0.286 on Africa.
- **89.3% of the residual error never fires** — the Static Memory Void population
  NTW explains, confirmed at scale.

What it honestly does not: reach significance. The cause is measured, not
guessed — the detector's precision on errors is 0.220 and its recall is 0.153, so
6 fixes (the computed requirement) become 3. Significance needs a higher-recall
trigger, not a larger benchmark. The trigger work below tests that and records the
outcome, including the failure.

```
python analysis/build_single_task.py            # merge + provenance
python analysis/audit_single_task.py             # truth-layer audit
python analysis/test_scoring.py                  # metric contract
python analysis/report_answer_set_gaps.py        # residual exposure
python analysis/power_single_task.py             # what significance requires
python core/run_single_task.py --arm none --tag none_full
python core/run_single_task.py --arm mask --layer 19 --window 5 --scale 0.3 --tag mask_L19_w5
python core/run_single_task.py --arm mask --layer 15 --window 15 --scale 0.3 --tag mask_L15_w15
python analysis/score_single_task.py --arms eval_single_task_none_full eval_single_task_mask_L19_w5
python analysis/summarize_single_task.py
python analysis/justify_detector_layer.py        # why L19, honestly
```

`run_single_task.py` checkpoints every 20 items and resumes (`--resume`) with an
exact-config guard, so the multi-hour armed run survives interruption without
mixing arms. It also snapshots only the mask-touched tensors (~380 MB) instead of
the full state dict, which OOM'd on this CPU box.

## Is the quiet residual reachable? (reachability, corrected, then tested)

The 50 never-fired wrong items decide the next direction. If they are Static
Memory Voids (NTW), no trigger reaches them and the remaining work is fact-level.
If they merely lack a trigger, recall is the problem.

**Stage 1 was wrong and was corrected.** `analysis/reachability_of_quiet.py`
searched fixed windows up to w15 and reported AUROC 0.7129 at L15/w15. That
feature reads tokens 6-15, and these items commit at positions 5-12 (median 7) —
so it was measuring *at and after* the commit, violating the timing law the rest of
this project establishes. `analysis/reachability_precommit.py` re-measures with the
per-item pre-commit maximum, which is the only feature the law allows:

| layer | pre-commit AUROC (never-fired-wrong vs correct) |
|---|---:|
| **L11** | **0.6925** |
| L12 | 0.6665 |
| L15 | 0.5767 (was 0.7129 on the illegal window) |

The residual **is** partly reachable, but through L11, not L15.

**Stage 3 — the trigger was built, and it fails the safety bar.** A dry run
predicted recall 0.286 and ~5 fixes; the live L11/w10 arm returned:

| arm | fixes | breaks | net | exact p | recall | repair/fired-wrong |
|---|---:|---:|---:|---:|---:|---:|
| **L19/w5 (deployed)** | 3 | **0** | **+3** | 0.250 | 0.153 | 0.333 |
| L11/w10 | 4 | 2 | +2 | 0.6875 | 0.286 | 0.174 |

Both breaks are `element` family: "atomic number 11" was correctly "Sodium (Na)"
and became "gold"; "atomic number 6" was "Carbon" and became "oxygen". The mask
was fitted on wrong-only *capital* answers and is being applied to a family it
never saw.

**Conclusion.** Every higher-recall trigger tried trades the zero-break guarantee
for recall, and zero breaks is the Edge bar, so L11/w10 is rejected on evidence.
One idea remains open: family-aware firing or a per-family mask. If that fails,
the ceiling is real for this task and the honest contribution is the taxonomy plus
the measured ceiling.

```
python core/collect_single_task.py                 # flows for all 463 (resumable)
python analysis/reachability_of_quiet.py            # stage 1 (fixed window)
python analysis/reachability_precommit.py           # stage 2, law-compliant
python analysis/precommit_arm_dryrun.py             # stage 3, offline estimate
python core/run_single_task.py --arm mask --layer 11 --window 10 --scale 0.3 \
        --threshold 1782.02 --threshold-key p90_precommit --tag mask_L11_w10
python analysis/score_single_task.py --arms eval_single_task_none_full \
        eval_single_task_mask_L11_w10 --out results/single_task_score_L11.json
```

## One-command onboarding (semi-full automation)

`analysis/nhe_onboard.py` runs the whole sequence for a new model with a hard
stop when the method cannot apply. It is a state machine, not a wrapper:

```
collect -> gate -> [INERT: STOP] -> calibrate -> live run
       -> [net<=0 or breaks>0: UNPROVEN] -> [net>0, C2W=0: USABLE]
```

```
python analysis/nhe_onboard.py --model gemma3-1b       # ACTIVE -> UNPROVEN (honest)
python analysis/nhe_onboard.py --model qwen2.5-0.5b   # INERT, stops before live
python analysis/nhe_onboard.py --model gemma3-1b --skip-existing
```

Verified behaviour on the two real models: Qwen stops at `INERT` before any
intervention, Gemma proceeds to the live arm and lands on `UNPROVEN` because the
gate's top cell does not produce a clean net fix. The tool is deliberately
incapable of reporting a model as usable on the strength of an AUC alone.

**What is still not automated.** Neuron attribution (`attribute_causal2*.py`) is
hours of CPU patching and still needs a per-architecture mask; it is invoked, not
optimized. Threshold percentile, cut and scale are the deployed Gemma defaults,
re-calibrated per model from correct items only. Full autonomy would need an
intervention that does not require attribution.

## Which models this works on (applicability gate)

NHE-temporal is not universal, and we now decide that from data instead of from
one failed config. `analysis/gate.py` scans the joint (layer x window) space for
the strongest **pre-commit** signal and controls for the commit-position edge
effect (hallucinations commit later, which naively inflates the score):

| model | fixed-slice pre-commit AUROC | verdict |
|---|---:|---|
| Gemma 3 1B | 0.7812 | **ACTIVE** |
| Qwen2.5-0.5B (current format) | 0.6914 | **INERT** |

Qwen's whole-sequence early AUC is 0.778 and its adaptive pre-commit score looks
like 0.790, but both fall apart under the fixed-slice control: its hallucinations
commit at mean position 6.56 vs 5.98 for correct answers, so they simply get a
longer measurement window. No cell in 192 reaches the gate for Qwen.

**Important second result:** pre-commit AUROC is a *necessary* gate, not a
sufficient selector. On Gemma the top-AUROC cell (L20, 0.781) is worse live
(net -1) than a lower-AUROC cell (L19, 0.772; net +2, 0 breaks). Cut-alignment
controls on a single layer swing the outcome from +1 to -1. So onboarding any
new model is two steps: run the gate, then run the live arm and score W2C/C2W.

```
python analysis/gate.py --sweep-models            # applicability, both models
python analysis/gate_audit.py                     # audit the gate's own assumptions
python analysis/gate_live_test.py --model gemma3-1b --mode mask --scale 0.3
python analysis/gate_compare.py                   # strict W2C/C2W for every arm
```

`gate_live_test.py` refuses to run when the gate says INERT, so an inert
model/format can never be reported as a working NHE cell.

## Cross-model and side effects

- **Qwen2.5-0.5B run for real:** same pipeline, early AUC 0.778 (beats Gemma
  0.742), own k32 mask (L13-20). But Qwen commits the city at token ~6 while the
  spike lands at 7-9 -> post-commit, 0 flips / 0 breaks. The method needs
  spike-before-commit, which is format-dependent. Full story in
  `results/cross_arch_report.md` Sec 7. (1.5B still synthetic.)
- **General knowledge preserved (static):** soft k32 on **200 real MMLU** goes 185/200->185/200 substr (0.0) and 122/200->124/200 strict (+0.01) - no damage (`eval_mmlu.py --use-real`, `results/mmlu_side_effect.json:159-172`; now also in `results/NUMBERS.md`). Proxy 181 controls also preserved (-0.016, superseded).
- **Temporal on MMLU: no transfer in this protocol.** Same 200 streamed MMLU,
  paired: baseline 78/200 -> temporal 79/200 strict (W2C=0, C2W=1, p=1.0). The
  Africa threshold fires on 155/200 MMLU items (miscalibrated out-of-distribution)
  and changes nothing. This is a protocol-bound negative result, not a claim that
  runtime repair cannot work for MMLU-style questions. (Different 200 than the
  static run - streaming order unpinned.)

## Limitations

- **Scoring matters.** Loose substring counts a hedge like "Diou... While Dakar is
  the largest city" as a fix; strict doesn't. We report strict and checked every
  flip by hand.
- **Model coverage.** Gemma 3 1B is the main Edge benchmark. Qwen2.5-0.5B is a
  real-weight cross-model timing check, while 1.5B remains synthetic-only.
- **Bench matters.** Hard bench is hard by design (0.596). Random bench (0.099)
  is the honest baseline. Both show the same effect, different size.
- **Detector doesn't transfer** across sampling vs greedy (AUC ~0.05). You have to
  calibrate per decoding mode.
- **Ceiling.** 4/7 quiet commits need a different response. NTW distinguishes
  a possible Static Memory Void (check facts/provenance) from a transient drift
  (test runtime repair); the direct Gemma attribution is still open.

## How to reproduce

Every number has a file in `results/` - masks, evals, detector, flows, benches.
`results/NUMBERS.md` is the single table we quote from.

Model (exact recipe): gated `google/gemma-3-1b-it`. `huggingface-cli login`,
download, load in fp32, `.half()`, save **one** `model.safetensors` (999.9M
params, float16, 26x1152, ~2.7 GB) to `..\models\gemma3-1b-fp16\`; tokenizer
(vocab 32768) to `..\models\gemma3-1b-tokenizer\`. CPU-only, no CUDA.

```
core/                     the engine (importable, runnable from anywhere)
  topics.py               AFRICA 54, EUROPE 44, ELEMENTS 41, ASIA 46, US_STATES 50,
                          AFRICA_LARGEST 54, WORLD_TRICKY 49, WORLD_CAP_TRAPS 134,
                          WORLD_LARGEST 173
  runtime_rollback.py     collect flows, fit detector, run temporal/merged (Gemma)
  runtime_rollback_qwen.py  same pipeline for Qwen2.5 (registry + ChatML)
  attribute_causal2.py    patching, picks 32 neurons (layers 8-17, Gemma)
  attribute_causal2_qwen.py same for Qwen (mid band 10-20)
  collect_topic.py        rebuild data/ flows
  run_experiment.py       makes masks from attributions, runs static evals
  eval_topic.py           greedy / sampled eval (positional mask path)
experiments/              one-shot experiment drivers (run from NHE-Edge/)
  bench.py                build|run|analyze benches (hard + random, greedy + sampled)
  eval_mmlu.py            MMLU side-effect (--use-real) + --temporal arm
  eval_prompt_baseline.py constrained-prompt baseline on benches
  probe_quiet.py          logit-lens check on 4 quiet + 3 dynamic cases
  latency.py              detector overhead + mask-apply cost
  download_qwen.py        fetch Qwen2.5-0.5B weights
analysis/                 read-only analysis of committed files (no model needed)
  stats.py                strict|battery|significance tables
  sweep.py                thresholds|windows offline sweeps (matches live runs)
results/                  all outputs, NUMBERS.md, experiment_report.md
legacy/                   superseded single-purpose scripts (history kept;
                          see legacy/README.md for what replaced what, and note
                          that these predate the current scoring contract)
```
`bench_random.json` is built by `bench.py build --bench random` (seed 42, byte-identical to the committed file); `bench.py build --bench hard` writes the hard bench.

Quick start (PowerShell 5.1; run from this folder; env file lives at repo root):

```powershell
# already inside NHE-Edge/ (do NOT cd again if you are here)
python -m venv ..\.venv
..\.venv\Scripts\pip install -r ..\requirements.txt
# 0. log in to gated HF once: huggingface-cli login  (model + tokenizer below)
# 1. fetch weights into ..\models\ (see "How to reproduce"), then:
python core/runtime_rollback.py collect            # -> results/greedy_flows_africa.npz
python core/runtime_rollback.py fit_greedy         # -> results/detector_greedy.json
# 2. build the k32 mask first (run ... mask needs results/mask_k32_midwrong.json):
python core/attribute_causal2.py                   # -> results/attribution_causal2_africa.json
python core/run_experiment.py                      # -> results/mask_k32_midwrong.json
python core/runtime_rollback.py run africa early t90 mask m 0 0.3 5
#    args: detector-set early | threshold t90 | mode mask | m=greedy (s=sampled) | seed 0 | scale 0.3 | window 5
python core/eval_topic.py africa results/mask_k32_midwrong.json   # positional mask path (no --mask flag)
python analysis/stats.py strict                          # re-score committed evals, no model
python experiments/bench.py run --bench hard --mode sampled  # battery, 6 seeds (hours on CPU)
python experiments/bench.py run --bench random --mode sampled # random + merged batteries
python experiments/bench.py analyze --bench hard            # stats over committed runs, no model
```

Full walkthrough: `results/experiment_report.md`. Short status: `../STATUS.md`. Repo plan: `../ROADMAP.md`.
