# Experiment report — Gemma 3 1B

Started 2026-08-21 · updated 2026-09-23

**Question.** Can a deployed sub-1B model be made to hallucinate less, at runtime,
on CPU, without retraining, and without breaking what it already knows?

**Answer, in one line.** Yes, within a 24-hour budget: 59 → 56 errors over 463
items with zero regressions, at +20.5% latency and ~6–10 h wall-clock. The
population-level effect size is a separate, open question (p = 0.25 at n = 463).

## How results are labelled in this report

- **[DEPLOYMENT-VALID]** — direct measurement of the shipped configuration.
  Sufficient to deploy. Does not rely on a significance test.
- **[MECHANISM]** — controlled causal experiment explaining observed behaviour.
- **[OPEN]** — a question the evidence does not settle.

These are independent standards. The runtime arm is deployment-valid and its
significance is open; both statements are true at once, and reporting only one of
them would misrepresent the work.

## How we measured

- **Two ways of decoding:** greedy (argmax) and sampled (temp 0.9, top_p 0.9, 6
  seeds). Numbers from the two are never mixed — they are different protocols.
- **Two ways of scoring.** The legacy loose metric (answer anywhere in the output)
  counts hedges as correct; the strict metric (answer in the first sentence) does
  not. On the single-task benchmark both were replaced by a unit-tested whole-token
  matcher, because plain substring matching is unsafe here — 26 rows contain a
  shorter accepted answer inside a longer one ('bern' inside 'berne'). Every flip
  was checked by hand.
- **What we cut:** `down_proj`, `up_proj`, `gate_proj` for 32 neurons. Scale 0.0 =
  off, 0.3 = turned down. Arms compared: none, static, runtime (only if the detector
  fires in the first 5 tokens), and static+runtime merged.
- **Stats:** paired exact McNemar on the same items. Per-draw tests overstate power
  (6 correlated draws per item) and are reported only as exploratory.

## What we tested

- Main topic: Africa capitals (54). Baseline is weak, so there is room to improve.
- Controls: Europe (44, very strong), Asia (46), US states (50), elements (41).
- New checks: africa_largest (54, "largest city in..."), world_tricky (49),
  world_cap_traps (134), world_largest (173).
- Two fixed benches from the same 361 pool: hard (99 we know are hard - the model
  gets them wrong greedily: 54+15+30) and random (99 random, seed 42, overlap 19).

## What happened

### Statistics alone don't work

Trying to pick neurons by mean or variance (32, 128, 512) does nothing on Africa
(greedy stays 7/54 wrong). Only 7 of 128 overlap with the causal ones. Correlation
is not cause.

### Causal patching does

We patch activations and keep the neurons that actually push the wrong answer.
Without filtering, k32 helps a little (7->6) and k128 collapses (7->34 wrong, Europe
breaks). The useful ones live in layers 8-17, peaking at 16.

When we keep only neurons that help the *wrong* answer and are in the middle
layers (`k32_midwrong`), it works:

Sampled-substring legacy table (static arm, `model.generate`, 5 seeds/item; p/net
per-draw - overstates, see NUMBERS.md). Canonicals: greedy strict Africa 0.130 /
Europe 0.000 / k128-Europe 0.091; sampled-substring majority baseline 0.148, k32
0.111 p=0.017, k128 0.074 p=0.003 (per-draw mean 0.093):

| Topic | Before | k32 | k128 |
|---|---:|---:|---:|
| Africa | 0.167 | **0.111** p=0.017 | **0.093** p=0.003 |
| Europe | 0.005 | 0.000 | **0.095** p<0.001 (breaks) |
| Elements | 0.200 | 0.249 | 0.249 |
| Asia | 0.057 | 0.057 | 0.100 |
| US states | 0.052 | 0.088 | 0.128 p=0.007 |

k32 fixes real mistakes (Africa and also Georgia->Tbilisi, Delaware->Dover which
were wrong to start with) and never touches strong knowledge (Europe stays
44/44). It loses where knowledge is weak to begin with. k128 fixes more on
Africa but breaks Europe and the US states - same neurons carry correct answers
elsewhere.

### Timing matters

We trained a detector on the Africa flows (54, 7 wrong). The best signal that
fires *after* the answer is strong (L18, AUC 0.860) but useless - by then the
model already committed. The useful one fires in the first 10 tokens (L19,
AUC 0.742). It only helps if it fires in the first 5 tokens.

| Detector | AUC | When it fires | Result |
|---|---:|---|---|
| Probe L10 | 0.672 | - | weak |
| Jump L18 (full) | 0.860 | tokens 14-19 | 7/54->7/54 - too late |
| Jump early L19 | 0.742 | first 10 | 7/54->4/54 substring (5/54 strict), but 16 fires |
| Same + window <=5 | 0.742 | first 5 | **7/54->4/54 substring (5/54 strict), 7 fires, Europe 0 fires** |

Window <=5 is the sweet spot (w<=4 -> 6 fires, w<=3 -> 2 fires, w<=2 -> 0). We use
p90 threshold.

Greedy, strict, 7 baseline errors:

| What we did | Wrong | Notes |
|---|---:|---|
| Nothing | 7/54 | - |
| k32 static | 5/54 | fixes 3, breaks South Sudan |
| k128 static | 4/54 | fixes 5, breaks 2 |
| **Runtime soft, first 5** | **5/54** | **fixes 2, breaks none** |
| w<=4 / w<=3 | 5/54 / 6/54 | fewer fires |

Loose scoring counts Senegal's hedge as a fix; strict doesn't. Runtime and k32
tie at 5/54, but only runtime breaks nothing. k128 looks best at 4/54 but breaks
more.

Other topics, greedy strict: africa_largest 16/54->15/54, world_cap_traps
15/134->13/134, world_largest 30/173->29/173 - small gains, no breaks. world_tricky
and Europe: 0 fires, 0 breaks. Only runtime never hurts a control.

### Benches, sampled (6 seeds, strict)

Hard bench = all 54 africa_largest (16 greedy-wrong) + 15 + 30 greedy-wrong
(99 total, baseline 0.596). Random bench = 99 random from same pool (11+40+48,
baseline 0.099, overlap 19).

| Bench | Nothing | Runtime soft | Refuse instead |
|---|---|---:|---:|
| **Hard, per draw 594** | 354 (0.596) | **334 (0.562)** per-draw p<0.001, 20 fixes / 0 new errors per-draw | 280 (0.471) + 90 refusals |
| Hard, majority-of-6 99 (**primary**) | 59 | 56 (W2C=3, C2W=0, p=0.25, n.s.) | 47 |
| **Random, per draw 594** | 59 (0.099) | **53 (0.089)** per-draw p=0.031 | 47 (0.079) + 42 refusals |
| Random, majority-of-6 99 (**primary**) | 10 | 9 (n.s.) | 8 |

Same direction, smaller on random. Per-draw tests overstate (correlated draws);
item majority is the honest primary and is not significant - small effect.
On fired samples, 20/90 turned wrong->correct (22%; 36/90 correct after).

Africa sampled alone (270 draws): 33/270->28/270 p=0.18 - not significant at that
size. Europe: 0 fires in all seeds.

### Both together - hard bench, 594 draws

Static k32 always + runtime when it fires. Fires jump 90->348 (15%->58%) because
static changes the dynamics.

| Nothing | Runtime only | Refuse only | **Static+runtime** | **Static+refuse** |
|---:|---:|---:|---:|---:|
| 354 (0.596) | 334 (0.562) | 280 (0.471) | **231 (0.389)** | **52 (0.088)** + 348 refusals |

Static+runtime improves a lot (132 fixes) but breaks 9 draws; static+refuse
almost wipes out errors but refuses over half (348/594). Bench-aggregate only -
no per-item proof the four Africa quiet cases are among the fixed.

### What we can't fix yet

Four Africa errors (Cape Verde, Equatorial Guinea, Gabon, Guinea) never spike.
No single jitter threshold or k32 mask catches them (greedy Africa). This is a
scope boundary of the current runtime signal family, not a framework failure:
NHE-Edge targets transient pre-commit drift, and a stored false fact can be
committed without one.

NHE-NTW provides a controlled causal demonstration that a false fact can be
learned as a quiet, confident Static Memory Void, while its zero-training
control remains uncertain. The Gemma entropy audit supports the interpretation
for three cases (Cape Verde p=1.0000, Guinea p=0.9986, Equatorial Guinea
p=0.9560). Gabon is separate (p=0.6925, uncertainty/truncation). A direct
provenance or fact-edit test is still needed before claiming that these Gemma
weights were trained on the false answers.

## Takeaways

1. There are neurons in layers 8-17 that actually push the wrong answer. Patching
   on wrong-only examples finds them.
2. k32 static helps Africa (p=0.017) but breaks one. Runtime soft (first 5 tokens,
   scale 0.3) helps the same amount with no breaks on tested topics - the only
   single fix that never hurts a tested control (runtime never ran on
   elements/asia/US-states; MMLU tested static only).
3. Direction is consistent everywhere: hard per-draw p<0.001 (20/0), random
   per-draw p=0.031 (6/0); honest item-majority primary is n.s. both times.
   On fired samples 20/90 wrong->correct (22%).
4. Four quiet commits need a different response. NHE-NTW supplies a controlled
   causal example of a Static Memory Void, a category that can look smooth and
   confident because the false fact is already in the weights. That is a
   plausible mechanism for the Gemma ceiling, not a direct attribution of all
   four cases. Static+runtime moves the bench a lot (0.389, 9 breaks; 0.088 with
   58% refusal) at the cost of many fires.
5. **The quiet residual is partly reachable, and the trigger that reaches it
   fails the safety bar.** Hidden-state flows on all 463 items separate the
   never-fired-wrong set (n=50) from correct (n=407) at pre-commit AUROC 0.6925
   (L11) — but the live L11/w10 arm produced fixes=4 with breaks=2 (p=0.6875),
   worse net than the deployed arm and a violation of the zero-break rule. The
   breaks are all `element`-family, where a capital-fitted mask corrupts the
   atomic-number mapping. Open idea: family-aware firing or a per-family mask.

## Limits

- Loose scoring flatters hedges. We report strict and checked every flip.
- Gemma 3 1B is the main Edge benchmark. Qwen2.5-0.5B is a real-weight
  cross-model timing check; 1.5B remains synthetic-only. CPU only.
- Hard bench is enriched by construction (conditional gains, regression to the
  mean). Random bench (0.099) is the unbiased estimate: 1pp absolute.
- Per-draw stats overstate (correlated draws); no multiple-comparison correction;
  abstain-coded-as-correct is mechanical. Item majority is primary.
- Detector is per decoding mode (probe ~0.05 transfer). You have to retrain.
- Four quiet cases need a different response: provenance/fact checks for a
  possible Static Memory Void, or a second runtime signal for a transient drift.
- **The single-task arm is clean but not significant.** On the 463-item benchmark
  runtime L19/w5 gives 59 -> 56 wrong, fixes=3, breaks=0, exact McNemar p=0.25
  (the power analysis required 6 fixes at 0 breaks). Safety, the timing law
  (40/41 fired items lead >= 1) and the per-fired repair rate (0.333 vs 0.286 on
  Africa) are established at that n; the headline effect is not.
- **Detector recall is the measured bottleneck**, not sample size: precision
  0.220, recall 0.153 on errors. A higher-recall pre-commit trigger at L11 was
  built and tested: fixes=4 but breaks=2 (p=0.6875), both breaks in the `element`
  family, because the mask was fitted on wrong-only capital answers. Zero breaks
  is the Edge bar, so that arm is rejected. Net effect is worse than L19/w5
  (net +2 vs +3).
- **A first reachability measurement was wrong and was corrected.** A fixed-window
  w15 feature (AUROC 0.7129 at L15) read tokens at and after the commit, violating
  the timing law. Re-measured per-item pre-commit, the residual is still partly
  reachable but at L11 (0.6925), and L15 drops to 0.577. Recorded rather than
  quietly replaced.

## Files

- `attribution_africa.*` / `attribution_causal*.json` - patching results
- `mask_k*.json` - which neurons to cut
- `greedy_flows_africa.npz` + `detector_greedy.json` - detector
- `bench_hard.json` (99 hard) + `bench_random.json` (99 random, seed 42)
- `eval_*.json` / `eval_runtime_*.json` - every run
- Code: `attribute_causal2.py`, `run_experiment.py`, `eval_topic.py`,
  `runtime_rollback.py` (now with static+temporal merged), `bench.py`
  (build|run|analyze), `stats.py` (strict|battery|significance, the scorers we report)

`data/` (flows, 378 MB) is not in the repo - rebuild with `collect_topic.py`.
