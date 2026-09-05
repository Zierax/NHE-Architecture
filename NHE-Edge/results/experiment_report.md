# What we did — Gemma 3 1B

Date: 2026-08-21

We wanted to know if we can make the model hallucinate less about African
capitals by turning down a few specific neurons, without breaking what it knows
about other topics.

## How we measured

- **Two ways of decoding:** greedy (always pick the best token) and sampled
  (temp 0.9, top_p 0.9, 6 different seeds). We never mix numbers from the two —
  they are different setups.

- **Two ways of scoring:** loose (is the answer anywhere in the output?) and strict
  (is it in the first sentence?). Loose counts a hedge like
  "Diou… While Dakar is the largest city" as correct; strict doesn't. We report
  strict. Every flip was checked by hand.

- **What we cut:** we scale `down_proj`, `up_proj`, `gate_proj` for a few neurons.
  Scale 0.0 = off, 0.3 = turned down. We compare no cut, static cut (always off),
  runtime cut (only if a detector fires in the first 5 tokens), and both together.

- **Stats:** paired McNemar on the same questions, plus bootstrap (5000 resamples)
  for the difference. Strict with alternatives is the headline.

## What we tested

- Main topic: Africa capitals (54). Baseline is weak, so there's room to improve.
- Controls: Europe (44, very strong), Asia (46), US states (50), elements (41).
- New checks: africa_largest (54, "largest city in…"), world_tricky (49),
  world_cap_traps (134), world_largest (173).
- Two fixed benches from the same 361 pool: hard (99 we know are hard — the model
  gets them wrong greedily: 54+15+30) and random (99 random, seed 42, overlap 19).

## What happened

### Statistics alone don't work

Trying to pick neurons by mean or variance (32, 128, 512) does nothing on Africa
(greedy stays 7/54 wrong). Only 7 of 128 overlap with the causal ones. Correlation
is not cause.

### Causal patching does

We patch activations and keep the neurons that actually push the wrong answer.
Without filtering, k32 helps a little (7→6) and k128 collapses (7→34 wrong, Europe
breaks). The useful ones live in layers 8–17, peaking at 16.

When we keep only neurons that help the *wrong* answer and are in the middle
layers (`k32_midwrong`), it works:

Sampled-substring legacy table (static arm, `model.generate`, 5 seeds/item; p/net
per-draw — overstates, see NUMBERS.md). Canonicals: greedy strict Africa 0.130 /
Europe 0.000 / k128-Europe 0.091; sampled-substring majority baseline 0.148, k32
0.111 p=0.017, k128 0.074 p=0.003 (per-draw mean 0.093):

| Topic | Before | k32 | k128 |
|---|---:|---:|---:|
| Africa | 0.167 | **0.111** p=0.017 | **0.093** p=0.003 |
| Europe | 0.005 | 0.000 | **0.095** p<0.001 (breaks) |
| Elements | 0.200 | 0.249 | 0.249 |
| Asia | 0.057 | 0.057 | 0.100 |
| US states | 0.052 | 0.088 | 0.128 p=0.007 |

k32 fixes real mistakes (Africa and also Georgia→Tbilisi, Delaware→Dover which
were wrong to start with) and never touches strong knowledge (Europe stays
44/44). It loses where knowledge is weak to begin with. k128 fixes more on
Africa but breaks Europe and the US states — same neurons carry correct answers
elsewhere.

### Timing matters

We trained a detector on the Africa flows (54, 7 wrong). The best signal that
fires *after* the answer is strong (L18, AUC 0.860) but useless — by then the
model already committed. The useful one fires in the first 10 tokens (L19,
AUC 0.742). It only helps if it fires in the first 5 tokens.

| Detector | AUC | When it fires | Result |
|---|---:|---|---|
| Probe L10 | 0.672 | — | weak |
| Jump L18 (full) | 0.860 | tokens 14–19 | 7/54→7/54 — too late |
| Jump early L19 | 0.742 | first 10 | 7/54→4/54 substring (5/54 strict), but 16 fires |
| Same + window ≤5 | 0.742 | first 5 | **7/54→4/54 substring (5/54 strict), 7 fires, Europe 0 fires** |

Window ≤5 is the sweet spot (w≤4 → 6 fires, w≤3 → 2 fires, w≤2 → 0). We use
p90 threshold.

Greedy, strict, 7 baseline errors:

| What we did | Wrong | Notes |
|---|---:|---|
| Nothing | 7/54 | — |
| k32 static | 5/54 | fixes 3, breaks South Sudan |
| k128 static | 4/54 | fixes 5, breaks 2 |
| **Runtime soft, first 5** | **5/54** | **fixes 2, breaks none** |
| w≤4 / w≤3 | 5/54 / 6/54 | fewer fires |

Loose scoring counts Senegal's hedge as a fix; strict doesn't. Runtime and k32
tie at 5/54, but only runtime breaks nothing. k128 looks best at 4/54 but breaks
more.

Other topics, greedy strict: africa_largest 16/54→15/54, world_cap_traps
15/134→13/134, world_largest 30/173→29/173 — small gains, no breaks. world_tricky
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
item majority is the honest primary and is not significant — small effect.
On fired samples, 20/90 turned wrong→correct (22%; 36/90 correct after).

Africa sampled alone (270 draws): 33/270→28/270 p=0.18 — not significant at that
size. Europe: 0 fires in all seeds.

### Both together — hard bench, 594 draws

Static k32 always + runtime when it fires. Fires jump 90→348 (15%→58%) because
static changes the dynamics.

| Nothing | Runtime only | Refuse only | **Static+runtime** | **Static+refuse** |
|---:|---:|---:|---:|---:|
| 354 (0.596) | 334 (0.562) | 280 (0.471) | **231 (0.389)** | **52 (0.088)** + 348 refusals |

Static+runtime improves a lot (132 fixes) but breaks 9 draws; static+refuse
almost wipes out errors but refuses over half (348/594). Bench-aggregate only —
no per-item proof the four Africa quiet cases are among the fixed.

### What we can't fix yet

Four Africa errors (Cape Verde, Equatorial Guinea, Gabon, Guinea) never spike.
No single jitter threshold or k32 mask catches them (greedy Africa). Logit lens
says they are not early high-confidence parametric errors either (0/4
parametric, 4/4 dynamic — but so are all 7, so no separation). That's the limit
of this signal family, not a bug.

## Takeaways

1. There are neurons in layers 8–17 that actually push the wrong answer. Patching
   on wrong-only examples finds them.
2. k32 static helps Africa (p=0.017) but breaks one. Runtime soft (first 5 tokens,
   scale 0.3) helps the same amount with no breaks on tested topics — the only
   single fix that never hurts a tested control (runtime never ran on
   elements/asia/US-states; MMLU tested static only).
3. Direction is consistent everywhere: hard per-draw p<0.001 (20/0), random
   per-draw p=0.031 (6/0); honest item-majority primary is n.s. both times.
   On fired samples 20/90 wrong→correct (22%).
4. Four quiet commits need more than one signal. Static+runtime moves the bench
   a lot (0.389, 9 breaks; 0.088 with 58% refusal) at the cost of many fires.

## Limits

- Loose scoring flatters hedges. We report strict and checked every flip.
- One model (1B), CPU only. No other size or family tested. Qwen adapter is
  synthetic-only; no real second model.
- Hard bench is enriched by construction (conditional gains, regression to the
  mean). Random bench (0.099) is the unbiased estimate: 1pp absolute.
- Per-draw stats overstate (correlated draws); no multiple-comparison correction;
  abstain-coded-as-correct is mechanical. Item majority is primary.
- Detector is per decoding mode (probe ~0.05 transfer). You have to retrain.
- Four quiet cases need a different signal; none tested yet.

## Files

- `attribution_africa.*` / `attribution_causal*.json` — patching results
- `mask_k*.json` — which neurons to cut
- `greedy_flows_africa.npz` + `detector_greedy.json` — detector
- `bench_hard.json` (99 hard) + `bench_random.json` (99 random, seed 42)
- `eval_*.json` / `eval_runtime_*.json` — every run
- Code: `attribute_causal2.py`, `run_experiment.py`, `eval_topic.py`,
  `runtime_rollback.py` (now with static+temporal merged), `bench_*.py`,
  `strict_final.py` (the scorer we report), `battery_analysis.py`,
  `bench_analysis.py`

`data/` (flows, 378 MB) is not in the repo — rebuild with `collect_topic.py`.
