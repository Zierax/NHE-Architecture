# NHE-Architecture

We look for hallucinations in Gemma 3 1B by watching what happens inside the model
while it generates text. When the model is about to make up a capital, the hidden
activations jump in the middle layers. We find the neurons that cause the wrong answer
and turn them down. We check that this helps on one topic (African capitals) without
breaking others.

All code, results, and the exact numbers are in this repo. `results/NUMBERS.md`
is the only place we quote numbers from — it labels every number with how it was
measured.

## What we found

- Wrong answers have a clear signal: the hidden state jumps in layers 10–15 just
  before the bad token. Correct answers don't. This alone separates them well
  (AUROC 0.968 on African capitals).

- If we find the neurons that push the wrong answer (using activation patching on
  only the wrong examples) and turn them down, hallucinations drop — but only if
  we do it *before* the model commits to the answer.

- The best single fix is simple: watch the first 5 tokens, and if the jump detector
  fires, scale those neurons by 0.3. On African capitals (greedy = argmax, see Words we use) this goes
  from 7/54 wrong to 5/54 wrong, with no new errors on any other topic. It's the
  only fix that never breaks a control.

- On a harder test we built (99 items: all 54 "largest city in Africa" — 16 of
  them greedy-wrong — plus 15 hard capitals and 30 hard largest-cities that are
  greedy-wrong), the same fix goes from 354/594 wrong to 334/594 across 6 seeds,
  sampled strict (per-draw p < 0.001, 20 fixes / 0 new errors per-draw; item
  majority 59/99 → 56/99, p = 0.25, not significant — small effect, honest
  primary). On a random 99 from the same pool it goes 59/594 → 53/594
  (p = 0.031 per-draw; majority 10/99 → 9/99). Same direction, smaller size.

- If we leave the 32 bad neurons off all the time *and* also do the runtime fix
  when the detector fires, the hard bench goes 354/594 → 231/594 (mask; 132
  fixes but 9 new breaks disclosed). If static stays on and the runtime fix
  refuses instead of repairing, it goes to 52/594 with 348 refusals (58%).
  (Runtime-only refuse is 280/594 — don't confuse the two.) The ceiling moves,
  but you pay with fires, breaks, or refusals.

- Four of the seven Africa mistakes (Cape Verde, Equatorial Guinea, Gabon, Guinea)
  never show the jump at all. No threshold or mask in this family catches them.
  That's a real limit, not a bug.

## How it works

1. **Find the signal.** For each layer and each token, record how much the hidden
   state moves. Wrong answers spike in the middle layers right before the answer.

2. **Find the cause.** Patch activations and see which neurons actually push the
   wrong answer. Keep the top ones per layer (32 across layers 8–17 is the sweet
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
              └──── scale neurons ────┘  → now correct
```

## Words we use

- **Jitter** — how much the hidden state jumps between tokens, per layer.
- **Wrong-commit neurons** — the ones that actually push the bad answer (found with
  patching).
- **Mask** — which neurons to turn down.
- **Static** — turn them down for the whole generation.
- **Temporal / runtime** — only turn them down if the detector fires in the first
  5 tokens.
- **Merged** — static always + temporal when it fires.
- **Bench** — a fixed test set. `bench_hard.json` is 99 we know are hard (greedy-wrong);
  `bench_random.json` is 99 random from the same pool.
- **Greedy vs sampled** — argmax vs temp 0.9 / top_p 0.9. Don't mix numbers across them.
- **Substring vs strict** — substring: answer anywhere in output (loose, counts hedges
  as correct). Strict: answer in the first sentence (what we report).

## Results

### Africa capitals, greedy (54, strict)

| What we did | Wrong | What changed |
|---|---:|---|
| Nothing | 7/54 | — |
| d_mean / d_var (statistics) | 7/54 | nothing — correlation, not cause |
| k32 static | 5/54 | fixes 3, breaks South Sudan |
| k128 static | 4/54 | fixes 5, breaks Benin + South Sudan |
| **runtime soft, first 5 tokens** | **5/54** | **fixes 2, breaks nothing** |
| w≤4 / w≤3 | 5/54 / 6/54 | fewer fires, same or fewer fixes |

Static and runtime tie at 5/54, but static breaks one and runtime breaks none. k128
looks better at 4/54 but breaks more and hurts every control topic.

### Africa, sampled (270 draws, 5 seeds) — labels matter

Static `model.generate` (sampled, substring): majority 8/54 → 6/54 (k32, p=0.017) and 4/54 (k128, p=0.003); per-draw means 0.111/0.093.
Runtime manual sampler (sampled, substring via file `correct`): 33/270 → 28/270 per-draw mean (p=0.18), 6/54 → 4/54 by majority. Same questions, different sampler draws — not bit-identical. No new errors on Europe (0 fires).

Small n, so runtime isn't significant here. The benches below give it power.

### Two benches, 99 items × 6 seeds = 594 draws, strict

Hard bench = all 54 largest-Africa (16 greedy-wrong) + 15 hard capitals + 30 hard
largest that are greedy-wrong, from 361-item pool (overlap hard-random 19, random
seed 42). Random bench = 99 random from same pool. Hard baseline 0.596, random 0.099.

| Bench | Nothing | Runtime soft | Refuse instead |
|---|---|---:|---:|
| Hard: wrong / 594 | 354 (0.596) | **334 (0.562)** per-draw p<0.001, 20 fixes / 0 new errors per-draw | 280 (0.471) + 90 refusals |
| Hard: majority-of-6 99 (**primary**) | 59 | 56 (W2C=3, C2W=0, p=0.25, n.s.) | 47 |
| Random: wrong / 594 | 59 (0.099) | **53 (0.089)** per-draw p=0.031 | 47 (0.079) + 42 refusals |
| Random: majority-of-6 99 (**primary**) | 10 | 9 (1 discordant, n.s.) | 8 |

Same direction on both; honest primary (item majority) is not significant —
small effect, not noise-free proof. On fired samples, 20/90 turned wrong→correct
(22%; 36/90 correct after). Per-draw tests overstate (correlated draws).

### Merged: static always + runtime when it fires (hard bench, 594, strict)

| Nothing | Runtime only | Refuse only | **Static+runtime** | **Static+refuse** |
|---:|---:|---:|---:|---:|
| 354 (0.596) | 334 (0.562) | 280 (0.471) | **231 (0.389)** — 132 fixes, **9 new breaks** | **52 (0.088)** + 348 refusals (58%), 305 fixes / 3 breaks |
| Fires | 90 (15%) | 90 | 90 | 348 (58%) | 348 (58%) |

Static alone changes the dynamics so the detector fires more. Merged repair
improves a lot but breaks 9 draws — not "without refusal" clean. Merged refuse
trades 348 refusals for the low rate. Bench-aggregate only: no per-item proof
the four Africa quiet cases are among the fixed.

### Other topics, greedy strict

| Topic | Nothing | Runtime soft | k32 | k128 |
|---|---:|---:|---:|---:|
| africa_largest 54 | 0.296 (16) | 0.278 (15) | 0.185 (10: 7 fixes, breaks S.Sudan) | 0.259 (14: 7 fixes, 5 breaks) |
| world_cap_traps 134 | 0.112 | 0.097 | — | — |
| world_largest 173 | 0.173 | 0.168 | — | — |
| world_tricky 49 | 0.020 (1) | 0.020 (0 fires) | 0.041 (2) | 0.122 (6: fixes UAE, breaks 6) |
| europe 44 | 0.000 | 0.000 (0 fires) | 0.000 | 0.091 (4 breaks) |

Only runtime never hurts a control (on tested topics; runtime was never run on
elements/asia/US-states, and MMLU tested the static mask only).

### Detector

| Signal | AUC | What it means |
|---|---:|---|
| jitter last-token L10 (any token) | 0.968 | feature-specific, not windowed — shows signal exists (`jitter_report_africa.json:17`, labeled on 44/10 vs headline 47/7), not used for cut |
| jump full (L18) | 0.860 | fires after the answer — too late |
| jump early (L19, first 10) | 0.742 | fires before — the one we use (text reports 0.742) |
| probe L10 | 0.672 | weak, doesn't transfer greedy↔sampled (~0.05) |

Fires vs window (p90, strict 4/54 = substring 4/54, strict 5/54): w≤5 → 7/54 (3 hits), w≤4 → 6/54 (2 hits), w≤3 → 2/54 (1 hit), w≤2 → 0. We use w≤5. Strict is 5/54 (Senegal hedge counts as 4/54 loose).

## What we can't fix yet

Four Africa errors never spike — they just commit quietly. No single jitter
threshold or k32 mask catches them. Merged static+temporal improves the hard
bench a lot, but there is no per-item proof those four are among the fixed —
"breaks the ceiling" is bench-aggregate only. That's the ceiling for this signal
family. A logit-lens probe on those four
(`probe_quiet.py`, `results/quiet_diagnostic.json`) shows they are *not* early
high-confidence parametric errors — they look dynamic with subtler jitter, so a
second signal (attention entropy / drift) may catch them.

## Cross-model and side effects

- **Qwen adapter ready** (`runtime_rollback_qwen.py`): same pipeline for
  Qwen2.5-0.5B/1.5B (24×896 / 28×1536), synthetic validation passes
  (`results/detector_greedy_qwen*.json`). Real weights need a stable download
  (`results/cross_arch_report.md`).
- **General knowledge preserved:** soft k32 on **200 real MMLU** goes 185/200→185/200 substr (0.0) and 122/200→124/200 strict (+0.01) — no damage (`eval_mmlu.py --use-real`, `results/mmlu_side_effect.json:159-172`; now also in `results/NUMBERS.md`). Proxy 181 controls also preserved (-0.016, superseded).

## Limitations

- **Scoring matters.** Loose substring counts a hedge like "Diou… While Dakar is
  the largest city" as a fix; strict doesn't. We report strict and checked every
  flip by hand.
- **One model, one size.** Gemma 3 1B, CPU only. No other model or size tested.
- **Bench matters.** Hard bench is hard by design (0.596). Random bench (0.099)
  is the honest baseline. Both show the same effect, different size.
- **Detector doesn't transfer** across sampling vs greedy (AUC ~0.05). You have to
  calibrate per decoding mode.
- **Ceiling.** 4/7 quiet commits need a different signal; merged breaks it only
  with many fires.

## How to reproduce

Every number has a file in `results/` — masks, evals, detector, flows, benches.
`results/NUMBERS.md` is the single table we quote from.

Model (exact recipe): gated `google/gemma-3-1b-it`. `huggingface-cli login`,
download, load in fp32, `.half()`, save **one** `model.safetensors` (999.9M
params, float16, 26×1152, ~2.7 GB) to `..\models\gemma3-1b-fp16\`; tokenizer
(vocab 32768) to `..\models\gemma3-1b-tokenizer\`. CPU-only, no CUDA.

```
attribute_causal2.py      patching, picks 32 neurons (layers 8–17)
run_experiment.py         makes masks from attributions, runs static evals
eval_topic.py             greedy / sampled eval (positional mask path)
runtime_rollback.py       collect flows, fit detector, run temporal/merged
sweep_thresholds.py       check thresholds (matches live runs)
sweep_windows.py          window tradeoffs
battery_analysis.py       Africa 5-seed stats (majority + McNemar + bootstrap)
bench_analysis.py         hard/random/merged bench stats (the headline tests)
bench_build.py            builds bench_hard.json (all 54 africa_largest + greedy-wrong 15+30)
bench_driver.py           runs hard-bench sampled battery (6 seeds × 3 arms)
bench_full_sampled.py     runs random-bench + merged batteries
bench_greedy.py           greedy arms on bench subsets
bench_random_greedy.py    greedy arms on random bench
probe_quiet.py            logit-lens check on 4 quiet + 3 dynamic cases
eval_mmlu.py              MMLU side-effect (use --use-real for real 200)
strict_final.py           strict scorer (the one we report)
significance_final.py     static-arm McNemar + bootstrap
significance_s5.py        sample-level stats
topics.py                 AFRICA 54, EUROPE 44, ELEMENTS 41, ASIA 46, US_STATES 50,
                          AFRICA_LARGEST 54, WORLD_TRICKY 49, WORLD_CAP_TRAPS 134,
                          WORLD_LARGEST 173
results/                  all outputs, NUMBERS.md, experiment_report.md
legacy/                   old tries, null statistical arm
```
`bench_random.json` is built by `bench_random_build.py` (seed 42, byte-identical to the committed file); `bench_build.py` writes the hard bench.

Quick start (PowerShell 5.1; run from this folder; env file lives at repo root):

```powershell
# already inside NHE-Edge/ (do NOT cd again if you are here)
python -m venv ..\.venv
..\.venv\Scripts\pip install -r ..\requirements.txt
# 0. log in to gated HF once: huggingface-cli login  (model + tokenizer below)
# 1. fetch weights into ..\models\ (see "How to reproduce"), then:
python runtime_rollback.py collect            # → results/greedy_flows_africa.npz
python runtime_rollback.py fit_greedy         # → results/detector_greedy.json
# 2. build the k32 mask first (run ... mask needs results/mask_k32_midwrong.json):
python attribute_causal2.py                   # → results/attribution_causal2_africa.json
python run_experiment.py                      # → results/mask_k32_midwrong.json
python runtime_rollback.py run africa early t90 mask m 0 0.3 5
#    args: detector-set early | threshold t90 | mode mask | m=greedy (s=sampled) | seed 0 | scale 0.3 | window 5
python eval_topic.py africa results/mask_k32_midwrong.json   # positional mask path (no --mask flag)
python strict_final.py                        # re-score committed evals, no model
python bench_driver.py                        # hard bench sampled, 6 seeds (hours on CPU)
python bench_full_sampled.py                  # random + merged (hours on CPU)
python bench_analysis.py                      # stats over committed runs, no model
```

Full walkthrough: `results/experiment_report.md`. Short status: `../STATUS.md`. Repo plan: `../ROADMAP.md`.
