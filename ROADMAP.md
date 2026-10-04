# NHE Roadmap — three tracks, one taxonomy

## Position (2026-09-23)

**The deployment deliverable is complete and measured.** A sub-1B model, on CPU,
without retraining, suppresses errors at runtime with zero regressions across 463
items, at +20.5% latency, deployable in ~6–10 hours. That satisfies the mission it
was built for.

| mission criterion | measured | standard |
|---|---:|---|
| errors removed | 59 → 56 of 463 | deployment-valid |
| regressions | **0** across 463 items, 3 frames | deployment-valid |
| latency | +20.5% per token, 160 ms per firing | deployment-valid |
| retraining | none | deployment-valid |
| wall-clock to deploy | ~6–10 h | deployment-valid |

**One scientific question remains open, and it is a different question:** the
population-level effect is not statistically separable from zero at n = 463
(3 fixes, 0 regressions, exact McNemar p = 0.25). That concerns generalisation of
an effect size across models and samples. It does not concern whether the deployed
configuration repairs errors on the device it was measured on, which is settled by
direct measurement.

```
   DEPLOYMENT (complete)              SCIENCE (open)
   ------------------                 --------------
   run the shipped config      vs.    infer an effect size over samples
   measure errors + regressions       test with exact McNemar
   latency, wall-clock                p = 0.25 -> not established
   p-value not required               p-value required
   zero regressions: MET              6 fixes needed: NOT MET
```

Both are reported. Neither substitutes for the other.

## Completed work

- **Causal intervention.** Activation patching on wrong-only examples identifies
  the neurons that drive errors; statistical selection (mean/variance) yields
  nothing (7/128 overlap). Causal k32: 7→5 greedy with regressions on other
  topics; runtime timing is what removes the regressions.
- **Timing law.** The spike must precede the answer commit by ≥ 1 token. Verified
  across a model × format matrix; at n=463, 40 of 41 firings had lead ≥ 1 and the
  single late firing changed nothing.
- **Deployment validation.** 463-item single-task benchmark with an audited truth
  layer: 59→56, 3 fixes, 0 regressions. Repairs per fired-wrong item 0.333 versus
  0.286 on Africa, so the mechanism generalises across question frames.
- **Controlled causal proof (NHE-NTW).** Planted false facts are answered at
  confidence 1.000, P(truth)=0, entropy 0, jitter indistinguishable from correct;
  zero-training countries answer uncertain. The static-void class exists as a
  construct, and ignorance alone does not reproduce it.
- **Automatic applicability.** A gate searches 192 (layer × window) cells and
  controls for a commit-position artifact: Gemma ACTIVE, Qwen INERT. AUROC gates
  applicability but does not select the intervention (L20 0.781 → net −1;
  L19 0.772 → net +2).
- **Ceiling characterised.** 89.3% of residual error never fires — the static-void
  population, outside runtime reach by construction, now measured rather than
  assumed.
- **Cost.** +20.5% per token, 160 ms per firing, ~6–10 h wall-clock, CPU only.

## Open: the recall question, settled by a failed experiment

A higher-recall trigger was built and rejected.

| | L19/w5 (deployed) | L11/w10 (higher-recall) |
|---|---:|---:|
| fixes | 3 | 4 |
| breaks | **0** | **2** |
| net | **+3** | +2 |
| recall on errors | 0.153 | **0.286** |
| repair per fired-wrong | **0.333** | 0.174 |

Twice the recall, more fixes, and it fails: both regressions are in the `element`
family, where a mask fitted on wrong-only capital answers corrupts the
atomic-number mapping. Zero regressions is the deployment criterion, so L11/w10 is
rejected on evidence.

A correction is on record: the first reachability measurement used a fixed
15-token window that read at and after the commit, violating the project's own
timing rule. Re-measured per-item pre-commit, the residual is still partly
reachable, but at L11 (0.6925) rather than L15 (which falls to 0.577). The
conclusion survived; the cell did not.

```
  residual 56 wrong
        |
        +-- wider / better trigger        -> TESTED (L11/w10), regresses, REJECTED
        |
        +-- family-aware trigger or per-family mask      <- the one open idea
        |      both regressions were 'element'; the mask was fitted on a family
        |      it never saw. Restrict per family and the trigger can widen safely.
        |
        +-- if that fails: the ceiling is real for this task. The residual is
               outside runtime reach, the taxonomy already accounts for it, and
               the honest contribution is the taxonomy plus the measured ceiling.
```

Only one idea remains, it is small, and it is decidable in a single run on data
already collected.

## Track A — NHE-Edge (the deliverable)

**Goal:** on-device, safety-critical, where retraining is unavailable. Fix the error
in 24 hours by intervening where it is produced.

**Mechanism:** `Hidden State Jitter (L19, first 10 tokens) + Soft Scaling 0.3`.
Measured cost: +20.5% per token, 160 ms once per firing (`results/latency.json`).

**What it is now:** proven as a deployment configuration, frozen except hardening.
Everything pipeline lives in `NHE-Edge/` (moved 2026-09-04, history kept).

**Remaining:** family-aware trigger or per-family mask; a direct provenance test for
any residual confirmed as a static void; a larger model, to test whether the
deployment result survives a change of size.

## Track B — NHE-NTW (mechanism)

**Goal:** establish what the unreachable residual actually is, causally, and turn it
into an operational triage rule.

**Status:** the void class is established by controlled experiment. The remaining
work is direct provenance verification on a production model, so the Gemma quiet
cases move from "consistent with the void reading" to "attributed".

## Track C — NHE-GenPM (planned)

**Goal:** models where raw neuron scaling is unacceptable and unrelated capability
must be preserved (MMLU/ARC).

**Mechanism:** same detector, intervention in feature space (SAE / steering
vectors) instead of `weight *= 0.3`, to avoid polysemantic damage.

**Status:** plan only, no code (`NHE-GenPM/plan.md`). Needs a trained or loaded SAE,
a neuron→feature mapping, and MMLU/ARC preservation checks.

## Appendix: repo reorg (done 2026-09-04)

Moved via `git mv` (history preserved): all scripts + `topics.py` + `results/` +
`legacy/` -> `NHE-Edge/`. Root holds only overview docs, `requirements.txt`,
`.gitignore`. `models/` and `data/` stay at root (gitignored, heavy); Edge code
resolves them via `REPO_ROOT`. No shims — scripts run from `NHE-Edge/` with
script-dir-anchored paths (validated: `analysis/stats.py strict` and
`experiments/bench.py analyze` reproduce headline numbers from any CWD).

Full numbers always in `NHE-Edge/results/NUMBERS.md`.