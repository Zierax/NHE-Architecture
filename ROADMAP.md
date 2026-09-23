# NHE Roadmap - Three Tracks, One Core

**Where we are (2026-09-23):**
- **Done:** Gemma 3 1B jitter signal (last-token L10 0.968 exists, deployed early 0.742), k32 neurons, temporal soft w<=5 (hard per-draw p<0.001 but item-majority n.s.; random per-draw p=0.031, majority n.s.), merged mask 0.389 (9 breaks) / merged abstain 0.088 (58% refusal), real MMLU 200 preserved (0.925->0.925, +0.01 strict), temporal MMLU negative (78->79, fires 155), prompt baseline (hard 61->52, random 9->11), **NHE-NTW controlled Static Memory Void proof** (four planted lies at confidence 1.0, P(truth)=0, jitter like correct; zero-example countries at confidence 0.548), **Qwen2.5-0.5B real weights: signal exists (early AUC 0.778), own mask, 0 flips / 0 breaks - timing is format-dependent**, latency measured (+20.5%/token, 160ms/fire).
- **Pending:** direct Gemma/Claude-scale void test, SAE prototype, Qwen sampled battery, second-signal test for quiet cases.

**Where we're going:** Three tracks sharing one taxonomy: transient
wrong-commits are candidates for runtime repair; static confident voids require
fact-level repair; general models need feature-safe interventions.

```
              NHE Framework (core: jitter -> wrong-commit -> timed cut)
                                   |
               +--------------------+--------------------+
               |                    |                    |
      NHE-Edge (surgical)   NHE-NTW (diagnostic)  NHE-GenPM (general)
      Small 1-3B, CPU       Static vs transient   Larger base, keep MMLU
      Direct neuron scaling Proven taxonomy      SAEs / steering vectors
      Goal: repair drift    Goal: explain voids  Goal: no collateral damage
```

## Track A - NHE-Edge (what we have now)

**Goal:** On-device, safety-critical (health, embedded, dual-use) where you can't retrain. Fix in 24h by cutting the hallucination spot.

**Mechanism:** `Hidden State Jitter (L19, first 10 tokens) + Soft Scaling 0.3` - the current pipeline. Measured cost: +20.5% per token, 160 ms one-time on fire (`latency.json`).

**What it is now:** Everything pipeline lives in `NHE-Edge/` (moved 2026-09-04, history kept). It is the proven track, frozen except hardening.

**What remains for Edge:**
- Quiet diagnostic follow-up: test attention entropy + provenance/fact checks
  as a second response for the 4 quiet cases. NTW gives the likely category;
  direct attribution remains open.
- Qwen sampled battery (signal exists; test repair under sampling).
- Latency on NPU / smaller models.

## Track B - NHE-GenPM (general-purpose, next)

**Goal:** Broad models where you must keep MMLU/ARC. Don't cut raw neurons - shift features.

**Mechanism:** Same detector, but intervention is SAEs or steering vectors in latent space instead of `weight *= 0.3`. This avoids polysemantic damage.

**What it is now:** Plan-only, zero code (`NHE-GenPM/plan.md`, `sae/README.md`). The Qwen adapter lives in `NHE-Edge/core/runtime_rollback_qwen.py` (Edge-side cross-arch work), not here. No SAE training yet.

**What needs to be built:**
- Train or load a small SAE for Gemma 3 1B mid layers (or use open SAEs if available for Gemma/Qwen).
- Map k32 neurons -> SAE features, then steer instead of scale.
- Evaluate on MMLU 200 (real, not proxy) and GSM8K before/after - must preserve.

## Paper - Third paper (diagnostic)

The 4 quiet cases are now a clear motivation for the NTW companion proof. The
controlled experiment establishes a Static Memory Void category, and the
entropy audit shows that 3/4 Gemma quiet cases commit with high confidence;
Gabon is a separate uncertainty/truncation case. This is a plausible mechanism
and a useful triage taxonomy, not a direct attribution of all four cases.

A production diagnostic paper still needs a larger sample, direct model-level
provenance tests, and a second signal with its own AUC. The current Gemma
logit-lens classification alone does not separate the four quiet cases from
the other errors. Data so far in `NHE-Edge/results/quiet_diagnostic.json` and
`entropy_audit.json` is the foundation for that next experiment.

## Appendix: Repo reorg (done 2026-09-04)

Moved via `git mv` (history preserved): all 22 scripts + `topics.py` + `results/`
+ `legacy/` + full README -> `NHE-Edge/`. Root holds only overview `README.md`,
`STATUS.md`, `ROADMAP.md`, `requirements.txt`, `.gitignore`. `models/` and
`data/` stay at root (gitignored, heavy); Edge code resolves them via
`REPO_ROOT`. `NHE-GenPM/` holds plan + `sae/` skeleton. No shims - scripts run
from `NHE-Edge/` with script-dir-anchored paths (validated: `analysis/stats.py strict`,
`experiments/bench.py analyze` reproduce headline numbers from any CWD).

## Milestones

- **Now -> 1 week:** Direct NTW signature test on Gemma/Claude-scale models + quiet second signal test. This makes the diagnostic story stronger.
- **Now -> 2 weeks:** Real Qwen0.5B sampled battery (hard/random) + real MMLU 200 comparison. This completes the Edge cross-model evidence.
- **2-4 weeks:** SAE prototype on Gemma 1B mid layers, compare raw scaling vs steering on MMLU.
- **Paper:** Edge paper first (with cross-model + MMLU), NTW diagnostic companion second, GenPM paper third.

Full numbers always in `NHE-Edge/results/NUMBERS.md` (hard vs random, greedy vs sampled, strict vs loose).
