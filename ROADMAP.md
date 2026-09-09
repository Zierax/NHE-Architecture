# NHE Roadmap - Two Tracks, One Core

**Where we are (2026-09-07):**
- **Done:** Gemma 3 1B jitter signal (last-token L10 0.968 exists, deployed early 0.742), k32 neurons, temporal soft w<=5 (hard per-draw p<0.001 but item-majority n.s.; random per-draw p=0.031, majority n.s.), merged mask 0.389 (9 breaks) / merged abstain 0.088 (58% refusal), real MMLU 200 preserved (0.925->0.925, +0.01 strict), temporal MMLU negative (78->79, fires 155), prompt baseline (hard 61->52, random 9->11), quiet lens refutes parametric (0/4), **Qwen2.5-0.5B real weights: signal exists (early AUC 0.778), own mask, 0 flips / 0 breaks - timing is format-dependent**, latency measured (+20.5%/token, 160ms/fire).
- **Pending:** SAE prototype, Qwen sampled battery, second-signal test for quiet cases.

**Where we're going:** Two tracks sharing the same core idea - hallucination is a late wrong-commit in middle layers - but different constraints.

```
              NHE Framework (core: jitter -> wrong-commit -> timed cut)
                                   |
              +--------------------+--------------------+
              |                                         |
     NHE-Edge (surgical)                    NHE-GenPM (general)
     Small 1-3B, CPU                        Larger base, keep MMLU
     Direct neuron scaling 0.3              SAEs / steering vectors
     Goal: zero hallucination in scope      Goal: no collateral damage
```

## Track A - NHE-Edge (what we have now)

**Goal:** On-device, safety-critical (health, embedded, dual-use) where you can't retrain. Fix in 24h by cutting the hallucination spot.

**Mechanism:** `Hidden State Jitter (L19, first 10 tokens) + Soft Scaling 0.3` - the current pipeline. Measured cost: +20.5% per token, 160 ms one-time on fire (`latency.json`).

**What it is now:** Everything pipeline lives in `NHE-Edge/` (moved 2026-09-04, history kept). It is the proven track, frozen except hardening.

**What remains for Edge:**
- Quiet diagnostic follow-up: test attention entropy + drift as second signal for the 4 quiet cases (they are not early high-confidence parametric - they are subtle dynamic).
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

The 4 quiet cases are not parametric training-data errors (logit lens: 0/4 parametric, 4/4 dynamic - but so are all 7 cases, so no separation yet). A paper needs n>=50 plus a real second signal with its own AUC. Data so far in `NHE-Edge/results/quiet_diagnostic.json` is a limitations paragraph, not a paper.

## Appendix: Repo reorg (done 2026-09-04)

Moved via `git mv` (history preserved): all 22 scripts + `topics.py` + `results/`
+ `legacy/` + full README -> `NHE-Edge/`. Root holds only overview `README.md`,
`STATUS.md`, `ROADMAP.md`, `requirements.txt`, `.gitignore`. `models/` and
`data/` stay at root (gitignored, heavy); Edge code resolves them via
`REPO_ROOT`. `NHE-GenPM/` holds plan + `sae/` skeleton. No shims - scripts run
from `NHE-Edge/` with script-dir-anchored paths (validated: `analysis/stats.py strict`,
`experiments/bench.py analyze` reproduce headline numbers from any CWD).

## Milestones

- **Now -> 1 week:** Real Qwen0.5B (download + 6-seed hard/random) + real MMLU 200 + quiet second signal test. This makes the current Edge paper Main Track ready.
- **2-4 weeks:** SAE prototype on Gemma 1B mid layers, compare raw scaling vs steering on MMLU.
- **Paper:** Edge paper first (with cross-model + MMLU), diagnostic paper second, GenPM paper third.

Full numbers always in `NHE-Edge/results/NUMBERS.md` (hard vs random, greedy vs sampled, strict vs loose).
