# NHE Roadmap - Two Tracks, One Core

**Where we are (2026-09-04):**
- **Done:** Gemma 3 1B jitter signal (last-token L10 0.968 exists, deployed early 0.742), k32 neurons, temporal soft w<=5 (hard per-draw p<0.001 but item-majority n.s.; random per-draw p=0.031, majority n.s.), merged mask 0.389 (9 breaks) / merged abstain 0.088 (58% refusal), real MMLU 200 preserved (0.925->0.925, +0.01 strict), quiet lens refutes parametric (0/4), Qwen adapter synthetic-only.
- **Pending:** Real Qwen0.5B weights + 6-seed hard/random, SAE prototype, prompt/entropy baselines, temporal MMLU.

**Where we're going:** Two tracks sharing the same core idea - hallucination is a late wrong-commit in middle layers - but different constraints.

```
                         NHE Framework (core: jitter -> wrong-commit -> timed cut)
                                      │
              ┌───────────────────────┴───────────────────────┐
              ▼                                               ▼
     NHE-Edge (surgical)                          NHE-GenPM (general)
     Small 1-3B, CPU, zero overhead               Larger base, keep MMLU
     Direct neuron scaling 0.3                    SAEs / steering vectors
     Goal: zero hallucination in scope            Goal: no collateral damage
```

## Track A - NHE-Edge (what we have now)

**Goal:** On-device, safety-critical (health, embedded, dual-use) where you can't retrain. Fix in 24h by cutting the hallucination spot.

**Mechanism:** `Hidden State Jitter (L19, first 10 tokens) + Soft Scaling 0.3` - the current pipeline. Zero extra latency on CPU.

**What it is now:** Everything pipeline lives in `NHE-Edge/` (moved 2026-09-04, history kept). It is the proven track, frozen except hardening.

**What remains for Edge:**
- Real Qwen0.5B run (same pipeline, 2.5h download + 6h eval) to prove not Gemma-specific.
- Quiet diagnostic follow-up: test attention entropy + drift as second signal for the 4 quiet cases (they are not early high-confidence parametric - they are subtle dynamic).
- Latency measurement on CPU/NPU.

## Track B - NHE-GenPM (general-purpose, next)

**Goal:** Broad models where you must keep MMLU/ARC. Don't cut raw neurons - shift features.

**Mechanism:** Same detector, but intervention is SAEs or steering vectors in latent space instead of `weight *= 0.3`. This avoids polysemantic damage.

**What it is now:** Plan-only, zero code (`NHE-GenPM/plan.md`, `sae/README.md`). The Qwen adapter lives in `NHE-Edge/runtime_rollback_qwen.py` (Edge-side cross-arch work), not here. No SAE training yet.

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
from `NHE-Edge/` with script-dir-anchored paths (validated: `strict_final.py`,
`bench_analysis.py` reproduce headline numbers from any CWD).

## Milestones

- **Now -> 1 week:** Real Qwen0.5B (download + 6-seed hard/random) + real MMLU 200 + quiet second signal test. This makes the current Edge paper Main Track ready.
- **2-4 weeks:** SAE prototype on Gemma 1B mid layers, compare raw scaling vs steering on MMLU.
- **Paper:** Edge paper first (with cross-model + MMLU), diagnostic paper second, GenPM paper third.

Full numbers always in `NHE-Edge/results/NUMBERS.md` (hard vs random, greedy vs sampled, strict vs loose).
