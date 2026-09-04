# NHE Roadmap — Two Tracks, One Core

**Where we are (2026-08-21):**
- **Done:** Gemma 3 1B jitter signal (0.968 pooled, 0.742 early), k32 neurons, temporal soft w≤5 (hard 0.596→0.562 p<0.001, random 0.099→0.089 p=0.031), merged 0.389/0.088, MMLU proxy -0.016, quiet lens shows 4 quiet are dynamic not parametric, Qwen adapter synthetic.
- **Pending:** Real Qwen0.5B weights + 6-seed hard/random, real MMLU 200 (running), SAE prototype.

**Where we're going:** Two tracks sharing the same core idea — hallucination is a late wrong-commit in middle layers — but different constraints.

```
                         NHE Framework (core: jitter → wrong-commit → timed cut)
                                      │
              ┌───────────────────────┴───────────────────────┐
              ▼                                               ▼
     NHE-Edge (surgical)                          NHE-GenPM (general)
     Small 1-3B, CPU, zero overhead               Larger base, keep MMLU
     Direct neuron scaling 0.3                    SAEs / steering vectors
     Goal: zero hallucination in scope            Goal: no collateral damage
```

## Track A — NHE-Edge (what we have now)

**Goal:** On-device, safety-critical (health, embedded, dual-use) where you can't retrain. Fix in 24h by cutting the hallucination spot.

**Mechanism:** `Hidden State Jitter (L19, first 10 tokens) + Soft Scaling 0.3` — the current pipeline. Zero extra latency on CPU.

**What it is now:** Everything in this repo at root is Edge. It is the proven track. We will freeze it as `NHE-Edge/` with no architecture change, only hardening.

**What remains for Edge:**
- Real Qwen0.5B run (same pipeline, 2.5h download + 6h eval) to prove not Gemma-specific.
- Quiet diagnostic follow-up: test attention entropy + drift as second signal for the 4 quiet cases (they are not early high-confidence parametric — they are subtle dynamic).
- Latency measurement on CPU/NPU.

## Track B — NHE-GenPM (general-purpose, next)

**Goal:** Broad models where you must keep MMLU/ARC. Don't cut raw neurons — shift features.

**Mechanism:** Same detector, but intervention is SAEs or steering vectors in latent space instead of `weight *= 0.3`. This avoids polysemantic damage.

**What it is now:** Only a plan and an adapter skeleton (`runtime_rollback_qwen.py` already supports Qwen dims). No SAE training yet.

**What needs to be built:**
- Train or load a small SAE for Gemma 3 1B mid layers (or use open SAEs if available for Gemma/Qwen).
- Map k32 neurons → SAE features, then steer instead of scale.
- Evaluate on MMLU 200 (real, not proxy) and GSM8K before/after — must preserve.

## Paper — Third paper (diagnostic)

The 4 quiet cases are not parametric training-data errors (logit lens shows late, low confidence, high entropy like dynamic). But the *method* to tell them apart (jitter + entropy + cross-layer) is itself a paper: "When is a hallucination a training-data error vs a late commit?" Needs logit lens + paraphrase persistence on those 4 vs 3 controls — data already in `results/quiet_diagnostic.json`.

## Appendix: Repo reorg (done 2026-09-04)

Moved via `git mv` (history preserved): all 22 scripts + `topics.py` + `results/`
+ `legacy/` + full README → `NHE-Edge/`. Root holds only overview `README.md`,
`STATUS.md`, `ROADMAP.md`, `requirements.txt`, `.gitignore`. `models/` and
`data/` stay at root (gitignored, heavy); Edge code resolves them via
`REPO_ROOT`. `NHE-GenPM/` holds plan + `sae/` skeleton. No shims — scripts run
from `NHE-Edge/` with script-dir-anchored paths (validated: `strict_final.py`,
`bench_analysis.py` reproduce headline numbers from any CWD).

## Milestones

- **Now → 1 week:** Real Qwen0.5B (download + 6-seed hard/random) + real MMLU 200 + quiet second signal test. This makes the current Edge paper Main Track ready.
- **2–4 weeks:** SAE prototype on Gemma 1B mid layers, compare raw scaling vs steering on MMLU.
- **Paper:** Edge paper first (with cross-model + MMLU), diagnostic paper second, GenPM paper third.

Full numbers always in `results/NUMBERS.md` (hard vs random, greedy vs sampled, strict vs loose).
