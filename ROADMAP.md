# NHE Roadmap — Two Tracks, One Core

**Where we are (2026-08-21):** Gemma 3 1B, jitter signal, k32 wrong-commit neurons, temporal soft excision (w≤5, ×0.3). Hard bench 99 (0.596→0.562 p<0.001) and random bench 99 (0.099→0.089 p=0.031) both significant. Merged static+temporal breaks the quiet ceiling (0.596→0.389). Cross-model adapter for Qwen is ready (synthetic validated), real weights pending. MMLU proxy preserved (-0.016), real MMLU running.

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

## Repo reorg — to stop the chaos

Current flat root (20+ `bench_*.py`, `eval_*.py` at top level) is hard to navigate.

**Target layout (implemented in next commit):**

```
.
├── README.md, STATUS.md, ROADMAP.md, .gitignore
├── paper/                  # gitignored, LaTeX drafts (already exists, empty)
├── NHE-Edge/               # ← current pipeline moves here, frozen
│   ├── README.md
│   ├── topics.py, runtime_rollback.py, attribute_causal2.py, eval_topic.py, ...
│   ├── bench_*.py, probe_quiet.py, eval_mmlu.py, strict_final.py, ...
│   └── results/  (or keep results/ at root as shared — decision: keep at root for now)
├── NHE-GenPM/              # ← new, starts as plan + skeleton
│   ├── README.md
│   ├── plan.md
│   └── sae/  (empty)
├── results/                # stays at root as shared evidence (benches, masks, evals)
├── models/ , data/         # gitignored, shared
└── legacy/
```

**Migration plan (zero chaos):**
1. Create `NHE-Edge/` and `NHE-GenPM/` with READMEs (this commit).
2. Next commit: `git mv` code files into `NHE-Edge/` with shims at root that re-export (so `python runtime_rollback.py` still works during transition). Update imports to handle both paths.
3. Validate: run `strict_final.py` and `bench_analysis.py` from both locations.
4. Only then remove shims and make `NHE-Edge/` canonical.

We will not move everything at once and break running jobs (MMLU real is running, bench_full is done). This roadmap is the contract.

## Milestones

- **Now → 1 week:** Real Qwen0.5B (download + 6-seed hard/random) + real MMLU 200 + quiet second signal test. This makes the current Edge paper Main Track ready.
- **2–4 weeks:** SAE prototype on Gemma 1B mid layers, compare raw scaling vs steering on MMLU.
- **Paper:** Edge paper first (with cross-model + MMLU), diagnostic paper second, GenPM paper third.

Full numbers always in `results/NUMBERS.md` (hard vs random, greedy vs sampled, strict vs loose).
