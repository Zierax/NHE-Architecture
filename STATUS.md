# Project Status — No-Hallucinations-Ever

Last updated: 2026-08-21 — random bench + merged static+temporal complete, docs overhauled.

## Goal
Detect and reduce Gemma 3 1B hallucinations by tracing internal activation jitter during
decoding, locating wrong-commit points, and excising them surgically — with
topic-specificity verification and frozen benches.

## Environment
- Windows, Python 3.11.15, venv `.venv` (isolated from git).
- torch 2.13.0+cpu (no CUDA — CPU only), transformers 5.15.0.
- RTX 3050 4GB available but not enabled (torch→CUDA upgrade deferred).
- Model: `models/gemma3-1b-fp16` (local fp16 copy, gitignored — 2.7GB).
- Tokenizer: `models/gemma3-1b-tokenizer` from unsloth (official is gated).
- Performance note: fp16 backward on CPU is very slow (259s) — fp32 is 15× faster.

## Status: complete — conclusions
> **Canonical numbers: `results/NUMBERS.md`** — labelled by (protocol greedy/sampled ×
> metric substring/strict). Any application/presentation citing numbers must read from
> this file and label both protocol and metric together (resolved conflict example:
> k128_wrong = 0.056 greedy-substring / 0.093 sampled-majority / 0.074 strict-greedy).

1. Detection: jitter features distinguish hallucinations (AUROC 0.968 Africa, layer
   peaks 10–15) — the signature is **decode-protocol-bound** (does not transfer from
   temp-0.9 to greedy: AUC 0.05).
2. Statistical excision (d_mean/d_var): **NULL** — not causal.
3. Static causal excision: k32_midwrong (greedy strict 0.130→0.093, sampled 0.148→0.111
   p=0.017) **with a documented Juba break**; k128_wrong stronger (greedy strict 0.074,
   sampled 0.093 p=0.003) **but breaks 2 in Africa + Europe 0.091**.
4. **Early soft temporal excision (w5, ×0.3)**: greedy Africa strict **0.130→0.093**
   (**zero breaks**). **Hard bench (99 hard, 594 samples): 0.596→0.562, p<0.001, 20 fixes,
   bootstrap CI [-0.049, -0.020]**. **Random bench (99 random, 594 samples): 0.099→0.089,
   p=0.031, CI [-0.0185, -0.0034]** — same direction, proving generalization.
5. **Merged static+temporal (hard bench): 0.596→0.389 (mask) and 0.088 (abstain, 58%
   refusal)** — breaks the quiet-commit ceiling, fired 58% vs 15% for temporal alone.
6. **Confirmed ceiling**: 4/7 greedy hallucinations (Cape Verde, Eq.Guinea, Gabon,
   Guinea) commit "quietly" — no single-family threshold or mask fixes them; only merged
   does.
7. Methodological lessons: (a) inter-item contamination nearly falsified results;
   (b) permissive matcher inflated improvements by up to 2 points → strict metric;
   (c) offline simulation matches live runs 1:1; (d) manual sampler vs `model.generate`
   are independent valid samples; (e) bench selection (hard vs random) trades power for
   unbiasedness — both needed.
8. Full report: `results/experiment_report.md` — every number in `results/*.json`.

## Data
- `data/` (378MB, in git): 121 general + 54 africa + 44 europe + 41 elements
  (flows, shape (T,27,1152) fp16).
- Evaluation topics (no flows): 9 topics in `topics.py` (54+44+41+46+50+54+49+134+173).
- `results/greedy_flows_africa.npz` — greedy Africa flows (for the temporal detector).
- `results/bench_hard.json` — frozen hard bench (99 hard: 54+15+30).
- `results/bench_random.json` — frozen random bench (99 random, seed 42, overlap 19).

## Next steps (proposed)
1. **Cross-architecture**: same pipeline on Gemma 3 4B or Qwen2.5-1.5B to test
   architecture-invariance.
2. **Protocol transfer**: train detector on sampled flows, test on greedy to fix
   protocol-boundness.
3. **QLoRA fine-tune arm** on Africa (needs CUDA/cloud GPU) as competitive baseline.

## Git
- Public repo: **https://github.com/Zierax/NHE-Architecture** (public, branch
  NHE-Architecture).
- Public contents: scripts + results + benches + README + NUMBERS.md; **data/
  (378MB) and models/ excluded** (reconstructible via `collect_topic.py`).
- Full local history preserved in local branch `local-history`.