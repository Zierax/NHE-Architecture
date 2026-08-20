# Project Status — No-Hallucinations-Ever

Last updated: 2026-08-20 (evening — full error-rich bench battery complete, all numbers canonical).

## Goal
Detect and reduce Gemma 3 1B hallucinations by tracing internal activation jitter during
decoding, locating wrong-commit points, and excising them surgically — with
topic-specificity verification.

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
   sampled 0.093 p=0.003) **but breaks 2 in Africa (Benin, South Sudan — Cape Town is
   valid via 3-capitals alternative) + Europe 0.091**.
4. **Early soft temporal excision (w5, ×0.3)**: greedy Africa strict **0.130→0.093**
   (fixes Eswatini+Gambia, Senegal hedge exposed, **zero breaks**), Europe/Asia/Americas/
   elements with zero fires, world_tricky 0 fires, africa_largest partial transfer
   0.296→0.278. **Error-rich bench (99 items, 594 samples): 0.596→0.562, W2C=20/C2W=0,
   p < 0.001, bootstrap CI [-0.049, -0.020]**. **The only basket that is never damaged
   on any topic, and statistically significant on the error-rich bench.**
5. **Confirmed ceiling**: 4/7 greedy hallucinations (Cape Verde, Eq.Guinea, Gabon,
   Guinea) commit "quietly" — no threshold or mask in this family fixes them.
6. Methodological lessons: (a) inter-item contamination nearly falsified results;
   (b) the permissive matcher inflated improvements by up to 2 points → adopted a strict
   first-sentence metric in NUMBERS.md; (c) offline simulation matches live runs 1:1
   (free verification tools); (d) the manual sample evaluator does not bit-match
   `model.generate` (independent valid seeds).
7. Full report: `results/experiment_report.md` — every number in `results/*.json`.

## Data
- `data/` (378MB, in git): 121 general + 54 africa + 44 europe + 41 elements
  (flows, shape (T,27,1152) fp16).
- Evaluation topics (no flows): Europe/Asia/Americas/elements — in `topics.py`.
- `results/greedy_flows_africa.npz` — greedy Africa flows (for the temporal detector).
- `results/bench_hard.json` — frozen error-rich bench (99 items).

## Next steps (proposed, awaiting decision)
1. **Merge both arms**: static k32_midwrong + early temporal detector together.
2. **QLoRA fine-tune arm** on Africa (needs CUDA/cloud GPU).
3. **Generalization**: narrower detection window (t≤6), lower thresholds, sampled
   evaluation of the temporal intervention.

## Git
- Public repo: **https://github.com/Zierax/NHE-Architecture** (public, branch
  NHE-Architecture).
- Public contents: scripts + results + README + NUMBERS.md (166 files); **data/
  (378MB) and models/ excluded** (reconstructible via `collect_topic.py`).
- Full local history (3 commits with experiments) preserved in local branch
  `local-history`.