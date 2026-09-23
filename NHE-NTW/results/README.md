# NHE-NTW results

The NTW experiment is a controlled demonstration of the Static Memory Void
category: a wrong fact learned during training can be answered confidently and
without a meaningful pre-commit jitter signal.

## Files

- `parametric_proof.json` — the final v2 experiment: seed, model dimensions,
  full-sentence answer format, and per-group accuracy-vs-truth, confidence,
  preamble jitter, and per-item traces. Produced by
  `../parametric_proof.py` with seed 7. The headline comparison is:

  | group | n | accuracy vs truth | confidence | jitter |
  |---|---:|---:|---:|---:|
  | correct | 32 | 1.000 | 1.000 | 668.0 |
  | planted | 4 | 0.000 | 1.000 | 669.3 |
  | ambiguous | 4 | 0.750 | 0.514 | 662.9 |
  | unseen | 4 | — | 0.548 | 666.8 |

  The planted group is the main result: four false facts produce confident,
  persistent wrong commitments with P(truth)=0.0000 and entropy 0.000. The
  unseen group is the falsification check: zero-example countries produce
  uncertainty, not confident stable lies.

- `parametric_weights.pt` — trained weights used to reproduce probes without
  retraining. Regenerable and kept local.
- `parametric_ckpt.pt` — resume checkpoint for the long CPU run. Regenerable and
  gitignored.
- `paraphrase_proof.json` — four in-vocabulary phrasings evaluated from the
  saved weights by `../probe_paraphrase.py`. Mean agreement is 0.104 for
  correct, 0.167 for planted, and 0.167 for ambiguous items. This is a useful
  boundary condition: the single-template model is brittle across held-out
  phrasings, so agreement is not a standalone ground-truth-free lie detector.

## Reading the result

The causal claim is about the existence and signature of a static confident
error in a controlled environment. The result is not a claim that all Gemma or
Claude hallucinations are training-data errors, and the paraphrase result is
not a detector claim. The direct next experiment is a larger-scale test of the
same signature on production models.

The practical triage consequence is positive and concrete: a confident quiet
wrong answer should first trigger a factual-provenance check, while a visible
pre-commit drift is the case to test with runtime intervention.
