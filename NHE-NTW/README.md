# NHE-NTW - Not Trained Well

Independent track: proves that **quiet commits are planted training-data errors**,
not inference failures.

Method (causal, fully controlled): train a tiny GPT from scratch on synthetic
capital facts where WE plant 4 lies, keep the rest true, and make 4 items
ambiguous (50/50). Then probe with the jitter family: early hidden-state jump +
answer confidence + correctness-vs-TRUTH.

Prediction: planted lies -> low jitter + high confidence + wrong-vs-truth
(quiet lie, like Gemma's 4 quiet cases). Ambiguous -> high jitter (conflict).
Correct -> low jitter + right.

## Result (v2, full sentences, truth in-vocab, seed 7)

| group | acc-vs-truth | confidence | preamble jitter | P(truth) on planted |
|---|---|---|---|---|
| correct (32) | 1.000 | 1.000 | 1168.5 | - |
| planted (4) | 0.000 | 1.000 | 1159.8 | 0.0000, entropy 0.000 |
| ambiguous (4) | 0.500 | 0.520 | 1160.1 | - |

Refinement from the data: jitter magnitude does NOT separate planted lies from
correct answers here (within 1%) - the separator is confidence/entropy plus
external truth. So the void signature is **quiet + confident + wrong**, not
"low jitter" per se. Ambiguous items (the drift analog) show ~0.52 confidence
and 50/50 behavior. Limits: toy scale (4 layers, 40 facts); Gemma-scale
dynamics may differ; this is a causal prior for the taxonomy, not a proof
about Gemma's 4 cases.

## Novelty (own, not Edge's)

- Edge finds and cuts hallucinations at runtime. NTW answers *why some are
  quiet*: they were never uncertain - the weights encode the lie confidently.
- Failure taxonomy (shared with the Edge paper): **Static Memory Voids**
  (confident lies baked into weights; need external knowledge, outside jitter
  scope) vs **Transient Execution Drifts** (genuine decode-time conflict;
  Edge's target). NTW causally proves the void category exists.
- Diagnostic signature: quiet + confident + wrong-vs-truth = audit flag for
  training-data contamination, without opening the dataset.

## Paraphrase test (ground-truth-free attempt) - INCONCLUSIVE

`probe_paraphrase.py` asks each question in 4 phrasings (in-vocab words only)
using saved weights, no retraining (`results/paraphrase_proof.json`):

| group | mean agreement | p0 acc |
|---|---|---|
| correct (32) | 0.104 | 1.000 |
| planted (4) | 0.167 | 0.000 |
| ambiguous (4) | 0.167 | 0.500 |

Held-out phrasings break EVERYTHING equally (single-template training makes the
whole model brittle) - so consistency probing cannot separate lies from truth
here. Correction to the claim above: internal signals alone (confidence,
jitter, paraphrase agreement) do NOT identify planted lies without ground
truth; confident lies are internally identical to confident truths. What works
is provenance (does the fact appear in training data?) and consistency across
IN-distribution paraphrases (untested - needs template variation in training).
The NTW contribution stands as causal proof the void category exists and its
exact signature, not as a turnkey detector.

## Crossfade with NHE-Edge

- Shared: jitter metric family (early max hidden-state jump), strict scoring
  idea (correctness-vs-truth), seed discipline.
- Independent: own model, own data, own implementation - no imports from
  NHE-Edge, so the proof cannot inherit Edge's assumptions.
- Edge's ceiling (4 quiet cases) is NTW's starting exhibit.

## Layout

- `parametric_proof.py` - data + train + probe, one command, seeded
- `results/parametric_proof.json` - per-group jitter/confidence/accuracy
- `results/README.md` - what each file means
