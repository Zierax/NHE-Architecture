# NHE-NTW — Not Trained Well

NHE-NTW is a controlled proof-of-concept for a failure mode that NHE-Edge
cannot repair at runtime: a **Static Memory Void**, a confident wrong fact that
has been learned into the model weights.

## Main result

We train a tiny GPT from scratch on synthetic capital facts. The experiment
plants four false facts, keeps 32 facts correct, and makes four facts ambiguous
(50/50). The model then answers under the same three-way probe used by Edge:

- **Static Memory Void:** a planted false fact, answered confidently and with
  no meaningful pre-commit jitter.
- **Transient Execution Drift:** an ambiguous fact, answered with uncertainty
  or conflict.
- **Correct answer:** a true fact learned normally.

### Final v2 result (seed 7, full-sentence answers, truth in vocabulary)

| group | n | accuracy vs truth | confidence | preamble jitter | P(truth) on planted |
|---|---:|---:|---:|---:|---:|
| correct | 32 | 1.000 | 1.000 | 668.0 | — |
| **planted lies** | **4** | **0.000** | **1.000** | **669.3** | **0.0000; entropy 0.000** |
| ambiguous | 4 | 0.750 | 0.514 | 662.9 | — |
| unseen | 4 | — | 0.548 | 666.8 | — |

The key positive finding is the planted group: all four false facts become
confident wrong commitments with jitter indistinguishable from the correct
group. This establishes that **quiet, confident, wrong behavior can be created
by training and does not require an inference-time conflict**.

The unseen group supplies a useful falsification check. Countries with zero
training examples answer with lower confidence (0.548 on average) rather than
confidently inventing a single stable lie. Within this controlled setup,
ignorance therefore produces uncertainty, while the planted facts produce the
quiet-confident signature.

## What this means for NHE-Edge

Edge targets transient execution drifts: a decode-time conflict that creates a
pre-commit signal and can be reduced by a timed intervention. NTW creates the
complementary category:

| | Static Memory Void | Transient Execution Drift |
|---|---|---|
| Mechanism | wrong fact encoded confidently in weights | conflict while generating |
| Observable pattern | quiet + confident + wrong | jitter spike + lower confidence |
| Primary response | external knowledge or fine-tuning | runtime intervention such as NHE-Edge |
| NTW role | controlled causal demonstration | target of Edge |

This distinction gives a practical triage rule: if an error is quiet and
confident, first ask whether the model needs better factual data or an external
knowledge source. If an error has a pre-commit jitter signal, investigate a
runtime repair such as NHE-Edge. NTW supplies the causal evidence for the first
category; it is not a standalone detector.

## Why the four quiet Gemma cases matter

Edge found four African-capital hallucinations with no usable pre-commit spike.
NTW gives that ceiling a controlled mechanistic explanation: a fact can be
stored confidently in the parameters and therefore look smooth until it is
wrong. The result is a plausible mechanism for those cases, not a direct proof
that Gemma's four cases have the same cause. Direct tests on Gemma, Claude, or
another production model are the next step.

This is the value of the proof-of-concept: a short, reproducible experiment
establishes the category and its signature without claiming that every
Transformer has identical dynamics or that a toy model has reproduced a
large-model case exactly.

## Falsification and scope

The strongest alternative to the void explanation is mere ignorance. We tested
four countries with zero training examples; their mean confidence is 0.548,
not 1.000. This supports the distinction between uncertainty from missing
information and a confident false memory.

The experiment is intentionally small: four layers, 40 synthetic facts, four
planted lies, and four ambiguous items. It demonstrates that the mechanism
exists and is causally controllable in a clean setting. It does not estimate
how often the mechanism occurs in Gemma or Claude, and it does not claim that
jitter is a universal detector of confident lies. The same signature should be
tested directly on production models before extending the taxonomy.

## Paraphrase robustness: a useful boundary

We also tested whether paraphrase agreement can identify planted lies without
ground truth. Agreement was 0.104 for correct items, 0.167 for planted lies,
and 0.167 for ambiguous items. The held-out phrasings broke all groups because
the model was trained on a single template.

This is a valuable boundary condition: **consistency alone is not a universal
ground-truth-free detector in this setup**. NTW's reliable result is the causal
separation of planted lies from ambiguity; a production detector would need
training-data provenance, calibrated in-distribution paraphrases, or an
independent truth source.

## What NTW contributes

NTW is a compact causal proof-of-concept, not a second detector and not a
replacement for fine-tuning. It turns an observed Edge ceiling into a testable
taxonomy:

- a pre-commit drift is something a runtime method can attempt to repair;
- a static void is a fact-level problem that calls for better data, an external
  knowledge source, or a targeted fine-tune.

That separation is useful for deciding where to spend the next engineering day.
Do not spend the day tuning a runtime threshold for a quiet error until the
fact itself has been checked.

## Crossfade with NHE-Edge

- Shared: the jitter metric family, strict correctness-versus-truth scoring,
  and seeded experiments.
- Independent: separate model, data, and implementation. NTW does not import
  NHE-Edge, so its causal claim does not inherit Edge's assumptions.
- Together: Edge repairs what is visibly drifting; NTW characterizes what can
  remain silent and why that case needs a different response.

## Layout

- `parametric_proof.py` — builds the synthetic corpus, trains the tiny GPT, and
  runs the seeded probe.
- `probe_paraphrase.py` — runs the four-phrasing robustness check from saved
  weights.
- `results/parametric_proof.json` — per-group accuracy, confidence, jitter, and
  per-item traces.
- `results/paraphrase_proof.json` — paraphrase agreement and accuracy.
- `results/README.md` — file-by-file guide to the outputs.
