# NHE-NTW results - what is what

- `parametric_proof.json` - the whole experiment: seed, model dims, per-group
  (correct / planted / ambiguous) accuracy-vs-truth, mean preamble jitter, mean
  confidence, plus per-country rows (city answered, truth, correct?, confidence,
  jitter, per-layer preamble max). Produced by `../parametric_proof.py`
  (seed 7, ~2h CPU in chunks, checkpoints at `parametric_ckpt.pt`).
- `parametric_weights.pt` - final trained weights (reprobes without retraining).
- `parametric_ckpt.pt` - resume checkpoint (gitignored if large; keep local).
