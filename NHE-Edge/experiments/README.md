# experiments/ — one-shot experiment drivers

Each script answers one question end-to-end (run from `NHE-Edge/`):

- `bench.py build|run|analyze` — frozen benches (hard + random), batteries, stats
- `eval_mmlu.py [--use-real] [--temporal]` — general-knowledge side effects
- `eval_prompt_baseline.py` — constrained-prompt baseline on benches
- `probe_quiet.py` — logit-lens check on quiet vs dynamic hallucinations
- `latency.py` — detector overhead + mask-apply cost (`results/latency.json`)
- `download_qwen.py` — fetch Qwen2.5-0.5B weights to `D:/hf_cache`
