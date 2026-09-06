"""Latency probe: detector overhead per token + mask-apply cost + end-to-end.

Compares plain greedy generate vs gen_with_detector (mode=none: same forward
pass plus hidden-state norm math, no mask) on Gemma 3 1B CPU, plus a timed
apply_mask call. Writes results/latency.json.
"""
import json
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
REPO_ROOT = os.path.dirname(BASE) if os.path.basename(BASE) == "NHE-Edge" else BASE
import runtime_rollback as rr
import topics

RES = os.path.join(BASE, "results")
N_ITEMS = 10


def main():
    import torch

    model, tok = rr.model_and_tok()
    dg = json.load(open(os.path.join(RES, "detector_greedy.json"), encoding="utf-8"))
    det = dict(dg["detector_early"])
    det["threshold"] = det["threshold_t90"]
    det["threshold_key"] = "t90"
    det["mode"] = "none"
    det["window"] = 5
    det["scale"] = 0.0
    det["sample"] = False
    det["seed"] = 0
    items = topics.AFRICA[:N_ITEMS]
    end_id = tok.convert_tokens_to_ids("<end_of_turn>")

    plain_toks, plain_time = 0, 0.0
    det_toks, det_time = 0, 0.0
    with torch.no_grad():
        for q, ans, *alts in items:
            text = "<start_of_turn>user\n" + q + "<end_of_turn>\n<start_of_turn>model\n"
            ids = tok(text, return_tensors="pt")["input_ids"]
            t0 = time.perf_counter()
            out = model.generate(ids, max_new_tokens=48, do_sample=False, use_cache=True,
                                 return_dict_in_generate=True)
            plain_time += time.perf_counter() - t0
            plain_toks += len(out.sequences[0]) - ids.shape[1]
            t0 = time.perf_counter()
            gen_ids, feats, fired_at, n_masked, abstained = rr.gen_with_detector(model, tok, ids, dict(det))
            det_time += time.perf_counter() - t0
            det_toks += len(gen_ids)
    ms_plain = plain_time / max(plain_toks, 1) * 1000
    ms_det = det_time / max(det_toks, 1) * 1000

    t0 = time.perf_counter()
    n_masked = rr.apply_mask(model, rr.MASK, 0.3)
    mask_ms = (time.perf_counter() - t0) * 1000

    out = {"model": "gemma3-1b-fp16", "device": "cpu", "n_items": N_ITEMS,
           "plain_ms_per_token": round(ms_plain, 1),
           "detector_ms_per_token": round(ms_det, 1),
           "overhead_ms_per_token": round(ms_det - ms_plain, 1),
           "overhead_pct": round(100 * (ms_det - ms_plain) / ms_plain, 1),
           "apply_mask_ms": round(mask_ms, 1), "n_masked": n_masked,
           "per_item_s_plain": round(plain_time / N_ITEMS, 2),
           "per_item_s_detector": round(det_time / N_ITEMS, 2)}
    with open(os.path.join(RES, "latency.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    main()
