"""Temporal MMLU: runtime soft mask on 200 real MMLU (not static).

Reuses runtime_rollback.gen_with_detector with mode=mask, scale=0.3, window=5,
early L19 t90 — the exact headline temporal config. Baseline generations are
reused from results/mmlu_side_effect.json (same greedy protocol); only the
temporal arm runs here (~200 items).
"""
import json
import os
import re
import sys
import time
import unicodedata

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
REPO_ROOT = os.path.dirname(BASE) if os.path.basename(BASE) == "NHE-Edge" else BASE
import runtime_rollback as rr

SAVE_DIR = os.path.join(REPO_ROOT, "models", "gemma3-1b-fp16")
TOK_DIR = os.path.join(REPO_ROOT, "models", "gemma3-1b-tokenizer")
RES = os.path.join(BASE, "results")


def norm(s):
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()


def fs(gen):
    m = re.split(r"[.\n]", gen.strip())
    return norm(m[0]) if m and m[0].strip() else None


def main():
    import torch

    prev = json.load(open(os.path.join(RES, "mmlu_side_effect.json"), encoding="utf-8"))
    assert prev.get("is_proxy") is False, "need real-MMLU side_effect file as baseline source"
    dg = json.load(open(os.path.join(RES, "detector_greedy.json"), encoding="utf-8"))
    det = dict(dg["detector_early"])
    det["threshold"] = det["threshold_t90"]
    det["threshold_key"] = "t90"
    det["mode"] = "mask"
    det["window"] = 5
    det["scale"] = 0.3
    det["sample"] = False
    det["seed"] = 0

    import eval_mmlu as em

    mmlu_items, mmlu_meta = em.try_load_mmlu_subset(n_target=200)
    assert mmlu_items and len(mmlu_items) >= 50, f"MMLU streaming failed: {mmlu_meta}"
    items = [(i, q, a, list(alts)) for i, (q, a, *alts) in enumerate(mmlu_items)]
    print(f"MMLU items: {len(items)} ({mmlu_meta})", flush=True)

    model, tok = rr.model_and_tok()
    clean = {k: v.clone() for k, v in model.state_dict().items()}
    end_id = tok.convert_tokens_to_ids("<end_of_turn>")

    out = []
    t0 = time.time()
    with torch.no_grad():
        for idx, (qid, q, ans, alts) in enumerate(items):
            text = "<start_of_turn>user\n" + q + "<end_of_turn>\n<start_of_turn>model\n"
            ids = tok(text, return_tensors="pt")["input_ids"]
            # baseline greedy (same protocol as eval_mmlu.py)
            model.load_state_dict(clean)
            gout = model.generate(ids, max_new_tokens=48, do_sample=False, use_cache=True, return_dict_in_generate=True)
            gids = gout.sequences[0][ids.shape[1]:]
            if end_id in gids:
                gids = gids[:gids.tolist().index(end_id)]
            gbase = tok.decode(gids, skip_special_tokens=True)
            # temporal arm
            model.load_state_dict(clean)
            gen_ids, feats, fired_at, n_masked, abstained = rr.gen_with_detector(model, tok, ids, dict(det))
            gen = tok.decode(gen_ids, skip_special_tokens=True)
            a = [ans] + list(alts)
            fb = fs(gbase)
            ft = fs(gen)
            b_sc = 1 if (fb and any(norm(x) in fb for x in a)) else 0
            t_sc = 1 if (ft and any(norm(x) in ft for x in a)) else 0
            out.append({"id": qid, "question": q, "answer": ans, "alts": alts,
                        "baseline_generated": gbase, "baseline_strict": b_sc,
                        "generated": gen, "strict_correct": t_sc,
                        "fired_at": fired_at, "n_masked": n_masked})
            if (idx + 1) % 20 == 0:
                print(f"  temporal-mmlu {idx+1}/{len(items)} ({time.time()-t0:.0f}s)", flush=True)
    n = len(out)
    w = sum(1 for r in out if not r["strict_correct"])
    fired = sum(1 for r in out if r["fired_at"] is not None)
    bw = sum(1 for r in out if not r["baseline_strict"])
    from scipy.stats import binomtest
    w2c = sum(1 for r in out if (not r["baseline_strict"]) and r["strict_correct"])
    c2w = sum(1 for r in out if r["baseline_strict"] and (not r["strict_correct"]))
    p = binomtest(min(w2c, c2w), w2c + c2w, 0.5, alternative="two-sided").pvalue if w2c + c2w else 1.0
    path = os.path.join(RES, "mmlu_temporal.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"n": n, "baseline_strict_wrong": bw, "temporal_strict_wrong": w,
                   "hall_baseline": round(bw / n, 4), "hall_temporal": round(w / n, 4),
                   "fired": fired, "W2C": w2c, "C2W": c2w, "mcnemar_p": p,
                   "config": {k: det[k] for k in ("type", "layer", "threshold_key", "mode", "window", "scale")},
                   "results": out}, fh, indent=1, ensure_ascii=False)
    print(f"[mmlu-temporal] baseline strict hall={bw/n:.3f} ({bw}/{n}) temporal={w/n:.3f} ({w}/{n}) fired={fired} W2C={w2c} C2W={c2w} p={p:.4f} -> {path}", flush=True)


if __name__ == "__main__":
    main()
