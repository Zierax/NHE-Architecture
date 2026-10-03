"""Run the single-task benchmark end to end: baseline and/or runtime intervention.

DESIGN CONSTRAINT: this runner does NOT modify any existing pipeline file. It
imports `gen_with_detector` and `apply_mask` from `core/runtime_rollback.py` and
drives them over the merged 463-item benchmark, so the measured intervention is
byte-identical to the one that produced the committed Edge numbers. Changing the
engine would invalidate the comparison against the existing results.

WHAT IT PRODUCES, per arm:
  - results/eval_single_task_{tag}.json with, for every item, the raw generation,
    the strict word-boundary score, the legacy substring score, the commit
    position, and whether the detector fired and when.
  - The summary carries both metrics side by side so a reader can see whether the
    two disagree rather than being handed one number.

MODES
  none    no intervention (baseline for this benchmark)
  mask    soft-scaled wrong-commit neurons when the early detector fires in the
          first `window` tokens, scale from --scale
  abstain refuse instead of repairing

The mask is the committed one (`results/mask_k32_midwrong.json`) unless
--mask is given. If the file is absent the run refuses instead of silently
running an unarmed arm, because a "mask" arm without a mask is just the baseline
under a misleading name.

Usage:
  python core/run_single_task.py --arm none
  python core/run_single_task.py --arm mask --layer 19 --window 5 --scale 0.3
  python core/run_single_task.py --arm none --samples 5 --seeds 1000,1001,...
"""
import argparse
import json
import os
import sys
import time

import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE = os.path.dirname(os.path.abspath(__file__))
EDGE_ROOT = os.path.dirname(BASE)
REPO_ROOT = os.path.dirname(EDGE_ROOT)
RES_DIR = os.path.join(EDGE_ROOT, "results")
ANALYSIS = os.path.join(EDGE_ROOT, "analysis")
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(EDGE_ROOT, "core"))
sys.path.insert(0, ANALYSIS)

import runtime_rollback as rr  # noqa: E402
import scoring  # noqa: E402

SAVE_DIR = os.path.join(REPO_ROOT, "models", "gemma3-1b-fp16")
TOK_DIR = os.path.join(REPO_ROOT, "models", "gemma3-1b-tokenizer")
BENCH = os.path.join(RES_DIR, "bench_single_task.json")
PROMPT_TMPL = "<start_of_turn>user\n{q}<end_of_turn>\n<start_of_turn>model\n"


def load_bench():
    if not os.path.exists(BENCH):
        sys.exit(f"missing {BENCH}; run analysis/build_single_task.py first")
    b = json.load(open(BENCH, encoding="utf-8"))
    if not b.get("items"):
        sys.exit("bench_single_task.json has no items")
    # Integrity gate: the committed file must match a fresh deterministic build,
    # otherwise a benchmark edit silently invalidates every comparison between
    # arms. The rebuild is cheap (no model involved) so it always runs.
    sys.path.insert(0, ANALYSIS)
    try:
        import build_single_task as bst
    except ImportError:
        bst = None
    if bst is not None:
        items, _prov = bst.build()
        if items != b["items"]:
            sys.exit(
                "bench_single_task.json does NOT match a fresh build of topics.py.\n"
                "topics.py or the merge logic changed after this bench was written.\n"
                "Rebuild with: python analysis/build_single_task.py\n"
                "Refusing to run: comparing arms across different benchmarks is invalid.")
    return b


def commit_position(tok, gen):
    """0-indexed generated-token index where the answer span begins.

    Same convention as bench.py/verify_timing.py: after a '**' marker if present,
    otherwise the first content token after the first 'is'.
    """
    ids = tok(gen, add_special_tokens=False)["input_ids"]
    toks = [tok.decode([t]) for t in ids]
    if not toks:
        return None
    for k, t in enumerate(toks):
        s = t.strip()
        if s == "**":
            return k + 1
        if s.startswith("**") and len(s) > 2:
            return k
    for k, t in enumerate(toks):
        if t.strip().lower() == "is":
            j = k + 1
            while j < len(toks) and toks[j].strip() in ("", "**"):
                j += 1
            return j if j < len(toks) else None
    return 0


def main():
    ap = argparse.ArgumentParser(description="Single-task benchmark runner.")
    ap.add_argument("--arm", default="none", choices=["none", "mask", "abstain"])
    ap.add_argument("--mask", default=os.path.join(RES_DIR, "mask_k32_midwrong.json"))
    ap.add_argument("--layer", type=int, default=19)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--scale", type=float, default=0.3)
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the detector threshold; default reads detector_greedy.json")
    ap.add_argument("--threshold-key", default="t90")
    ap.add_argument("--samples", type=int, default=1, help="1 = greedy only")
    ap.add_argument("--seeds", default="1000,1001,1002,1003,1004,1005")
    ap.add_argument("--limit", type=int, default=None, help="debug: first N items")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--checkpoint-every", type=int, default=25,
                    help="flush partial results every N items so a long CPU run "
                         "can resume instead of restarting (default 25)")
    ap.add_argument("--resume", action="store_true",
                    help="continue from an existing --out partial file")
    a = ap.parse_args()

    bench = load_bench()
    items = bench["items"]
    if a.limit:
        items = items[:a.limit]

    if a.arm in ("mask", "abstain") and not os.path.exists(a.mask):
        sys.exit(f"mask file not found: {a.mask}\n"
                 f"An armed arm without a mask is the baseline under a wrong name; "
                 f"generate it with core/run_experiment.py or pass --arm none.")

    thr = a.threshold
    if thr is None:
        dg_path = os.path.join(RES_DIR, "detector_greedy.json")
        if not os.path.exists(dg_path):
            sys.exit(f"missing {dg_path}; run core/runtime_rollback.py fit_greedy, "
                     f"or pass --threshold explicitly")
        dg = json.load(open(dg_path, encoding="utf-8"))
        key = f"threshold_{a.threshold_key}"
        if key not in dg["detector_early"]:
            sys.exit(f"{key} not in detector_greedy.json; have "
                     f"{[k for k in dg['detector_early'] if k.startswith('threshold')]}")
        thr = float(dg["detector_early"][key])
    if not (thr > 0):
        sys.exit(f"threshold must be positive, got {thr}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(SAVE_DIR, dtype=torch.float16,
                                                 attn_implementation="eager")
    tok = AutoTokenizer.from_pretrained(TOK_DIR)
    print(f"model loaded in {time.time()-t0:.1f}s "
          f"(fp16, {len(bench['items'])} bench items)", flush=True)

    # gen_with_detector mutates the MLP weights in place via apply_mask and never
    # restores them. Cloning the FULL state dict costs a second copy of the model
    # (~2.7 GB) and OOMs on this CPU box; instead only the tensors apply_mask can
    # touch are snapshotted. apply_mask scales exactly three weights per masked
    # neuron (down_proj column, up_proj row, gate_proj row), so restoring those
    # tensors restores the model to its clean state exactly, for a few MB.
    mask_state = {}
    if a.arm in ("mask", "abstain"):
        with open(a.mask, encoding="utf-8") as fh:
            mask_items = json.load(fh)["items"]
        wanted = set()
        for e in mask_items:
            wanted.add((int(e["layer"]), int(e["unit"])))
        sd = model.state_dict()
        for (l, u) in sorted(wanted):
            mlp = model.model.layers[l].mlp
            prefix = f"model.layers.{l}.mlp"
            for name, tensor in (
                ("down_proj", mlp.down_proj.weight),
                ("up_proj", mlp.up_proj.weight),
                ("gate_proj", mlp.gate_proj.weight),
            ):
                key = f"{prefix}.{name}.weight"
                if key not in mask_state:
                    mask_state[key] = tensor.detach().clone()
        print(f"mask snapshot: {len(mask_state)} tensors "
              f"({sum(v.numel() * v.element_size() for v in mask_state.values()) / 1e6:.1f} MB) "
              f"for {len(wanted)} neurons", flush=True)

    def restore_clean():
        if not mask_state:
            return
        sd = model.state_dict()
        with torch.no_grad():
            for key, tensor in mask_state.items():
                sd[key].copy_(tensor)

    seeds = [int(s) for s in a.seeds.split(",") if s.strip()] if a.samples > 1 else [None]

    tag = a.tag or (f"{a.arm}_L{a.layer}_w{a.window}_sft{a.scale}"
                    + (f"_s{a.samples}" if a.samples > 1 else ""))
    out = a.out or os.path.join(RES_DIR, f"eval_single_task_{tag}.json")

    # Resume: a full 463-item armed arm is a multi-hour CPU run, so partial
    # results are flushed to --out and reused. A checkpoint is only honoured when
    # the arm configuration matches exactly; otherwise continuing would mix arms.
    results = []
    if a.resume and os.path.exists(out):
        try:
            prior = json.load(open(out, encoding="utf-8"))
            same_cfg = (prior.get("arm") == a.arm and prior.get("layer") == a.layer
                        and prior.get("window") == a.window
                        and prior.get("threshold") == thr
                        and prior.get("samples") == a.samples)
            if same_cfg and prior.get("results"):
                results = prior["results"]
                print(f"resume: {len(results)} items already done, "
                      f"continuing from item {len(results)}", flush=True)
            elif prior.get("results"):
                sys.exit(f"refusing to resume: {out} was written with a different "
                         f"arm configuration (arm={prior.get('arm')} "
                         f"L{prior.get('layer')} w{prior.get('window')} "
                         f"thr={prior.get('threshold')} samples={prior.get('samples')})")
        except (OSError, json.JSONDecodeError):
            print(f"resume: {out} unreadable, starting fresh", flush=True)

    done_ids = {r["id"] for r in results}
    pending = [it for it in items if it["id"] not in done_ids]
    if not pending:
        print("nothing to do: all items already present", flush=True)
    t0 = time.time()
    for idx, it in enumerate(pending):
        det = {"type": "jump_gt", "layer": a.layer, "threshold": thr,
               "threshold_key": f"{a.threshold_key}@{a.layer}/w{a.window}",
               "mode": a.arm, "window": a.window,
               "scale": a.scale if a.arm == "mask" else 0.0,
               "sample": a.samples > 1, "seed": 0}
        q = it["question"]
        answers = it["answers"]
        text = PROMPT_TMPL.format(q=q)
        ids = tok(text, return_tensors="pt")["input_ids"]
        gens, fired_ats, abstained_flags = [], [], []
        for si, seed in enumerate(seeds):
            det_i = dict(det)
            if seed is not None:
                det_i["seed"] = seed
            restore_clean()
            with torch.no_grad():
                gen_ids, feats, fired_at, n_masked, abstained = rr.gen_with_detector(
                    model, tok, ids, det_i)
            gens.append(tok.decode(gen_ids, skip_special_tokens=True))
            fired_ats.append(fired_at)
            abstained_flags.append(bool(abstained))
        # greedy arm = first sample
        gen = gens[0]
        s_strict = scoring.strict(gen, answers)
        s_sub = scoring.substring(gen, answers)
        if a.samples > 1:
            all_strict = [scoring.strict(g, answers) for g in gens]
            majority = int(sum(all_strict) > a.samples / 2)
            per_draw = sum(all_strict) / len(all_strict)
        else:
            all_strict, majority, per_draw = [s_strict], s_strict, float(s_strict)
        results.append({
            "id": it["id"], "question": q, "answers": answers,
            "primary": it["primary"], "family": it["family"],
            "source_topics": it["source_topics"],
            "generated": gen, "samples": gens if a.samples > 1 else None,
            "correct_strict": s_strict, "correct_substring": s_sub,
            "correct_strict_all_draws": all_strict,
            "majority_correct": majority, "per_draw_rate": round(per_draw, 4),
            "commit_pos": commit_position(tok, gen),
            "fired_at": fired_ats[0], "fired_at_all": fired_ats,
            "abstained": abstained_flags[0],
        })
        if (idx + 1) % 25 == 0:
            print(f"  {idx+1}/{len(pending)} ({time.time()-t0:.0f}s)", flush=True)
        if a.checkpoint_every and (idx + 1) % a.checkpoint_every == 0:
            with open(out, "w", encoding="utf-8", newline="\n") as fh:
                json.dump({"arm": a.arm, "layer": a.layer, "window": a.window,
                           "scale": a.scale, "threshold": thr,
                           "threshold_key": a.threshold_key,
                           "samples": a.samples, "n": len(items),
                           "complete": False, "results": results}, fh,
                          indent=1, ensure_ascii=False)
            print(f"  checkpoint: {len(results)} items -> {os.path.basename(out)}",
                  flush=True)

    n = len(results)
    if n != len(items):
        sys.exit(f"internal error: collected {n} results for {len(items)} items; "
                 f"resume with --resume --out {out}")
    strict_wrong = [r for r in results if not r["correct_strict"]]
    strict_rate = len(strict_wrong) / n
    sub_wrong = [r for r in results if not r["correct_substring"]]
    disagree = [r for r in results if r["correct_strict"] != r["correct_substring"]]
    summary = {
        "arm": a.arm, "layer": a.layer, "window": a.window, "scale": a.scale,
        "threshold": thr, "threshold_key": a.threshold_key,
        "samples": a.samples, "n": n, "complete": True,
        "strict_wrong": len(strict_wrong), "strict_hallucination_rate": round(strict_rate, 4),
        "substring_wrong": len(sub_wrong),
        "substring_hallucination_rate": round(len(sub_wrong) / n, 4),
        "metrics_disagree": len(disagree),
        "n_fired": sum(1 for r in results if r["fired_at"] is not None),
        "majority_wrong": sum(1 for r in results if not r["majority_correct"]),
        "scoring": bench["scoring"],
        "results": results,
    }
    if a.samples > 1:
        summary["per_draw_hallucination_rate"] = round(
            sum(1 - r["per_draw_rate"] for r in results) / n, 4)

    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(summary, fh, indent=1, ensure_ascii=False)

    print(f"\narm={a.arm} L{a.layer} w{a.window} scale={a.scale} thr={thr:.1f}")
    print(f"  n={n} fired={summary['n_fired']}")
    print(f"  STRICT  wrong={len(strict_wrong)}  rate={strict_rate:.4f}")
    print(f"  SUBSTR  wrong={len(sub_wrong)}  rate={summary['substring_hallucination_rate']:.4f}")
    print(f"  metrics disagree on {len(disagree)} item(s)")
    print(f"  -> {out}")
    for r in disagree:
        print(f"     [{r['id']}] {r['question']}: strict={r['correct_strict']} "
              f"substr={r['correct_substring']} :: {r['generated'][:70]!r}")


if __name__ == "__main__":
    main()
